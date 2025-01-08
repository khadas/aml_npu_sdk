from argparse import ArgumentParser
from acuitylib.acuitylog import AcuityLog as al
import json
from collections import OrderedDict
from acuitylib.converter.onnx.onnx_tensor_model import OnnxTensorModel
from acuitylib.converter.tensor_model import OUT, IN
import onnx
from onnx import ModelProto
import copy
from acuitylib.converter.onnx.onnx_util import ONNXProto_Util as opu

class smart_graph:
    def __init__(self, onnx_model, inputs, outputs):
        # Pre_process
        # Step 1 Optimizing Graph
        onnx_model = opu.optim_model(onnx_model)
        # Step 2 Polishing Graph
        onnx_model = opu.polishing_model(onnx_model)
        # Use built-in shape inference
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        self.onnx_model = onnx_model
        self.tensor_model = OnnxTensorModel()
        self.tensor_factory_map, self.tensor_map = \
            self.tensor_model.build_model(self.onnx_model, with_node=True, with_attribute=True)
        self.input_tensors = inputs
        self.output_tensors = outputs
        # Include Tensor and TensorFactory
        self.op_alias_map = dict()
        self.tensor_alias_map = dict()
        self.op_scan_order = list()
        self.internal_flow = list()
        self.__setup_smart_graph()

    def __setup_smart_graph(self):
        in_tensors = [self.tensor_map[tid] for tid in self.input_tensors]
        out_tensors = [self.tensor_map[tid] for tid in self.output_tensors]
        op_map = dict()
        tensor_scan_order = copy.copy(out_tensors)
        tensor_foot_print = set()
        op_foot_print = set()
        while len(tensor_scan_order) > 0:
            to_scan_tensor = tensor_scan_order.pop(0)
            tensor_foot_print.add(to_scan_tensor)
            if to_scan_tensor in in_tensors:
                continue
            tensor_factory = self.tensor_model.tensor_factory(self.tensor_model.product_by(to_scan_tensor))
            if tensor_factory.name in op_foot_print:
                continue
            self.op_scan_order.append(tensor_factory.name)
            op_foot_print.add(tensor_factory.name)
            if tensor_factory.op not in op_map:
                op_map[tensor_factory.op] = 0
            else:
                op_map[tensor_factory.op] += 1
            if op_map[tensor_factory.op] == 0:
                self.op_alias_map[tensor_factory.name] = tensor_factory.op
            else:
                self.op_alias_map[tensor_factory.name] = tensor_factory.op + "_{}".format(op_map[tensor_factory.op])

            for port_id, tensor in enumerate(self.tensor_model.product_to(tensor_factory.name)):
                self.tensor_alias_map[tensor] = self.op_alias_map[tensor_factory.name] + ':out{}'.format(port_id)
            for port_id, consum_tensor in enumerate(self.tensor_model.consume_from(tensor_factory.name)):
                self.tensor_alias_map[consum_tensor] = self.op_alias_map[tensor_factory.name] + ':in{}'.format(port_id)
                flow = self.tensor_model.flow(consum_tensor, 'from')
                src_tensor = flow[OUT]
                if src_tensor not in in_tensors:
                    self.internal_flow.append(flow)
                if src_tensor not in tensor_foot_print:
                    tensor_scan_order.append(src_tensor)
        for id, in_tensor in enumerate(in_tensors):
            self.tensor_alias_map[in_tensor] = 'I_{}:out0'.format(id)

    def gen_op_scan_order(self, use_alias=False):
        if use_alias == False:
            return self.op_scan_order
        else:
            return [self.op_alias_map[op] for op in self.op_scan_order]

    def gen_internal_flow(self, use_alias=False):
        if use_alias == False:
            return self.internal_flow
        else:
            return [[self.tensor_alias_map[flow[OUT]], self.tensor_alias_map[flow[IN]]] for flow in self.internal_flow]

    def gen_in_flow(self, use_alias=False):
        in_tensors = [self.tensor_map[tid] for tid in self.input_tensors]
        flows = list()
        for in_tensor in in_tensors:
            for flow in  self.tensor_model.flow(in_tensor, 'to'):
                if self.tensor_model.provide_to(flow[IN]) not in self.op_scan_order:
                    continue
                flows.append(flow)
        if use_alias == False:
            return flows
        else:
            return [[self.tensor_alias_map[flow[OUT]], self.tensor_alias_map[flow[IN]]] for flow in flows]

    def gen_out_tensor(self, use_alias=False):
        out_tensors = [self.tensor_map[tid] for tid in self.output_tensors]
        if use_alias == False:
            return out_tensors
        else:
            return [self.tensor_alias_map[ts] for ts in out_tensors]



def gen_input_map(src_in_anchor, acu_layers):
    port_map = set()
    input_map = list()
    for flow in src_in_anchor:
        src_tensor = flow[0]
        if src_tensor in port_map:
            continue
        else:
            port = len(port_map)
            port_map.add(src_tensor)
        input_map.append([src_tensor, acu_layers[0] + ':in' + str(port)])
    return input_map

def gen_output_map(dst_out_tensor, acu_layers):
    output_map = list()
    for port, out_tensor in enumerate(dst_out_tensor):
        output_map.append([out_tensor, acu_layers[-1] + ':out' + str(port)])
    return output_map



def gen_acu_internal_flow(acu_layers):
    if len(acu_layers) == 1:
        return []
    else:
        internal_edge_in_default = list()
        for id in range(len(acu_layers) - 1):
            internal_edge_in_default.append([acu_layers[id] + ':out0', acu_layers[id+1] + ':in0'])
            return internal_edge_in_default

def main():
    options = ArgumentParser(description='Tensorflow Ruler Generate Assistant tool.')
    options.add_argument('-file',
                         required=True,
                         help='ONNX model file')
    options.add_argument('-rulername',
                         default=None,
                         help='Provide a name for ruler')
    options.add_argument('-inputs',
                         default='input',
                         help='User give input nodes, include these select nodes.')
    options.add_argument('-outputs',
                         default='output',
                         help='User give output nodes, include these select nodes.')
    options.add_argument('-acuityalias',
                         default='noop',
                         help='Set acuity layer alias in select graph')
    options.add_argument('-rulerfile',
                         default=None,
                         help='For ruler generate provide a predef file for this pb')
    args = options.parse_args()

    if args.rulerfile == None:
        args.rulerfile = './gen_ruler_{}.json'.format(args.rulername)

    # Parse File to generate model object
    input_tensor_list = list()
    output_tensor_list = list()
    model = ModelProto()
    with open(args.file, 'rb') as model_f:
        model.ParseFromString(model_f.read())
        # Use built-in shape inference
        # model = onnx.shape_inference.infer_shapes(model)
    sg = smart_graph(model, args.inputs.split(), args.outputs.split())




    ruler_dict = OrderedDict()
    ruler_dict['ruler_name'] = args.rulername
    ruler_dict['src_ops_alias'] = sg.gen_op_scan_order(use_alias=True)
    # Build Internal Edge
    ruler_dict['src_inter_flow'] = sg.gen_internal_flow(use_alias=True)
    # TODO: modify it to input tensors
    ruler_dict['src_in_anchor'] = sg.gen_in_flow(use_alias=True)
    ruler_dict['src_out_tensor'] = sg.gen_out_tensor(use_alias=True)
    ruler_dict['acu_lys_alias'] = args.acuityalias.split()
    ruler_dict['src_acu_in_tensor_map'] = gen_input_map(ruler_dict['src_in_anchor'], args.acuityalias.split())
    ruler_dict['src_acu_out_tensor_map'] = gen_output_map(ruler_dict['src_out_tensor'], args.acuityalias.split())
    ruler_dict['acu_inter_flow'] = gen_acu_internal_flow(args.acuityalias.split())
    ruler_dict['param_map'] = {acu_ly:dict() for acu_ly in args.acuityalias.split()}
    ruler_dict['blob_map'] = {acu_ly:dict() for acu_ly in args.acuityalias.split()}
    ruler_dict['priority_tip'] = 0
    ruler_dict['pre_condition'] = None
    ruler_dict['src_ops_main_version'] = None
    ruler_dict['src_ops_minior_version'] = [1, -1]

    string_result = json.dumps(ruler_dict)
    # real_names_in_order = \
    #     NX_ACU('tensorflow').build_unique_graph(tfg.graph, input_tensor_list, output_tensor_list, use_alias=False)
    alias_real_map_str = list()
    for alia, real in zip(sg.gen_op_scan_order(use_alias=True), sg.gen_op_scan_order(use_alias=False)):
        alias_real_map_str.append('{}:{}'.format(alia, real))
    string_result = string_result + '\n#' + ';'.join(alias_real_map_str)

    for key in ruler_dict.keys():
        string_result = string_result.replace("\"{}\"".format(key), "\n\"{}\"".format(key))
    string_result = string_result.replace("\"pre_condition\": null", "\"pre_condition\": None")
    string_result = string_result.replace("\"src_ops_main_version\": null", "\"src_ops_main_version\": None")
    with open(args.rulerfile, 'w') as f:
        f.write(string_result)
    print(string_result)

main()
