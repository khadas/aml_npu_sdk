import re
import tensorflow as tf
from acuitylib.converter.tensorflow.tensorflowloader import TF_Graph_Preprocess
from acuitylib.converter.tensorflow.tf_util import TFProto_Util
from acuitylib.converter.tensorflow.nx_for_acu import NX_ACU
from argparse import ArgumentParser
from acuitylib.acuitylog import AcuityLog as al
import json
from collections import OrderedDict
from acuitylib.converter.tensor_model import IN, OUT

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

def gen_internal_flow(acu_layers):
    if len(acu_layers) == 1:
        return []
    else:
        internal_edge_in_default = list()
        for id in range(len(acu_layers) - 1):
            internal_edge_in_default.append([acu_layers[id] + ':out0', acu_layers[id+1] + ':in0'])
        return internal_edge_in_default

def prettify_rule(rule_content):
    def reconstruct_line(items, max_length=100, line_type='list'):
        indent = 4
        new_lines_items = []
        new_line = ''
        while len(items) > 0:
            cur_item = items.pop(0)
            if (len(new_line) + len(cur_item)) > max_length:
                new_lines_items.append(new_line) # append new line
                new_line = (' ' * indent + cur_item) if len(new_lines_items) > 0 else cur_item
            elif len(new_line):
                new_line = ', '.join([new_line, cur_item])
            else:
                new_line = (' ' * indent + cur_item) if len(new_lines_items) > 0 else cur_item

        if len(new_line) > 0:
            new_lines_items.append(new_line) # append new line

        if line_type == 'list':
            line = '{}[{}],'.format(prefix, ',\n'.join(new_lines_items))
        else:
            raise NotImplementedError('line type({}) is supported'.format(line_type))

        return line

    # limit line length
    new_lines = []
    lines = rule_content.split('\n')
    for line in lines:
        m = re.match(pattern=r'^("\w+":\s+)', string=line)
        if m:
            prefix = m.group(0)
            content = line.replace(prefix, '')
            if content.startswith('[['):
                items = re.findall(pattern=r'(\["\S+",\s+"\S+"\])', string=content)
                line = reconstruct_line(items, max_length=100)
            elif content.startswith('['):
                items = re.findall(pattern=r'("\S+")', string=content)
                line = reconstruct_line(items, max_length=100)
        new_lines.append(line)

    return '\n'.join(new_lines)

def main():
    options = ArgumentParser(description='Tensorflow Ruler Generate Assistant tool.')
    options.add_argument('-pbfile',
                         required=True,
                         help='Tensorflow model proto file')
    options.add_argument('-rulername',
                         default=None,
                         help='Provide a name for ruler')
    options.add_argument('-inputs',
                         default='input',
                         help='User give input nodes, include these select nodes.')
    options.add_argument('-inputsize',
                         default="1 1 1 1",
                         help='Set input tensor rank')
    options.add_argument('-outputs',
                         default='output',
                         help='User give output nodes, include thses select nodes.')
    options.add_argument('-namescope',
                         default=None,
                         help='User specify name scope to query nodes'
                         )
    options.add_argument('-acuityalias',
                         default='noop',
                         help='Set acuity layer alias in select graph')
    options.add_argument('-predefinefile',
                         default=None,
                         help='For ruler generate provide a predef file for this pb')
    options.add_argument('-rulerfile',
                         default=None,
                         help='For ruler generate provide a predef file for this pb')
    args = options.parse_args()

    if args.rulerfile == None:
        args.rulerfile = './gen_ruler_{}.json'.format(args.rulername)

    # Parse File to generate model object
    input_tensor_list = list()
    output_tensor_list = list()
    input_size_list = list()
    tp_u = TFProto_Util()
    nx_and_acu = NX_ACU('tensorflow')

    if args.namescope == None:
        input_size_list = [[int(s) for s in ins.split()] for ins in args.inputsize.split(';')]
        input_tensor_list, output_tensor_list = \
            nx_and_acu.input_output_to_tensor(args.inputs.split(), args.outputs.split())
    else:
        graph_def = tf.GraphDef()
        with open(args.pbfile, 'rb') as f:
            # Orig graph
            graph_def.ParseFromString(f.read())
        input_flows, output_flows = tp_u.query_namescope_in_out_flow(graph_def, args.namescope)
        if len(input_flows) == 0:
            al.e('The query scope donot have input tensor, please use -inputs -outputs to specify')
        if len(output_flows) == 0:
            al.e('The query scope donot have output tensor, please use -inputs -outputs to specify.')
        src_ref_slot = dict()
        for flow in input_flows:
            if flow[OUT] not in src_ref_slot:
                src_ref_slot[flow[OUT]] = 1
            else:
                src_ref_slot[flow[OUT]] += 1
        for flow in input_flows:
            if src_ref_slot[flow[OUT]] == 1:
                input_tensor_list.append(flow[IN])
            else:
                if flow[OUT] not in input_tensor_list:
                    input_tensor_list.append(flow[OUT])
        for flow in output_flows:
            if flow[OUT] not in output_tensor_list:
                output_tensor_list.append(flow[OUT])

        for in_tensor in input_tensor_list:
            input_size_list.append(list(map(int, input('Please give {} tensor shape:'.format(in_tensor)).split())))

    tfg = TF_Graph_Preprocess(args.pbfile, input_tensor_list, output_tensor_list, input_size_list, args.predefinefile)
    tfg.pre_proces()
    input_tensor_list = tfg.input_tensors
    nx_and_acu = NX_ACU('tensorflow')

    nodes_in_scan_order = \
        nx_and_acu.build_unique_graph(tfg.graph, input_tensor_list, output_tensor_list, use_alias=True)

    ruler_dict = OrderedDict()
    ruler_dict['ruler_name'] = args.rulername
    ruler_dict['src_ops_alias'] = nodes_in_scan_order
    # Build Internal Edge
    ruler_dict['src_inter_flow'] = nx_and_acu.flows(nodes_in_scan_order, use_alias=True)
    # TODO: modify it to input tensors
    ruler_dict['src_in_anchor'] = nx_and_acu.in_flows(use_alias=True)
    ruler_dict['src_out_tensor'] = nx_and_acu.out_tensors(use_alias=True)
    ruler_dict['acu_lys_alias'] = args.acuityalias.split()
    ruler_dict['src_acu_in_tensor_map'] = gen_input_map(ruler_dict['src_in_anchor'], args.acuityalias.split())
    ruler_dict['src_acu_out_tensor_map'] = gen_output_map(ruler_dict['src_out_tensor'], args.acuityalias.split())
    ruler_dict['acu_inter_flow'] = gen_internal_flow(args.acuityalias.split())
    ruler_dict['param_map'] = {acu_ly:dict() for acu_ly in args.acuityalias.split()}
    ruler_dict['blob_map'] = {acu_ly:dict() for acu_ly in args.acuityalias.split()}
    ruler_dict['priority_tip'] = 0
    ruler_dict['pre_condition'] = None

    string_result = json.dumps(ruler_dict)
    real_names_in_order = \
        NX_ACU('tensorflow').build_unique_graph(tfg.graph, input_tensor_list, output_tensor_list, use_alias=False)
    alias_real_map_str = list()
    for alia, real in zip(nodes_in_scan_order, real_names_in_order):
        alias_real_map_str.append('{}:{}'.format(alia, real))
    string_result = string_result + '\n#' + ';\n'.join(alias_real_map_str)

    for key in ruler_dict.keys():
        string_result = string_result.replace("\"{}\"".format(key), "\n\"{}\"".format(key))
    string_result = string_result.replace("\"pre_condition\": null", "\"pre_condition\": None")

    string_result = prettify_rule(string_result)

    with open(args.rulerfile, 'w') as f:
        f.write(string_result)
    print(string_result)

main()
