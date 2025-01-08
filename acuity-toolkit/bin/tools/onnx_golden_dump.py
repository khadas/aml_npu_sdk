#!/usr/bin/env python
# coding: utf-8
import numpy as np
import onnx
from acuitylib.onnx_ir.onnx_numpy_backend import onnx_backend as backend
#from onnx_tf import backend
#from onnx_tf import frontend
#import onnx_tf
import os
from onnx import numpy_helper
import sys
from argparse import ArgumentParser
from onnx import helper, shape_inference

#load onnx model
def loadOnnxModel(path):
    model = onnx.load(path)
    return model

#get the node from model
def getNode(node_name, model):
    for id, node in enumerate(model.graph.node):
        if node.name == node_name:
            return node
    print('Can\'t find the {}.'.format(node_name))

#remove some attribute, which the backend not support
def remove_attr(node, attrs):
    to_delete_attr = None
    for attr_name in attrs:
        for attr in node.attribute:
            if attr.name == attr_name:
                to_delete_attr = attr
                break
        if to_delete_attr is not None:
            node.attribute.remove(to_delete_attr)

#dump the node's outputs golden
def dumpNodeOutputs(node, ml_blob_tbl, ml_in_tensor_tbl, dump_path):
    inputs = list()
    for i in node.input:
        npy_input = os.path.join(dump_path, i.replace('/', '_') + '.npy')
        if i in ml_blob_tbl:
            np_data = onnx.numpy_helper.to_array(ml_blob_tbl[i])
            #np_data.flags['WRITEABLE'] = True
            inputs.append(np_data)
            continue
        elif os.path.exists(npy_input):
            np_data = np.load(npy_input)
            inputs.append(np_data)
        else:
            if i in ml_in_tensor_tbl:
                print("Please prepare input file '{}' with shape '{}' into dump path '{}'"
                      .format(i, ml_in_tensor_tbl[i], dump_path))
                sys.exit(-1)
            assert ('Not ready node input {}'.format(i))

    outs = backend.run_node(node, inputs)
    out_id = 0
    for out, out_name in zip(outs, node.output):
        out_name = os.path.join(dump_path, out_name.replace('/', '_'))
        np.save(out_name + '.npy', out)
        if len(out.shape) == 4:
            np.transpose(out, (0, 3, 1, 2))
        sp = '_'.join([str(s) for s in out.shape])
        #out.tofile(node.op_type + '_' + str(id) + '_' + sp + '.tensor', sep='\n')
        out.tofile(out_name + '_' + str(out_id) + '_' + sp + '.tensor', sep='\n')
        out_id += 1
        print('save ', out_name)

#dump model's golden, specified node or all nodes' golden
def dumpModelGolden(model_name, node_name, dump_path):
    model = loadOnnxModel(model_name)

    ml_blob_tbl = dict()
    for tp in model.graph.initializer:
        ml_blob_tbl[tp.name] = tp

    ml_in_tensor_tbl = dict()
    for in_tensor in model.graph.input:
        ml_in_tensor_tbl[in_tensor.name] = in_tensor

    if node_name:
        print('Dump the {} node\'s outputs'.format(node_name))
        node = getNode(node_name, model)
        dumpNodeOutputs(node, ml_blob_tbl, ml_in_tensor_tbl, dump_path)
    else:
        print('Dump all the {} nodes\' outputs'.format(len(model.graph.node)))
        for id, node in enumerate(model.graph.node):
            if node.op_type == 'BatchNormalization':
                remove_attr(node, ['momentum', 'spatial'])
            if node.op_type == 'Gemm':
                remove_attr(node, ['broadcast'])
            # if node.op_type == 'Mul':
            #     remove_attr(node, ['axis', 'broadcast'])
            if node.op_type == 'Dropout':
                remove_attr(node, ['is_test'])

            dumpNodeOutputs(node, ml_blob_tbl, ml_in_tensor_tbl, dump_path)

#get the node's inputs information
def getInputsTensorValueInfo(inputs_name, model):
    in_tvi = []
    for name in inputs_name:
        for input in model.graph.input:
            if input.name == name:
                in_tvi.append(input)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                in_tvi.append(inner_output)
    return in_tvi

#get the node's outputs information
def getOutputsTensorValueInfo(outputs_name, model):
    out_tvi = []
    for name in outputs_name:
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                out_tvi.append(inner_output)
        for output in model.graph.output:
            if output.name == name:
                out_tvi.append(output)
    return out_tvi

#get hyper-parameter value
def getInitTensorsValue(inputs_name, model):
    init_tv = []
    for name in inputs_name:
        for init in model.graph.initializer:
            if init.name == name:
                init_tv.append(init)
    return init_tv

#create a single onnx model by specified node
def createNodeOnnxModel(model_name, node_name, save_type='', save_path=''):
    model = loadOnnxModel(model_name)
    if node_name:
        #onnx.checker.check_model(model)
        print('Before shape inference, the shape info is:\n{}'.format(model.graph.value_info))
        inferred_model = shape_inference.infer_shapes(model)
        #print('After shape inference, the shape info is:\n{}'.format(
        #    inferred_model.graph.value_info))
        node = getNode(node_name, inferred_model)
        inputs_name = node.input
        outputs_name = node.output
        print('in_out_name', inputs_name, outputs_name)
        in_tvi = getInputsTensorValueInfo(inputs_name, inferred_model)
        out_tvi = getOutputsTensorValueInfo(outputs_name, inferred_model)
        init_tv = getInitTensorsValue(inputs_name, inferred_model)

        print(inputs_name, in_tvi, outputs_name, out_tvi, 'init_', init_tv)
        node_name = node_name.replace('/', '_')
        graph_def = helper.make_graph(
            [node],
            node_name,
            inputs=in_tvi,
            outputs=out_tvi,
            initializer=init_tv
        )
        model_def = helper.make_model(graph_def, producer_name='onnx_'+node_name)
        onnx.checker.check_model(model_def)
        #onnx.save_model()
        path_name = save_path+'/{}.onnx'.format(node_name)
        onnx.save(model_def, path_name)

        print('{} generated successfully!'.format(path_name))

    else:
        print('Please specify the node name.')


def main():
    options = ArgumentParser(description='Dump onnx model golden.')
    options.add_argument('--action',
                         required=True,
                         help='ONNX model options, dump|create_model')
    options.add_argument('--onnx-model',
                         required=True,
                         help='ONNX model file.')
    options.add_argument('--node-name',
                         default=None,
                         help='Give the node name, which you want to dump the golden.'
                            'If not given, will dump all the nodes.')
    options.add_argument('--path',
                         default='./onnx_dump',
                         help='Input the path to save dump data.')
    args = options.parse_args()

    model_name = args.onnx_model
    node_name = args.node_name
    path = args.path
    os.path.exists(path) or os.mkdir(path)

    #dump the node's output
    #you need prepare the input npy data of which node you want dump
    #eg. if the name of node's input is cls_score_reshape, the npy data name
    #should be cls_score_reshape.npy and put it under the '--path' folder
    if args.action == 'dump':
        print('Start dump golden...')
        dumpModelGolden(model_name, node_name, path)

    #create a single onnx model by specified node
    elif args.action == 'create_model':
        print('Start create ONNX model...')
        createNodeOnnxModel(model_name, node_name, save_path=path)

    else:
        print('Please input correct arguments.')

if __name__ == '__main__':
    main()