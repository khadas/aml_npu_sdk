#!/usr/bin/env python3

import sys
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

def tensor_2_npy(tensor_file, shape, npy_name=None, platform='caffe'):
    t = np.fromfile(tensor_file, sep='\n')
    t = np.reshape(t, shape)
    if platform == 'caffe':
        t = np.transpose(t, [1, 2, 0])
    if npy_name == None:
        npy_name = tensor_file + '.npy'
    np.save(npy_name, t.astype(np.float32))

def img_2_npy(img_file, npy_name=None ):
    image_tensor = tf.io.read_file(img_file)
    image_tensor = tf.io.decode_image(image_tensor)
    img_npy = image_tensor.numpy()
    if npy_name == None:
        npy_name = img_file + '.npy'
    np.save(npy_name, img_npy)

def main(argv):
    options = ArgumentParser(description='Tensor convert to numpy.')
    options.add_argument('--action', help='Provide [img2npy | tensor2npy]', default='tensor2npy')
    options.add_argument('--src-platform', help='Provide [tensorflow | caffe] only use in tensor2npy action', default='caffe')
    options.add_argument('--input-file', help='Tensor file or Image file path')
    options.add_argument('--shape', help='Tensor origin shape, only use for tensor2npy action, use space to split dim value')
    options.add_argument('--numpy-data-file', help='Output numpy file name, extension is .npy', default='acuity_input.npy')
    args = options.parse_args()
    if args.action == 'tensor2npy':
        tensor_2_npy(args.input_file, list(map(int, args.shape.split(' '))), npy_name=args.numpy_data_file, platform = args.src_platform)
    elif args.action == 'img2npy':
        img_2_npy(args.input_file, npy_name=args.numpy_data_file)


if __name__ == '__main__':
    main(sys.argv)
