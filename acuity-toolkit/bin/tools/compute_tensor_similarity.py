#!/usr/bin/env python3
import tensorflow
main_version = tensorflow.__version__.split('.')[0]
if int(main_version) == 2:
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
else:
    import tensorflow as tf
import numpy as np
import sys

def get_tensor(arg1,arg2):
    tensor1_name , tensor2_name = arg1,arg2
    tensor1 , tensor2 = np.loadtxt(tensor1_name),np.loadtxt(tensor2_name)
    return tensor1 , tensor2

def compute_cos_sim(a,b,dim):
    norma ,normb = tf.nn.l2_normalize(a,dim),tf.nn.l2_normalize(b,dim)
    cos_similarity = tf.subtract(tf.constant(1.0),tf.losses.cosine_distance(norma,normb,dim = dim))
    return cos_similarity

def compute_euclidean_dis(a,b):
    squart_dis = tf.square(tf.subtract(a,b))
    dis_sum = tf.reduce_sum(squart_dis)
    euclidean = tf.sqrt(dis_sum)
    return euclidean

def main(argv):
    tensor1 , tensor2 = get_tensor(sys.argv[1],sys.argv[2])
    if tensor1.shape == tensor2.shape:
        dim_now = 0
        sess = tf.Session()
        a ,b = tf.placeholder(tf.float32,shape=tensor1.shape),tf.placeholder(tf.float32,shape = tensor2.shape)
        cos_similarity = compute_cos_sim(a,b,dim_now)
        euclidean_distance = compute_euclidean_dis(a,b)
        cos_similarity_result , euclidean_distance_result = sess.run(
                                                        [cos_similarity,euclidean_distance],
                                                            feed_dict={a: tensor1,b:tensor2})
        print('euclidean_distance',round(euclidean_distance_result,6))
        print('cos_similarity',round(cos_similarity_result,6))
    else:
        print('[INPUT ERROR]please make sure input tensors have the same shape!')

if __name__=='__main__':
    #run 'python3 compute_tensor_similarity a.tensor b.tensor'
    main(sys.argv)
