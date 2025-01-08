#!/usr/bin/env python3

import tensorflow
main_version = tensorflow.__version__.split('.')[0]
if int(main_version) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
else:
    import tensorflow as tf

import os
import shutil
import signal
import tempfile
import sys
from tensorflow.python.framework import graph_util

flags = tf.app.flags
flags.DEFINE_string('tf_pb', '', 'Tensorflow PB file path')
flags.DEFINE_string('output_tensor', '', 'Output tensor name list, without port number')#'concat concat_1'
flags.DEFINE_string('ip_addr', '', 'Tensorboard Serving IP, the IP Tensorboard running on')#'192.168.32.21'
FLAGS = flags.FLAGS

tmp_graph_dir = tempfile.mkdtemp()
print(sys.path[0], tmp_graph_dir)

def exit(signum, frame):
    if os.path.exists(tmp_graph_dir):
        shutil.rmtree(tmp_graph_dir)
        print(tmp_graph_dir, 'Removed')
def main(_):
    with tf.Graph().as_default():
        import_graph_def = tf.GraphDef()

        output_names = list([] if FLAGS.output_tensor == '' else FLAGS.output_tensor.split(' '))
        with tf.gfile.GFile(FLAGS.tf_pb, "rb") as f:
            import_graph_def.ParseFromString(f.read())
            if len(output_names) != 0:
                import_graph_def = graph_util.extract_sub_graph(import_graph_def, output_names)
            tf.import_graph_def(import_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(logdir=tmp_graph_dir, graph=sess.graph)
            train_writer.close()

    tensorboard_cmd = 'tensorboard --logdir ' + tmp_graph_dir
    tensorboard_cmd += ' --port ' + '0'
    if FLAGS.ip_addr != '':
        tensorboard_cmd += ' --host ' + FLAGS.ip_addr
    else:
        print("***Please try to specify --ip_addr if you can't visit Tensorboard content via IE***")
    os.system(tensorboard_cmd)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)
    tf.app.run()
