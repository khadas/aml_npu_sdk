import os
import sys
import tensorflow
main_version = tensorflow.__version__.split('.')[0]
if int(main_version) == 2:
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
else:
    import tensorflow as tf
import  numpy as np
from tensorflow.python.framework import graph_util

flags = tf.app.flags
flags.DEFINE_string('tf_pb', '', 'Tensorflow PB or TFlite file path')

# seperated by space, like './dog_bike_car_300x300.jpg'
flags.DEFINE_string('input_images',
                    '',
                    'Input images file path, each image path split by space')

# seperated by space, like 'FeatureExtractor/MobilenetV2/MobilenetV2/input:0'
flags.DEFINE_string('input_tensors',
                    '',
                    'Input tensor names, each tensor split by space, with port number')

flags.DEFINE_string('input_size_list',
                    '',
                    'Set each input tensor size list, each input size split by #, each size split by ,'
                              'Each entry in the list should match an entry in \'--input_tensors\'')
flags.DEFINE_string('mean_values',
                    None,
                    'mean values parameter, comma-separated list of doubles,'
                              'Each entry in the list should match an entry in \'--input_tensors\'')

flags.DEFINE_string('std_values',
                    None,
                    'std values parameter, comma-separated list of doubles,'
                              'Each entry in the list should match an entry in \'--inputs\'')

# seperated by space, 'concat:0 concat_1:0'
flags.DEFINE_string('output_tensors',
                    '',
                    'Output tensor name list, each tensor split by space, with port number')
FLAGS = flags.FLAGS

# Note:
# if run pb file, please specify: --tf_pb,--input_images,--input_tensors,--input_size_list,
#                                 --mean_values, --std_values, --output_tensors
# if run tflite file, please specify: --tf_pb,--input_images

def run_tflite():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=FLAGS.tf_pb)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # change the following line to feed into your own data.
    img_npy = None
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            # Decode jpeg image to numpy
            with tf.gfile.GFile(FLAGS.input_images, "rb") as f:
                image_tensor = f.read()
            image_tensor = tf.image.decode_image(image_tensor)
            img_npy = sess.run(image_tensor)
            img_npy = np.reshape(img_npy, input_shape)

    img_npy.tofile('tflite_input.tensor', '\n')

    img_npy = img_npy.astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], img_npy)

    interpreter.invoke()
    for i in range(0, len(output_details)):
        output_data = interpreter.get_tensor(output_details[i]['index'])
        # print(output_details[i]['name'])
        output_data.tofile('tflite_output_{}.tensor'.format(output_details[i]['name']).replace('/', '_'), '\n')
    print("Run tflite done.")

def run_pb():
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()

        output_names_with_port = list([] if FLAGS.output_tensors == '' else FLAGS.output_tensors.split(' '))
        output_names = []
        for item in output_names_with_port:
            output_names.append(item.split(':')[0])

        with tf.gfile.GFile(FLAGS.tf_pb, "rb") as f:
            graph_def.ParseFromString(f.read())
            if len(output_names) != 0:
                graph_def = graph_util.extract_sub_graph(graph_def, output_names)
            _ = tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # convert parameters to list
            input_images = list(FLAGS.input_images.split(' '))
            input_tensors = list(FLAGS.input_tensors.split(' '))
            mean_values = list()
            if FLAGS.mean_values != None:
                mean_values = [float(mean) for mean in FLAGS.mean_values.split(',')]
                if len(mean_values) != len(input_tensors):
                    print('length of mean_values is not equal to input_tensors')
                    sys.exit(-1)
            std_values = list()
            if FLAGS.std_values != None:
                std_values = [float(std) for std in FLAGS.std_values.split(',')]
                if len(std_values) != len(input_tensors):
                    print('length of std_values is not equal to input_tensors')
                    sys.exit(-1)
            if len(mean_values) != len(std_values):
                print('length of mean_values is not equal to std_values')
                sys.exit(-1)
            input_size_list = list()
            if FLAGS.input_size_list != None:
                for sz in FLAGS.input_size_list.split('#'):
                    input_size_list.append([int(x) for x in sz.split(',')])
                if len(input_size_list) != 0 and len(input_size_list) != len(input_tensors):
                    print('length of input_size_list is not equal to input_tensors')
                    sys.exit(-1)
            tf_inputs = []
            img_npys = []
            for i in range(len(input_tensors)):
                input = sess.graph.get_tensor_by_name(input_tensors[i])
                tf_inputs.append(input)

                img_npy = None

                # Decode input image to numpy
                ext = os.path.splitext(os.path.split(input_images[i])[1])[1]
                if ext in ['.jpg','.png','.bmp']:
                    with tf.gfile.GFile(input_images[i], "rb") as f:
                        image_tensor = f.read()
                        image_tensor = tf.image.decode_image(image_tensor)
                        img_npy = sess.run(image_tensor)
                elif ext in ['.npy']:
                    img_npy = np.load(input_images[i], allow_pickle=True)
                elif ext in ['.tensor', '.txt']:
                    img_npy = np.fromfile(input_images[i], sep='\n')

                # get input shape for tensor
                input_shape = None
                input_dims = -1
                if input_size_list is not None:
                    input_shape = input_size_list[i]
                    input_dims = len(input_shape)
                else:
                    print("Warning: Try to get input shape by tensorflow, may be incorrect!")
                    try:
                        input_dims = len(input.shape)
                        if input_dims != -1:
                            input_shape = input.shape
                    except:
                        pass

                # reshape input tensor
                if input_shape is not None:
                    img_npy = np.reshape(img_npy, input_shape)
                else:
                    print("Error: Couldn't get correct input shape for tensor {}".format(input_tensors[i]))
                    sys.exit(-1)

                # Image Preprocess
                mean_value = 0
                std_value = 1
                if mean_values:
                    mean_value = mean_values[i]
                if std_values:
                    std_value = std_values[i]
                img_npy = (img_npy - mean_value) / std_value

                img_npys.append(img_npy)

                # Save pre-processed tensor to file
                img_npy.tofile('pb_input_{}.tensor'.format(input_tensors[i]).replace('/','_').replace(':','_'), '\n')

            # save output tensors
            for output_name_with_port in output_names_with_port:
                output = sess.graph.get_tensor_by_name(output_name_with_port)
                feed_dict = {}
                for i in range(len(tf_inputs)):
                    feed_dict[tf_inputs[i]]=img_npys[i]
                out = sess.run(output, feed_dict=feed_dict)

                # convert to NCHW layout data.
                # out = np.transpose(out, [0, 3, 1, 2])
                out.tofile('pb_output_{}.tensor'.format(output_name_with_port).replace('/','_').replace(':','_'), '\n')
                #np.save('pb_{}.npy'.format(output_name).replace('/','_'), out)

            print("Run pb done.")

def main(_):
    file_path = FLAGS.tf_pb
    file_ext = file_path.split('.')[-1]
    if file_ext == 'pb':
        run_pb()
    elif file_ext == 'tflite':
        run_tflite()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    tf.app.run()