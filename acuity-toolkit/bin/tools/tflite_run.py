import os
import numpy as np
import tensorflow
main_version = tensorflow.__version__.split('.')[0]
if int(main_version) == 2:
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
from argparse import ArgumentParser

def arguments():
    parser = ArgumentParser(description='TFLite Runner')
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='TFLite model to be executed',
    )
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='TFLite model input file, each input path split by space',
    )
    parser.add_argument(
        '--input-mean',
        help='Input channel mean value list, every value split by space. Each mean value list split by #'
             'Each entry in the list should match an entry in \'--input\'',
    )
    parser.add_argument(
        '--input-std',
        help='Input standard deviation. Each std split by #'
             'Each entry in the list should match an entry in \'--input\'',
    )
    parser.add_argument(
        '--dump-float',
        action='store_true',
        help='Force dump the output tensor as float32',
    )
    args = parser.parse_args()
    return args

def process_npy(input_detail, input_file):
    _fdata = np.load(input_file)
    if _fdata.dtype == np.float32 and input_detail['dtype'] != np.float32:
        scale = input_detail['quantization'][0]
        zp = input_detail['quantization'][1]
        data = np.rint(_fdata / scale).astype(input_detail['dtype']) + zp
    else:
        data = _fdata
    return data

def process_qtensor(input_detail, input_file):
    data = np.loadtxt(input_file, dtype=input_detail['dtype'])
    data = np.reshape(data, input_detail['shape'])
    return data

def process_tensor(input_detail, input_file):
    is_qtensor = input_file.endswith('.qnt.tensor')
    if is_qtensor:
        data = process_qtensor(input_detail, input_file)
    else:
        _fdata = np.loadtxt(input_file, dtype=np.float32)
        _fdata = np.reshape(_fdata, input_detail['shape'])
        if input_detail['dtype'] != np.float32:
            scale = input_detail['quantization'][0]
            zp = input_detail['quantization'][1]
            data = np.rint(_fdata / scale).astype(input_detail['dtype']) + zp
        else:
            data = _fdata
    return data

def process_image(input_detail, input_file, mean = None, std = None):
    height = input_detail['shape'][1]
    width = input_detail['shape'][2]
    dtype = input_detail['dtype']

    # decode the image file
    img_tensor = tf.io.decode_image(tf.io.read_file(input_file))
    img = img_tensor.numpy()
    img_height = img.shape[0]
    img_width = img.shape[1]
    if img_height != height or img_width != width:
        img_tensor = tf.image.resize(img_tensor, (height, width))
        img = img_tensor.numpy()
    _fdata = np.expand_dims(img, axis=0).astype(np.float32)

    # process the mean and std
    if mean is not None:
        mean = np.array([float(m) for m in mean.split(' ')])
        _fdata = _fdata - mean
    if std is not None:
        _std = float(std)
        _fdata = _fdata / _std

    # Convert float value to dtype value
    if dtype != np.float32:
        scale = input_detail['quantization'][0]
        zp = input_detail['quantization'][1]
        data = np.rint(_fdata / scale).astype(dtype) + zp
    else:
        data = _fdata.astype(dtype)
    return data

def show_top5(data, n = 5):
    res = data
    res = np.reshape(res, [-1])
    idx = np.argsort(res)[::-1]
    # Top 5
    print("Show Top 5")
    if len(idx) < n:
        n = len(idx)
    for i in idx[:n]:
        print('{}: {}'.format(i, res[i]))

def save_output(output_detail, data):
    shape = [str(s) for s in data.shape]
    name = output_detail['name'].replace(':', '_').replace('@', '').replace('/', '_')
    filename = name + '_'.join(shape) + '.tensor'
    print("Dump result to file {}".format(filename))
    data.tofile(filename, '\n')

def post_process_output(interpreter, args):
    output_details = interpreter.get_output_details()
    for idx in range(len(output_details)):
        data = interpreter.get_tensor(output_details[idx]['index'])
        if args.dump_float and output_details[idx]['dtype'] != np.float32:
            scale = output_details[idx]['quantization'][0]
            zp = output_details[idx]['quantization'][1]
            _data = data.astype(np.int32)
            data = ((_data - zp) * scale).astype(np.float32)
        save_output(output_details[idx], data)
        show_top5(data)

def pre_process_input(interpreter, args):
    input_files = args.input.split(' ')
    input_details = interpreter.get_input_details()
    if len(input_files) != len(input_details):
        raise ValueError("This model need {} inputs, but model input file number is {}".
             format(len(input_details), len(input_files)))

    input_data = list()
    if args.input_mean is not None:
        mean_list = args.input_mean.split('#')
    else:
        mean_list = None
    if args.input_std is not None:
        std_list = args.input_std.split('#')
    else:
        std_list = None
    for idx in range(len(input_files)):
        shuffix = os.path.splitext(input_files[idx])[-1]
        if shuffix == '.jpg' or shuffix == '.jpeg' or shuffix == '.bmp':
            if mean_list is not None:
                mean = mean_list[idx]
            else:
                mean = None
            if std_list is not None:
                std = std_list[idx]
            else:
                std = None
            data = process_image(input_details[idx], input_files[idx], mean, std)
        elif shuffix == '.tensor':
            data = process_tensor(input_details[idx], input_files[idx])
        elif shuffix == '.npy':
            data = process_npy(input_details[idx], input_files[idx])
        elif shuffix == '.qtensor':
            data = process_qtensor(input_details[idx], input_files[idx])
        else:
            raise ValueError("TFLite Runner can't handle ({}) input file type".format(shuffix))
        input_data.append(data)

    return input_data

def inference(interpreter, inputs):
    input_details = interpreter.get_input_details()

    for idx in range(len(input_details)):
        interpreter.set_tensor(input_details[idx]['index'], inputs[idx])

    # Inference
    interpreter.invoke()

def load_model(model):
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    return interpreter

if __name__ == '__main__':
    args = arguments()

    interpreter = load_model(args.model)
    inputs = pre_process_input(interpreter, args)
    inference(interpreter, inputs)
    post_process_output(interpreter, args)
