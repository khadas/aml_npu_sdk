# -*- mode: python -*-
import platform
import tensorflow as tf
import onnx
import torch
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# CAUTION*** use root account, PyInstaller 3.4, python 3.5 to pack acuity binary on ubuntu 16.04
# CAUTION*** use root account, PyInstaller 3.4, python 3.6 to pack acuity binary on ubuntu 18.04
# CAUTION*** use root account, PyInstaller 4.5.1, python 3.8 to pack acuity binary on ubuntu 20.04

# add hidden imports
acuity_hidden_import = ['acuitylib.converter.caffe.caffeloader',
                         'acuitylib.app.importer.import_tensorflow',
                         'acuitylib.converter.lite.tflite_loader',
                         'acuitylib.converter.darknet.convert_darknet',
                         'acuitylib.app.importer.import_onnx',
                         'acuitylib.app.importer.import_pytorch',
                         'acuitylib.onnx_ir.frontend.pytorch_frontend.pytorch_frontend',
                         'acuitylib.onnx_ir.frontend.frontend_base',
                         'acuitylib.onnx_ir.frontend.pytorch_frontend.pytorch_lower_to_onnx',
                         'acuitylib.app.importer.import_keras',
                         'acuitylib.app.exporter.ovxlib_case.export_ovxlib',
                         'acuitylib.app.exporter.timvx_case.export_timvx',
                         'acuitylib.app.exporter.tflite_case.export_tflite']
h5py_hidden_import = collect_submodules('h5py')
tf_hidden_import = []
if tf.__version__.startswith('1.'):
    pass
elif tf.__version__=='2.0.0':
    tf_hidden_import = collect_submodules('tensorflow_core')
elif tf.__version__=='2.3.0':
    tf_hidden_import = ['tensorflow.python.ops.while_v2']
else:
    pass
hidden_imports = acuity_hidden_import + h5py_hidden_import + tf_hidden_import

# add datas
tf_datas = []
if tf.__version__.startswith('1.'):
    pass
elif tf.__version__=='2.0.0':
    tf_datas = collect_data_files('tensorflow_core', subdir=None, include_py_files=True)
elif tf.__version__=='2.3.0':
    pass
else:
    pass
additional_datas = tf_datas

#add binaries for tensorflow.contrib, onnx
python_version_str = platform.python_version()
python_version = python_version_str.split('.')
machine_python_ver_major = python_version[0]
machine_python_ver_minor = python_version[1]
python_folder_name = "{}.{}".format(machine_python_ver_major,machine_python_ver_minor)
python_folder_name_squeeze_dot = "{}{}".format(machine_python_ver_major,machine_python_ver_minor)
third_binaries_tf_1_10_0_cpu=[
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/nccl/python/ops/_nccl_ops.so".format(python_folder_name), './tensorflow/contrib/nccl/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tpu/python/ops/_tpu_ops.so".format(python_folder_name), './tensorflow/contrib/tpu/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/text/python/ops/_skip_gram_ops.so".format(python_folder_name), './tensorflow/contrib/text/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/data/_dataset_ops.so".format(python_folder_name), './tensorflow/contrib/data/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/data/python/ops/__init__.py".format(python_folder_name), './tensorflow/contrib/data/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/periodic_resample/python/ops/_periodic_resample_op.so".format(python_folder_name), './tensorflow/contrib/periodic_resample/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/fused_conv/python/ops/_fused_conv2d_bias_activation_op.so".format(python_folder_name), './tensorflow/contrib/fused_conv/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/python/ops/_tensor_forest_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/python/ops/_stats_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/python/ops/_model_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/libforestprotos.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/hybrid/python/ops/_training_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/hybrid/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/rnn/python/ops/_gru_ops.so".format(python_folder_name), './tensorflow/contrib/rnn/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/rnn/python/ops/_lstm_ops.so".format(python_folder_name), './tensorflow/contrib/rnn/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/libsvm/python/ops/_libsvm_ops.so".format(python_folder_name), './tensorflow/contrib/libsvm/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/lite/python/interpreter_wrapper/_tensorflow_wrap_interpreter_wrapper.so".format(python_folder_name), './tensorflow/contrib/lite/python/interpreter_wrapper/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/lite/toco/python/_tensorflow_wrap_toco.so".format(python_folder_name), './tensorflow/contrib/lite/toco/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/factorization/python/ops/_factorization_ops.so".format(python_folder_name), './tensorflow/contrib/factorization/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/factorization/python/ops/_clustering_ops.so".format(python_folder_name), './tensorflow/contrib/factorization/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/layers/python/ops/_sparse_feature_cross_op.so".format(python_folder_name), './tensorflow/contrib/layers/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/input_pipeline/python/ops/_input_pipeline_ops.so".format(python_folder_name), './tensorflow/contrib/input_pipeline/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/resampler/python/ops/_resampler_ops.so".format(python_folder_name), './tensorflow/contrib/resampler/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/memory_stats/python/ops/_memory_stats_ops.so".format(python_folder_name), './tensorflow/contrib/memory_stats/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/proto/python/kernel_tests/libtestexample.so".format(python_folder_name), './tensorflow/contrib/proto/python/kernel_tests/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/ffmpeg/ffmpeg.so".format(python_folder_name), './tensorflow/contrib/ffmpeg/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/rpc/python/kernel_tests/libtestexample.so".format(python_folder_name), './tensorflow/contrib/rpc/python/kernel_tests/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/reduce_slice_ops/python/ops/_reduce_slice_ops.so".format(python_folder_name), './tensorflow/contrib/reduce_slice_ops/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/boosted_trees/python/ops/_boosted_trees_ops.so".format(python_folder_name), './tensorflow/contrib/boosted_trees/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/image/python/ops/_single_image_random_dot_stereograms.so".format(python_folder_name), './tensorflow/contrib/image/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/image/python/ops/_image_ops.so".format(python_folder_name), './tensorflow/contrib/image/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/image/python/ops/_distort_image_ops.so".format(python_folder_name), './tensorflow/contrib/image/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensorrt/python/ops/_trt_engine_op.so".format(python_folder_name), './tensorflow/contrib/tensorrt/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensorrt/_wrap_conversion.so".format(python_folder_name), './tensorflow/contrib/tensorrt/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/kafka/_dataset_ops.so".format(python_folder_name), './tensorflow/contrib/kafka/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/framework/python/ops/_variable_ops.so".format(python_folder_name), './tensorflow/contrib/framework/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/nearest_neighbor/python/ops/_nearest_neighbor_ops.so".format(python_folder_name), './tensorflow/contrib/nearest_neighbor/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/kinesis/_dataset_ops.so".format(python_folder_name), './tensorflow/contrib/kinesis/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/coder/python/ops/_coder_ops.so".format(python_folder_name), './tensorflow/contrib/coder/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/seq2seq/python/ops/_beam_search_ops.so".format(python_folder_name), './tensorflow/contrib/seq2seq/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/bigtable/python/ops/_bigtable.so".format(python_folder_name), './tensorflow/contrib/bigtable/python/ops/')]

third_binaries_tf_1_13_2_cpu=[
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/periodic_resample/python/ops/_periodic_resample_op.so".format(python_folder_name), './tensorflow/contrib/periodic_resample/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/resampler/python/ops/_resampler_ops.so".format(python_folder_name), './tensorflow/contrib/resampler/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/input_pipeline/python/ops/_input_pipeline_ops.so".format(python_folder_name), './tensorflow/contrib/input_pipeline/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/ffmpeg/ffmpeg.so".format(python_folder_name), './tensorflow/contrib/ffmpeg/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/hadoop/_dataset_ops.so".format(python_folder_name), './tensorflow/contrib/hadoop/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tpu/python/ops/_tpu_ops.so".format(python_folder_name), './tensorflow/contrib/tpu/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/ignite/_ignite_ops.so".format(python_folder_name), './tensorflow/contrib/ignite/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/image/python/ops/_distort_image_ops.so".format(python_folder_name), './tensorflow/contrib/image/python/ops'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/image/python/ops/_single_image_random_dot_stereograms.so".format(python_folder_name), './tensorflow/contrib/image/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/image/python/ops/_image_ops.so".format(python_folder_name), './tensorflow/contrib/image/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/reduce_slice_ops/python/ops/_reduce_slice_ops.so".format(python_folder_name), './tensorflow/contrib/reduce_slice_ops/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/coder/python/ops/_coder_ops.so".format(python_folder_name), './tensorflow/contrib/coder/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/fused_conv/python/ops/_fused_conv2d_bias_activation_op.so".format(python_folder_name), './tensorflow/contrib/fused_conv/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/bigtable/python/ops/_bigtable.so".format(python_folder_name), './tensorflow/contrib/bigtable/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensorrt/_wrap_conversion.so".format(python_folder_name), './tensorflow/contrib/tensorrt/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensorrt/python/ops/_trt_engine_op.so".format(python_folder_name), './tensorflow/contrib/tensorrt/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/boosted_trees/python/ops/_boosted_trees_ops.so".format(python_folder_name), './tensorflow/contrib/boosted_trees/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/memory_stats/python/ops/_memory_stats_ops.so".format(python_folder_name), './tensorflow/contrib/memory_stats/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/rnn/python/ops/_gru_ops.so".format(python_folder_name), './tensorflow/contrib/rnn/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/rnn/python/ops/_lstm_ops.so".format(python_folder_name), './tensorflow/contrib/rnn/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/nearest_neighbor/python/ops/_nearest_neighbor_ops.so".format(python_folder_name), './tensorflow/contrib/nearest_neighbor/python/ops'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/framework/python/ops/_variable_ops.so".format(python_folder_name), './tensorflow/contrib/framework/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/seq2seq/python/ops/_beam_search_ops.so".format(python_folder_name), './tensorflow/contrib/seq2seq/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/libsvm/python/ops/_libsvm_ops.so".format(python_folder_name), './tensorflow/contrib/libsvm/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/factorization/python/ops/_clustering_ops.so".format(python_folder_name), './tensorflow/contrib/factorization/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/factorization/python/ops/_factorization_ops.so".format(python_folder_name), './tensorflow/contrib/factorization/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/kinesis/_dataset_ops.so".format(python_folder_name), './tensorflow/contrib/kinesis/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/rpc/python/kernel_tests/libtestexample.so".format(python_folder_name), './tensorflow/contrib/rpc/python/kernel_tests/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/kafka/_dataset_ops.so".format(python_folder_name), './tensorflow/contrib/kafka/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/layers/python/ops/_sparse_feature_cross_op.so".format(python_folder_name), './tensorflow/contrib/layers/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/text/python/ops/_skip_gram_ops.so".format(python_folder_name), './tensorflow/contrib/text/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/hybrid/python/ops/_training_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/hybrid/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/libforestprotos.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/python/ops/_model_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/python/ops/_stats_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/contrib/tensor_forest/python/ops/_tensor_forest_ops.so".format(python_folder_name), './tensorflow/contrib/tensor_forest/python/ops/')]

third_binaries_tf_2_3_0_cpu=[
("/usr/local/lib/python{}/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so".format(python_folder_name), './tensorflow/core/kernels/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so".format(python_folder_name), './tensorflow/lite/experimental/microfrontend/python/ops'),
("/usr/local/lib/python{}/dist-packages/tensorflow/lite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper.so".format(python_folder_name), './tensorflow/lite/python/interpreter_wrapper/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/lite/python/optimize/_pywrap_tensorflow_lite_calibration_wrapper.so".format(python_folder_name), './tensorflow/lite/python/optimize/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/compiler/tf2tensorrt/_pywrap_py_utils.so".format(python_folder_name), './tensorflow/compiler/tf2tensorrt/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/compiler/tf2tensorrt/ops/gen_trt_ops.py".format(python_folder_name), './tensorflow/compiler/tf2tensorrt/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/compiler/tf2xla/ops/_xla_ops.so".format(python_folder_name), './tensorflow/compiler/tf2xla/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_mlir.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tfprof.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_stacktrace_handler.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_kernel_registry.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_file_io.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_record_io.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_device_lib.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tfcompile.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/data/experimental/service/_pywrap_server_lib.so".format(python_folder_name), './tensorflow/python/data/experimental/service/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_events_writer.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_py_func.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_py_exception_registry.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_python_op_gen.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_stat_summarizer.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tfe.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_tf_stack.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tf_cluster.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_op_def_registry.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tf_item.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/framework/fast_tensor_util.so".format(python_folder_name), './tensorflow/python/framework/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/profiler/internal/_pywrap_profiler.so".format(python_folder_name), './tensorflow/python/profiler/internal/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/profiler/internal/_pywrap_traceme.so".format(python_folder_name), './tensorflow/python/profiler/internal/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/autograph/impl/testing/pybind_for_testing.so".format(python_folder_name), './tensorflow/python/autograph/impl/testing/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_dtypes.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_util_port.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_quantize_training.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_debug_events_writer.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_utils.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_transform_graph.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tf32_execution.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tf_session.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_toco_api.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_checkpoint_reader.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_tf_optimizer.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_pywrap_bfloat16.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/_python_memory_checker_helper.so".format(python_folder_name), './tensorflow/python/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/keras/engine/base_layer_v1.py".format(python_folder_name), './tensorflow/python/keras/engine/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/__init__.py".format(python_folder_name), './tensorflow/python/ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/numpy_ops/__init__.py".format(python_folder_name), './tensorflow/python/ops/numpy_ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/numpy_ops/np_array_ops.py".format(python_folder_name), './tensorflow/python/ops/numpy_ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/numpy_ops/np_arrays.py".format(python_folder_name), './tensorflow/python/ops/numpy_ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/numpy_ops/np_dtypes.py".format(python_folder_name), './tensorflow/python/ops/numpy_ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/numpy_ops/np_math_ops.py".format(python_folder_name), './tensorflow/python/ops/numpy_ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/numpy_ops/np_random.py".format(python_folder_name), './tensorflow/python/ops/numpy_ops/'),
("/usr/local/lib/python{}/dist-packages/tensorflow/python/ops/numpy_ops/np_utils.py".format(python_folder_name), './tensorflow/python/ops/numpy_ops/'),
]

third_binaries_tf = []
if tf.__version__=='1.10.0':
    third_binaries_tf = third_binaries_tf_1_10_0_cpu
elif tf.__version__=='1.13.2':
    third_binaries_tf = third_binaries_tf_1_13_2_cpu
elif tf.__version__=='2.0.0':
    pass
elif tf.__version__=='2.3.0':
    third_binaries_tf = third_binaries_tf_2_3_0_cpu
else:
    pass

third_binaries_onnx_1_2_2=[
("/usr/local/lib/python{}/dist-packages/onnx/.libs/libprotobuf-487875fe.so.9.0.1".format(python_folder_name), '.'),
("/usr/local/lib/python{}/dist-packages/onnx/onnx_cpp2py_export.cpython-{}m-x86_64-linux-gnu.so".format(python_folder_name,python_folder_name_squeeze_dot), './onnx/')]

third_binaries_onnx_1_3_0=[
("/usr/local/lib/python{}/dist-packages/onnx/.libs/libprotobuf-024eded0.so.9.0.1".format(python_folder_name), '.'),
("/usr/local/lib/python{}/dist-packages/onnx/onnx_cpp2py_export.cpython-{}m-x86_64-linux-gnu.so".format(python_folder_name,python_folder_name_squeeze_dot), './onnx/')]

third_binaries_onnx_1_4_1=third_binaries_onnx_1_3_0

third_binaries_onnx_1_6_0=third_binaries_onnx_1_2_2

third_binaries_onnx_1_8_0=[
("/usr/local/lib/python{}/dist-packages/onnx/onnx_cpp2py_export.cpython-{}m-x86_64-linux-gnu.so".format(python_folder_name,python_folder_name_squeeze_dot), './onnx/')]

third_binaries_cp38_onnx_1_8_0=[
("/usr/local/lib/python{}/dist-packages/onnx/onnx_cpp2py_export.cpython-{}-x86_64-linux-gnu.so".format(python_folder_name,python_folder_name_squeeze_dot), './onnx/')]

third_binaries_onnx = []
if onnx.__version__=='1.2.2':
    third_binaries_onnx = third_binaries_onnx_1_2_2
elif onnx.__version__=='1.3.0':
    third_binaries_onnx = third_binaries_onnx_1_3_0
elif onnx.__version__=='1.4.1':
    third_binaries_onnx = third_binaries_onnx_1_4_1
elif onnx.__version__=='1.6.0':
    third_binaries_onnx = third_binaries_onnx_1_6_0
elif onnx.__version__=='1.8.0':
    if python_version_str >= '3.8.0':
        third_binaries_onnx = third_binaries_cp38_onnx_1_8_0
    else:
        third_binaries_onnx = third_binaries_onnx_1_8_0
else:
    pass

third_binaries = third_binaries_tf + third_binaries_onnx

excluded_torch_modules = []
if torch.__version__=='1.5.1':
    excluded_torch_modules = ['torch.distributions']
else:
    pass

excluded_modules = excluded_torch_modules

def default_analyze(script):
    a = Analysis([script],
                 pathex=['.'],
                 datas=additional_datas,
                 win_no_prefer_redirects=False,
                 win_private_assemblies=False,
                 hiddenimports=hidden_imports,
                 binaries=third_binaries,
                 excludes=excluded_modules,
                 )
    return a

def default_pyz_exe(a, name):
    pyz = PYZ(a.pure, a.zipped_data)
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name=name,
              debug=False,
              strip=False,
              upx=True,
              console=True )
    return pyz,exe

def merge_libs(analyzes):
    ret = list()
    for a in analyzes:
        ret.extend([a.binaries,a.zipfiles,a.datas])
    return ret

#convert_caffe = default_analyze('convertcaffe.py')
#convert_tf = default_analyze('convertensorflow.py')
#export = default_analyze('tensorconverter.py')
#tensorzonex = default_analyze('tensorzonex.py')
#prune = default_analyze('tensorreduce.py')
#ovx_generator = default_analyze('ovxgenerator.py')
#convert_tflite = default_analyze('convertflite.py')
#convert_darknet = default_analyze('convertdarknet.py')
#convert_onnx = default_analyze('convertonnx.py')
pegasus = default_analyze('pegasus.py')
#exportflite = default_analyze('exportflite.py')
#convert_pytorch = default_analyze('convertpytorch.py')
#convert_keras = default_analyze('convertkeras.py')
#convert_caffe = Analysis(['convertcaffe.py'],
#             pathex=['.'],
#             binaries=[],
#             datas=[],
#             hiddenimports=[],
#             hookspath=[],
#             runtime_hooks=[],
#             excludes=[],
#             win_no_prefer_redirects=False,
#             win_private_assemblies=False,
#             cipher=block_cipher)

# Do merge
MERGE(
#      (convert_caffe, 'convertcaffe', 'convertcaffe'),
#      (convert_tf, 'convertensorflow', 'convertensorflow'),
#      (export, 'tensorconverter', 'tensorconverter'),
#      (tensorzonex, 'tensorzonex', 'tensorzonex'),
#      (prune, 'tensorreduce', 'tensorreduce'),
#      (ovx_generator, 'ovxgenerator', 'ovxgenerator'),
#      (convert_tflite, 'convertflite', 'convertflite'),
#      (convert_darknet, 'convertdarknet', 'convertdarknet'),
#      (convert_onnx, 'convertonnx', 'convertonnx'),
      (pegasus, 'pegasus', 'pegasus'),
#      (exportflite, 'exportflite', 'exportflite'),
#      (convert_pytorch, 'convertpytorch', 'convertpytorch'),
#      (convert_keras, 'convertkeras', 'convertkeras'),
      )

#convert_caffe_pyz,convert_caffe_exe = default_pyz_exe(convert_caffe, 'convertcaffe')
#convert_tf_pyz,convert_tf_exe = default_pyz_exe(convert_tf, 'convertensorflow')
#export_pyz,export_exe = default_pyz_exe(export, 'tensorconverter')
#tensorzonex_pyz,tensorzonex_exe = default_pyz_exe(tensorzonex, 'tensorzonex')
#prune_pyz,prune_exe = default_pyz_exe(prune, 'tensorreduce')
#ovx_generator_pyz,ovx_generator_exe = default_pyz_exe(ovx_generator, 'ovxgenerator')
#convert_tflite_pyz,convert_tflite_exe = default_pyz_exe(convert_tflite, 'convertflite')
#convert_darknet_pyz,convert_darknet_exe = default_pyz_exe(convert_darknet, 'convertdarknet')
#convert_onnx_pyz,convert_onnx_exe = default_pyz_exe(convert_onnx, 'convertonnx')
pegasus_pyz,pegasus_exe = default_pyz_exe(pegasus, 'pegasus')
#exportflite_pyz,exportflite_exe = default_pyz_exe(exportflite, 'exportflite')
#convert_pytorch_pyz,convert_pytorch_exe = default_pyz_exe(convert_pytorch, 'convertpytorch')
#convert_keras_pyz,convert_keras_exe = default_pyz_exe(convert_keras, 'convertkeras')

# Collections
libs = merge_libs([
#                   convert_caffe,
#                   convert_tf,
#                   export,
#                   tensorzonex,
#                   prune,
#                   ovx_generator,
#                   convert_tflite,
#                   convert_darknet,
#                   convert_onnx,
                   pegasus,
#                   exportflite,
#                   convert_pytorch,
#                   convert_keras
                   ])

coll_exe = COLLECT(
#               convert_caffe_exe,
#               convert_tf_exe,
#               export_exe,
#               tensorzonex_exe,
#               prune_exe,
#               ovx_generator_exe,
#               convert_tflite_exe,
#               convert_darknet_exe,
#               convert_onnx_exe,
               pegasus_exe,
#               exportflite_exe,
#               convert_pytorch_exe,
#               convert_keras_exe,
               strip=None,
               upx=True,
               name = 'executable'
               )

coll_lib = COLLECT(*libs,
               strip=None,
               upx=True,
               name = 'libs'
               )

