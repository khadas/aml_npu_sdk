#!/bin/bash

NAME=mobilenet_tf
ACUITY_PATH=../bin/

pegasus=${ACUITY_PATH}pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=${ACUITY_PATH}pegasus.py
fi

#Tensorflow
$pegasus import tensorflow  \
		--model ./model/mobilenet_v1.pb \
		--inputs input \
		--outputs MobilenetV1/Predictions/Reshape_1 \
		--input-size-list '224,224,3' \
		--output-data ${NAME}.data \
		--output-model ${NAME}.json

#generate inpumeta
$pegasus generate inputmeta \
	--model ${NAME}.json \
	--input-meta-output ${NAME}_inputmeta.yml \
	--channel-mean-value "128 128 128 0.0078125"  \
	--source-file dataset_npy.txt


<<COMMENT
#Tensorflow
$pegasus import tensorflow  \
		--model ./model/mobilenet_v1.pb \
		--inputs input \
		--outputs MobilenetV1/Predictions/Reshape_1 \
		--input-size-list '224,224,3' \
		--output-data ${NAME}.data \
		--output-model ${NAME}.json

#CAFFE
$pegasus import caffe \
	--model caffe/alexnet.prototxt \
	--weights caffe/alexnet.caffemodel  \
	--output-data ${NAME}.data \
	--output-model ${NAME}.json \

#TFLITE
#$pegasus import tflite\
    --model  ${NAME}.tflite \
    --output-model ${NAME}.json \
    --output-data ${NAME}.data 

#Onnx
#$pegasus import onnx\
    --model  ${NAME}.onnx \
    --output-model ${NAME}.json \
    --output-data ${NAME}.data 

#Kreas
#$pegasus import keras\
    --model  ${NAME}.hdf5 \
    --output-model ${NAME}.json \
    --output-data ${NAME}.data 

#Pytorch
#$pegasus import pytorch\
    --model  ${NAME}.pt \
    --output-model ${NAME}.json \
    --output-data ${NAME}.data \
    --input-size-list '3,224,224' \

#Darknet
#$pegasus import darknet\
    --model  ${NAME}.cfg \
	--weights  ${NAME}.weights \
    --output-model ${NAME}.json \
    --output-data ${NAME}.data \
COMMENT


