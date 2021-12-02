#!/bin/bash

NAME=mobilenet_tf
ACUITY_PATH=../bin/

tensorzone=${ACUITY_PATH}tensorzonex

if [ ! -e "$tensorzone" ]; then
    tensorzone=${ACUITY_PATH}tensorzonex.py
fi


#snapshot inference , if set  action snapshot, will export the result of each layer
$tensorzone \
    --action inference \
	--reorder-channel '0 1 2' \
    --source text \
    --source-file ./data/validation_tf.txt \
    --channel-mean-value '128 128 128 128' \
    --model-input ${NAME}.json \
    --model-data ${NAME}.data \
    --dtype quantized \

#    --dtype quantized
#    --dtype float

#Note: inference float:  set--dtype float, inference float result is correct if other parameters arm set correct.
#Note: inference quantize:  set--dtype quantized, execute after quantize step


