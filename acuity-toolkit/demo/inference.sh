#!/bin/bash

NAME=mobilenet_tf
ACUITY_PATH=../bin/

tensorzone=${ACUITY_PATH}tensorzonex

if [ ! -e "$tensorzone" ]; then
    tensorzone=${ACUITY_PATH}tensorzonex.py
fi


#snapshots inference
$tensorzone \
    --action inference \
    --source text \
    --source-file ./data/validation_tf.txt \
    --channel-mean-value '128 128 128 128' \
    --model-input ${NAME}.json \
    --model-data ${NAME}.data \
    --dtype quantized \
#    --reorder-channel '0 1 2' 
#    --dtype quantized
#    --dtype float

#Note: inference float:  set--dtype float, inference float result is correct if other parameters arm set correct.
#Note: inference quantize:  set--dtype quantized, execute after quantize step


