#!/bin/bash

NAME=mobilenet_tf
ACUITY_PATH=../bin/

pegasus=$ACUITY_PATH/pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=$ACUITY_PATH/pegasus.py
fi

#snapshots inference quantized float32
$pegasus inference \
	--dtype quantized \
	--model ${NAME}.json \
	--model-data ${NAME}.data \
	--with-input-meta ${NAME}_inputmeta.yml \


