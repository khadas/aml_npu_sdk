#!/bin/bash

NAME=model
ACUITY_PATH=../../acuity-toolkit-binary-6.0.12/bin/

pegasus=${ACUITY_PATH}pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=${ACUITY_PATH}pegasus.py
fi

#quantized float32
$pegasus inference \
	--dtype quantized \
	--model ${NAME}.json \
	--model-data ${NAME}.data \
	--with-input-meta ${NAME}_inputmeta.yml \
	


