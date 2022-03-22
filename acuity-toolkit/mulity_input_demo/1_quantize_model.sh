#!/bin/bash

NAME=model
ACUITY_PATH=../../acuity-toolkit-binary-6.0.12/bin/

pegasus=${ACUITY_PATH}pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=${ACUITY_PATH}pegasus.py
fi

#dynamic_fixed_point  --> i8(int16) asymmetric_affine  --->  uint8
$pegasus quantize \
	--model ${NAME}.json \
	--model-data ${NAME}.data \
	--with-input-meta ${NAME}_inputmeta.yml \
	--quantizer asymmetric_affine  \
	--qtype uint8 --rebuild
	
