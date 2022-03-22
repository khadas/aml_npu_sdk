#!/bin/bash

NAME=model
ACUITY_PATH=../../acuity-toolkit-binary-6.0.12/bin/

pegasus=${ACUITY_PATH}pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=${ACUITY_PATH}pegasus.py
fi

	
$pegasus  import onnx \
   --model   ./onnx_model/${NAME}.onnx \
   --inputs "0 1 2" \
   --input-size-list "3,288,288#3,288,288#3,288,288" \
   --outputs "327 328" \
   --output-model ./${NAME}.json \
   --output-data ./${NAME}.data
  
$pegasus generate inputmeta --model ./${NAME}.json \
	--channel-mean-value "128 128 128 0.00715#128 128 128 0.00715#128 128 128 0.00715" \
	--source-file dataset.txt \
	#--separated-database \
	#--source-file "dataset1.txt#dataset2.txt#dataset3.txt"
