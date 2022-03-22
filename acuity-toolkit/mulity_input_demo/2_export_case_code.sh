#!/bin/bash

NAME=model
ACUITY_PATH=../../acuity-toolkit-binary-6.0.12/bin/

pegasus=${ACUITY_PATH}pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=${ACUITY_PATH}pegasus.py
fi

$pegasus 	export ovxlib\
    --model ${NAME}.json \
    --model-data ${NAME}.data \
    --model-quantize ${NAME}.quantize \
    --with-input-meta ${NAME}_inputmeta.yml \
    --dtype quantized \
    --optimize VIPNANOQI_PID0X88  \
    --viv-sdk ${ACUITY_PATH}vcmdtools \
    --pack-nbg-unify 
	

#save normal case demo export.data 
mkdir normal_case_demo
mv  *.h *.c .project .cproject *.vcxproj *.lib BUILD *.linux *.export.data normal_case_demo

#rm  *.h *.c .project .cproject *.vcxproj  BUILD *.linux *.export.data
