#!/bin/bash

NAME=mobilenet_tf
ACUITY_PATH=../bin/

pegasus=$ACUITY_PATH/pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=$ACUITY_PATH/pegasus.py
fi

$pegasus export ovxlib\
    --model ${NAME}.quantize.json \
    --model-data ${NAME}.data \
    --model-quantize ${NAME}.quantize \
    --with-input-meta ${NAME}_inputmeta.yml \
    --dtype quantized \
    --optimize VIPNANOQI_PID0X88  \
    --viv-sdk ${ACUITY_PATH}vcmdtools \
    --pack-nbg-unify

mkdir normal_case_demo
mv *.h *.c .project .cproject *.vcxproj *.export.data BUILD *.linux normal_case_demo
