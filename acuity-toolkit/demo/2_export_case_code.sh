#!/bin/bash

NAME=mobilenet_tf
ACUITY_PATH=../bin/

pegasus=$ACUITY_PATH/pegasus
if [ ! -e "$pegasus" ]; then
    pegasus=$ACUITY_PATH/pegasus.py
fi

$pegasus export ovxlib\
    --model ${NAME}.json \
    --model-data ${NAME}.data \
    --model-quantize ${NAME}.quantize \
    --with-input-meta ${NAME}_inputmeta.yml \
    --dtype quantized \
    --optimize VIPNANOQI_PID0X1E  \
    --viv-sdk ${ACUITY_PATH}vcmdtools \
    --pack-nbg-unify

rm -rf ${NAME}_nbg_unify

mv ../*_nbg_unify ${NAME}_nbg_unify

cd ${NAME}_nbg_unify

mv network_binary.nb ${NAME}.nb

cd ..

#save normal case demo export.data 
mkdir -p ${NAME}_normal_case_demo
mv  *.h *.c .project .cproject *.vcxproj BUILD *.linux *.export.data ${NAME}_normal_case_demo

# delete normal_case demo source
#rm  *.h *.c .project .cproject *.vcxproj  BUILD *.linux *.export.data

rm *.data *.quantize *.json *_inputmeta.yml

