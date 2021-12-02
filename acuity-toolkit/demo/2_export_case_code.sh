#!/bin/bash

NAME=mobilenet_tf
ACUITY_PATH=../bin/

export_ovxlib=${ACUITY_PATH}ovxgenerator

$export_ovxlib \
    --model-input ${NAME}.json \
    --data-input ${NAME}.data \
    --model-quantize ${NAME}.quantize \
    --reorder-channel '0 1 2' \
    --channel-mean-value '128 128 128 128' \
    --export-dtype quantized \
    --optimize VIPNANOQI_PID0XE8  \
    --viv-sdk ${ACUITY_PATH}vcmdtools \
    --pack-nbg-unify  \

#Note:
#	 --optimize VIPNANOQI_PID0XB9  
#	when exporting nbg case for different platforms, the paramsters are different.
#   you can set VIPNANOQI_PID0X7D	VIPNANOQI_PID0X88	VIPNANOQI_PID0X99
#				VIPNANOQI_PID0XA1	VIPNANOQI_PID0XB9	VIPNANOQI_PID0XBE	VIPNANOQI_PID0XE8
#	Refer to sectoin 3.4(Step 3) of the  <Model_Transcoding and Running User Guide_V0.8> documdent


rm -rf nbg_unify_${NAME}

mv ../*_nbg_unify nbg_unify_${NAME}

cd nbg_unify_${NAME}

mv network_binary.nb ${NAME}.nb

cd ..

#save normal case demo export.data 
mkdir -p ${NAME}_normal_case_demo
mv  *.h *.c .project .cproject *.vcxproj BUILD *.linux *.export.data ${NAME}_normal_case_demo

# delete normal_case demo source
#rm  *.h *.c .project .cproject *.vcxproj  BUILD *.linux *.export.data

rm *.data *.quantize *.json


