#!/bin/bash

set -e

CONTAINER_NAME="npu-vim3"
REPOSITORY_NAME="numbqq/npu-vim3"

if [ `docker inspect $CONTAINER_NAME &>>/dev/null &&  echo 0 || echo 1` -eq 0 ];then
    echo "The container $CONTAINER_NAME is exist!, stop and rm it at first"

    docker stop $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME
fi

#Environment required for configuring containers
DOCKER_RUN="docker run -it --name $CONTAINER_NAME \
	--rm \
	-v $(pwd):/home/khadas/npu \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/timezone:/etc/timezone:ro \
    -v /home/$(whoami):/home/$(whoami) \
	$REPOSITORY_NAME \
    "

echo $DOCKER_RUN

eval $DOCKER_RUN bash -c '"
	cd ~/npu/acuity-toolkit/demo
	bash 0_import_model.sh && bash 1_quantize_model.sh && bash 2_export_case_code.sh
	"'
# Remove the container
if [ `docker inspect $CONTAINER_NAME &>>/dev/null &&  echo 0 || echo 1` -eq 0 ];then
    echo "The container $CONTAINER_NAME is exist!, stop and rm it at last"

    docker stop $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME
fi

echo "Convert in Docker Done!!!"

