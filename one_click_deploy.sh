#!/bin/bash

echo 拉取镜像 
docker pull sillintang/mmlab:v2

if [ $? -ne 0 ]; then
	  echo "Error pulling image"
	    exit 1
fi

echo 创建容器mmdeploy-test
docker run --shm-size 32G --gpus all --name mmdeploy-test -d sillintang/mmlab:v2 tail -f /dev/null

if [ $? -ne 0 ]; then
	  echo "Error starting container"
	    exit 1  
fi

echo 容器mmdeploy-test已创建成功
