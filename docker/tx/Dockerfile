FROM cupy/nvidia-cuda:10.2-devel-ubuntu18.04

ARG MMCV_VERSION=">=2.0.0rc2"
ARG MMENGINE_VERSION=">=0.3.0"
ARG TENSORRT_VERSION=8.2.1.8
# tensorrt tar file url
ARG TENSORRT_URL=https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.1/tars/tensorrt-8.2.1.8.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
ARG CUDNN_URI=https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz


RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    echo "Use aliyun source for installing libs" ; 
    
### update apt and install libs
RUN apt update &&\
    apt install -y vim curl git wget --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

# ENV http_proxy=http://127.0.0.1:10809
# ENV https_proxy=$http_proxy

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.8 && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple; \
    echo "pip using tsinghua source" 


### install pytorch openmim
RUN /opt/conda/bin/conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia &&\
#     /opt/conda/bin/pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117 &&\
    /opt/conda/bin/pip install --no-cache-dir openmim

### pytorch mmcv onnxruntime
RUN /opt/conda/bin/mim install --no-cache-dir mmengine${MMENGINE_VERSION} &&\
    /opt/conda/bin/mim install mmdet &&\
    /opt/conda/bin/pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html &&\
    /opt/conda/bin/mim install mmpose

RUN /opt/conda/bin/pip install mmdeploy==1.2.0 &&\
    /opt/conda/bin/pip install mmdeploy-runtime-gpu==1.2.0
    
RUN wget http://192.168.100.88/mmdeploy.zip &&\
    apt update &&\
    apt install unzip &&\
    unzip mmdeploy.zip
    
ENV http_proxy=http://127.0.0.1:10809
ENV https_proxy=$http_proxy


RUN cd mmdeploy &&\
    wget -c $TENSORRT_URL && \
    tar -zxvf tensorrt-${TENSORRT_VERSION}*.tar.gz &&\
    /opt/conda/bin/pip install Tensor*/python/tensorrt-*-cp38-none-linux_x86_64.whl &&\
    /opt/conda/bin/pip install pycuda 

ENV TENSORRT_DIR=/mmdeploy/TensorRT-${TENSORRT_VERSION}
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
ENV PATH=$TENSORRT_DIR/bin:$PATH

RUN /opt/conda/bin/conda init bash

ENV http_proxy=
ENV https_proxy=

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update &&\
    apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libssl-dev libopencv-dev libspdlog-dev

## install cudnn
RUN cd mmdeploy &&\
    wget http://192.168.100.88/cudnn-11.3-linux-x64-v8.2.1.32.tgz && \
    tar -zxvf /mmdeploy/cudnn-11.3-linux-x64-v8.2.1.32.tgz &&\
    rm -rf cudnn-11.3-linux-x64-v8.2.1.32.tgz 

ENV CUDNN_DIR=/mmdeploy/cuda/
ENV LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

