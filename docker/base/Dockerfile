FROM nvidia/cuda:11.8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip cmake build-essential git wget \
    libeigen3-dev libsuitesparse-dev libboost-all-dev libgoogle-glog-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install pybind11 numpy scipy pytest evo torchmetrics matplotlib pyyaml \
    open3d plyfile
WORKDIR /workspace
