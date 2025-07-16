# 基础镜像使用官方Ubuntu 22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装必要工具和依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gnupg2 \
    ca-certificates \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 添加NVIDIA CUDA仓库公钥及源
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm cuda-keyring_1.0-1_all.deb

RUN apt-get update

# 安装 CUDA Toolkit 12.4
RUN apt-get install -y cuda-toolkit-12-4

# 设置环境变量，方便CUDA命令生效
ENV PATH=/usr/local/cuda-12.4/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}

# 默认进入bash
CMD ["/bin/bash"]
