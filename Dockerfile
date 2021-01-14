# CUDA
FROM  nvidia/cuda:10.0-runtime-ubuntu16.04

LABEL author="filippo.giruzzi@gmail.com"

# CUDNN
LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"
ENV CUDNN_VERSION 7.6.0.64
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN  apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# ARGS
ARG PYTHON_VERSION=3.7.3
ARG PYTHON_SUFFIX=3.7

# Basic
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    file \
    git \
    graphviz \
    gzip \
    libsqlite3-dev \
    openssl \
    unzip \
    wget \
    xmlsec1 \
    zip

# Python
RUN apt-get install -y \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev

WORKDIR /voice_activity_detection
RUN curl -L https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz | tar xfJ -

WORKDIR /voice_activity_detection/Python-${PYTHON_VERSION}
RUN ./configure --prefix=/usr \
    && make altinstall \
    && rm -r /voice_activity_detection/Python-${PYTHON_VERSION} \
    && ln -s /usr/bin/python${PYTHON_SUFFIX} /usr/bin/python \
    && ln -s /usr/bin/pip${PYTHON_SUFFIX} /usr/bin/pip

RUN \
  ln -f -s /usr/bin/python${PYTHON_SUFFIX} /usr/bin/python && \
  ln -f -s /usr/bin/pip${PYTHON_SUFFIX} /usr/bin/pip

RUN rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* /var/tmp/*

# Setup
WORKDIR /voice_activity_detection
COPY . .
RUN rm -r Python-${PYTHON_VERSION}

# Install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# Test global package import
RUN python -c "import vad"
