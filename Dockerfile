#| This Dockerfile provides a starting point for a ROCm installation of Tensorflow.
FROM rocm/dev-ubuntu-20.04:6.1.1-complete

ENV ROCM_PATH=/opt/rocm-6.1.1

ENV DEBIAN_FRONTEND noninteractive
ENV HOME /root/

# Install required python packages
RUN apt-get update --allow-insecure-repositories && apt-get install -y \
    python3-dev \
    python3-pip \
    rocprim \
    hipcub \
    python3-wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install pip enum34 mock setuptools cython --upgrade
RUN pip3 install keras_preprocessing --upgrade
RUN pip3 install keras_applications
RUN pip3 install jupyter 
RUN pip3 install numpy==1.18.5

# Install Bazel
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt-get update --allow-insecure-repositories && apt-get install -y openjdk-8-jdk openjdk-8-jre unzip wget git && apt-get clean && rm -rf /var/lib/apt/lists/* 
RUN cd ~ && rm -rf bazel*.sh && wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh && bash bazel*.sh && rm -rf ~/*.sh

# Clone TF
RUN cd ~ && git clone -b r1.15-mi308 https://github.com/fsx950223/tensorflow.git tensorflow
# RUN cd ~ && git clone -b r1.15 https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git tensorflow

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN cd ~/tensorflow && bash build_rocm_python3 && rm -rf ~/.cache

RUN cd ~ && git clone https://github.com/tensorflow/models.git

# TF/benchmarks with some workarounds for ImageNet
RUN cd ~ && git clone -b cnn_tf_v1.15_compatible https://github.com/tensorflow/benchmarks.git
WORKDIR ~
