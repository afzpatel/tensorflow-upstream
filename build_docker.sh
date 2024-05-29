ROCM_VERSION_SHORT=rocm6.1.1
TENSORFLOW_IMAGE=rocm/tensorflow-private:$ROCM_VERSION_SHORT-tf1.15-dev

docker build \
       -f ./Dockerfile \
       -t $TENSORFLOW_IMAGE \
       .

docker run --privileged --name build_tf1.15_tmp --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 16 docker.io/rocm/tensorflow-private:rocm6.1.1-tf1.15-dev \
    /bin/bash -c "cd /root/tensorflow && bash build_rocm_python3 && rm -rf /root/.cache" 

docker commit build_tf1.15_tmp $TENSORFLOW_IMAGE
docker rm build_tf1.15_tmp
