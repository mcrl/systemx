#!/bin/bash

# TODO: change the path to your nsys
NSYS=/home/n1/junyeol/cuda-11.7/bin/nsys

# set --kernel argument as profile name
PROFILE=$(echo $@ | awk -F'kernels=' '{print $2}' | awk -F' ' '{print $1}')
OUTPUT=/home/n1/junyeol/systemx/profiles/${PROFILE}

${NSYS} profile --force-overwrite=true \
                --trace=cuda,nvtx --stats=true \
                --sample=process-tree --cpuctxsw=process-tree \
                --output ${OUTPUT} \
                numactl --physcpubind 0-63 \
                $@