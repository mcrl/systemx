#!/bin/bash

# TODO: change the path to your ncu
NCU=/home/n1/junyeol/cuda-11.7/bin/ncu

# set --benchmark_file argument without extension as profile name
PROFILE=$(echo $@ | awk -F'--benchmark_file ' '{print $2}' | awk -F' ' '{print $1}' | \
          awk -F'/' '{print $NF}' | awk -F'.json' '{print $1}')
OUTPUT=/home/n1/junyeol/systemx/profiles/${PROFILE}

sudo ${NCU} --set full -o ${OUTPUT} --force $@