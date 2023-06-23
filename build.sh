#!/bin/bash

set -e
# set -x

cd "$( dirname $BASH_SOURCE )"

mkdir -p build
cd build
cmake .. $@
cmake --build . -j