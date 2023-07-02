# SystemX Installation Guide

## Prerequisites
- CMake 3.27.0-rc2
- cuda 11.7

## Clone
```bash
git clone --recurse-submodules git@github.com:mcrl/systemx.git
git submodule foreach --recursive "git checkout $(git remote show origin | grep 'HEAD branch' | sed 's/.*: //')"

# setup external libraries
cd external
cd spdlog && git checkout ad0e89c && cd .. # spdlog release v.1.11.0
cd jsoncpp && git checkout 5defb4e && cd .. # jsoncpp release v.1.9.5
```

## Install
```bash
./build.sh -DCUDA_INSTALL_PATH=/path/to/cuda/install
```