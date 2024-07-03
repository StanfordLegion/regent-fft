#!/bin/bash

set -ex

export INCLUDE_PATH="$INCLUDE_PATH;$PWD/fftw-3.3.8/install/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/fftw-3.3.8/install/lib"
export TERRA_PATH="$TERRA_PATH;$PWD/src/?.rg"

sudo apt-get update -qq
sudo apt-get install -qq software-properties-common

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

sudo apt-get update -qq
sudo apt-get install -qq cuda-toolkit-12-2
sudo apt-get install nvidia-utils-515

export CUDA="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export PATH="$PATH:$CUDA_PATH/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATH/lib:$CUDA_PATH/lib64"

git clone --depth 1 --branch master https://gitlab.com/StanfordLegion/legion.git
CC=gcc CXX=g++ USE_GASNET=0 USE_CUDA=1 DEBUG=${DEBUG} MAX_DIM=4 ./legion/language/scripts/setup_env.py
./install.py
./legion/language/regent.py test/fft_test.rg -fgpu cuda -fgpu-offline 1 -fgpu-arch pascal
./legion/language/regent.py test/fft_test.rg -fgpu none
