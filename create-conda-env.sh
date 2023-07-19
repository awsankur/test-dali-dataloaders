#!/bin/bash

# PyTorch 1.13
#conda activate pytorch

# PyTorch 2.0
source /home/ubuntu/.bashrc
conda create -n pytorch-2-dali python=3.9

conda activate pytorch-2-dali

conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia

pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

pip3 install -r requirements.txt

