#!/bin/bash

#create the necesary environment variables
source /esat/spchtemp/scratch/pwang/pre-training/.bashrc
source ~/anaconda2/etc/profile.d/conda.sh
conda activate env1
echo $CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA}

#run the original
$@
