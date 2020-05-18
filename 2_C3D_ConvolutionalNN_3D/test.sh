#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_veryshort
#SBATCH --job-name=gpujob
#SBATCH --time=00:53:00
#SBATCH --mem=120G
source activate pytorch
cd qb18517/11_DT/dynamicTextureMain/4_C3D_ConvolutionalNN_3D/
python 2_C3D_Model_BC.py

