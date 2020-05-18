#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --time=2:30:00
#SBATCH --mem=64G
source activate pytorch
cd qb18517/11_DT/dynamicTextureMain/4_C3D_ConvolutionalNN_3D/ 
python 2_C3D_Model_BC.py
