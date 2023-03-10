#!/bin/bash

#SBATCH --job-name=project
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=32GB

module purge

module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch

echo "Start"

python env.py

echo "Done"