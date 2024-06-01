#!/bin/bash

#SBATCH --partition=batch
#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J sam_1
#SBATCH -o sam_1.out
#SBATCH --nodes=1
#SBATCH --time=1-23:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:rtx2080ti:8

#run the application:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/xxx/anaconda3/envs/ldm/bin/python -u SAM.py
