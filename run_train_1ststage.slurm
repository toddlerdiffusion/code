#!/bin/bash

#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J 1stSketch_Sobel8_50steps_TinyUnet_lsun_LR4e4_4V100_b64
#SBATCH -o 1stSketch_Sobel8_50steps_TinyUnet_lsun_LR4e4_4V100_b64.out
#SBATCH --nodes=1
#SBATCH --time=0-23:30:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:4

scontrol show job $SLURM_JOB_ID to the jobscript

#config_file="celebhq/1stSketch_celebahq_pixelspace.yaml"
#config_file="celebhq/cot_celebahq_pixelspace_sketch2Palette.yaml"
config_file="lsun/1stSketch_lsun_pixelspace.yaml"
#config_file="lsun/1stSketch_lsun_tinyvqgan.yaml"
#config_file="lsun/cot_lsun_pixelspace_sketch2Palette.yaml"

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/xxx/anaconda3/envs/ldm/bin/python -u main.py \
    --scale_lr False --base "configs/latent-diffusion/"$config_file -t --gpus 0,1,2,3 \
    --logdir "/xxx/user/xxx/ldm_weights/cot/lsun/1stSketch_Sobel8_50steps_TinyUnet_lsun_LR4e4_4V100_b64/"
