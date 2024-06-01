#!/bin/bash

#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J Sketch_Pixel_LDM_lsun_LR4e4_1a100_b32
#SBATCH -o Sketch_Pixel_LDM_lsun_LR4e4_1a100_b32.out
#SBATCH -N 1
#SBATCH --time=0-23:30:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1

#run the application:

#config_file="lsun_churches-ldm-kl-8.yaml"
#config_file="lsun/lsun_churches-ldm-vq-4.yaml"
config_file="lsun/1sketch_ldm_pixel_lsun.yaml"
#config_file="celebhq/1stSketch_ldm_celebahq_pixelspace.yaml"
#config_file="lsun_churches-ldm-kl-8_UViT.yaml"
#config_file="lsun_churches-ldm-vq4_UViT.yaml"
#config_file="celebhq/celebahq-ldm-vq-4.yaml"
#config_file="coco/coco_vq4.yaml"

CUDA_VISIBLE_DEVICES=0 /home/xxx/anaconda3/envs/ldm/bin/python -u main.py \
    --scale_lr False --base "configs/latent-diffusion/"$config_file -t --gpus 0, \
    --logdir "/xxx/user/xxx/ldm_weights/ldm/lsun/Sketch_Pixel_LDM_lsun_LR4e4_1a100_b32"
