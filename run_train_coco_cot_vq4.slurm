#!/bin/bash

#SBATCH --partition=batch
#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J 1st_coco_black2sketch
#SBATCH -o 1st_coco_black2sketch.out
#SBATCH --nodes=1
#SBATCH --time=0-23:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:4


#run the application:
#10^-3 --> 5*10^-5
# Sketch_CoT_50_LsunChurches_vq4_1000steps_BlackBridgeNoise2Sketch_BBDM_X0_Noise2_weighting_SmallVqGAN_SmallUnet

#exp_dir="2nd_Y3DBinaryT_B32_LR5e-5_CoT_10steps_Sketch2RGB_BBDM_XVqGAN-LargeVqGAN_XLUnet248"
exp_dir="1st_coco_black2sketch_XVqGAN-SVqGAN_SUnet"
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/xxx/anaconda3/envs/ldm/bin/python -u main.py \
    --scale_lr False --base configs/latent-diffusion/coco/cot_coco-ldm-vq-4_1ststage.yaml -t --gpus 0,1,2,3, \
    --logdir "logs/"$exp_dir
    #--resume "/home/xxx/CoT_Gen/latent-diffusion-main/logs/2nd_Y3DBinaryT_B32_LR1e-3_CoT_50steps_Sketch2RGB_BBDM_XVqGAN_SmallVqGAN_Small/2023-11-16T01-41-00_cot_lsun_churches-ldm-vq-4/checkpoints/epoch=000049.ckpt"
