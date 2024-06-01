#!/bin/bash

#SBATCH --partition=batch
#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J retrival_results
#SBATCH -o retrival_results.out
#SBATCH --nodes=1
#SBATCH --time=0-3:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1

source ~/anaconda3/bin/activate ldm

cd retrival_img_img
python sketch_img2img_retrival.py --encoder_type "clip" /home/xxx/CoT_Gen/latent-diffusion-main/logs/1st_0.25linear_Gaussian_200_LsunChurches_vq4_200steps_Black1D2Sketch_BBDM_X0_Weighting_XVqGAN_XVqGAN_SmallUnet/2023-11-09T21-10-15_cot_lsun_churches-ldm-vq-4_1ststage/samples/img \
    /home/xxx/CoT_Gen/latent-diffusion-main/data/lsun/sketches_lsun_churches_all_binary
    
