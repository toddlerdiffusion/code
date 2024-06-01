#!/bin/bash

#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J CoT_VQ4_50Steps_Lsun_LR1e4_4v100_b32_NoiseMul1_BinaryY_3DY_NoMorph_Ch128_8Att8
#SBATCH -o CoT_VQ4_50Steps_Lsun_LR1e4_4v100_b32_NoiseMul1_BinaryY_3DY_NoMorph_Ch128_8Att8.out
#SBATCH --nodes=1
#SBATCH --time=1-23:30:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
scontrol show job $SLURM_JOB_ID to the jobscript

#config_file="cot_lsun_churches-ldm-kl-8.yaml"
#config_file="cot_lsun_churches-ldm-kl8_1ststage.yaml"
#config_file="cot_lsun_churches-ldm-vq-4_1ststage.yaml"
#config_file="cot_LsunChurches_VQ4_UViT.yaml"
#config_file="lsun/cot_cond_lsun-ldm-vq-4.yaml"
config_file="lsun/cot_lsun_churches-ldm-vq-4.yaml"
#config_file="coco/cot_coco_vq4.yaml"
#config_file="celebhq/cot_celebahq-ldm-vq-4.yaml"
#config_file="celebhq/cot_cond_celebahq-ldm-vq-4.yaml"

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/xxx/anaconda3/envs/ldm/bin/python -u main.py \
    --scale_lr False --base "configs/latent-diffusion/"$config_file -t --gpus 0,1,2,3 \
    --logdir "/xxx/user/xxx/ldm_weights/cot/lsun/CoT_VQ4_50Steps_Lsun_LR1e4_4v100_b32_NoiseMul1_BinaryY_3DY_NoMorph_Ch128_8Att8"
    #--logdir "/xxx/user/xxx/ldm_weights/cot/celebhq/CoT_Palette2RGB_CondConcat_VQ4_Celeb_LR4e4_4v100_b32_NoMorphTrans_Noise_NoBinaryY_blur12"
    #--resume "/xxx/user/xxx/ldm_weights/cot/coco/CoT_VQ4_Coco_LR4e5_8v100_b64_ClipL14_Precomputed_SpatialTrans/2024-02-03T23-45-28_cot_coco_vq4/checkpoints/epoch=000045.ckpt"
    #--resume "/xxx/user/xxx/ldm_weights/cot/coco/CoT_SketchPalette_VQ4_Coco_LR4e4_8v100_b64_ClipL14_Precomputed_SpatialTrans/2024-01-30T13-12-19_cot_coco_vq4/checkpoints/epoch=000099.ckpt"
