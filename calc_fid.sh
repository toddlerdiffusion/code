#!/bin/bash

#SBATCH --partition=batch
#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J fid_results_reproduce_10
#SBATCH -o fid_results_reproduce_10.out
#SBATCH --nodes=1
#SBATCH --time=0-01:30:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1

source ~/anaconda3/bin/activate ldm

# Run this command only once to generate GT for a new dataset:
#python random_sample_50k.py

# Calculate FID for Diffusion Output:
# -----------------------------------
# -----------------------------------
#               CelebHQ
# -----------------------------------
#python -m pytorch_fid /xxx/user/xxx/ldm_weights/cot/celebhq/CoT_VQ4_50Steps_Celeb_LR2e4_4A6000_b32_NoiseMul1_BinaryY_3DY_NoMorph_Ch224_8Att8/2024-03-26T19-49-38_cot_celebahq-ldm-vq-4/samples_150_SketchTiny500Noise0/img \
# /xxx/user/xxx/celebhq/rgb_processed_orgLDM/
#python -m pytorch_fid /xxx/user/xxx/ldm_weights/cot/celebhq/CoT_Palette2RGB_CondConcat_VQ4_Celeb_LR4e4_4a100_b32_NoMorphTrans_Noise_NoBinaryY_blur12/2024-03-14T06-36-50_cot_cond_celebahq-ldm-vq-4/samples_150_palette150_Noise20%/img \
# /xxx/user/xxx/celebhq/celebhq/
#python -m pytorch_fid  /xxx/user/xxx/ldm_weights/cot/celeb/CoT_Cond_VQ4_50Steps_Celeb_LR4e4_4a100_b32_Noise100200Mult1_BinaryY_3DY_NoMorph_8Att8/2024-03-16T18-28-49_cot_celebahq-ldm-vq-4/samples_150_SketchTiny500_Noise80/img \
#  /xxx/user/xxx/celebhq/celebhq/

# -----------------------------------
#               COCO
# -----------------------------------
#python -m pytorch_fid /xxx/user/xxx/ldm_weights/cot/coco/CoT_VQ4_Coco_LR4e4_8v100_b64_ClipL14_Precomputed_SpatialTrans/2024-01-28T13-34-24_cot_coco_vq4/samples_10k_650_eta0/img \
# /xxx/scratch/xxx/COCO_dataset/val2017_resized256/


# -----------------------------------
#               LSUN
# -----------------------------------
python -m pytorch_fid /xxx/user/xxx/ldm_weights/cot/lsun/CoT_VQ4_50Steps_Lsun_LR1e4_4v100_b32_NoiseMul1_BinaryY_3DY_NoMorph_Ch128_8Att8/2024-03-29T15-04-30_cot_lsun_churches-ldm-vq-4/samples_50_SketchTiny150Noise0/img \
 /xxx/user/xxx/lsun_churches_sampled_50k_processed_cot

# Calculate FID for VqGAN Output:
# -------------------------------
#python -m pytorch_fid logs/vqgan_10epochs/samples/img \
# /xxx/scratch/xxx/lsun_dataset/openimages_val_resized256_40k


# Calculate FID for VqGAN Output on ImageNet:
# ------------------------------------------
#python -m pytorch_fid logs/vqgan_10epochs/samples/img \
# /xxx/scratch/xxxßß/imagenet_val