#!/bin/bash

#SBATCH --partition=batch
#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J fid_sketch_results
#SBATCH -o fid_sketch_results.out
#SBATCH --nodes=1
#SBATCH --time=0-0:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1

source ~/anaconda3/bin/activate ldm

cd ldm/data
#python prepare_txt_train_val.py

cd ../../sketch_fid/src/pytorch_fid
python fid_score.py /xxx/user/xxx/ldm_weights/cot/celebhq/1stSketch_PixelSpace_50steps_TinyUnet_celeb_LR4e4_1A100_b32_2VarNoise_Bridge/2024-03-06T18-24-11_1stSketch_celebahq_pixelspace/samples_50/img \
    /xxx/user/xxx/celebhq/celebhq_sketch_64
