#!/bin/bash

#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH -J sample_RGB50
#SBATCH -o sample_RGB50.out
#SBATCH --nodes=1
#SBATCH --time=0-0:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1

#run the application:
source ~/anaconda3/bin/activate ldm

python sample_inference.py -n "$1" --batch_size "$2" -v --sp "$3" --save_dir "$4" -r "$5" --eta 0 --data_type "$6"