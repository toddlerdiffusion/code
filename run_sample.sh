# Sample Diffusion:
# ----------------
num_img=1000
counter=0

# Loop to increment and display the counter
for ((i = 0; i < 8; i++)); do
    counter=$((num_img * i))
    sbatch slurm_run_sampling.sh $num_img 32 $counter 'samples_50_SketchTiny150Noise0' \
      '/xxx/user/xxx/ldm_weights/cot/lsun/CoT_VQ4_50Steps_Lsun_LR1e4_4v100_b32_NoiseMul1_BinaryY_3DY_NoMorph_Ch128_8Att8/2024-03-29T15-04-30_cot_lsun_churches-ldm-vq-4/checkpoints/epoch=000049.ckpt' \
      'validation'
done
