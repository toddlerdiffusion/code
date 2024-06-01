import os
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import cv2

def resize_and_copy_image(source_file, destination_file):
    image = cv2.imread(source_file)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite(destination_file, image)


source_dir = "/xxx/scratch/xxx/ldm_weights/logs/churches/sec_stage/RGB_50_1000_LsunChurches_ldm_Largevq4RGB_LargeUnet_org_x0/2023-10-20T17-47-43_lsun_churches-ldm-vq-4/samples_ddpm1k/img"
destination_dir = "/xxx/scratch/xxx/ldm_weights/logs/churches/sec_stage/RGB_50_1000_LsunChurches_ldm_Largevq4RGB_LargeUnet_org_x0/2023-10-20T17-47-43_lsun_churches-ldm-vq-4/samples_1k_5k"
N = 5000

files = os.listdir(source_dir)
selected_files = random.sample(files, N)

# Define the number of worker processes (adjust as needed)
num_workers = 64

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for file in selected_files:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        futures.append(executor.submit(resize_and_copy_image, source_file, destination_file))

    # Wait for all tasks to complete
    for future in tqdm(futures, total=len(futures), desc="Processing images"):
        future.result()

print(f"Copied {N} random files from {source_dir} to {destination_dir}.")
