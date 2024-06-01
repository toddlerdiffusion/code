import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import os
from torch.utils.data import Dataset, DataLoader
from ldm.data.coco import COCOtrain, COCOval
from ldm.models.autoencoder import VQModelInterface
from ldm.modules.encoders.modules import FrozenCLIPEmbedder_HuggingFace


def pre_encode_vqgan4(dataset, output_directory, batch_size=32, device='cuda'):
    # Process Val data for proper FID calculation:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    os.makedirs(output_directory, exist_ok=True)
    ddconfig = {'double_z': False, 'z_channels': 3, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult':[1,2,4],
              'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    lossconfig = {'target': 'torch.nn.Identity'}
    vq_model = VQModelInterface(embed_dim=3, n_embed=8192, ckpt_path="models/first_stage_models/vq-f4/model.ckpt",
                                ddconfig=ddconfig, lossconfig=lossconfig).to(device)
    with torch.no_grad():
        for img_count, sample in tqdm(enumerate(dataloader)):
            image = sample['image']
            x = torch.tensor(image, device=device)  # [B, H, W, 3]
            x = x.permute(0, 3, 1, 2)  # [B, 3, H, W]
            # Run VQ-GAN-4:
            z = vq_model.encode(x).cpu().detach()
            # Saving:
            for idx, img_name in enumerate(sample["imgname"]):
                img_name = img_name.split('.jpg')[0]
                np.save(os.path.join(output_directory, f'{img_name}.npy'), z[idx])


def pre_encode_cliphuggingface(dataset, output_directory, batch_size=32, device='cuda'):
    # Encode COCO caption once to speed up the training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    os.makedirs(output_directory, exist_ok=True)
    ddconfig = {'double_z': False, 'z_channels': 3, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult':[1,2,4],
              'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    lossconfig = {'target': 'torch.nn.Identity'}
    model = FrozenCLIPEmbedder_HuggingFace(version="openai/clip-vit-large-patch14", device=device, max_length=77)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for img_count, sample in tqdm(enumerate(dataloader)):
            caption = sample['caption']
            #x = torch.tensor(caption, device=device)
            #x = x.permute(0, 3, 1, 2)
            # Run the model:
            z = model.encode(caption).cpu().detach()  # [B, Seq_Length, D]
            # Saving:
            for idx, img_name in enumerate(sample["imgname"]):
                img_name = img_name.split('.jpg')[0] + "_" + caption[idx].replace(" ", "").replace("/", "").replace(".", "")
                np.save(os.path.join(output_directory, f'{img_name}.npy'), z[idx])


if __name__ == "__main__":
    #"""
    dataset = COCOval(size=256, pre_compute_img_emd=True)
    pre_encode_vqgan4(dataset=dataset, output_directory="/xxx/scratch/xxx/COCO_dataset_embedding/train2017_vq4",
                      batch_size=128)
    #"""
