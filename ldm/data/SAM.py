import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from lsun import get_files_in_dir
import threading


def split_list(lst, n):
    """Split a list into sublists containing n elements."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def thread_function(i):
    print("Thread %s: starting", str(i))
    sam = SAM(sam_checkpoint="../../weights/sam_vit_h_4b8939.pth", model_type="vit_h",points_per_side=64, gpu_id=i)
    for file_name in tqdm(files_names[i]):
        sam(img_path=os.path.join(imgs_dir, file_name), saving_path=os.path.join(saving_dir, file_name))


class SAM:
    def __init__(self, sam_checkpoint, model_type, points_per_side, gpu_id):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device="cuda:"+str(gpu_id))
        self.mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)

    def __call__(self, img_path, saving_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        self._save_anns(anns=masks, saving_path=saving_path)

    def _save_anns(self, anns, saving_path):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)  # larger first

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        for ann in sorted_anns:
            m = ann['segmentation']  # 2D mask
            color_mask = np.random.random(3)
            img[m] = color_mask
        plt.imsave(saving_path, img)


if __name__ == '__main__':
    np.random.rand(4)
    imgs_dir = "../../data/lsun/churches/"
    saving_dir = "../../data/lsun/churches_sam/"

    # Test One image
    """
    sam = SAM(sam_checkpoint="../../weights/sam_vit_h_4b8939.pth", model_type="vit_h",points_per_side=64, gpu_id=0)
    sam(img_path=os.path.join(imgs_dir, "50df56048f09bb22ec032dedea6aaa529a086342.webp"),
        saving_path=os.path.join(saving_dir, "SAM-50df56048f09bb22ec032dedea6aaa529a086342.webp"))
    """

    files_names = get_files_in_dir(in_dir=imgs_dir)
    files_names = np.array(files_names)
    files_names = np.array_split(files_names, 24)
    files_names = [list(array) for array in files_names]

    files_names = files_names[:8]
    #files_names = files_names[8:16]
    #files_names = files_names[16:]

    for chunk in files_names:
        print(len(chunk))

    threads = list()
    for index in range(8):
        x = threading.Thread(target=thread_function, args=(index,))
        threads.append(x)
        x.start()

    #126527