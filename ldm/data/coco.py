import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import json
import random
from tqdm import tqdm
from ldm.data.lsun import pixelate_image_numpy


class Base(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 caption_root = "",
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 pre_compute_img_emd=False
                 ):
        self.data_paths = txt_file
        self.caption_root = caption_root
        self.data_root = data_root
        self.pre_compute_img_emd = pre_compute_img_emd
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
        }

        # Load Captions:
        with open(self.caption_root, 'r') as fp:
            json_data = json.load(fp)
        image_id_to_file_name = {}
        for img in tqdm(json_data['images']):
            #if img['file_name'] in self.image_paths:
            image_id_to_file_name[img['id']] = img['file_name']
        self.caption_image_pairs = []
        for ann in tqdm(json_data['annotations']):
            image_id = ann['image_id']
            caption = ann['caption']
            file_name = image_id_to_file_name.get(image_id)
            if file_name:
                self.caption_image_pairs.append((caption, file_name))

        if self.pre_compute_img_emd:
            self._length = len(image_id_to_file_name)
        else:
            self._length = len(self.caption_image_pairs)
        print("The length is : ", txt_file, "  -->  ", self._length)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        self.images_names = json_data['images']

    def __getitem__(self, i):
        example = dict()
        if self.pre_compute_img_emd:
            imgname = self.images_names[i]['file_name']
        else:
            caption, imgname = self.caption_image_pairs[i]
            example["caption"] = caption
        image_path = os.path.join(self.data_root, imgname)
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["imgname"] = imgname
        
        return example

    def __len__(self):
        return self._length
    
    
class CocoCoT(Base):
    def __init__(self, cot_stages, cot_process_type, gray_flg, palette, aug_y_p, aug_types, drop_y_p,
                 binary_y=False, add_noise_y=False, timesteps=1000, timesteps_1ststage=200, gray_y=False,
                 truncation_ratio=0.025, **kwargs):
        super().__init__(**kwargs)
        self.cot_stages = cot_stages
        self.gray_flg = gray_flg
        self.cot_process_type = cot_process_type
        self.palette = palette
        self.aug_y_p = aug_y_p
        self.aug_types = aug_types
        self.drop_y_p = drop_y_p
        self.binary_y = binary_y
        self.gray_y = gray_y
        self.truncation_ratio = truncation_ratio
        self.timesteps = timesteps
        self.timesteps_1ststage = timesteps_1ststage
        self.add_noise_y = add_noise_y
        self.cot_data_roots = []

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }
        for i, cot_stage in enumerate(self.cot_stages):
            if (cot_stage == "Black1D") or (cot_stage == "Black_White"):
                cot_stage = "sketch"
            elif cot_stage == "sam_edges":
                cot_stage = "sam"
            self.cot_data_roots.append(self.data_root + "_" + cot_stage)
            self.labels["file_path_"+cot_stage+"_"] = [os.path.join(self.cot_data_roots[i], l) for l in self.image_paths]

        # Load Captions:
        with open(self.caption_root, 'r') as fp:
            json_data = json.load(fp)
        image_id_to_file_name = {img['id']: img['file_name'] for img in json_data['images']}
        caption_image_pairs = {}
        for ann in json_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            image_id = str(image_id).zfill(12)+'.jpg'
            if image_id in caption_image_pairs.keys():
                caption_image_pairs[image_id].append(caption)
            else:
                caption_image_pairs[image_id] = [caption]
        if "val" in self.data_paths:
            self.labels["caption"] = [caption_image_pairs[l.replace('1', '0', 1)] for l in self.image_paths]
        else:
            self.labels["caption"] = [caption_image_pairs[l] for l in self.image_paths]

        # Augmentation:
        if "sketch" in aug_types:
            self.labels["file_path_"+"sketch"+"_"] = [os.path.join(self.data_root + "_" + "sketch", l) for l in self.image_paths]
        elif "edges" in aug_types:
            self.labels["file_path_"+"edges"+"_"] = [os.path.join(self.data_root + "_" + "edges", l) for l in self.image_paths]

        if self.add_noise_y:
            m_t_s_2nd = np.linspace(0.001, 0.999, self.timesteps)
            m_t_s_1st = np.linspace(0.001, 0.999, self.timesteps_1ststage)
            self.S2 = np.sqrt(m_t_s_2nd - m_t_s_2nd ** 2)
            self.S1 = np.sqrt(m_t_s_1st)

    def aug_mask(self, image, mask_p, patch_size = 3):
        """
        image: PIL Image
        mask_p: Masking Probability
        patch_size: Define the patch size (3x3 in this example)
        """
        image = np.array(image).astype(np.uint8)
        
        # Calculate the number of patches to be masked
        h, w, _ = image.shape
        num_patches_w = w//patch_size
        num_patches_h = h//patch_size
        num_patches_x_y_min = min(num_patches_w, num_patches_h)
        num_patches = num_patches_w*num_patches_h
        
        # Calculate the number of patches to mask based on the percentage
        num_patches_to_mask = int(num_patches * mask_p / 100)
        # Create a mask of the same size as the image
        mask = np.ones((h, w, 3))
        # Determine the positions of patches to mask
        masked_indices = np.random.choice(num_patches, num_patches_to_mask, replace=False)
        row_indices = masked_indices // num_patches_w
        col_indices = masked_indices % num_patches_w
        y_coords = row_indices
        x_coords = col_indices

        # Apply the masking by setting pixels in the patches to zero
        for y, x in zip(y_coords, x_coords):
            idx_y = y*patch_size
            idx_x = x*patch_size
            mask[idx_y:idx_y+patch_size, idx_x:idx_x+patch_size, :] = 0

        # Apply the mask to the image
        masked_img = (image * mask).astype(np.uint8)

        return masked_img

    def aug_sampling(self, image, sample_p):
        """
        image: PIL Image
        mask_p: Masking Probability
        patch_size: Define the patch size (3x3 in this example)
        """
        image = np.array(image).astype(np.uint8)
        mask = np.zeros_like(image, dtype=int)
        h, w, _ = image.shape
        total_elements = h * w
        keep_p = 100 - sample_p
        num_ones = int((keep_p / 100) * total_elements)
        # Generate random positions for ones
        indices = np.random.choice(total_elements, num_ones, replace=False)
        row_indices, col_indices = np.unravel_index(indices, (h, w))
        # Fill the array with ones at the random positions
        mask[row_indices, col_indices] = 1
        # Mask the image:
        sampled_img = (image * mask).astype(np.uint8)

        return sampled_img

    def _norm_img(self, image, norm_flag, norm_range):
        if norm_flag:
            if norm_range == [1, -1]:
                return (image / 127.5 - 1.0).astype(np.float32)
            elif norm_range == [1, 0]:
                return (image / image.max()).astype(np.float32)
            else:
                return None
        else:
            return image

    def _process_one_img(self, img_path, norm_range, img_type, gray_flg, norm_flag, get_edges_flag,
                         aug_y_p=0, aug_types=[], drop_y_p=0):
        """
        img_path: Take the image path.
        norm_range: [1, 0] or [1, -1]
        img_type: "rgb", "binary"
        out: Return the image itself after croping, augmentations, etc...
        """
        image = Image.open(img_path)
        if gray_flg:
            image = image.convert('L')
        else:
            if not image.mode == "RGB":
                image = image.convert("RGB")

        if get_edges_flag:
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
            image = image.filter(ImageFilter.MaxFilter)
            img = np.expand_dims(np.array(image), axis=-1)
            img = np.repeat(img, 3, axis=-1).astype(np.uint8)
        else:
            img = np.array(image).astype(np.uint8)

        # default to score-sde preprocessing
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # Augmentations:
        if self.rand_num > self.flip_p:
            image = self.flip(image)
        #"""
        aug_rand_int = random.randint(1, 100)
        if 0 <= aug_rand_int < aug_y_p*len(aug_types):
            aug_id = int(aug_rand_int // aug_y_p)
            if aug_types[aug_id] == "masking":
                image = self.aug_mask(image, mask_p=random.randint(5, 20), patch_size=3)
            elif aug_types[aug_id] == "downsampling":
                image = self.aug_sampling(image, sample_p=random.randint(5, 20))
            #elif aug_types[aug_id] == "edges":
            #elif aug_types[aug_id] == "sketch":
        #elif aug_y_p*len(aug_types) <= aug_rand_int <= aug_y_p*len(aug_types)+drop_y_p:
        #"""
        
        image = np.array(image).astype(np.uint8)

        if gray_flg:
            image = image.reshape((self.size, self.size, 1))

        if img_type == "binary":
            mid_val = (image.max()-image.min())/2
            image = (image>mid_val).astype(np.uint8)

        return self._norm_img(image, norm_flag, norm_range)

    def __getitem__(self, i):
        get_edges_flag = False
        self.rand_num = random.random()
        example = dict((k, self.labels[k][i]) for k in self.labels)
        # Load Caption:
        example["caption"] = random.choice(example["caption"])
        #import pdb; pdb.set_trace()
        # Load the Target Image:
        if self.cot_stages[0] == "Black_White" or self.cot_stages[0] == "Black1D":
            example["image"] = self._process_one_img(img_path=example["file_path_sketch_"],
                                                     norm_range=[1, 0], img_type="binary", 
                                                     gray_flg=True, norm_flag=(True and not self.palette),
                                                     get_edges_flag=get_edges_flag)
        else:
            example["image"] = self._process_one_img(img_path=example["file_path_"], norm_range=[1, -1], img_type="rgb",
                                                     gray_flg=self.gray_flg, norm_flag=(True and not self.palette),
                                                     get_edges_flag=get_edges_flag)
        # Load the Cond Images:
        for cot_stage in self.cot_stages:
            if cot_stage == "Black_White":
                example["image"+"_"+cot_stage] = np.zeros((self.size, self.size, 1))
                white_ratio = 0.1
                # Generate random numbers for each pixel
                random_numbers = np.random.rand(self.size, self.size, 1)
                # Set pixels to white based on the ratio
                example["image"+"_"+cot_stage][random_numbers < white_ratio] = 1.0
            elif cot_stage == "Black1D":
                example["image"+"_"+cot_stage] = np.zeros((self.size, self.size, 1))
            else:
                if cot_stage == "sam_edges":
                    cot_stage = "sam"
                    get_edges_flag = True
                
                if self.cot_process_type == "with_norm":
                    example["image"+"_"+cot_stage] = self._process_one_img(img_path=example["file_path_"+cot_stage+"_"],
                                                                        norm_range=[1, 0], gray_flg=self.gray_y,
                                                                        img_type="binary" if self.binary_y else None, 
                                                                        norm_flag=(True and not self.palette),
                                                                        get_edges_flag=get_edges_flag,
                                                                        aug_y_p=self.aug_y_p, aug_types=self.aug_types,
                                                                        drop_y_p=self.drop_y_p)
                elif self.cot_process_type == "no_norm":
                    example["image"+"_"+cot_stage] = self._process_one_img(img_path=example["file_path_"+cot_stage+"_"],
                                                                        norm_range=None, img_type=None, 
                                                                        gray_flg=False, norm_flag=False,
                                                                        get_edges_flag=get_edges_flag,
                                                                        aug_y_p=self.aug_y_p, aug_types=self.aug_types,
                                                                        drop_y_p=self.drop_y_p)
            
            # Add Noise to the Condition Y to work with the truncation method:
            if self.add_noise_y:
                if self.truncation_ratio > 0:
                    idx = np.argmin(np.abs(self.S2[int(self.timesteps/2):] - self.S1[int(self.truncation_ratio*self.timesteps_1ststage)]))
                    var_noise = self.S2[idx+int(self.timesteps/2)]
                else:
                    var_noise = self.S2[random.randrange(int(len(self.S2)/1))]
                example["image"+"_"+cot_stage] += var_noise*np.random.randn(*example["image"+"_"+cot_stage].shape)
                example["image"+"_"+cot_stage] = self._norm_img(example["image"+"_"+cot_stage], True, [1, 0])
            
            # Overlay the Palette on the Sketch:
            if self.palette:
                # Convert the mask to binary values:
                img_sketch = example["image"+"_"+cot_stage]
                img_sketch = np.array(img_sketch).astype(np.uint8)
                mid_val = (img_sketch.max()-abs(img_sketch.min()))/2
                img_sketch_binary = (img_sketch>mid_val).astype(np.uint8)
                # Pixelate the RGB:
                img_pixelate = pixelate_image_numpy(example["image"], block_size=32, avg=True)
                # Overlay the palette on the sketch:
                fused_img = (img_sketch_binary*0.5*img_sketch) + (img_sketch_binary*(1.-0.5)*img_pixelate) + ((1-img_sketch_binary)*img_pixelate)
                fused_img = fused_img.astype(np.uint8)
                # Normalize:
                if self.cot_process_type == "with_norm":
                    example["image"+"_"+cot_stage] = self._norm_img(fused_img, norm_flag=True, norm_range=[1, 0])
                elif self.cot_process_type == "no_norm":
                    example["image"+"_"+cot_stage] = self._norm_img(fused_img, norm_flag=False, norm_range=None)
                example["image"] = self._norm_img(example["image"], norm_flag=True, norm_range=[1, -1])

        return example


class COCOtrain(Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/coco/coco_train.txt",
                         caption_root="/xxx/ai/reference/CV/COCO/cocoapi/data/2017/annotations/captions/train/captions_train2017.json",
                         data_root = "/xxx/scratch/xxx/COCO_dataset/train", **kwargs)
        
class COCOval(Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/coco/coco_val.txt",
                         caption_root="/xxx/ai/reference/CV/COCO/cocoapi/data/2017/annotations/captions/val/captions_val2017.json",
                         data_root = "/xxx/scratch/xxx/COCO_dataset/val2017", **kwargs)


def keystoint(x):
    return {int(k): v for k, v in x.items()}


class COCO_load_emb(Base):
    # This class should be activated to load the stored embeddings.
    def __init__(self, img_emb_path, txt_emb_path, caption_root, imageid2filename_dict_path=None, size=None, interpolation="bicubic", flip_p=0.5):
        self.caption_root = caption_root
        self.img_emb_path = img_emb_path
        self.txt_emb_path = txt_emb_path

        # Load COCO json:
        with open(self.caption_root, 'r') as fp:
            json_data = json.load(fp)

        # Load images names that we have emb for them:
        self.imgs_emb_names_wo_ext = [f.split(".npy")[0] for f in os.listdir(self.img_emb_path) if os.path.isfile(os.path.join(self.img_emb_path, f))]

        # Create imgID2imgname mapping dict:
        if imageid2filename_dict_path is not None:
            with open(imageid2filename_dict_path, 'r') as fp:
                image_id_to_file_name = json.load(fp, object_hook=keystoint)
        else:
            image_id_to_file_name = {}
            for img in tqdm(json_data['images']):
                # Skip the images that we don't have emb for them:
                if img['file_name'].split(".")[0] in self.imgs_emb_names_wo_ext:
                    image_id_to_file_name[img['id']] = img['file_name']
            with open('coco_val_imgid2name_dict.json', 'w') as fp:
                json.dump(image_id_to_file_name, fp)

        self.caption_image_pairs = []
        for ann in tqdm(json_data['annotations']):
            image_id = ann['image_id']
            caption = ann['caption']
            file_name = image_id_to_file_name.get(image_id)
            if file_name:
                self.caption_image_pairs.append((caption, file_name))

        self._length = len(self.caption_image_pairs)
        print("The length is : ", txt_emb_path, "  -->  ", self._length)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        self.images_names = json_data['images']

    def __getitem__(self, i):
        example = dict()
        caption, imgname = self.caption_image_pairs[i]
        # Load pre-computed emb:
        image_path = os.path.join(self.img_emb_path, imgname.split(".")[0]+".npy")
        example["image"] = np.load(image_path, allow_pickle=True)  # [B, H, W] [B, 64, 64]
        txt_embfile_name = imgname.split(".")[0] + "_" + caption.replace(" ", "").replace("/", "").replace(".", "")
        txt_path = os.path.join(self.txt_emb_path, txt_embfile_name+".npy")
        example["caption"] = np.load(txt_path, allow_pickle=True)
        example["caption_as_txt"] = caption
        
        return example
        

class CoT_COCO_load_emb(COCO_load_emb):
    # This class should be activated to load the stored embeddings in the CoT case.
    def __init__(self, img_emb_path, txt_emb_path, caption_root, 
                 cot_stages, cot_process_type, gray_flg, palette, aug_y_p, aug_types, drop_y_p, data_root="",
                 binary_y=False, add_noise_y=False, timesteps=1000, timesteps_1ststage=200, gray_y=False, truncation_ratio=0.025,
                 imageid2filename_dict_path=None, size=None, interpolation="bicubic", flip_p=0.5):
        super().__init__(img_emb_path, txt_emb_path, caption_root, imageid2filename_dict_path, size, interpolation, flip_p)
        self.cot_stages = cot_stages
        self.cot_process_type = cot_process_type
        self.gray_flg = gray_flg
        self.palette = palette
        self.aug_y_p = aug_y_p
        self.aug_types = aug_types
        self.drop_y_p = drop_y_p
        self.data_root = data_root
        self.binary_y = binary_y
        self.add_noise_y = add_noise_y
        self.timesteps = timesteps
        self.timesteps_1ststage = timesteps_1ststage
        self.gray_y = gray_y
        self.truncation_ratio = truncation_ratio
        
        self.cot_data_roots = []

        # Load the images names for the Y in CoT:
        self.labels = {}
        for i, cot_stage in enumerate(self.cot_stages):
            if (cot_stage == "Black1D") or (cot_stage == "Black_White"):
                cot_stage = "sketch"
            elif cot_stage == "sam_edges":
                cot_stage = "sam"
            self.cot_data_roots.append(self.data_root + "_" + cot_stage)
            self.labels["file_path_"+cot_stage+"_"] = [os.path.join(self.cot_data_roots[i], l['file_name']) for l in self.images_names]

        # Augmentation:
        if "sketch" in aug_types:
            self.labels["file_path_"+"sketch"+"_"] = [os.path.join(self.data_root + "_" + "sketch", l['file_name']) for l in self.images_names]
        elif "edges" in aug_types:
            self.labels["file_path_"+"edges"+"_"] = [os.path.join(self.data_root + "_" + "edges", l['file_name']) for l in self.images_names]

        if self.add_noise_y:
            m_t_s_2nd = np.linspace(0.001, 0.999, self.timesteps)
            m_t_s_1st = np.linspace(0.001, 0.999, self.timesteps_1ststage)
            self.S2 = np.sqrt(m_t_s_2nd - m_t_s_2nd ** 2)
            self.S1 = np.sqrt(m_t_s_1st)

    def aug_mask(self, image, mask_p, patch_size = 3):
        """
        image: PIL Image
        mask_p: Masking Probability
        patch_size: Define the patch size (3x3 in this example)
        """
        image = np.array(image).astype(np.uint8)
        
        # Calculate the number of patches to be masked
        h, w, _ = image.shape
        num_patches_w = w//patch_size
        num_patches_h = h//patch_size
        num_patches_x_y_min = min(num_patches_w, num_patches_h)
        num_patches = num_patches_w*num_patches_h
        
        # Calculate the number of patches to mask based on the percentage
        num_patches_to_mask = int(num_patches * mask_p / 100)
        # Create a mask of the same size as the image
        mask = np.ones((h, w, 3))
        # Determine the positions of patches to mask
        masked_indices = np.random.choice(num_patches, num_patches_to_mask, replace=False)
        row_indices = masked_indices // num_patches_w
        col_indices = masked_indices % num_patches_w
        y_coords = row_indices
        x_coords = col_indices

        # Apply the masking by setting pixels in the patches to zero
        for y, x in zip(y_coords, x_coords):
            idx_y = y*patch_size
            idx_x = x*patch_size
            mask[idx_y:idx_y+patch_size, idx_x:idx_x+patch_size, :] = 0

        # Apply the mask to the image
        masked_img = (image * mask).astype(np.uint8)

        return masked_img

    def aug_sampling(self, image, sample_p):
        """
        image: PIL Image
        mask_p: Masking Probability
        patch_size: Define the patch size (3x3 in this example)
        """
        image = np.array(image).astype(np.uint8)
        mask = np.zeros_like(image, dtype=int)
        h, w, _ = image.shape
        total_elements = h * w
        keep_p = 100 - sample_p
        num_ones = int((keep_p / 100) * total_elements)
        # Generate random positions for ones
        indices = np.random.choice(total_elements, num_ones, replace=False)
        row_indices, col_indices = np.unravel_index(indices, (h, w))
        # Fill the array with ones at the random positions
        mask[row_indices, col_indices] = 1
        # Mask the image:
        sampled_img = (image * mask).astype(np.uint8)

        return sampled_img

    def _norm_img(self, image, norm_flag, norm_range):
        if norm_flag:
            if norm_range == [1, -1]:
                return (image / 127.5 - 1.0).astype(np.float32)
            elif norm_range == [1, 0]:
                return (image / image.max()).astype(np.float32)
            else:
                return None
        else:
            return image

    def _process_one_img(self, img_path, norm_range, img_type, gray_flg, norm_flag, get_edges_flag,
                         aug_y_p=0, aug_types=[], drop_y_p=0):
        """
        img_path: Take the image path.
        norm_range: [1, 0] or [1, -1]
        img_type: "rgb", "binary"
        out: Return the image itself after croping, augmentations, etc...
        """
        image = Image.open(img_path)
        if gray_flg:
            image = image.convert('L')
        else:
            if not image.mode == "RGB":
                image = image.convert("RGB")

        if get_edges_flag:
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
            image = image.filter(ImageFilter.MaxFilter)
            img = np.expand_dims(np.array(image), axis=-1)
            img = np.repeat(img, 3, axis=-1).astype(np.uint8)
        else:
            img = np.array(image).astype(np.uint8)

        # default to score-sde preprocessing
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # Augmentations:
        if self.rand_num > self.flip_p:
            image = self.flip(image)
        #"""
        aug_rand_int = random.randint(1, 100)
        if 0 <= aug_rand_int < aug_y_p*len(aug_types):
            aug_id = int(aug_rand_int // aug_y_p)
            if aug_types[aug_id] == "masking":
                image = self.aug_mask(image, mask_p=random.randint(5, 20), patch_size=3)
            elif aug_types[aug_id] == "downsampling":
                image = self.aug_sampling(image, sample_p=random.randint(5, 20))
            #elif aug_types[aug_id] == "edges":
            #elif aug_types[aug_id] == "sketch":
        #elif aug_y_p*len(aug_types) <= aug_rand_int <= aug_y_p*len(aug_types)+drop_y_p:
        #"""
        
        image = np.array(image).astype(np.uint8)

        if gray_flg:
            image = image.reshape((self.size, self.size, 1))

        if img_type == "binary":
            mid_val = (image.max()-image.min())/2
            image = (image>mid_val).astype(np.uint8)

        return self._norm_img(image, norm_flag, norm_range)

    def __getitem__(self, i):
        example = dict()
        caption, imgname = self.caption_image_pairs[i]
        # Load pre-computed emb:
        image_path = os.path.join(self.img_emb_path, imgname.split(".")[0]+".npy")
        example["image"] = np.load(image_path, allow_pickle=True)  # [B, H, W] [B, 64, 64]
        txt_embfile_name = imgname.split(".")[0] + "_" + caption.replace(" ", "").replace("/", "").replace(".", "")
        txt_path = os.path.join(self.txt_emb_path, txt_embfile_name+".npy")
        example["caption"] = np.load(txt_path, allow_pickle=True)
        example["caption_as_txt"] = caption

        # Load the CoT Y:
        get_edges_flag = False
        self.rand_num = random.random()
        # Overwrite the Target Image in case we are training for other output rather than the RGB:
        if self.cot_stages[0] == "Black_White" or self.cot_stages[0] == "Black1D":
            example["image"] = self._process_one_img(img_path=os.path.join(self.cot_data_roots[0], imgname.split(".")[0]+".png"),
                                                     norm_range=[1, 0], img_type="binary", 
                                                     gray_flg=True, norm_flag=(True and not self.palette),
                                                     get_edges_flag=get_edges_flag)

        # Load the Cond Images:
        for j, cot_stage in enumerate(self.cot_stages):
            if cot_stage == "Black_White":
                example["image"+"_"+cot_stage] = np.zeros((self.size, self.size, 1))
                white_ratio = 0.1
                # Generate random numbers for each pixel
                random_numbers = np.random.rand(self.size, self.size, 1)
                # Set pixels to white based on the ratio
                example["image"+"_"+cot_stage][random_numbers < white_ratio] = 1.0
            elif cot_stage == "Black1D":
                example["image"+"_"+cot_stage] = np.zeros((self.size, self.size, 1))
            else:
                if cot_stage == "sam_edges":
                    cot_stage = "sam"
                    get_edges_flag = True
                
                if self.cot_process_type == "with_norm":
                    example["image"+"_"+cot_stage] = self._process_one_img(img_path=os.path.join(self.cot_data_roots[j], imgname.split(".")[0]+".png"),
                                                                           norm_range=[1, 0], gray_flg=self.gray_y,
                                                                           img_type="binary" if self.binary_y else None, 
                                                                           norm_flag=(True and not self.palette),
                                                                           get_edges_flag=get_edges_flag,
                                                                           aug_y_p=self.aug_y_p, aug_types=self.aug_types,
                                                                           drop_y_p=self.drop_y_p)
                elif self.cot_process_type == "no_norm":
                    example["image"+"_"+cot_stage] = self._process_one_img(img_path=os.path.join(self.cot_data_roots[j], imgname.split(".")[0]+".png"),
                                                                           norm_range=None, img_type=None, 
                                                                           gray_flg=False, norm_flag=False,
                                                                           get_edges_flag=get_edges_flag,
                                                                           aug_y_p=self.aug_y_p, aug_types=self.aug_types,
                                                                           drop_y_p=self.drop_y_p)
            
            # Add Noise to the Condition Y to work with the truncation method:
            if self.add_noise_y:
                if self.truncation_ratio > 0:
                    idx = np.argmin(np.abs(self.S2[int(self.timesteps/2):] - self.S1[int(self.truncation_ratio*self.timesteps_1ststage)]))
                    var_noise = self.S2[idx+int(self.timesteps/2)]
                else:
                    var_noise = self.S2[random.randrange(int(len(self.S2)/1))]
                example["image"+"_"+cot_stage] += var_noise*np.random.randn(*example["image"+"_"+cot_stage].shape)
                example["image"+"_"+cot_stage] = self._norm_img(example["image"+"_"+cot_stage], True, [1, 0])
            
            # Overlay the Palette on the Sketch:
            if self.palette:
                # Convert the mask to binary values:
                img_sketch = example["image"+"_"+cot_stage]
                img_sketch = np.array(img_sketch).astype(np.uint8)
                mid_val = (img_sketch.max()-abs(img_sketch.min()))/2
                img_sketch_binary = (img_sketch>mid_val).astype(np.uint8)
                # Pixelate the RGB:
                rgb_image_path = os.path.join(self.data_root, imgname)
                rgb_img = self._process_one_img(img_path=rgb_image_path, norm_range=[1, -1], img_type="rgb",
                                                gray_flg=False, norm_flag=False, get_edges_flag=False)
                img_pixelate = pixelate_image_numpy(rgb_img, block_size=32, avg=True)
                # Overlay the palette on the sketch:
                fused_img = (img_sketch_binary*0.5*img_sketch) + (img_sketch_binary*(1.-0.5)*img_pixelate) + ((1-img_sketch_binary)*img_pixelate)
                fused_img = fused_img.astype(np.uint8)
                # Normalize:
                if self.cot_process_type == "with_norm":
                    example["image"+"_"+cot_stage] = self._norm_img(fused_img, norm_flag=True, norm_range=[1, 0])
                elif self.cot_process_type == "no_norm":
                    example["image"+"_"+cot_stage] = self._norm_img(fused_img, norm_flag=False, norm_range=None)

        return example