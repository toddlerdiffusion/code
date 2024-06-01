import os
import numpy as np
import PIL
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import random
import cv2


def generate_random_int_with_prob(min_val, max_val, probs):
    # Generate a range of integers from min_val to max_val (inclusive)
    numbers = np.arange(min_val, max_val + 1)

    # Generate a random integer based on the specified probabilities
    random_int = np.random.choice(numbers, p=probs)

    return random_int


def generate_blurred_image_palette(img, blur_radius, blurred_image_path=None):
    """
    Blurs an image to create a color palette effect, based on a specified blur radius, and saves the result.
    
    Parameters:
    - image_path: The path to the source image file.
    - blur_radius: The radius of the Gaussian blur to apply. Higher values result in a more blurred image.
    
    The function saves the blurred image in the same directory as the original, appending '_blurred' to the file name.
    """
    # Apply Gaussian blur
    blurred_img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    if blurred_image_path:
        blurred_img.save(blurred_image_path)
        print(f"Blurred image saved as: {blurred_image_path}")

    return blurred_img


def pixelate_image_torch(img, block_size, stride, avg=True):
    """
    img: tensor in shape of [B, C, H, W]
    block_size: [B] ints represent kernel size. Higher leads to more pixelization.
    stride: [B] ints represent stride size. Higher leads to less pixelization.
    avg: True for averaging colors in each block, False for keeping the color of the top-left pixel.
    Return:
    img_pixelated: tensor in shape of [B, C, H, W]
    """
    # Get the image dimensions
    batch_size, channels, height, width = img.size()

    # Reshape image to operate on blocks
    img_reshaped = img.view(batch_size, channels, height // block_size, block_size, width // block_size, block_size)
    
    if avg:
        # Calculate the average color in each block
        avg_color = img_reshaped.mean(dim=(3, 5), keepdim=True)
        avg_color = avg_color.expand(-1, -1, -1, block_size, -1, block_size)
    else:
        avg_color = img_reshaped[:, :, :, 0:1, :, 0:1]

    # Expand the color block to match the original image size
    avg_color = avg_color.expand(batch_size, channels, height // block_size, block_size, width // block_size, block_size)

    # Reshape back to the original image size
    img_pixelated = avg_color.contiguous().view(batch_size, channels, height, width)

    return img_pixelated


def pixelate_image_numpy(img, block_size, avg=True):
    """
    img: NumPy array in shape of [H, W, C]
    block_size: int represent kernel size. Higher leads to more pixelization.
    avg: True for averaging colors in each block, False for keeping the color of the top-left pixel.
    """
    # Get the image dimensions
    height, width, channels = img.shape

    # Reshape image to operate on blocks
    img_reshaped = img.reshape(height // block_size, block_size, width // block_size, block_size, channels)

    if avg:
        # Calculate the average color in each block
        avg_color = img_reshaped.mean(axis=(1, 3), keepdims=True)
        avg_color = np.tile(avg_color, (1, block_size, 1, block_size, 1))
    else:
        avg_color = img_reshaped[:, 0:1, :, 0:1, :]

    # Reshape back to the original image size
    img_pixelated = avg_color.reshape(height, width, channels)

    return img_pixelated


class LSUNBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5, gray_flg=False, norm_flag=True
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        self.gray_flg = gray_flg
        self.norm_flag = norm_flag

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        self.rand_num = random.random()
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])

        if self.gray_flg:
            image = image.convert('L')
        else:
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

        if (self.flip_p>0) and (self.rand_num > self.flip_p):
            image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        if self.norm_flag:
            example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        else:
            example["image"] = image.astype(np.float32)
        return example


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, out_modality="sketch", **kwargs):
        data_root="data/lsun/churches"
        if out_modality != "RGB":
            data_root += "_" + out_modality
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root=data_root, **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., out_modality="sketch", **kwargs):
        data_root="data/lsun/churches"
        if out_modality != "RGB":
            data_root += "_" + out_modality
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root=data_root,
                         flip_p=flip_p, **kwargs)


class LSUNChurchesCoT(LSUNBase):
    def __init__(self, cot_stages, cot_process_type, gray_flg, palette, aug_y_p, aug_types, drop_y_p,
                 binary_y=False, add_noise_y=False, timesteps=1000, timesteps_1ststage=200, gray_y=False,
                 truncation_ratio=0.0, morphological_trans=False, morphological_range=[1,1],
                 out_palette=False, blur_radius=None, noise_mult=1, sketch_variants_prob=None, downsize_y=None, **kwargs):
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
        self.noise_mult = noise_mult
        self.cot_data_roots = []
        self.morphological_trans = morphological_trans
        self.morphological_range = morphological_range 
        self.blur_radius = blur_radius
        self.out_palette = out_palette
        self.sketch_variants_prob = sketch_variants_prob
        self.downsize_y = downsize_y
        if self.morphological_trans:
            # TODO: this one should be configurable
            self.structuredEdgeDetection = cv2.ximgproc.createStructuredEdgeDetection("ldm/data/model.yml.gz")

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
            if self.sketch_variants_prob:
                sketch_variants_paths = []
                for s_var_id, s_var_prob in enumerate(self.sketch_variants_prob):
                    sketch_variants_paths.append(os.path.join(self.data_root + "_" + cot_stage, str(s_var_id)))
                    self.labels["file_path_"+cot_stage+"_"+str(s_var_id)+"_"] = [os.path.join(sketch_variants_paths[s_var_id], l.replace('.webp','.jpg')) for l in self.image_paths]
                self.cot_data_roots.append(sketch_variants_paths)
            else:
                self.cot_data_roots.append(self.data_root + "_" + cot_stage)
                self.labels["file_path_"+cot_stage+"_"] = [os.path.join(self.cot_data_roots[i], l) for l in self.image_paths]
                """
                s_var_id = 8  # TODO: make this configurable
                self.cot_data_roots.append(os.path.join(self.data_root + "_" + cot_stage, str(s_var_id)))
                self.labels["file_path_"+cot_stage+"_"] = [os.path.join(self.cot_data_roots[i], l.replace('.webp','.jpg')) for l in self.image_paths]
                """

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
                return (image / 127.5 - 1.0).astype(np.float32)  # [1, 0] --> [2, 0]  --> [1, -1] 
                #return (((image / image.max())*2)-1).astype(np.float32)  # [1, 0] --> [2, 0]  --> [1, -1] 
            elif norm_range == [1, 0]:
                return (image / image.max()).astype(np.float32)
            else:
                return None
        else:
            return image

    def _process_one_img(self, img_path, norm_range, img_type, gray_flg, norm_flag, get_edges_flag, downsize_y=None,
                         aug_y_p=0, aug_types=[], drop_y_p=0, image=None, out_palette=False, blur_radius=None):
        """
        img_path: Take the image path.
        norm_range: [1, 0] or [1, -1]
        img_type: "rgb", "binary"
        out: Return the image itself after croping, augmentations, etc...
        """
        if image is None:
            image = Image.open(img_path)

        if gray_flg:
            image = image.convert('L')
        else:
            if not image.mode == "RGB":
                image = image.convert("RGB")

        if out_palette:
            image = generate_blurred_image_palette(img=image, blur_radius=random.randint(blur_radius[0], blur_radius[1]))
            
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
        if downsize_y:
            image = image.resize((downsize_y, downsize_y), resample=self.interpolation)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # Augmentations:
        if (self.flip_p>0) and (self.rand_num > self.flip_p):
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
        # Load the Target Image:
        if self.cot_stages[0] == "Black_White" or self.cot_stages[0] == "Black1D":
            example["image"] = self._process_one_img(img_path=example["file_path_sketch_"],
                                                     norm_range=[1, -1], img_type="rgb", 
                                                     gray_flg=True, norm_flag=(True and not self.palette),
                                                     get_edges_flag=get_edges_flag)
        else:
            example["image"] = self._process_one_img(img_path=example["file_path_"], norm_range=[1, -1], img_type="rgb",
                                                     gray_flg=self.gray_flg, norm_flag=(True and not self.palette),
                                                     get_edges_flag=get_edges_flag, out_palette=self.out_palette, blur_radius=self.blur_radius)
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
                #example["image"+"_"+cot_stage] = np.zeros((self.size, self.size, 1))
                example["image"+"_"+cot_stage] = np.ones((self.size, self.size, 1))*-1
            else:
                if cot_stage == "sam_edges":
                    cot_stage = "sam"
                    get_edges_flag = True
                
                if self.cot_process_type == "with_norm":
                    if self.morphological_trans:
                        if random.random() < self.morphological_trans: # Code to execute with x% probability
                            min_line_length = random.randint(self.morphological_range[0], self.morphological_range[1])
                            example["file_path_"+cot_stage+"_"] = example["file_path_"+cot_stage+"_"].replace("_sketch", "")
                            image_vec = cv2.imread(example["file_path_"+cot_stage+"_"], cv2.IMREAD_COLOR)
                            g_blurred = cv2.GaussianBlur(image_vec, (3, 3), 0)
                            blurred_float = g_blurred.astype(np.float32) / 255.0
                            edges = self.structuredEdgeDetection.detectEdges(blurred_float)
                            _, binary_edges = cv2.threshold(edges*255, 40, 255, cv2.THRESH_BINARY)
                            line_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, min_line_length))
                            sketch = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, line_structure)
                            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
                            sketch = sketch.astype(np.uint8)
                            sketch_pil = Image.fromarray(sketch)
                            example["image"+"_"+cot_stage] = self._process_one_img(img_path=None, image=sketch_pil,
                                                                                    norm_range=[1, 0], gray_flg=self.gray_y,
                                                                                    img_type="binary" if self.binary_y else None, 
                                                                                    norm_flag=(True and not self.palette),
                                                                                    get_edges_flag=get_edges_flag,
                                                                                    aug_y_p=self.aug_y_p, aug_types=self.aug_types,
                                                                                    drop_y_p=self.drop_y_p)
                        else:
                            example["image"+"_"+cot_stage] = self._process_one_img(img_path=example["file_path_"+cot_stage+"_"],
                                                                                    norm_range=[1, 0], gray_flg=self.gray_y,
                                                                                    img_type="binary" if self.binary_y else None, 
                                                                                    norm_flag=(True and not self.palette),
                                                                                    get_edges_flag=get_edges_flag,
                                                                                    aug_y_p=self.aug_y_p, aug_types=self.aug_types,
                                                                                    drop_y_p=self.drop_y_p)
                    else:
                        if self.sketch_variants_prob:
                            s_var_id = generate_random_int_with_prob(min_val=0, max_val=len(self.sketch_variants_prob)-1,
                                                                     probs=self.sketch_variants_prob)
                            img_path = example["file_path_"+cot_stage+"_"+str(s_var_id)+"_"]
                        else:
                            img_path = example["file_path_"+cot_stage+"_"]
                        example["image"+"_"+cot_stage] = self._process_one_img(img_path=img_path,
                                                                                norm_range=[1, 0] if self.binary_y else [1, -1], 
                                                                                img_type="binary" if self.binary_y else None, 
                                                                                norm_flag=(True and not self.palette),
                                                                                get_edges_flag=get_edges_flag, gray_flg=self.gray_y,
                                                                                aug_y_p=self.aug_y_p, aug_types=self.aug_types,
                                                                                drop_y_p=self.drop_y_p, downsize_y=self.downsize_y)
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
                    var_noise = self.S2[random.randrange(int(len(self.S2)/1))]*self.noise_mult
                example["image"+"_"+cot_stage] += var_noise*np.random.randn(*example["image"+"_"+cot_stage].shape)
                example["image"+"_"+cot_stage] = self._norm_img(example["image"+"_"+cot_stage], True, [1, 0])
            
            # Overlay the Palette on the Sketch:
            if self.palette:
                # Convert the mask to binary values:
                img_sketch = example["image"+"_"+cot_stage]
                img_sketch = np.array(img_sketch).astype(np.uint8)
                mid_val = (img_sketch.max()-abs(img_sketch.min()))/2
                img_sketch_binary = (img_sketch>mid_val).astype(np.uint8)
                # Pixelate/Blur the RGB:
                img_pixelate = generate_blurred_image_palette(img=Image.fromarray(example["image"]), 
                                                              blur_radius=random.randint(self.blur_radius[0], self.blur_radius[1]))
                #img_pixelate = pixelate_image_numpy(example["image"], block_size=32, avg=True)
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


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)


class OpenImagesTrain(LSUNBase):
    def __init__(self, **kwargs):
        dataroot = "/xxx/ai/reference/openimagesv4/val/validation"
        super().__init__(txt_file="data/openimagesv4/openimagesv4_val.txt", data_root=dataroot, **kwargs)


class OpenImagesValidation(LSUNBase):
    def __init__(self, **kwargs):
        dataroot = "/xxx/ai/reference/openimagesv4/val/validation"
        txt_file = "data/openimagesv4/openimagesv4_val.txt"
        super().__init__(txt_file=txt_file, data_root=dataroot, **kwargs)


class ImageNetValidation(LSUNBase):
    def __init__(self, **kwargs):
        dataroot = "/xxx/scratch/xxx/imagenet_val"
        txt_file = "data/imagenet/imagenet_val.txt"
        super().__init__(txt_file=txt_file, data_root=dataroot, **kwargs)


def get_edges(img_pth, method, saving_pth=None):
    image = Image.open(img_pth)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = image.convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)
    image = image.filter(ImageFilter.MaxFilter)  # make edges thick. Comment it to get original edges. https://stackoverflow.com/questions/47063224/python-pil-outline-image-increase-thicknesses
    if saving_pth:
        image.save(saving_pth)


def get_files_in_dir(in_dir):
    files_names = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
    return files_names


def get_edges_in_batch(in_dir, method, saving_dir=None):
    files_names = get_files_in_dir(in_dir)
    for file_name in tqdm(files_names):
        get_edges(img_pth=os.path.join(in_dir, file_name), method=None, saving_pth=os.path.join(saving_dir, file_name))
        

def process_data_proper_FID_calc(input_directory, output_directory):
    # Process Val data for proper FID calculation:
    dataset = LSUNBase(txt_file="/xxx/user/xxx/LSUN_GT_fid_images_50k.txt", 
                       flip_p=0, size=256, data_root=input_directory)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(output_directory, exist_ok=True)
    for img_count in tqdm(range(dataset.__len__())):
        sample = dataset.__getitem__(img_count)
        image = sample['image']
        image = ((image + 1) / 2.0 * 255).astype(np.uint8)
        #img_name = sample['relative_file_path_']
        img_name = str(img_count).zfill(5) + ".webp"
        output_path = os.path.join(output_directory, os.path.basename(img_name))
        transforms.ToPILImage()(image).save(output_path)


if __name__ == '__main__':
    process_data_proper_FID_calc(input_directory="../../lsun_churches_sampled_50k",
                                 output_directory="/xxx/user/xxx/lsun_churches_sampled_50k_processed_cot")
