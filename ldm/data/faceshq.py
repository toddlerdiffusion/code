import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import bisect
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
import cv2
from ldm.data.lsun import pixelate_image_numpy, generate_blurred_image_palette
from torch.utils.data import Dataset, ConcatDataset, DataLoader


def norm_img(image, norm_flag, norm_range):
        if norm_flag:
            if norm_range == [1, -1]:
                return (image / 127.5 - 1.0).astype(np.float32)  # [1, 0] --> [2, 0]  --> [1, -1]
            elif norm_range == [1, 0]:
                return (image / image.max()).astype(np.float32)
            else:
                return None
        else:
            return image


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None, gray_flg=False, norm_flag=True,
                 out_palette=False, blur_radius=None, flip_p=0.0):
        self.size = size
        self.random_crop = random_crop
        self.gray_flg = gray_flg
        self.norm_flag = norm_flag
        self.out_palette = out_palette
        self.blur_radius = blur_radius
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, rand_num):
        image = Image.open(image_path)
        if self.gray_flg:
            image = image.convert('L')
        else:
            if not image.mode == "RGB":
                image = image.convert("RGB")
        
        if self.out_palette:
            image = generate_blurred_image_palette(img=image, blur_radius=random.randint(self.blur_radius[0], self.blur_radius[1]))

        # Augmentations:
        if (self.flip_p>0) and (rand_num > self.flip_p):
            image = self.flip(image)

        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.gray_flg:
            image = image.reshape((self.size, self.size, 1))
        
        if self.norm_flag:
            image = (image/127.5 - 1.0).astype(np.float32)

        return image

    def __getitem__(self, i, rand_num):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i], rand_num=rand_num)
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class ImagePathsCoT(ImagePaths):
    def __init__(self, paths, size=None, random_crop=False, labels=None, palette=False,
                 add_noise_y=False, truncation_ratio=0, timesteps=0, timesteps_1ststage=0, S2=0, S1=0,
                 norm_range=None, img_type=None, gray_flg=None, norm_flag=None, get_edges_flag=None,
                 aug_y_p=0, aug_types=[], drop_y_p=0, cot_process_type=None, morphological_trans=0, flip_p=0.0):
        super().__init__(paths=paths, size=size, random_crop=random_crop, labels=labels)
        self.size = size
        self.random_crop = random_crop
        self.add_noise_y = add_noise_y
        self.palette = palette
        self.truncation_ratio = truncation_ratio
        self.timesteps = timesteps; self.timesteps_1ststage = timesteps_1ststage
        self.S1 = S1; self.S2 = S2
        self.norm_range=norm_range; self.img_type=img_type; self.gray_flg=gray_flg
        self.norm_flag=norm_flag; self.get_edges_flag=get_edges_flag; self.aug_y_p=aug_y_p
        self.aug_types=aug_types; self.drop_y_p=drop_y_p
        self.cot_process_type = cot_process_type
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.morphological_trans = morphological_trans
        if self.morphological_trans:
            # TODO: this one should be configurable
            self.rgb_paths = [path.replace("celebhq_sketch", "celebhq") for path in paths]
            self.structuredEdgeDetection = cv2.ximgproc.createStructuredEdgeDetection("ldm/data/model.yml.gz")

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def _process_one_cot_img(self, img_path, rand_num, image=None):
        """
        img_path: Take the image path.
        norm_range: [1, 0] or [1, -1]
        img_type: "rgb", "binary"
        out: Return the image itself after croping, augmentations, etc...
        """
        if image is None:
            image = Image.open(img_path)
        if self.gray_flg:
            image = image.convert('L')
        else:
            if not image.mode == "RGB":
                image = image.convert("RGB")

        # Augmentations:
        if (self.flip_p>0) and (rand_num > self.flip_p):
            image = self.flip(image)

        if self.get_edges_flag:
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
            image = image.filter(ImageFilter.MaxFilter)
            img = np.expand_dims(np.array(image), axis=-1)
            img = np.repeat(img, 3, axis=-1).astype(np.uint8)
        else:
            image = np.array(image).astype(np.uint8)

        image = self.preprocessor(image=image)["image"]

        if self.gray_flg:
            image = image.reshape((self.size, self.size, 1))

        if self.img_type == "binary":
            mid_val = (image.max()-image.min())/2
            image = (image>mid_val).astype(np.uint8)

        return norm_img(image, self.norm_flag, self.norm_range)

    def __getitem__(self, i, rand_num):
        example = dict()
        if self.morphological_trans:
            if random.random() < self.morphological_trans: # Code to execute with X% probability
                min_line_length = random.randint(3, 7)  # [3, 7]
                image_vec = cv2.imread(self.rgb_paths[i], cv2.IMREAD_COLOR)
                g_blurred = cv2.GaussianBlur(image_vec, (3, 3), 0)
                blurred_float = g_blurred.astype(np.float32) / 255.0
                edges = self.structuredEdgeDetection.detectEdges(blurred_float)
                _, binary_edges = cv2.threshold(edges*255, 40, 255, cv2.THRESH_BINARY)
                line_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, min_line_length))
                sketch = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, line_structure)
                sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
                sketch = sketch.astype(np.uint8)
                sketch_pil = Image.fromarray(sketch)
                #sketch_pil.save('test_sketch.jpg')
                example["image"] = self._process_one_cot_img(img_path=None, image=sketch_pil)
            else:
                example["image"] = self._process_one_cot_img(img_path=self.labels["file_path_"][i])
        else:
            example["image"] = self._process_one_cot_img(img_path=self.labels["file_path_"][i], rand_num=rand_num)

        # Add Noise to the Condition Y to work with the truncation method:
        if self.add_noise_y:
            if self.truncation_ratio > 0:
                idx = np.argmin(np.abs(self.S2[int(self.timesteps/2):] - self.S1[int(self.truncation_ratio*self.timesteps_1ststage)]))
                var_noise = self.S2[idx+int(self.timesteps/2)]
            else:
                var_noise = self.S2[random.randrange(int(len(self.S2)/1))]
            np.add(example["image"], var_noise*np.random.randn(*example["image"].shape), out=example["image"], casting="unsafe")
            if not self.palette: 
                example["image"] = norm_img(example["image"], True, [1, 0])

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQTrain(FacesBase):
    def __init__(self, root, size, txt_file, keys=None, out_modality="RGB", gray_flg=False, norm_flag=True):
        if out_modality != "RGB":
            root += "_" + out_modality
        super().__init__()
        with open(txt_file, "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, gray_flg=gray_flg, norm_flag=norm_flag)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, root, size, txt_file, keys=None, out_modality="RGB"):
        if out_modality != "RGB":
            root += "_" + out_modality
        super().__init__()
        with open("data/celebhq_val.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CoTCelebAHQ(FacesBase):
    def __init__(self, data_root, size, txt_file, cot_stages, cot_process_type, gray_flg, palette, aug_y_p, aug_types, drop_y_p,
                 binary_y=False, add_noise_y=False, timesteps=1000, timesteps_1ststage=200, gray_y=False, truncation_ratio=0.0,
                 keys=None, morphological_trans=False, out_palette=False, blur_radius=None, inference=False,
                 noise_mult=1, flip_p=0.0):
        super().__init__()
        self.data_root = data_root
        self.size = size
        self.palette = palette
        self.cot_stages = cot_stages
        self.cot_process_type = cot_process_type
        self.cot_data_roots = []
        self.cot_data = {}
        self.blur_radius = blur_radius
        self.inference = inference
        self.add_noise_y = add_noise_y
        self.noise_mult = noise_mult
        self.timesteps = timesteps
        self.timesteps_1ststage = timesteps_1ststage
        self.truncation_ratio = truncation_ratio
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

        # RGB:
        with open(txt_file, "r") as f:
            relpaths = f.read().splitlines()
        
        # Load the Target Image:
        if self.cot_stages[0] == "Black_White" or self.cot_stages[0] == "Black1D":
            paths = [os.path.join(data_root+"_sketch", relpath) for relpath in relpaths]
            self.data = ImagePaths(paths=paths, size=size, random_crop=False, gray_flg=True)
        else:
            paths = [os.path.join(data_root, relpath) for relpath in relpaths]
            self.data = ImagePaths(paths=paths, size=size, random_crop=False, norm_flag= not (palette or self.cot_stages[0]=="palette"),
                                   out_palette=out_palette, blur_radius=blur_radius, flip_p=flip_p)
        self.keys = keys
        # Load Palette:
        if self.palette and self.inference:
            self.sketch_paths = [os.path.join(data_root+"_sketch", relpath) for relpath in relpaths]
            self.palette_paths = [os.path.join(data_root+"_palette", relpath) for relpath in relpaths]
        elif self.cot_stages[0] == "palette" and self.inference:
            self.palette_paths = [os.path.join(data_root+"_palette", relpath) for relpath in relpaths]
        
        # CoT:
        m_t_s_2nd = np.linspace(0.001, 0.999, self.timesteps)
        m_t_s_1st = np.linspace(0.001, 0.999, self.timesteps_1ststage)
        self.S2 = np.sqrt(m_t_s_2nd - m_t_s_2nd ** 2)
        self.S1 = np.sqrt(m_t_s_1st)

        # Augmentation:
        if add_noise_y:
            m_t_s_2nd = np.linspace(0.001, 0.999, timesteps)
            m_t_s_1st = np.linspace(0.001, 0.999, timesteps_1ststage)
            self.S2 = np.sqrt(m_t_s_2nd - m_t_s_2nd ** 2)
            self.S1 = np.sqrt(m_t_s_1st)
        else:
            S1=None; S2=None

        for i, cot_stage in enumerate(self.cot_stages):
            if cot_stage == "Black1D" or cot_stage == "palette":
                continue
            
            self.cot_data_roots.append(self.data_root + "_" + cot_stage)
            cot_paths = [os.path.join(self.cot_data_roots[i], relpath) for relpath in relpaths]
            # Load the Cond Images:
            if self.cot_process_type == "with_norm":
                norm_range=[1, 0]; img_type="binary" if binary_y else None
                norm_flag=(True and not palette); get_edges_flag=False
            elif self.cot_process_type == "no_norm":
                norm_range=None; img_type=None; gray_flg=False; norm_flag=False; get_edges_flag=False
                
            self.cot_data[cot_stage] = ImagePathsCoT(paths=cot_paths, size=size, random_crop=False, cot_process_type=self.cot_process_type,
                                                     add_noise_y=add_noise_y, truncation_ratio=truncation_ratio, palette=palette,
                                                     timesteps=timesteps, timesteps_1ststage=timesteps_1ststage, S2=self.S2, S1=self.S1, 
                                                     norm_range=norm_range, img_type=img_type, gray_flg=gray_flg, norm_flag=norm_flag,
                                                     get_edges_flag=get_edges_flag, aug_y_p=aug_y_p, aug_types=[], drop_y_p=drop_y_p,
                                                     morphological_trans=morphological_trans, flip_p=flip_p)

    def _add_noise(self, img):
        # Add Noise to the Condition Y to work with the truncation method
        if self.truncation_ratio > 0:
            idx = np.argmin(np.abs(self.S2[int(self.timesteps/2):] - self.S1[int(self.truncation_ratio*self.timesteps_1ststage)]))
            var_noise = self.S2[idx+int(self.timesteps/2)]
        else:
            var_noise = self.S2[random.randrange(int(len(self.S2)/1))]*self.noise_mult
        g_noise = np.random.randn(*img.shape)
        np.add(img, var_noise*g_noise, out=img, casting="unsafe")

        return img

    def __getitem__(self, i):
        ex = {}
        # Load the Target Image:
        #example = self.data[i]
        rand_num = random.random()
        example = self.data.__getitem__(i, rand_num)
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example

        # Load the Cond Images:
        for cot_stage in self.cot_stages:
            if cot_stage == "Black1D":
                ex["image_"+cot_stage] = np.ones((self.size, self.size, 1))*-1
            elif cot_stage == "palette":
                if self.inference:
                    img_pixelate = Image.open(self.palette_paths[i])
                    #img_pixelate = img_pixelate.resize((64, 64), Image.ANTIALIAS)  #TODO: should make it generic
                    img_pixelate = np.array(img_pixelate).astype(np.uint8)
                else:
                    # Pixelate/Blur the RGB:
                    img_pixelate = generate_blurred_image_palette(img=Image.fromarray(ex["image"]),
                                                                blur_radius=random.randint(self.blur_radius[0], self.blur_radius[1]))
                    img_pixelate = np.array(img_pixelate).astype(np.uint8)
                
                # Normalize:
                if self.cot_process_type == "with_norm":
                    ex["image_"+cot_stage] = norm_img(img_pixelate, norm_flag=True, norm_range=[1, -1])
                elif self.cot_process_type == "no_norm":
                    ex["image_"+cot_stage] = norm_img(img_pixelate, norm_flag=False, norm_range=None)
                ex["image"] = norm_img(ex["image"], norm_flag=True, norm_range=[1, -1])

                # Add Noise to the Condition Y to work with the truncation method:
                if self.add_noise_y:
                    ex["image_"+cot_stage] = self._add_noise(ex["image_"+cot_stage])
                    #if self.cot_process_type == "with_norm":  # Normalize
                    #    ex["image_"+cot_stage] = norm_img(ex["image_"+cot_stage], norm_flag=True, norm_range=[1, -1])
            else:
                #ex["image_"+cot_stage] = self.cot_data[cot_stage][i]["image"]
                ex["image_"+cot_stage] = self.cot_data[cot_stage].__getitem__(i, rand_num)["image"]

            # Overlay the Palette on the Sketch:
            if self.palette:
                if self.inference:
                    img_sketch = Image.open(self.sketch_paths[i])
                    img_sketch = img_sketch.convert("RGB")
                    img_sketch = img_sketch.resize((64, 64), Image.ANTIALIAS)  #TODO: should make it generic
                    img_sketch = np.array(img_sketch).astype(np.uint8)
                    mid_val = (img_sketch.max()-abs(img_sketch.min()))/2
                    img_sketch_binary = (img_sketch>mid_val).astype(np.uint8)
                    img_sketch = img_sketch_binary
                    img_pixelate = Image.open(self.palette_paths[i])
                    img_pixelate = img_pixelate.resize((64, 64), Image.ANTIALIAS)  #TODO: should make it generic
                    img_pixelate = np.array(img_pixelate).astype(np.uint8)
                else:
                    # Convert the mask to binary values:
                    img_sketch = ex["image_"+cot_stage]
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
                    ex["image_"+cot_stage] = norm_img(fused_img, norm_flag=True, norm_range=[1, 0])
                elif self.cot_process_type == "no_norm":
                    ex["image_"+cot_stage] = norm_img(fused_img, norm_flag=False, norm_range=None)
                ex["image"] = norm_img(ex["image"], norm_flag=True, norm_range=[1, -1])

        return ex


class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


def process_data_proper_FID_calc(input_directory, output_directory):
    # Process Val data for proper FID calculation:
    dataset = CelebAHQTrain(root=input_directory, size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(output_directory, exist_ok=True)
    for img_count in tqdm(range(dataset.__len__())):
        sample = dataset.__getitem__(img_count)
        image = sample['image']
        image = ((image + 1) / 2.0 * 255).astype(np.uint8)
        #img_name = sample['relative_file_path_']
        img_name = str(img_count).zfill(5) + ".png"
        output_path = os.path.join(output_directory, os.path.basename(img_name))
        transforms.ToPILImage()(image).save(output_path)


if __name__ == '__main__':
    process_data_proper_FID_calc(input_directory="/xxx/user/xxx/celebhq/celebhq_rgb/",
                                 output_directory="/xxx/user/xxx/celebhq/rgb_processed_orgLDM")