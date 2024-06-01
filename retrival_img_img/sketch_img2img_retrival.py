import torch
import torch.nn.functional as F
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import clip
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import cv2
from PIL import Image
from tqdm import tqdm

from inception import InceptionV3


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--save-stats', action='store_true',
                    help=('Generate an npz archive from a directory of samples. '
                          'The first path is used as input and the second as output.'))
parser.add_argument('--encoder_type', type=str, default="inception")
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}


def find_closest_images(query_set, reference_set):
    """
    Find the closest images in the reference set for each image in the query set.

    Args:
    - query_set: Generated Images Tensor of shape [num_queries, feature_dim]
    - reference_set: GT data Tensor of shape [num_references, feature_dim]

    Returns:
    - closest_indices: Tensor of shape [num_queries] containing indices of closest images in the reference set.
    """

    # Normalize the feature vectors
    query_set_normalized = F.normalize(query_set, p=2, dim=1)
    reference_set_normalized = F.normalize(reference_set, p=2, dim=1)

    # Calculate cosine similarities
    cosine_similarities = torch.mm(query_set_normalized, reference_set_normalized.t())

    # Find the indices of the closest images
    closest_indices = torch.argmax(cosine_similarities, dim=1)

    return closest_indices


def run_test_example():
    # Dummy Example usage:
    # Assuming feature_set1 has shape [100000, d] and feature_set2 has shape [5000, d]
    # where d is the feature dimension
    feature_set1 = torch.randn(100000, 256)  # Replace 256 with your actual feature dimension
    feature_set2 = torch.randn(5000, 256)  # Replace 256 with your actual feature dimension
    # Find the closest images in feature_set1 for each image in feature_set2
    closest_indices = find_closest_images(feature_set2, feature_set1)

    # Access the closest images in feature_set1 based on the indices
    closest_images = feature_set1[closest_indices]
    print("closest_images = ", closest_indices.shape)
    print("closest_images = ", closest_images.shape)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def compute_features_of_path(path, model, batch_size, dims, device, num_workers=1):
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])
    act = get_activations(files, model, batch_size, dims, device, num_workers)

    return act, np.array(files)


def get_clip_features(path, model, batch_size, num_workers=1, preprocess=None, device=None):
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])
    dataset = ImagePathDataset(files, transforms=preprocess)

    all_features = []
    with torch.no_grad():
        for images in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))
            all_features.append(features)

    return torch.cat(all_features).cpu().numpy(), np.array(files)


def get_closest_imgs_given_paths(paths, batch_size, device, dims, num_workers=1, encoder_type="inception"):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if encoder_type == "inception":
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        features1, files1 = compute_features_of_path(paths[0], model, batch_size, dims, device, num_workers)
        features2, files2 = compute_features_of_path(paths[1], model, batch_size, dims, device, num_workers)
        features1 = torch.tensor(features1)
        features2 = torch.tensor(features2)
    elif encoder_type == "clip":
        model, preprocess = clip.load('ViT-B/32', device)
        features1, files1 = get_clip_features(path=paths[0], model=model, batch_size=batch_size, num_workers=num_workers, 
                                              preprocess=preprocess, device=device)
        query_set_normalized = F.normalize(torch.tensor(features1).to(dtype=torch.float64), p=2, dim=1)
        features2, files2 = get_clip_features(path=paths[1], model=model, batch_size=batch_size, num_workers=num_workers, 
                                              preprocess=preprocess, device=device)
        features1 = torch.tensor(features1).to(dtype=torch.float64)
        features2 = torch.tensor(features2).to(dtype=torch.float64)

    closest_indices = find_closest_images(query_set=features1, reference_set=features2)
    closest_images_names = files2[closest_indices]

    return files1, closest_images_names


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    quary_files_names, closest_images_names = get_closest_imgs_given_paths(args.path, args.batch_size, device,
                                                                           args.dims, num_workers, args.encoder_type)
    print("Saving Image for debuging")
    for i in tqdm(range(len(quary_files_names))):
        img1 = np.array(Image.open(quary_files_names[i]))
        img2 = np.array(Image.open(closest_images_names[i]))
        stacked_image = np.concatenate((img1, img2), axis=1)
        result_image = Image.fromarray(stacked_image, 'L')
        result_image.save(os.path.join("stacked_sketchs_for_debug", str(i)+".png"))
    #import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
