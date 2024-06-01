"""
python sample_vqgan_recon.py -n 64 --batch_size 32 --sp 0 -r 'logs/vqgan_4epochs/checkpoints/epoch=000004.ckpt'
"""

import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
import pdb
from tqdm import tqdm

from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T
import PIL

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from ldm.data.base import Txt2ImgIterableBaseDataset

import torch.nn as nn


rescale = lambda x: (x + 1.) / 2.


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def cycle(dl):
    while True:
        for data in dl:
            yield data


def custom_to_np(y):
    y = y.detach().cpu()
    y = torch.clamp(y, -1., 1.)
    y = (y + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    y = y.transpose(1, 2).transpose(2, 3)  # [256, 256, 3]
    y = y.numpy()
    y = (y * 255).astype(np.uint8)
    return y


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


def run(model, logdir, config,  batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None, sp=0):
    tstart = time.time()
    n_saved = sp
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    is_iterable_dataset = isinstance(data.datasets['validation'], Txt2ImgIterableBaseDataset)
    cond_dataloader = DataLoader(data.datasets["validation"], batch_size=batch_size, pin_memory=True, shuffle=False if is_iterable_dataset else True,persistent_workers=True, num_workers=12)
    tmp = cycle(cond_dataloader)
    for _ in tqdm(trange(sp // batch_size,(n_samples+sp) // batch_size, desc="Sampling Batches (CoT)")):
        batch = next(tmp)
        x = batch["image"].permute(0,3,1,2).float().cuda()  # [32, 3, 256, 256]
        y, _ = model(x)  # [32, 3, 256, 256]
        y = custom_to_np(y)
        n_saved = save_logs(y, logdir, n_saved=n_saved, key="sample")
        if n_saved >= (n_samples+sp):
            print(f'Finish after generating {n_saved} samples')
            break

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(imgs, path, n_saved=0, key="sample", np_path=None):
    for img in imgs:
        img = Image.fromarray(img)
        imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
        img.save(imgpath)
        n_saved += 1
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--sp",
        type=int,
        nargs="?",
        help="the start point",
        default=0
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-2])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    # model = nn.DataParallel(model)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    #logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    logdir = os.path.join(logdir, "samples")
    imglogdir = os.path.join(logdir, "img")

    if not os.path.exists(imglogdir):
        os.makedirs(imglogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    run(model, imglogdir, config, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=None, sp = sampling_conf["sp"])
    print("done.")
