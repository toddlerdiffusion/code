import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
from einops import rearrange, repeat

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from ldm.data.base import Txt2ImgIterableBaseDataset

import torch.nn as nn
from torchvision import transforms


rescale = lambda x: (x + 1.) / 2.


def _save_sketches(image_tensor, path, n_saved_cond=0):
    """
    image_tensor: [32, 256, 256, 3] range [0,1] or [-1,1]
    """
    if image_tensor.min() < 0:  # detect input range
        image_tensor = (image_tensor+1)*0.5  # shift it from [-1,1] --> [0,1]
    image_tensor = image_tensor * 255
    image_tensor = image_tensor.to(torch.uint8)

    # Iterate through the batch dimension (assuming the first dimension is the batch size)
    for i in range(image_tensor.size(0)):
        # Select one image from the batch
        image = image_tensor[i]
        # Convert the tensor to a PIL Image
        image_pil = transforms.ToPILImage()(image)
        image_pil = image_pil.resize((256, 256), Image.ANTIALIAS)
        # Save the image
        imgpath = os.path.join(path, f"{n_saved_cond:06}.png")
        image_pil.save(imgpath)
        n_saved_cond += 1

    return n_saved_cond


def custom_to_pil(x):
    x = x.detach().cpu()
    if x.shape[0] == 1:
        x = torch.clamp(x, 0., 1.)
        x = x.permute(1, 2, 0).numpy()
        x = (255 * x).astype(np.uint8)
        x = Image.fromarray(x[:,:, 0], 'L')
    else:
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


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


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


@torch.no_grad()
def convsample(model, shape, return_intermediates=True, verbose=True, make_prog_row=False, cond=None, conditioning=None):
    if not make_prog_row:
        return model.p_sample_loop(None, shape, return_intermediates=return_intermediates, verbose=verbose)
    else:
        # Run Forward process on t=T to get the correct starting point for the different cases:
        N = cond.shape[0]
        t = repeat(torch.tensor([model.num_timesteps-1]), '1 -> b', b=N)
        t = t.to(cond.device).long()
        model.z_cot = cond
        x_start = cond[:N]
        if x_start.shape[1] != model.channels:
            x_start = x_start.repeat(1, model.channels, 1, 1)
        xt = model.q_sample(x_start=x_start, t=t, noise=torch.randn_like(cond[:N]))
        return model.progressive_denoising(conditioning, shape, verbose=True, x_T=xt)


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0, parameterization="x0", conditioning=None):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, parameterization=parameterization,
                                         conditioning=conditioning)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None,
                              eta=1.0, cond=None, parameterization="x0", conditioning=None):
    log = dict()
    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape, make_prog_row=True, cond=cond, conditioning=conditioning)
            sample = progrow[-1]
        else:
            sample, intermediates = convsample_ddim(model, steps=custom_steps, shape=shape,
                                                    eta=eta, parameterization=parameterization, conditioning=conditioning)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, config,  batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000,
        nplog=None, sp=0, input_cond_log=None, data_type='validation'):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = sp
    n_saved_cond = sp
    if not model.CoT_flag:
        all_images = []
        # Handle the conditioning case:
        if (model.cond_stage_model is None) or (model.cond_stage_model=="None") \
                or (model.cond_stage_key is None) or (model.cond_stage_key=="None"):
            cond_dataloader = None
            conditioning = None
        else:
            data = instantiate_from_config(config.data)
            data.prepare_data()
            data.setup()
            is_iterable_dataset = isinstance(data.datasets[data_type], Txt2ImgIterableBaseDataset)
            cond_dataloader = DataLoader(data.datasets[data_type], batch_size=batch_size, pin_memory=True, shuffle=False if is_iterable_dataset else True,persistent_workers=True, num_workers=12)
            cond_dataloader = cycle(cond_dataloader)

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(sp // batch_size,(n_samples+sp) // batch_size, desc="Sampling Batches (unconditional)"):
            if cond_dataloader is not None:
                batch = next(cond_dataloader)
                conditioning = batch[config.model.params.cond_stage_key].cuda()  # [32, 77, 768]

            logs = make_convolutional_sample(model, batch_size=batch_size, vanilla=vanilla, 
                                             custom_steps=custom_steps, eta=eta,
                                             parameterization=model.parameterization, conditioning=conditioning)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= (n_samples+sp):
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)
    else:
        print(f"Running CoT sampling for {n_samples} samples")
        all_images = []
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        data_type
        is_iterable_dataset = isinstance(data.datasets[data_type], Txt2ImgIterableBaseDataset)
        cond_dataloader = DataLoader(data.datasets[data_type], batch_size=batch_size, pin_memory=True,
                                     shuffle=False if is_iterable_dataset else True,
                                     persistent_workers=True, num_workers=12)
        tmp = cycle(cond_dataloader)
        for _ in trange(sp // batch_size,(n_samples+sp) // batch_size, desc="Sampling Batches (CoT)"):
            batch = next(tmp)
            x_cot = batch["image_" + config.model.params.cot_stage_key].permute(0,3,1,2).float().cuda()
            z_cot = model.get_learned_cot_conditioning(x_cot)

            if (model.cond_stage_model is None) or (model.cond_stage_model=="None") \
                or (model.cond_stage_key is None) or (model.cond_stage_key=="None"):
                conditioning = None
            else:
                conditioning = batch["image_" + config.model.params.cond_stage_key].permute(0,3,1,2).float().cuda()  # [32, 77, 768]
                conditioning = model.get_learned_conditioning(conditioning)
                print("conditioning = ", conditioning.shape)
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, cond=z_cot, conditioning=conditioning,
                                             parameterization=model.parameterization)
            n_saved_cond = _save_sketches(batch["image_" + config.model.params.cot_stage_key].permute(0,3,1,2),
                                          input_cond_log, n_saved_cond=n_saved_cond)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= (n_samples+sp):
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
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
    parser.add_argument(
        "--save_dir",
        type=str,
        nargs="?",
        help="save dir",
        default="none"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        nargs="?",
        help="train or validation",
        default="validation"
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    for key in ["m_t_s", "variance_t_s", "betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
                 "sqrt_one_minus_alphas_cumprod", "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
                 "sqrt_recipm1_alphas_cumprod", "posterior_variance", "posterior_log_variance_clipped",
                 "posterior_mean_coef1", "posterior_mean_coef2"]:
        sd.pop(key, None)
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
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

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
    logdir = os.path.join(logdir, opt.save_dir)
    imglogdir = os.path.join(logdir, "img")
    input_cond_imglogdir = os.path.join(logdir, "Input_cond_img")
    numpylogdir = os.path.join(logdir, "numpy")

    if not os.path.exists(imglogdir):
        os.makedirs(imglogdir)
    if not os.path.exists(input_cond_imglogdir):
        os.makedirs(input_cond_imglogdir)
    if not os.path.exists(numpylogdir):
        os.makedirs(numpylogdir)
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
        batch_size=opt.batch_size, nplog=numpylogdir, sp = sampling_conf["sp"],
        input_cond_log=input_cond_imglogdir, data_type=opt.data_type)

    print("done.")
