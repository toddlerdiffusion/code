"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.contrastive_patch_loss import PatchNCELoss


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class ResizeLayer(nn.Module):
    def __init__(self, target_size, channels):
        super(ResizeLayer, self).__init__()
        self.target_size = target_size
        self.channels = channels

    def forward(self, x):
        # Resize the input tensor 'x' to the target size
        resized_x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        if resized_x.shape[1] != self.channels:
            resized_x = torch.cat([resized_x, resized_x[:,-1,:,:].unsqueeze(1)], dim=1)
        return resized_x


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


def pixelate_image(in_img, block_size, stride, avg=True):
    """
    img: tensor in shape of [B, C, H, W]
    block_size: [B] ints represent kernel size. Higher leads to more pixilization.
    stride: [B] ints represent stride size. Higher leads to less pixilization.
    pixel_space: True-->means operate in the image space. False-->means operate in the latent space. 
    """
    # Get the image dimensions
    img = in_img.clone().permute(0, 2, 3, 1)  # [B, H, W, C]
    _, height, width, _ = img.shape

    # TODO: optimize this
    for b in range(0, len(block_size)):  # loop through the batch size
        block_size_i = int(block_size[b].item())
        stride_i = int(stride[b].item())
        # Calculate the number of blocks in each dimension
        num_blocks_x = width // block_size_i
        num_blocks_y = height // block_size_i
        # Iterate through each block and compute the average color
        for i in range(0, num_blocks_y, stride_i):
            for j in range(0, num_blocks_x, stride_i):
                # Define the block boundaries
                y1 = i * block_size_i
                y2 = (i + 1) * block_size_i
                x1 = j * block_size_i
                x2 = (j + 1) * block_size_i

                if avg:
                    # Get the average color in the block
                    block = img[b, y1:y2, x1:x2]
                    avg_color = torch.mean(block, dim=(0, 1))
                    # Fill the block with the average color
                    img[b, y1:y2, x1:x2] = avg_color
                else:
                    img[b, y1:y2, x1:x2] = img[b, y1, x1]

    return img.permute(0, 3, 1, 2)  # [B, C, H, W]


def downscale_feature_map(feature_map, scale_ratio):
    # Get the feature dimensions
    batch_size, num_channels, height, width = feature_map.shape

    # Calculate the new dimensions after scaling
    new_height = height // scale_ratio
    new_width = width // scale_ratio

    # Reshape the feature map to prepare for pooling
    reshaped_feature_map = feature_map.reshape(batch_size, num_channels, new_height, scale_ratio, new_width, scale_ratio)

    # Perform average pooling along the appropriate axes
    downscaled_feature_map = torch.mean(reshaped_feature_map, dim=(3, 5))

    # Convert the mask to binary values again:
    mid_val = (downscaled_feature_map.max()-downscaled_feature_map.min())/2
    downscaled_feature_map = (downscaled_feature_map>mid_val).type(torch.uint8)

    return downscaled_feature_map
    

def heaviside_step(x, threshold=0):
    return torch.where(x >= threshold, torch.tensor(1.0).to(x.device), torch.tensor(0.0).to(x.device))


def aug_sampling(image, sample_p):
    """
    image: PIL Image
    sample_p: Sampling Probability (Tensor of size [B])
    """
    _, _, h, w = image.shape
    total_elements = h * w

    # Calculate the number of ones to keep based on the probability
    num_ones = torch.round((100 - sample_p) / 100 * total_elements).long()

    # Generate random positions for ones for each image in the batch
    indices = [torch.randperm(total_elements, device=image.device)[:num] for num in num_ones]

    # Create a mask tensor
    mask = torch.zeros((image.size(0), 1, h, w), device=image.device)

    # Set ones at the generated positions for each image in the batch
    for i in range(image.size(0)):
        row_indices = torch.div(indices[i], w, rounding_mode='floor')
        col_indices = indices[i] % w
        mask[i, 0, row_indices, col_indices] = 1

    # Apply the mask to the image
    sampled_img = image * mask
    return sampled_img


def aug_mask_batch(image, mask_p, patch_size=3):
    """
    image: Tensor [B, C, H, W]
    mask_p: Masking Probability (Tensor of size [B])
    patch_size: Define the patch size (3x3 in this example)
    """
    # Calculate the number of patches to be masked
    _, c, h, w = image.shape
    num_patches_w = w // patch_size
    num_patches_h = h // patch_size
    num_patches_x_y_min = min(num_patches_w, num_patches_h)
    num_patches = num_patches_w * num_patches_h

    # Calculate the number of patches to mask based on the percentage
    num_patches_to_mask = torch.round(num_patches * mask_p / 100).long()

    # Create a mask of the same size as the image
    mask = torch.ones((1, c, h, w), device=image.device)

    # Determine the positions of patches to mask
    for i in range(image.size(0)):
        masked_indices = torch.randperm(num_patches, device=image.device)[:num_patches_to_mask[i]]
        #row_indices = masked_indices // num_patches_w
        row_indices = torch.div(masked_indices, num_patches_w, rounding_mode='floor')
        col_indices = masked_indices % num_patches_w
        y_coords = row_indices
        x_coords = col_indices

        # Apply the masking by setting pixels in the patches to zero
        for y, x in zip(y_coords, x_coords):
            idx_y = y * patch_size
            idx_x = x * patch_size
            mask[:, :, idx_y:idx_y + patch_size, idx_x:idx_x + patch_size] = 0

    # Apply the mask to the image
    masked_img = image * mask
    return masked_img

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 forward_type="weighting",
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "XtminX0", "YandN", "Weps", "WYandN"], 'currently only supporting "eps", "XtminX0", "YandN", "WYandN, "Weps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.forward_type = forward_type
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # self.use_scheduler = scheduler_config is not None and not "None"
        if (scheduler_config is not None) and (scheduler_config is not "None"):
            self.use_scheduler = True
        else:
            self.use_scheduler = False

        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        if exists(given_betas):
            betas = given_betas
        elif self.CoT_flag:
            # Brownian Bridge Diffusion Models:
            if beta_schedule == "linear":
                m_t_s = to_torch(np.linspace(0.001, 0.999, self.CoT_timesteps))
            elif beta_schedule == "sin":
                m_t_s = 1.0075 ** np.linspace(0, self.CoT_timesteps, self.CoT_timesteps)
                m_t_s = m_t_s / m_t_s[-1]
                m_t_s[-1] = 0.999
                m_t_s = to_torch(m_t_s)

            self.register_buffer('m_t_s', m_t_s)

            self.max_var = 1.0
            if self.BBDM_noise_schedule=="bridge":
                variance_t_s = self.BBDM_noise * (self.m_t_s - self.m_t_s ** 2) * self.max_var
            elif self.BBDM_noise_schedule=="log":
                sigma_t_s = to_torch(np.log10(self.m_t_s))
                variance_t_s = self.BBDM_noise*(sigma_t_s-sigma_t_s.min())/(sigma_t_s.max()-sigma_t_s.min())
            elif self.BBDM_noise_schedule=="linear":
                variance_t_s = self.BBDM_noise * self.m_t_s * self.max_var
            elif self.BBDM_noise_schedule=="triangle":
                variance_t_s = self.BBDM_noise * (0.5 - np.abs(self.m_t_s - 0.5)) * self.max_var
            
            self.register_buffer('variance_t_s', to_torch(variance_t_s))
            betas = self.variance_t_s.cpu().detach().numpy()

            if (self.aug_x == "masking") and ("Black" in self.cot_stage_key):
                self.register_buffer('aug_x_p', to_torch(np.linspace(0, 30, self.CoT_timesteps)))  # bounded from 0 to 100 as it is probability
            elif (self.aug_x == "downsampling") and ("Black" in self.cot_stage_key):
                self.register_buffer('aug_x_p', to_torch(np.linspace(0, 80, self.CoT_timesteps)))  # bounded from 0 to 100 as it is probability

            if self.forward_type == "pixelate_2bridges":
                self.register_buffer('pixel_alpha_s', to_torch(np.linspace(1, 8, self.CoT_timesteps).astype(np.uint8)))
                self.register_buffer('stride_alpha_s', to_torch(np.linspace(3, 1, self.CoT_timesteps).astype(np.uint8)))
                # Brdige #2:
                alpha_t_s = 1.0075 ** np.linspace(0, self.CoT_timesteps, self.CoT_timesteps)
                alpha_t_s = alpha_t_s / alpha_t_s[-1]
                alpha_t_s[-1] = 0.999
                self.register_buffer('alpha_t_s', to_torch(alpha_t_s))
                self.register_buffer('variance_alpha_t_s', to_torch(self.BBDM_noise * (self.alpha_t_s - self.alpha_t_s ** 2) * self.max_var))
            elif self.forward_type == "pixelate":
                self.register_buffer('pixel_alpha_s', to_torch(np.linspace(1, 8, self.CoT_timesteps).astype(np.uint8)))
                self.register_buffer('stride_alpha_s', to_torch(np.linspace(3, 1, self.CoT_timesteps).astype(np.uint8)))
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        #import pdb; pdb.set_trace()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        print("self.num_timesteps = ", self.num_timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # [Forward] calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # [Reverse] calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # TODO xxx: lvlb_weights is the weights for each timestep which represents its importance!
        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0" or self.parameterization == "XtminX0" or \
             self.parameterization == "YandN" or self.parameterization ==  "Weps" or self.parameterization == "WYandN":
            #lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
        #print("lvlb_weights Len = ", lvlb_weights.shape)
        #print("lvlb_weights 1 = ", self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        #print("lvlb_weights 2 = ", 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod)))
        #print("*"*50)
        # print("betas = ", betas)
        # print("betas = ", betas.shape)
        # print("variance_t_s = ", self.variance_t_s)
        # print("variance_t_s = ", self.variance_t_s.shape)
        # print("lvlb_weights = ", lvlb_weights)
        # print("---------------------------------")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.CoT_flag:
            if "Black" in self.cot_stage_key:
                if isinstance(self.first_stage_model, IdentityFirstStage):
                    # Make sure Forward Process are operating on 1D:
                    x_binary = x_start.mean(1).unsqueeze(1)
                    if self.binary_out:
                        mid_value = (x_binary.max() + x_binary.min()) / 2
                        binary_tensor = torch.where(x_binary >= mid_value, torch.ones_like(x_binary), torch.zeros_like(x_binary))
                        x_start = binary_tensor
                    else:
                        x_start = x_binary  # Just gray

                if self.aug_x == "masking":
                    x_start = aug_mask_batch(image=x_start, mask_p=self.aug_x_p[t], patch_size=3)
                elif self.aug_x == "downsampling":
                    x_start = aug_sampling(image=x_start, sample_p=self.aug_x_p[t])
                
                noise = torch.randn_like(x_start)
                if self.binary_noise:
                    noise = heaviside_step(noise, threshold=0)
                if self.masking_noise_p:
                    self.masking_noise = torch.rand(noise.shape, device=noise.device) > self.masking_noise_p  # [B, 1, H, W]
                    noise *= self.masking_noise

            y = self.z_cot[:x_start.shape[0]]
            # Brownian Bridge Diffusion Models:
            m_t = extract_into_tensor(self.m_t_s, t, x_start.shape)
            var_t = extract_into_tensor(self.variance_t_s, t, x_start.shape)
            if ("Black" in self.cot_stage_key) and y.shape[1] != 1:
                y = y[:, 0].unsqueeze(1)

            sigma_t = torch.sqrt(var_t)

            if self.forward_type == "weighting":
                x_noisy = (1. - m_t) * x_start + m_t * y + sigma_t * noise
                if self.parameterization == "XtminX0":
                    self.objective = m_t * (y - x_start) + sigma_t * noise
                elif self.parameterization == "YandN":
                    self.objective = (m_t * y) + (sigma_t * noise)
                elif self.parameterization == "WYandN":
                    self.objective = ((m_t * y) + (sigma_t * noise))/(1. - m_t)
                elif self.parameterization == "Weps":
                    self.objective = (sigma_t * noise)/((1. - m_t))
                """
                print("x_noisy = ", x_noisy.max(), x_noisy.min())
                print("y = ", y.max(), y.min())
                print("noise = ", noise.max(), noise.min())
                print("m_t = ", m_t.max(), m_t.min())
                print("sigma_t = ", sigma_t.max(), sigma_t.min())
                Numerator = x_noisy - (m_t*y) - (sigma_t*noise)
                x_recon = Numerator / (1-m_t)
                print("x_start = ", x_start.max(), x_start.min())
                print("x_recon = ", x_recon.max(), x_recon.min())
                print("sigma_t * noise = ", (sigma_t * noise).max(), (sigma_t * noise).min())
                print("Numerator = ", Numerator.max(), Numerator.min())
                print("Den = ", (1-m_t).max(), (1-m_t).min())
                print("Numerator = ", Numerator[2])
                print("="*50)
                """
            elif self.forward_type == "pixelate_2bridges":
                mid_val = (self.x_cot.max()+self.x_cot.min())/2
                self.pixel_alpha = self.pixel_alpha_s.gather(-1, t)
                self.stride_alpha = self.stride_alpha_s.gather(-1, t)
                x_start_p = pixelate_image(x_start, block_size=self.pixel_alpha, stride=self.stride_alpha)  # [64, 4, 32, 32]
                # Downsample the mask
                y_binary_mask = downscale_feature_map(self.x_cot, scale_ratio=int(self.x_cot.shape[-1]/x_start_p.shape[-1]))  # [64, 3, 256, 256] --> [64, 3, 32, 32]
                self.y_binary_mask = y_binary_mask[:, 1].unsqueeze(1).repeat(1, x_start_p.shape[1], 1, 1)
                y_binary_mask = self.y_binary_mask[:x_start_p.shape[0]]

                # Forward Equation:
                x_noisy_B1 = (y_binary_mask*m_t*y) + (y_binary_mask*(1.-m_t)*x_start_p)
                alpha_t = extract_into_tensor(self.alpha_t_s, t, x_start_p.shape)
                x_noisy_B2 = ((1-y_binary_mask)*alpha_t*y) + ((1-y_binary_mask)*(1.-alpha_t)*x_start_p)
                x_noisy = x_noisy_B1 + x_noisy_B2 + (sigma_t*noise)
                """
                t0_mask = (t==self.num_timesteps-1)  # [B] Check if this is the last step or not
                if torch.any(t0_mask):
                    x_noisy[t0_mask] = y[t0_mask]
                """
            elif self.forward_type == "pixelate":
                # Get the mask
                mid_val = (self.x_cot.max()+self.x_cot.min())/2
                self.pixel_alpha = self.pixel_alpha_s.gather(-1, t)
                self.stride_alpha = self.stride_alpha_s.gather(-1, t)
                x_start_p = pixelate_image(x_start, block_size=self.pixel_alpha, stride=self.stride_alpha)  # [64, 4, 32, 32]
                # Downsample the mask
                y_binary_mask = downscale_feature_map(self.x_cot, scale_ratio=int(self.x_cot.shape[-1]/x_start_p.shape[-1]))  # [64, 3, 256, 256] --> [64, 3, 32, 32]
                self.y_binary_mask = y_binary_mask[:, 1].unsqueeze(1).repeat(1, x_start_p.shape[1], 1, 1)
                y_binary_mask = self.y_binary_mask[:x_start_p.shape[0]]
                # Forward Equation:
                x_noisy = (y_binary_mask*m_t*y) + (y_binary_mask*(1.-m_t)*x_start_p) + ((1-y_binary_mask)*x_start_p) + (sigma_t*noise)

            return x_noisy
        else:
            x_noisy = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                    extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
            return x_noisy

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if self.l1_sketch_w > 1:
                loss_weighted = loss * (target > 0).float() * self.l1_sketch_w
                loss = loss + loss_weighted
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == 'cls':
            if mean:
                loss = F.cross_entropy(input=pred, target=target, reduction='mean')
            else:
                loss = F.cross_entropy(input=pred, target=target, reduction='none', weight=torch.tensor([1, 10], device=pred.device)).unsqueeze(1)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if (not hasattr(self.cond_stage_model, "load_precomp_emd")) or (not self.cond_stage_model.load_precomp_emd):
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # TODO xxx: I have to debug this.
        # As a workaround for now I have to set the learning rate again here.
        # As when resume is called to continue training it will overwrite the LR.
        #for g in self.optimizers().param_groups:
        #    g['lr'] = self.learning_rate

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 CoT_flag=False,
                 highres_distill_loss=False,
                 CoT_timesteps=1000,
                 BBDM_noise=None,
                 forward_type="weighting",
                 identity_epochs=0,
                 contrast_patch_loss_flag=False,
                 l_contrast_patch_loss_weight=1.0,
                 BBDM_noise_schedule="bridge",
                 binary_out=False,
                 binary_noise=False,
                 masking_noise_p=0,
                 aug_x="None",
                 steps_gap=1,
                 l1_sketch_w=1,
                 cot_stage_key=None,
                 cot_stage_config=None,
                 cot_stage_trainable=False,
                 *args, **kwargs):
        self.CoT_flag = CoT_flag
        self.identity_epochs = identity_epochs
        self.contrast_patch_loss_flag = contrast_patch_loss_flag
        self.l_contrast_patch_loss_weight = l_contrast_patch_loss_weight
        self.highres_distill_loss = highres_distill_loss
        self.CoT_timesteps = CoT_timesteps
        self.BBDM_noise = BBDM_noise
        self.steps_gap = steps_gap
        self.BBDM_noise_schedule = BBDM_noise_schedule
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.binary_out = binary_out
        self.binary_noise = binary_noise
        self.masking_noise_p = masking_noise_p
        self.l1_sketch_w = l1_sketch_w
        self.aug_x = aug_x
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.cond_stage_key = cond_stage_key
        self.cot_stage_key = cot_stage_key
        super().__init__(conditioning_key=conditioning_key, forward_type=forward_type, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cot_stage_trainable = cot_stage_trainable
        
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        if self.CoT_flag:
            self.instantiate_cot_stage(cot_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        if self.contrast_patch_loss_flag:
            self.contrast_patch_loss = PatchNCELoss()

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            if (x.shape[1] != self.channels) and ("Black" in self.cot_stage_key):
                x = x.repeat(1,self.channels,1,1)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        # Freeze the first stage:
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                # self.cond_stage_model = None
                self.cond_stage_model = ResizeLayer(target_size=self.image_size, channels=self.channels)
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    
    def instantiate_cot_stage(self, config):
        if not self.cot_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cot_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                # self.cot_stage_model = None
                self.cot_stage_model = ResizeLayer(target_size=self.image_size, channels=self.channels)
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cot_stage_model = model.eval()
                self.cot_stage_model.train = disabled_train
                for param in self.cot_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cot_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
    
    def get_learned_cot_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cot_stage_model, 'encode') and callable(self.cot_stage_model.encode):
                c = self.cot_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cot_stage_model(c)
        else:
            assert hasattr(self.cot_stage_model, self.cond_stage_forward)
            c = getattr(self.cot_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        """
        xc: is the original condition
        c: is the embedding of the condition
        """
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)  # [72, 3, 256, 256]

        if (x.shape[1] != self.channels) and ("Black" in self.cot_stage_key):
            x = x.repeat(1,self.channels,1,1)

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()  # [72, 4, 32, 32]
        # Encode CoT:
        if self.CoT_flag:
            if self.cot_stage_key == "RGB" or (self.current_epoch < self.identity_epochs):  # Special Case (Learn Identity)
                self.x_cot = x
                self.z_cot = z
            else:
                self.x_cot = batch["image_"+self.cot_stage_key].permute(0, 3, 1, 2).float()
                self.z_cot = self.get_learned_cot_conditioning(self.x_cot)

        if (self.model.conditioning_key is not None) and (self.model.conditioning_key!="None"):
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if ('caption' in cond_key) or ('coordinates_bbox' in cond_key):
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                # TODO: xxx this one should be revisted as currently conditioning on Y is not supported.
                elif cond_key in ['edges', 'sam', 'sketch', 'palette']:
                    xc = batch["image_"+self.cot_stage_key]
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            
            # Encode the condition if it is freezed:
            if self.CoT_flag and False:
                # TODO: xxx this one should be revisted as currently conditioning on Y is not supported.
                # To avoid encoding the condition twice!
                xc = self.x_cot
                c = self.z_cot
            elif not self.cond_stage_trainable or force_c_encode: # TODO: xxx should add cond here.
                if isinstance(xc, dict) or isinstance(xc, list) or self.cond_stage_model.load_precomp_emd:
                    c = self.get_learned_conditioning(xc)
                else:
                    xc = xc.permute(0, 3, 1, 2).float()
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}
        elif (self.model.conditioning_key == 'crossattn') and self.CoT_flag and (self.cond_stage_key is not None) and False:
            # TODO: xxx this one should be revisted as currently conditioning on Y is not supported.
            if cond_key is None:
                cond_key = self.cond_stage_key
            xc = self.x_cot
            c = self.z_cot
        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)  # call the forward function
        return loss

    def forward(self, x, c, *args, **kwargs):
        num_timesteps = self.CoT_timesteps if self.CoT_flag else self.num_timesteps
        t = torch.randint(0, num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)  # Encode the condition if it is trainable
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # if noise is None will return the randn_like()
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # Forward process
        if self.CoT_flag and ("Black" in self.cot_stage_key) and (isinstance(self.first_stage_model, IdentityFirstStage)):
            if x_start.shape[1] != self.channels:
                x_start = x_start.repeat((1, self.channels, 1, 1))
            if x_noisy.shape[1] != self.channels:
                x_noisy = x_noisy.repeat((1, self.channels, 1, 1))
            if self.z_cot.shape[1] != self.channels:
                self.z_cot = self.z_cot.repeat(1, self.channels, 1, 1)

        model_output = self.apply_model(x_noisy, t, cond)  # [72, 4, 32, 32]

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if (self.parameterization=="XtminX0" or self.parameterization=="YandN" or self.parameterization=="WYandN" or self.parameterization=="Weps") and self.CoT_flag:
            target = self.objective
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)  # self.original_elbo_weight=0
        loss_dict.update({f'{prefix}/loss': loss})

        if self.highres_distill_loss:
            print("distill_loss = ", torch.abs(self.z_cot - model_output).mean())  # [72, 4, 32, 32] --> single value

        if self.contrast_patch_loss_flag and (self.current_epoch>10):
            loss_contrastive_patch = self.contrast_patch_loss(model_output, target)
            loss_contrastive_patch = loss_contrastive_patch.mean()*self.l_contrast_patch_loss_weight
            loss += loss_contrastive_patch
            loss_dict.update({f'{prefix}/loss_contrastive_patch': loss_contrastive_patch})

        return loss, loss_dict

    def predict_start_from_y(self, x, x_recon, t, idx, noise):
        """
        This function decodes the model output. Brownian Bridge Diffusion Models
        """
        if self.iterator[idx] == 0:  # Last step in the generation:
            if self.loss_type == 'cls':
                return torch.argmax(x_recon, dim=1, keepdim=True).repeat(1, self.channels, 1, 1), torch.argmax(x_recon, dim=1, keepdim=True).repeat(1, self.channels, 1, 1)
            else:
                return x_recon, x_recon
        else:
            b = self.y.shape[0]
            nt = torch.full((b,), self.iterator[idx+1], device=self.device, dtype=torch.long)
            m_t = extract_into_tensor(self.m_t_s, t, x.shape)
            m_nt = extract_into_tensor(self.m_t_s, nt, x.shape)
            var_t = extract_into_tensor(self.variance_t_s, t, x.shape)
            var_nt = extract_into_tensor(self.variance_t_s, nt, x.shape)

            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t  # Eq13 BBDM
            sigma_t = torch.sqrt(sigma2_t)

            if self.CoT_flag and ("Black" in self.cot_stage_key):
                if self.binary_noise:
                    noise = heaviside_step(noise, threshold=0)
                if self.masking_noise_p:
                    noise *= self.masking_noise

            if self.forward_type == "weighting":
                """
                x_tminus_mean = (1. - m_nt) * x_recon + m_nt * self.y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                                (x - (1. - m_t) * x_recon - m_t * self.y)
                """
                # TODO xxx --> Understand this equation as it directly map to the Eq in section 3.2
                #""" Decompose the org implementation:
                C_xt_2 = torch.sqrt((var_nt - sigma2_t) / var_t)
                C_y_2 = m_nt - (m_t*C_xt_2)
                C_x0_2 = (1. - m_nt) - ((1. - m_t)*C_xt_2)
                x_tminus_mean = C_xt_2*x + C_x0_2*x_recon + C_y_2*self.y
                #"""
                """ Appendix Derivation:
                C_xt_3 = (var_nt/var_t)*((1-m_t)/(1-m_nt))
                C_y_3 = m_nt - (m_t*C_xt_3)
                C_x0_3 = 1. - (m_nt*(sigma2_t/var_nt))
                """
                """ My Derivation:
                C_xt_4 = (2*var_nt*(1-m_t))/((-1*var_nt*(1-m_t)**2)-var_t)
                C_xt_4 = ((2*(m_t-1))/var_t) / (((m_t-1)**2/var_t)+(1/var_nt))
                C_y_4 = (1-2*(m_nt/var_nt)) / (((m_t-1)**2/var_t)+(1/var_nt))
                C_x0_4 = ((2*(m_nt-1))/var_nt) / (((m_t-1)**2/var_t)+(1/var_nt))
                """
                #"""
                #import pdb; pdb.set_trace()
                #print("Xt = ", (C_xt_4 - C_xt_3).abs().sum())
                #print("y = ", (C_y_4 - C_y_3).abs().sum())
                #print("X0 = ", (C_x0_4 - C_x0_3).abs().sum())
                #print("1-2 ", (x_tminus_mean_2 - x_tminus_mean).abs().sum() , (x_tminus_mean_2 - x_tminus_mean).abs().max())
                #print("2-3 ", (x_tminus_mean_2 - x_tminus_mean_3).abs().sum() , (x_tminus_mean_2 - x_tminus_mean_3).abs().max())
                #print("1-3 ", (x_tminus_mean - x_tminus_mean_3).abs().sum() , (x_tminus_mean - x_tminus_mean_3).abs().max())
                #print((x_tminus_mean_3 - x_tminus_mean).abs().sum())
                #print("1-2 ", torch.allclose(x_tminus_mean, x_tminus_mean_2))
                #print("1-3 ", torch.allclose(x_tminus_mean, x_tminus_mean_3))
                #print("2-3 ", torch.allclose(x_tminus_mean_2, x_tminus_mean_3))
                #print("-"*50)
                #"""
            elif self.forward_type == "pixelate_2bridges":
                y_binary_mask = self.y_binary_mask[:x.shape[0]]
                # Bridge #1:
                Mm_t = m_t*y_binary_mask
                Mm_nt = m_nt*y_binary_mask
                x_tminus_mean_B1 = (1. - Mm_nt) * x_recon + Mm_nt * self.y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                                   (x - (1. - Mm_t) * x_recon - Mm_t * self.y)
                # Bridge #2:
                alpha_t = extract_into_tensor(self.alpha_t_s, t, x.shape)
                alpha_nt = extract_into_tensor(self.alpha_t_s, nt, x.shape)
                var_alpha_t = extract_into_tensor(self.variance_alpha_t_s, t, x.shape)
                var_alpha_nt = extract_into_tensor(self.variance_alpha_t_s, nt, x.shape)
                sigma2_alpha_t = (var_alpha_t - var_alpha_nt * (1. - alpha_t) ** 2 / (1. - alpha_nt) ** 2) * var_alpha_nt / var_alpha_t
                Malpha_t = alpha_t * (1-y_binary_mask)
                Malpha_nt = alpha_nt * (1-y_binary_mask)
                x_tminus_mean_B2 = (1. - Malpha_nt) * x_recon + Malpha_nt * self.y + torch.sqrt((var_alpha_nt - sigma2_alpha_t) / var_alpha_t) * \
                                   (x - (1. - Malpha_t) * x_recon - Malpha_t * self.y)
                x_tminus_mean = x_tminus_mean_B1 + x_tminus_mean_B2
            elif self.forward_type == "pixelate":
                y_binary_mask = self.y_binary_mask[:x.shape[0]]
                x_t_coeff = (1-y_binary_mask*m_t)
                y_coeff = (y_binary_mask*(y_binary_mask*torch.square(m_t)+m_nt-m_t))
                x_0_coeff = 1-(y_binary_mask*m_nt)
                var_t += (1-m_t)
                var_nt += (1-m_nt)
                shared_coeff = (-2*var_nt / (var_t*(1+torch.square(1-y_binary_mask*m_t))))
                self.pixel_alpha = self.pixel_alpha_s.gather(-1, nt)
                self.stride_alpha = self.stride_alpha_s.gather(-1, nt)
                x_recon_p = pixelate_image(x_recon, block_size=self.pixel_alpha, stride=self.stride_alpha)
                #x_p = pixelate_image(x, block_size=self.pixel_alpha, stride=self.stride_alpha)
                #x_recon_p = x_recon
                x_p = x
                x_tminus_mean = shared_coeff*((x_t_coeff*x_p) + (y_coeff*self.y) + (x_0_coeff*x_recon_p))

            return x_tminus_mean + sigma_t * noise, x_recon  # img, x0_partial

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None, idx=None, noise=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "XtminX0" and self.CoT_flag:
            x_recon = x - model_out
        elif self.parameterization == "Weps" and self.CoT_flag:
            m_t = extract_into_tensor(self.m_t_s, t, x.shape)
            x_recon = ((x - (m_t*self.y))/ (1-m_t)) - model_out
        elif self.parameterization == "WYandN" and self.CoT_flag:
            m_t = extract_into_tensor(self.m_t_s, t, x.shape)
            x_recon = (x/(1-m_t)) - model_out
        elif self.parameterization == "YandN" and self.CoT_flag:
            #if t[0]-2 >=0:
            #    t = t - 2
            #    idx = idx + 2
            m_t = extract_into_tensor(self.m_t_s, t, x.shape)
            x_recon = (x - model_out) / (1-m_t)
        elif self.parameterization == "eps":
            if self.CoT_flag:
                #if t[0]-1 >=0:
                #    t = t - 1
                #    idx = idx + 1
                var_t = extract_into_tensor(self.variance_t_s, t, x.shape)
                m_t = extract_into_tensor(self.m_t_s, t, x.shape)
                x_recon = (x - (m_t*self.y) - (torch.sqrt(var_t)*model_out)) / (1-m_t)
            else:
                x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        if self.CoT_flag:
            return self.predict_start_from_y(x=x, x_recon=x_recon, t=t, idx=idx, noise=noise)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, idx=None, noise=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, idx=idx, noise=noise)
        if self.CoT_flag:
            if return_x0:
                return outputs  # img, x0_partial
            else:
                return outputs[0]

        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T[:b]
            # TODO: xxx check this one I think it should be (self.y = cond[:b])
            self.y = x_T[:b]
            #self.y = cond[:b]
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        
        # [999, 998, 997, .., 3, 2, 1, 0]
        """
        gap = 2
        samples_2 = []
        i = 0
        while i <50:
            samples_2.append(i)
            i = i + gap
            gap = 1 if gap==2 else 2
        samples_1 = np.array(range(50, 100, 3))
        iterator = np.concatenate((samples_2, samples_1))
        print("iterator = ", iterator)
        """
        #"""
        iterator = range(0, timesteps, self.steps_gap)  # xxx: Uniform
        #"""
        self.iterator = list(reversed(iterator))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for idx in tqdm(range(len(self.iterator))):
            noise = torch.randn_like(img)
            i = self.iterator[idx]
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            if ("Black" in self.cot_stage_key) and (img.shape[1] != self.channels) and self.CoT_flag and isinstance(self.first_stage_model, IdentityFirstStage):
                img = img.repeat((1, self.channels, 1, 1))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, idx=idx, noise=noise)
            if mask is not None:
                assert x0 is not None
                if self.CoT_flag:
                    img_orig, objective = self.q_sample(x0, ts)
                else:
                    img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
            self.y = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        
        self.iterator = list(reversed(range(0, timesteps)))
        self.tqdm_iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for idx, i in enumerate(self.tqdm_iterator):
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised, idx=idx, noise=noise)
            if mask is not None:
                if self.CoT_flag:
                    img_orig, objective = self.q_sample(x0, ts)
                else:
                    img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps,**kwargs):
        if ddim:
            print("--------------------------------")
            print("Start Sampling using DDIM .......")
            print("--------------------------------")
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps, batch_size, shape, cond,
                                                        parameterization=self.parameterization, verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None
        num_timesteps = self.CoT_timesteps if self.CoT_flag else self.num_timesteps
        if use_ddim:
            ddim_steps = int(num_timesteps/5)

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif "caption" in self.cond_stage_key:
                if self.cond_stage_model.load_precomp_emd:
                    xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption_as_txt"])
                else:
                    xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        # Plot forward process:
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(num_timesteps):
                if t % self.log_every_t == 0 or t == num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        inpaint = False  # disable inpainting and outpainting
        quantize_denoised = False
        plot_progressive_rows = False  # TODO: xxx I set it to false to speedup the training
        if self.CoT_flag:
            # Run Forward process on t=T to get the correct starting point for the different cases:
            t = repeat(torch.tensor([num_timesteps-1]), '1 -> b', b=N)
            t = t.to(self.device).long()
            x_T = self.q_sample(x_start=z[:N], t=t, noise=torch.randn_like(z[:N]))
            start_T = None
        else:
            x_T = None
            start_T = None

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                if self.CoT_flag:
                    _, samples = self.progressive_denoising(c, shape=(self.channels, self.image_size, self.image_size),
                                                            batch_size=N, x_T=x_T, start_T=start_T)
                    samples = samples[-1]
                else:
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                            ddim_steps=ddim_steps,eta=ddim_eta, x_T=x_T)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True, x_T=x_T)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N, x_T=x_T, start_T=start_T)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            """
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            """
            print("Setting up MilestonesLR scheduler...")
            scheduler = [{"scheduler": scheduler.create_scheduler(opt=opt), "interval": "epoch"}]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        # self.conditioning_key = str(self.conditioning_key) if not isinstance(self.conditioning_key, str) else self.conditioning_key
        assert self.conditioning_key in [None, "None", 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key == "None" or self.conditioning_key == None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            #print("----- x = ", x.shape)                          # torch.Size([72, 256, 256, 3])
            #print("----- c_concat = ", c_concat[0].shape)         # torch.Size([72, 256, 256, 3])
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)  # [72, 4, 32, 32]
            # TODO: xxx make this configurable.
            # xxx: The following reshape is only needed in the case of 2D spatial cond:
            #cc = cc.view(cc.shape[0], cc.shape[1], -1).permute(0, 2, 1)  # [B, C, h, w]-->[B, C, h*W]-->[B, h*W, C]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs
