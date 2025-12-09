# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like
from .utils.timer import Timer


class WanI2I:
    def optimize_t5(self):
        self.text_encoder.model = torch.compile(self.text_encoder.model)
        texts = ["Alper example input"]
        for i in range(10):
            self.timer.start("text_encoding_in_optimize")
            self.text_encoder(texts, device="cuda")
            self.timer.end("text_encoding_in_optimize")

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        deepcache_blocks=10,
        deepcache_interval=2,
        engine=False
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            deepcache_blocks (`int` *optional*, defaults to 10):
                Num of blocks to skip on deepcache
            deepcache_interval (`int` *optional*, defaults to 2):
                Interval to skip blocks
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.timer = Timer()
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint if not engine else config.engine_path),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)
    
        # self.optimize_t5()
        if not engine:
            self.text_encoder.model.eval().to(self.device)
        

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device if not engine else "cpu")

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
            deepcache_blocks=deepcache_blocks,
            deepcache_interval=deepcache_interval)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype, deepcache_blocks, deepcache_interval):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            deepcache_blocks (`int` *optional*, defaults to 10):
                Num of blocks to skip on deepcache
            deepcache_interval (`int` *optional*, defaults to 2):
                Interval to skip blocks

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)
        
        model.CACHE_INTERVAL = deepcache_interval
        model.CACHE_BLOCKS = deepcache_blocks

        print(f"Configuring the model with following deepcache configuration: Interval:{model.CACHE_INTERVAL}, Blocks:{model.CACHE_BLOCKS}")

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)



        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 creativity=0.1,
                 extra_images=[]):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            creativity (`float`, defaults to 0.1. Range: [0,1])
                For 0 model outputs almost identical image. 1 corresponds to t2i pipeline where image guidance is simply ignored


        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2i
        return self.i2i(input_prompt=input_prompt,
            img=img,
            max_area=max_area,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
            creativity=creativity,
            extra_images=extra_images)

    def load_extra_images(self, extra_images, outsize):
        assert all([os.path.exists(i) for i in extra_images]), "Plase ensure all of the extra images exists"
        for i in range(len(extra_images)):
            impath = extra_images[i]
            try:
                img = Image.open(impath).convert("RGB")
                img = img.resize(outsize)
                img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
                extra_images[i] = img
            except Exception as e:
                print(f"While loading the following image, a problem has occured :{impath} error: {e} ")
        return extra_images
    

                

    def i2i(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=100,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            creativity=0.1,
            extra_images=[]):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            creativity (`float`, defaults to 0.1. Range: [0,1])
                For 0 model outputs almost identical image. 1 corresponds to t2i pipeline where image guidance is simply ignored

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        assert creativity >  0 and creativity < 1, "creativity parameter should be in range: [0,1]"
        

        self.timer.start("preprocess")
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh
        
        
        if extra_images:
            extra_images = self.load_extra_images(extra_images, (ow, oh))

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
        self.timer.end("preprocess")





        F = 1
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        
        # We do not need to recompute negative prompt over and over again
        context_null = torch.load(open("negative_prompt_emb.pth", "rb"))
        

        # preprocess
        if not self.t5_cpu:
            self.timer.start("text_encoding")
            context = self.text_encoder([input_prompt], self.device)
            # context_null = self.text_encoder([n_prompt], self.device)
            self.timer.end("text_encoding")
            if offload_model:
                if self.text_encoder.engine is not None:
                    del self.text_encoder.context  
                    del self.text_encoder.engine
                    gc.collect()
                    torch.cuda.empty_cache()  
                else:
                    self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        
        self.vae.model.to("cuda")

        self.vae.scale[0] = self.vae.scale[0].to("cuda")
        self.vae.scale[1] = self.vae.scale[1].to("cuda")
        
        self.timer.start("image_encoding")
        z = self.vae.encode([img,*extra_images])
        self.timer.end("image_encoding")

        z[0] = torch.stack(z, dim=0).mean(dim=0)



        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps                
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample images
            """
            Current mechanism generates first image heavily based on the given image
            We want to generate iamges in one shot so we need to update the latent here
            without going too far from the original image but we need to add a little 
            variance to it thus adding creativity parameter
            """
            
            image_guidance_tendancy = 1-creativity
            latent = z[0]
            latent = sample_scheduler.add_noise(latent, torch.randn_like(latent), torch.Tensor([timesteps[int(len(timesteps)*image_guidance_tendancy)]]))

            arg_c = {
                'context': context,
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()
            for step_ind, t in enumerate(tqdm(timesteps[int(len(timesteps)*image_guidance_tendancy):])):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]
                timestep = torch.stack(timestep).to(self.device)
                self.timer.start("noise_pred_cond")
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, step_ind=step_ind, **arg_c)[0]
                self.timer.end("noise_pred_cond")
                if offload_model:
                    torch.cuda.empty_cache()
                self.timer.start("noise_pred_uncond")
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, step_ind=step_ind, **arg_null)[0]
                self.timer.end("noise_pred_uncond")
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                self.timer.start("scheduler_step")
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                self.timer.end("scheduler_step")
                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        print(self.timer.summarize())
        return videos[0] if self.rank == 0 else None
