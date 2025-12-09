import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool, save_image

import numpy as np
import traceback

class T2IInput(BaseModel):
    prompt: str = Field(description="The prompt to generate an image from")

class I2IInput(BaseModel):
    prompt: str = Field(description="The prompt to generate an image from")
    images: list[Image] = Field(description="The other input Image")
    creativity: float = Field(description="The independency level from image input")

class Output(BaseModel):
    image: Image = Field(description="The generated image")

class MyApp(fal.App):
    machine_type = "GPU-H100"
    app_name = "wan"
    # requirements = ["accelerate==1.12.0","fal==1.58.0","flash-attn==2.5.8","huggingface-hub==0.36.0","numpy==1.26.4","scipy==1.15.3","pandas==2.3.3","scikit-learn==1.7.2","torch==2.8.0","torchvision==0.23.0","torchaudio==2.8.0","triton==3.4.0","torch-tensorrt==2.8.0","opencv-python==4.11.0.86","imageio==2.37.2","imageio-ffmpeg==0.6.0","pillow==11.3.0","onnx==1.20.0","onnxruntime-gpu==1.22.0","onnx-simplifier==0.4.36","onnxconverter-common==1.16.0","tensorrt==10.12.0.36","tensorrt-cu12==10.12.0.36","tensorrt-cu12-bindings==10.12.0.36","tensorrt-cu12-libs==10.12.0.36","tensorrt-cu13==10.14.1.48.post1","tensorrt-cu13-bindings==10.14.1.48.post1","tensorrt-cu13-libs==10.14.1.48.post1","pycuda==2025.1.2","numba==0.62.1","llvmlite==0.45.1","transformers==4.53.3","tokenizers==0.21.4","peft==0.18.0","safetensors==0.7.0","nvidia-ml-py==13.580.82","nvidia-modelopt==0.33.1","nvidia-modelopt-core==0.33.1"]
    requirements = [
        "torch==2.6.0",
        "torchvision==0.21.0",
        "opencv-python>=4.9.0.80",
        "diffusers==0.32.2",
        "transformers==4.49.0",
        "tokenizers==0.21.0",
        "accelerate==1.4.0",
        "tqdm",
        "imageio",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        "numpy==1.24.4",
        "xfuser==0.4.1",
        "fal-client",
        "tensorrt",
        "decord",
        "librosa",
        "peft==0.17.1",
        "fal"
    ]
    def setup(self):
    
        
        # first get updated repository
        from fal.toolkit import FAL_MODEL_WEIGHTS_DIR, File, clone_repository
            # Clone repository with commit pinning
        target_dir = clone_repository(
            "https://github.com/alperbahcekapili/fal-interview.git",
            target_dir="/data",  # Use temp directory to avoid conflicts
            include_to_path=True,
            commit_hash="7159813b984883995cc42ac3d448322bfd02e63d",
            repo_name="wan2_2",
        )


        # download model weights from the huggingface
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Wan-AI/Wan2.2-TI2V-5B",
            local_dir="/data/fal-interview/Wan2.2-TI2V-5B",
            local_files_only=True  # Skip checks if cached
        )


        os.chdir(target_dir)
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS["i2i-5B"]
        self.wan_t2i = wan.WanT2I(
            config=cfg,
            checkpoint_dir="/data/fal-interview/Wan2.2-TI2V-5B",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )


        self.wan_i2i = wan.WanI2I(        
            config=cfg,
            checkpoint_dir="/data/fal-interview/Wan2.2-TI2V-5B",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            deepcache_interval=2,
            deepcache_blocks=10
        )

    def tensor_to_pil(self, tensor):
        from PIL import Image
        print("Converting tensor to image...")
        tensor = (tensor + 1.0) / 2.0
        img_array = tensor * 255
        print("Converting to numpy...")
        img_array = img_array.permute(1, 2, 0).cpu().numpy()
        img_array = img_array.astype(np.uint8)
        print("Converting to PIL Image...")
        pil_image = Image.fromarray(img_array)
        return pil_image
        

    @fal.endpoint("/t2i")
    def runt2i(self, request: T2IInput) -> Output:
        try:
            image4d = self.wan_t2i.generate(
                request.prompt,
                size=SIZE_CONFIGS["1280*704"],
                max_area=MAX_AREA_CONFIGS["1280*704"],
                shift=5.0,
                sample_solver="unipc",
                sampling_steps=50,
                offload_model=False,
                guide_scale=5.0,
                seed=-1
            )
            image = image4d[:,-1,...]
            value_range=(-1, 1)
            image = image.clamp(min(value_range), max(value_range))
            image = self.tensor_to_pil(image)
            print("PIL conversion completed...")
            to_ret_image = Image.from_pil(image)
            print("Fal Image conversion completed. Returning...")
            response_obj = Output(image=to_ret_image)
            print("Response obj constructed...")
            print(response_obj)
            return response_obj
        except Exception as e:
            traceback.print_exc()   # prints full stack trace
        



    @fal.endpoint("/i2i")
    def runi2i(self, request: I2IInput) -> Output:
        try:
            
            print("Got the request printing all: ")
            print("|||||||||||||||<<")
            print(request)
            print("|||||||||||||||<<")

            print("Converting image to PIL")
            image = request.images[0].to_pil()
            if len(request.images) > 1:
                extra_images = [f.to_pil() for f in request.images[1:]] # request.extra_image.to_pil()
            else:
                extra_images = []


            image4d = self.wan_i2i.generate(
                request.prompt,
                img= image,
                size=SIZE_CONFIGS["1280*704"],
                max_area=MAX_AREA_CONFIGS["1280*704"],
                offload_model=False,
                creativity=request.creativity,
                extra_images=extra_images)
        
            image = image4d[:,-1,...]
            value_range=(-1, 1)
            image = image.clamp(min(value_range), max(value_range))
            image = self.tensor_to_pil(image)
            print("PIL conversion completed...")
            to_ret_image = Image.from_pil(image)
            print("Fal Image conversion completed. Returning...")
            response_obj = Output(image=to_ret_image)
            print("Response obj constructed...")
            print(response_obj)
            return response_obj
        except Exception as e:
            traceback.print_exc()   # prints full stack trace
        