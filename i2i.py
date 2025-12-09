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

class Input(BaseModel):
    prompt: str = Field(description="The prompt to generate an image from")

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
        import torch
        
        "--task", "i2i-5B",
        "--size", "1280*704",
        "--ckpt_dir", "./Wan2.2-TI2V-5B",
        "--offload_model", "True",
        "--convert_model_dtype",
        "--prompt", "Boxer cat wears red gloves. Looking angry and has a fine texture.",
        "--image", "/home/alpfischer/Wan2.2/input.jpg",
        "--sample_steps", "100",
        "--creativity", "0.3",
        "--sample_guide_scale", "5.0",
        "--sample_solver", "unipc",
        "--deepcache_intervals", "3",
        "--deepcache_blocks", "10"
        
        # first get updated repository
        from fal.toolkit import FAL_MODEL_WEIGHTS_DIR, File, clone_repository
            # Clone repository with commit pinning
        target_dir = clone_repository(
            "https://github.com/alperbahcekapili/fal-interview.git",
            target_dir="/data",  # Use temp directory to avoid conflicts
            include_to_path=True,
            commit_hash="76492f830e52cac1e496bb64eda3e76172ef9c2c",
            repo_name="fal-interview",
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
            checkpoint_dir="Wan2.2-TI2V-5B",
            device_id=0,
        )


    def tensor_to_pil(self, tensor):
        tensor = (tensor + 1.0) / 2.0
        img_array = tensor * 255
        img_array = img_array.permute(1, 2, 0).cpu().numpy()
        img_array = img_array.astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        return pil_image
        

    @fal.endpoint("/")
    def run(self, request: Input) -> Output:
        try:
            image4d = self.wan_t2i.generate(
                request.prompt,
                size=SIZE_CONFIGS["1280*704"],
                max_area=MAX_AREA_CONFIGS["1280*704"],
                shift=5.0,
                sample_solver="unipc",
                sampling_steps=50,
                guide_scale=5.0,
                seed=-1
            )
            image = image4d[:,-1,...]
            value_range=(-1, 1)
            image = image.clamp(min(value_range), max(value_range))
            image = self.tensor_to_pil(image)
            return Output(image=image)
        except Exception as e:
            print(e)
        