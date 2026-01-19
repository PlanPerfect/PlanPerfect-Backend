"""
Stable Diffusion Pipeline Setup

This file initializes the Stable Diffusion img2img pipeline and
configures it based on available hardware (GPU or CPU).

Several memory optimisations are applied to ensure the model
can run on low-VRAM environments such as Colab.
"""

import torch
from diffusers import StableDiffusionImg2ImgPipeline


# ================================
# Device detection
# ================================
use_cuda = torch.cuda.is_available()


# ================================
# Load Stable Diffusion pipeline
# ================================
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    # Use half precision on GPU to reduce memory usage
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    # Disable safety checker for faster inference
    safety_checker=None,
)


# ================================
# Hardware-specific optimisations
# ================================
if use_cuda:
    # Offloada parts of the model to CPU, reduces GPU memory usage
    pipe.enable_model_cpu_offload()

    # Use channels_last memory format for improved performance
    pipe.unet.to(memory_format=torch.channels_last)
else:
    # Fallback to CPU if CUDA GPU is not available
    pipe = pipe.to("cpu")


print("SD device:", "CUDA" if use_cuda else "CPU")
