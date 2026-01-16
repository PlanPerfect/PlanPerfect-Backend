import torch
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# extra VRAM optimization
pipe.unet.to(memory_format=torch.channels_last)
