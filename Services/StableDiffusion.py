import torch
from diffusers import StableDiffusionImg2ImgPipeline

use_cuda = torch.cuda.is_available()

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    safety_checker=None,
)

# comment out line 13-18 below if using CPU only
if use_cuda:
    pipe.enable_model_cpu_offload()
    # pipe.enable_attention_slicing()
    pipe.unet.to(memory_format=torch.channels_last)
else:
    pipe = pipe.to("cpu")

# uncomment the code below if using CPU only
# pipe = pipe.to("cpu")

print("SD device:", "CUDA (4GB-safe)" if use_cuda else "CPU")
