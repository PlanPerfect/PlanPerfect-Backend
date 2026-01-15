# imageGeneration.py
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image
import uuid
import json

from Services.StableDiffusion import pipe
from Services.LLMGroq import generate_sd_prompt


router = APIRouter(prefix="/image", tags=["Image Generation"])

@router.post("/generate")
async def generate_image(
    file: UploadFile = File(...),
    styles: str = Form(...),
    preferences: str = Form(...)
):
    # Save input image
    input_path = f"static/input_{uuid.uuid4()}.png"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    init_image = Image.open(input_path).convert("RGB")

    # Generate prompt using LLM
    llm_response = generate_sd_prompt(styles, preferences)

    parsed = json.loads(llm_response)

    prompt = parsed["prompt"]
    negative_prompt = parsed["negative"]

    # Run Stable Diffusion img2img
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=0.65,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]


    output_path = f"static/output_{uuid.uuid4()}.png"
    result.save(output_path)

    return FileResponse(output_path)
