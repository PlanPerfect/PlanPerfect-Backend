from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image
import uuid
import json
import torch

from Services.StableDiffusion import pipe
from Services.LLMGemini import generate_sd_prompt

def extract_json(text: str) -> dict:
    """
    Safely extract JSON object from LLM output
    """
    try:
        # Remove markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No JSON object found")

        return json.loads(text[start:end])

    except Exception as e:
        raise ValueError(f"Failed to parse LLM JSON: {text}") from e


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

    parsed = extract_json(llm_response)

    prompt = parsed["prompt"]
    negative_prompt = parsed["negative"]
    torch.cuda.empty_cache()

    # Run Stable Diffusion img2img
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image.resize((512, 512)),
        strength=0.6,
        guidance_scale=7.0,
        num_inference_steps=15               
    ).images[0]


    output_path = f"static/output_{uuid.uuid4()}.png"
    result.save(output_path)

    return FileResponse(output_path)
