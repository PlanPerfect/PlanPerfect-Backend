from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import uuid
import json
import torch

from Services.StableDiffusion import pipe
from Services.LLMGemini import generate_sd_prompt

def extract_json(text: str) -> dict:
    """
    Safely extract JSON object from LLM output with robust error handling
    """
    try:
        text = text.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                # Remove 'json' language identifier
                if text.lower().startswith("json"):
                    text = text[4:]

        text = text.strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        json_str = text[start:end]
        parsed = json.loads(json_str)

        # Validate required fields
        if "prompt" not in parsed or "negative" not in parsed:
            raise ValueError("Missing required fields in JSON")

        return parsed

    except json.JSONDecodeError as e:
        # Try to salvage partial JSON
        try:
            # Attempt to close incomplete JSON
            if not text.endswith("}"):
                # Find last complete field
                last_quote = text.rfind('"')
                if last_quote > 0:
                    text = text[:last_quote + 1] + "}"

            start = text.find("{")
            end = text.rfind("}") + 1
            parsed = json.loads(text[start:end])

            if "prompt" in parsed and "negative" in parsed:
                return parsed
        except:
            pass

        raise ValueError(f"Failed to parse JSON: {e}\nRaw text: {text[:200]}...") from e

    except Exception as e:
        raise ValueError(f"Unexpected error parsing LLM response: {str(e)}") from e


router = APIRouter(prefix="/image", tags=["Image Generation"])

@router.post("/generate")
async def generate_image(
    file: UploadFile = File(...),
    styles: str = Form(...),
    preferences: str = Form(...)
):
    try:
        # Save input image
        input_path = f"static/input_{uuid.uuid4()}.png"
        with open(input_path, "wb") as f:
            f.write(await file.read())

        init_image = Image.open(input_path).convert("RGB")

        # Generate prompt using LLM with retry logic
        max_retries = 3
        llm_response = None
        parsed = None

        for attempt in range(max_retries):
            try:
                llm_response = generate_sd_prompt(styles, preferences)
                parsed = extract_json(llm_response)
                break
            except ValueError as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Use fallback prompts
                    print("All retries failed, using fallback prompts")
                    parsed = {
                        "prompt": f"{styles}, detailed, high quality, professional",
                        "negative": "blurry, low quality, distorted, ugly, bad anatomy"
                    }
                    break

        prompt = parsed["prompt"]
        negative_prompt = parsed["negative"]

        # Ensure prompts aren't empty
        if not prompt or prompt.strip() == "":
            prompt = "detailed, high quality, professional photograph"
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted"

        torch.cuda.empty_cache()

        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")
        init_image = init_image.resize((512, 512))

        # Run Stable Diffusion img2img
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image.resize((512, 512)),
            strength=0.35,
            guidance_scale=7.0,
            num_inference_steps=15
        ).images[0]

        output_path = f"static/output_{uuid.uuid4()}.png"
        result.save(output_path)

        return FileResponse(output_path)

    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))