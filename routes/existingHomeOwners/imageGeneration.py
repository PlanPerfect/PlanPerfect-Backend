from fastapi import APIRouter, Form, Request, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import uuid
import io
import json
import tempfile
import os
import requests
from datetime import datetime
from Services.LLMGemini import generate_interior_design
from Services.FileManager import FileManager
from Services import DatabaseManager as DM
from Services import Logger
from middleware.auth import _verify_api_key
from google import genai
from google.genai import types


class UploadFileAdapter:
    def __init__(self, file_content: bytes, filename: str):
        self.file = io.BytesIO(file_content)
        self.filename = filename


def extract_keywords(user_prompt: str) -> list:
    if not user_prompt or not user_prompt.strip():
        return []

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        extraction_prompt = f"""
You are a professional interior design consultant summarizing a client's request into concise notes for the design team.

Client request: "{user_prompt}"

Return ONLY 2-4 short bullet points using "-" as bullet.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=extraction_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=50
            )
        )

        notes_text = response.text.strip()
        notes = [line.strip().lstrip('-').strip() for line in notes_text.splitlines()]
        notes = [n for n in notes if n and len(n) > 3][:6]
        return notes

    except Exception as e:
        Logger.log(f"[IMAGE GENERATION] - Error generating designer notes: {str(e)}")
        return []


def download_image_as_pil(url: str) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        Logger.log(f"[IMAGE GENERATION] - Failed to download image: {str(e)}")
        return None


router = APIRouter(prefix="/image", tags=["Image Generation"], dependencies=[Depends(_verify_api_key)])


@router.post("/generate")
async def generate_image(
    styles: str = Form(...),
    user_prompt: str = Form(None),
    furniture_urls: str = Form(None),
    furniture_descriptions: str = Form(None),
    request: Request = None
):
    tmp_file_path = None
    try:
        user_id = getattr(request.state, 'user_id', None) if request else None
        if not user_id:
            user_id = request.headers.get('X-User-ID') if request else None

        if not user_id:
            return JSONResponse(status_code=400, content={"error": "Invalid user."})

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(status_code=404, content={"error": "Please login again."})

        analysis_path = ["Users", user_id, "Existing Homeowner", "Style Analysis"]
        analysis_data = DM.peek(analysis_path)

        if not analysis_data or not analysis_data.get("image_url"):
            return JSONResponse(status_code=404, content={"error": "Please complete style analysis first."})

        response = requests.get(analysis_data["image_url"])
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        init_image = Image.open(tmp_file_path).convert("RGB")

        furniture_images = []
        if furniture_urls:
            try:
                urls_list = json.loads(furniture_urls)
                for url in urls_list:
                    img = download_image_as_pil(url)
                    if img:
                        furniture_images.append(img)
            except:
                pass

        furniture_descs = []
        if furniture_descriptions:
            try:
                furniture_descs = json.loads(furniture_descriptions)
            except:
                pass

        result_image = generate_interior_design(
            init_image=init_image,
            styles=styles,
            user_modifications=user_prompt,
            furniture_images=furniture_images if furniture_images else None,
            furniture_descriptions=furniture_descs if furniture_descs else None
        )

        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)

        file_adapter = UploadFileAdapter(
            img_byte_arr.getvalue(),
            f"generated_design_{uuid.uuid4()}.png"
        )

        cloudinary_result = FileManager.store_file(
            file=file_adapter,
            subfolder="Generated Designs"
        )

        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        designer_notes = extract_keywords(user_prompt) if user_prompt else []

        generation_id = str(uuid.uuid4())

        generation_data = {
            "image_url": cloudinary_result["url"],
            "file_id": cloudinary_result["file_id"],
            "filename": cloudinary_result["filename"],
            "styles": styles,
            "designer_notes": designer_notes,
            "furniture_count": len(furniture_images),
            "created_at": datetime.utcnow().isoformat()
        }

        user_home = DM.data["Users"][user_id]["Existing Homeowner"]

        if "Image Generations" not in user_home:
            user_home["Image Generations"] = {}

        user_home["Image Generations"][generation_id] = generation_data

        user_home["Preferences"]["selected_styles"] = styles.split(", ")

        DM.save()

        return {
            "url": cloudinary_result["url"],
            "file_id": cloudinary_result["file_id"],
            "filename": cloudinary_result["filename"],
            "generation_id": generation_id
        }

    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        Logger.log(f"[IMAGE GENERATION] - Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/history")
async def get_generation_history(request: Request):
    try:
        user_id = getattr(request.state, 'user_id', None)
        if not user_id:
            user_id = request.headers.get('X-User-ID')

        if not user_id:
            return JSONResponse(status_code=400, content={"error": "Not authenticated."})

        DM.load()

        history = DM.peek(["Users", user_id, "Existing Homeowner", "Image Generations"])

        if not history:
            return {"success": True, "designs": []}

        designs = []

        for gen_id, data in history.items():
            designs.append({
                "id": gen_id,
                "url": data.get("image_url"),
                "styles": data.get("styles"),
                "created_at": data.get("created_at")
            })

        designs.sort(key=lambda x: x["created_at"], reverse=True)

        return {"success": True, "designs": designs}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/getFurniture")
async def get_furniture(request: Request):
    try:
        user_id = getattr(request.state, 'user_id', None)
        if not user_id:
            user_id = request.headers.get('X-User-ID')

        if not user_id:
            return JSONResponse(status_code=400, content={"error": "Not authenticated."})

        DM.load()

        furniture_data = DM.peek(["Users", user_id, "Existing Homeowner", "Saved Recommendations", "recommendations"])

        if not furniture_data:
            return JSONResponse(status_code=404, content={"error": "No saved recommendations found."})

        furniture_list = []
        for item_id, item_data in furniture_data.items():
            if isinstance(item_data, dict) and item_data.get("image"):
                furniture_list.append({
                    "id": item_id,
                    "name": item_data.get("name", "Unknown"),
                    "image_url": item_data["image"],
                    "description": item_data.get("description", ""),
                    "type": item_data.get("description", "")
                })

        return {"success": True, "furniture": furniture_list}

    except Exception as e:
        Logger.log(f"[GET FURNITURE] - Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})