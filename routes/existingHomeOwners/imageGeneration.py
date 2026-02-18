"""
Image Generation API - Using Gemini Imagen 3

Workflow:
1. Receives input image, style preferences, optional furniture URLs, and optional user prompt
2. Downloads room image + selected furniture images from Cloudinary/Firebase URLs
3. Uses Gemini Imagen 3 to transform the room design, referencing the furniture pieces
4. Uploads generated image to Cloudinary
5. Saves generated image data to Firebase (overwriting previous generations)
6. Returns the Cloudinary URL in the response
"""

from fastapi import APIRouter, Form, Request, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import uuid
import io
import json
import tempfile
import os
import re
import requests
import traceback
from Services.LLMGemini import generate_interior_design
from Services.FileManager import FileManager
from Services import DatabaseManager as DM
from Services import Logger
from middleware.auth import _verify_api_key
from google import genai
from google.genai import types


class UploadFileAdapter:
    """Adapter to make image bytes compatible with FileManager's expected UploadFile interface"""
    def __init__(self, file_content: bytes, filename: str):
        self.file = io.BytesIO(file_content)
        self.filename = filename


def extract_keywords(user_prompt: str) -> list:
    """
    Convert user prompt into structured designer notes using Gemini LLM.
    """
    if not user_prompt or not user_prompt.strip():
        return []

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        extraction_prompt = f"""You are a professional interior design consultant summarizing a client's request into concise notes for the design team.

        Convert the client's raw input into clear, actionable designer notes.
        Focus on: specific changes requested, colors, materials, furniture, lighting, mood, and spatial preferences.
        Write in a professional but brief tone, as if handing off notes to the designer.

        Client request: "{user_prompt}"

        Return ONLY 2-4 short bullet points (using "-" as bullet), nothing else.
        Example output:
        - Lighten wall colour to a warm off-white or pale beige
        - Replace overhead lighting with warmer, ambient light fixtures
        - Introduce natural wood accents and indoor greenery
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
        Logger.log(f"[IMAGE GENERATION] - Error generating designer notes with LLM: {str(e)}")
        return _simple_keyword_extraction(user_prompt)


def _simple_keyword_extraction(user_prompt: str) -> list:
    """Fallback simple keyword extraction if LLM fails."""
    if not user_prompt or not user_prompt.strip():
        return []

    text = user_prompt.lower()

    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'make',
        'add', 'more', 'some', 'any', 'it', 'this', 'that', 'these', 'those',
        'look', 'less', 'get', 'want', 'need'
    }

    words = re.findall(r'\b[a-z]+\b', text)
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)

    return unique_keywords[:10]


def download_image_as_pil(url: str) -> Image.Image | None:
    """
    Download an image from a URL and return a PIL Image in RGB mode.
    Returns None if the download fails.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        Logger.log(f"[IMAGE GENERATION] - Failed to download image from {url}: {str(e)}")
        return None


router = APIRouter(prefix="/image", tags=["Image Generation"], dependencies=[Depends(_verify_api_key)])


@router.post("/generate")
async def generate_image(
    styles: str = Form(...),
    user_prompt: str = Form(None),
    furniture_urls: str = Form(None),  # JSON-encoded list of furniture image URLs
    request: Request = None
):
    """
    Generate a new room design using Gemini Imagen 3.

    Args:
        styles:         Comma-separated string of selected interior design styles
        user_prompt:    Optional user-provided instructions for customization
        furniture_urls: Optional JSON array of furniture image URLs to reference
                        e.g. '["https://...", "https://..."]'
        request:        FastAPI request object to extract user information

    Returns:
        JSON with generated image URL, file_id, and filename
    """
    tmp_file_path = None
    try:
        user_id = None
        if request:
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                user_id = request.headers.get('X-User-ID')

        if not user_id:
            return JSONResponse(
                status_code=400,
                content={"error": "UERROR: One or more required fields are invalid / missing."}
            )

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={"error": "UERROR: Please login again."}
            )

        analysis_path = ["Users", user_id, "Existing Homeowner", "Style Analysis"]
        analysis_data = DM.peek(analysis_path)

        if not analysis_data:
            return JSONResponse(
                status_code=404,
                content={"error": "UERROR: Please complete style analysis first."}
            )

        image_url = analysis_data.get('image_url')
        if not image_url:
            return JSONResponse(
                status_code=404,
                content={"error": "UERROR: Please complete style analysis first."}
            )

        response = requests.get(image_url)
        response.raise_for_status()

        suffix = '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        init_image = Image.open(tmp_file_path).convert("RGB")

        furniture_images: list[Image.Image] = []
        if furniture_urls:
            try:
                urls_list = json.loads(furniture_urls)
                if isinstance(urls_list, list):
                    for url in urls_list:
                        if isinstance(url, str) and url.strip():
                            img = download_image_as_pil(url.strip())
                            if img is not None:
                                furniture_images.append(img)
                    Logger.log(
                        f"[IMAGE GENERATION] - Downloaded {len(furniture_images)}/{len(urls_list)} furniture images"
                    )
            except json.JSONDecodeError:
                Logger.log(f"[IMAGE GENERATION] - Invalid furniture_urls JSON, skipping furniture: {furniture_urls}")

        try:
            result_image = generate_interior_design(
                init_image=init_image,
                styles=styles,
                user_modifications=user_prompt,
                furniture_images=furniture_images if furniture_images else None
            )
        except Exception as e:
            Logger.log(
                f"[IMAGE GENERATION] - ERROR in generate_interior_design: {str(e)}. "
                f"Type: {type(e).__name__}. Traceback: {traceback.format_exc()}"
            )
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
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

        generation_data = {
            "image_url": cloudinary_result["url"],
            "file_id": cloudinary_result["file_id"],
            "filename": cloudinary_result["filename"],
            "styles": styles,
            "designer_notes": designer_notes,
            "furniture_count": len(furniture_images),
        }

        DM.data["Users"][user_id]["Existing Homeowner"]["Image Generation"] = generation_data
        DM.data["Users"][user_id]["Existing Homeowner"]["Preferences"]["selected_styles"] = styles.split(", ")
        DM.save()

        return {
            "url": cloudinary_result["url"],
            "file_id": cloudinary_result["file_id"],
            "filename": cloudinary_result["filename"]
        }

    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        Logger.log(f"[IMAGE GENERATION] - Error in generate_image: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@router.get("/getFurniture")
async def get_furniture(request: Request):
    try:
        user_id = getattr(request.state, 'user_id', None)
        if not user_id:
            user_id = request.headers.get('X-User-ID')

        if not user_id:
            return JSONResponse(status_code=400, content={"error": "UERROR: Not authenticated."})

        # Force reload from Firebase to get latest data
        DM.load()

        furniture_path = ["Users", user_id, "Existing Homeowner", "Saved Recommendations", "recommendations"]
        furniture_data = DM.peek(furniture_path)

        if not furniture_data:
            return JSONResponse(status_code=404, content={"error": "UERROR: No saved recommendations found."})

        furniture_list = []
        for item_id, item_data in furniture_data.items():
            if isinstance(item_data, dict) and item_data.get("image"):
                furniture_list.append({
                    "id": item_id,
                    "name": item_data.get("name", "Unknown"),
                    "image_url": item_data["image"],
                    "description": item_data.get("description", ""),  # ADD THIS
                    "type": item_data.get("description", "")
                })

        return {
            "success": True,
            "furniture": furniture_list
        }

    except Exception as e:
        Logger.log(f"[GET FURNITURE] - Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})