"""
Image Generation API - Using Gemini Imagen 3

This file handles image generation using Gemini's native image generation (Imagen 3).
This replaces the Stable Diffusion pipeline for faster, higher-quality results.

Workflow:
1. Receives input image, style preferences, and optional user prompt from frontend
2. Downloads image from Cloudinary
3. Uses Gemini Imagen 3 to transform the room design
4. If user_prompt is provided, incorporates it into the generation
5. Uploads generated image to Cloudinary
6. Saves generated image data to Firebase database (overwriting previous generations)
7. Returns the Cloudinary URL in the response

Benefits over Stable Diffusion:
- Much faster generation (no GPU required)
- Higher quality, sharper images
- Better at preserving room layout
- No blur or artifacts
- Simpler pipeline (no model loading, VRAM management, etc.)
- Can incorporate user-specific instructions dynamically
"""

from fastapi import APIRouter, Form, HTTPException, Request, Depends
from PIL import Image
import uuid
import io
import tempfile
import os
import requests
from Services.LLMGemini import generate_interior_design
from Services.FileManager import FileManager
from Services import DatabaseManager as DM
from Services import Logger
from middleware.auth import _verify_api_key
import re


class UploadFileAdapter:
    """Adapter to make image bytes compatible with FileManager's expected UploadFile interface"""
    def __init__(self, file_content: bytes, filename: str):
        self.file = io.BytesIO(file_content)
        self.filename = filename


def extract_keywords(user_prompt: str) -> list:
    """
    Convert user prompt into structured designer notes using Gemini LLM.

    Args:
        user_prompt: Raw user input (e.g., "Make the walls brighter, add more lights and decorations")

    Returns:
        List of actionable designer notes (e.g., ["Lighten wall colour to a warm off-white", "Add ambient lighting fixtures"])
    """
    if not user_prompt or not user_prompt.strip():
        return []

    try:
        # Import Gemini client
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Use Gemini 2.5 Flash Lite for fast, designer note generation
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
                temperature=0.1,  # Low temperature for consistency
                max_output_tokens=50
            )
        )

        # Parse the response - split bullet points into a list
        notes_text = response.text.strip()

        # Split by newline and clean up bullet formatting
        notes = [line.strip().lstrip('-').strip() for line in notes_text.splitlines()]

        # Filter out empty lines
        notes = [n for n in notes if n and len(n) > 3][:6]

        return notes

    except Exception as e:
        Logger.log(f"[IMAGE GENERATION] - Error generating designer notes with LLM: {str(e)}")
        # Fallback to simple extraction if LLM fails
        return _simple_keyword_extraction(user_prompt)


def _simple_keyword_extraction(user_prompt: str) -> list:
    """
    Fallback simple keyword extraction if LLM fails.

    Args:
        user_prompt: Raw user input

    Returns:
        List of basic extracted keywords
    """
    if not user_prompt or not user_prompt.strip():
        return []

    # Convert to lowercase
    text = user_prompt.lower()

    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'make',
        'add', 'more', 'some', 'any', 'it', 'this', 'that', 'these', 'those',
        'look', 'less', 'get', 'want', 'need'
    }

    # Extract words (alphanumeric only)
    words = re.findall(r'\b[a-z]+\b', text)

    # Filter out stop words and keep meaningful keywords
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)

    # Limit to top 10 keywords
    return unique_keywords[:10]


router = APIRouter(prefix="/image", tags=["Image Generation"], dependencies=[Depends(_verify_api_key)])

@router.post("/generate")
async def generate_image(
    styles: str = Form(...),
    user_prompt: str = Form(None),  # NEW: Optional user customization prompt
    request: Request = None
):
    """
    Generate a new room design using Gemini Imagen 3.

    Args:
        styles: Comma-separated string of selected interior design styles
        user_prompt: Optional user-provided instructions for customization
                    (e.g., "Make the walls darker blue, add more plants")
        request: FastAPI request object to extract user information

    Returns:
        JSON with generated image URL, file_id, and filename
    """
    tmp_file_path = None
    try:
        # Extract user ID from request
        user_id = None
        if request:
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                user_id = request.headers.get('X-User-ID')

        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="User ID is required"
            )

        # Fetch image info from Style Analysis node
        analysis_path = ["Users", user_id, "Existing Homeowner", "Style Analysis"]
        analysis_data = DM.peek(analysis_path)

        if not analysis_data:
            raise HTTPException(
                status_code=404,
                detail="No style analysis found. Please complete style analysis first."
            )

        image_url = analysis_data.get('image_url')
        if not image_url:
            raise HTTPException(
                status_code=404,
                detail="Image URL not found in style analysis data."
            )

        # Download image from Cloudinary
        response = requests.get(image_url)
        response.raise_for_status()

        suffix = '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        init_image = Image.open(tmp_file_path).convert("RGB")

        # Generate new design with Imagen 3
        try:
            result_image = generate_interior_design(
                init_image=init_image,
                styles=styles,
                user_modifications=user_prompt
            )
        except Exception as gen_error:
            Logger.log(f"[IMAGE GENERATION] - ERROR in generate_interior_design: {str(gen_error)}. Error type: {type(gen_error).__name__}. Traceback: {traceback.format_exc()}")
            import traceback
            raise HTTPException(
                status_code=500,
                detail=f"Image generation failed: {str(gen_error)}"
            )

        # Upload to Cloudinary
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

        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        # Save generated image to database
        # Extract designer notes from user prompt for better storage
        designer_notes = extract_keywords(user_prompt) if user_prompt else []

        generation_data = {
            "image_url": cloudinary_result["url"],
            "file_id": cloudinary_result["file_id"],
            "filename": cloudinary_result["filename"],
            "styles": styles,
            "designer_notes": designer_notes,
        }

        # Path: Users/{userId}/Existing Homeowner/Image Generation
        generation_path = ["Users", user_id, "Existing Homeowner", "Image Generation"]
        DM.set_value(generation_path, generation_data)

        # Keep Preferences/selected_styles in sync with the styles used for this generation
        preferences_styles_path = ["Users", user_id, "Existing Homeowner", "Preferences", "selected_styles"]
        DM.set_value(preferences_styles_path, styles.split(", "))

        DM.save()

        # Return Cloudinary URL and file_id
        return {
            "url": cloudinary_result["url"],
            "file_id": cloudinary_result["file_id"],
            "filename": cloudinary_result["filename"]
        }

    except Exception as e:
        # Error handling and cleanup
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        Logger.log(f"[IMAGE GENERATION] - Error in generate_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))