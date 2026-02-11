"""
Image Generation API - Using Gemini Imagen 3

This file handles image generation using Gemini's native image generation (Imagen 3).
This replaces the Stable Diffusion pipeline for faster, higher-quality results.

Workflow:
1. Receives input image and style preferences from frontend
2. Downloads image from Cloudinary
3. Uses Gemini Imagen 3 to transform the room design
4. Uploads generated image to Cloudinary
5. Returns the Cloudinary URL in the response

Benefits over Stable Diffusion:
- Much faster generation (no GPU required)
- Higher quality, sharper images
- Better at preserving room layout
- No blur or artifacts
- Simpler pipeline (no model loading, VRAM management, etc.)
"""

from fastapi import APIRouter, Form, HTTPException, Request, Depends
from PIL import Image
import uuid
import io
import tempfile
import os
import requests
from Services.LLMGemini import generate_interior_design, generate_interior_design_with_mask
from Services.FileManager import FileManager
from Services import DatabaseManager as DM
from middleware.auth import _verify_api_key


class UploadFileAdapter:
    """Adapter to make image bytes compatible with FileManager's expected UploadFile interface"""
    def __init__(self, file_content: bytes, filename: str):
        self.file = io.BytesIO(file_content)
        self.filename = filename


router = APIRouter(prefix="/image", tags=["Image Generation"], dependencies=[Depends(_verify_api_key)])

@router.post("/generate")
async def generate_image(
    styles: str = Form(...),
    request: Request = None
):
    """
    Generate a new room design using Gemini Imagen 3.

    Args:
        styles: Comma-separated string of selected interior design styles
        request: FastAPI request object to extract user information

    Returns:
        JSON with generated image URL, file_id, and filename
    """
    tmp_file_path = None
    try:
        # ================================
        # Extract user ID from request
        # ================================
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
        
        # ================================
        # Fetch image info from Style Analysis node
        # ================================
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
        
        print(f"=== FETCHING IMAGE FROM STYLE ANALYSIS ===")
        print(f"User ID: {user_id}")
        print(f"Image URL: {image_url}")
        
        # ================================
        # Download image from Cloudinary
        # ================================
        response = requests.get(image_url)
        response.raise_for_status()

        suffix = '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        init_image = Image.open(tmp_file_path).convert("RGB")

        print(f"=== GENERATING DESIGN WITH IMAGEN 3 ===")
        print(f"Input image size: {init_image.size}")
        print(f"Target styles: {styles}")

        # ================================
        # Generate new design with Imagen 3
        # ================================
        # Try the edit mode first (better at preserving layout)
        try:
            result_image = generate_interior_design_with_mask(
                init_image=init_image,
                styles=styles,
                preserve_layout=True  # Keep room structure, only change aesthetics
            )
            print("Generated using Imagen edit mode")
        except Exception as e:
            print(f"Edit mode failed: {str(e)}, falling back to generate mode")
            # Fallback to standard generation
            result_image = generate_interior_design(
                init_image=init_image,
                styles=styles
            )
            print("Generated using Imagen generate mode")

        print(f"Generated image size: {result_image.size}")

        # ================================
        # Upload to Cloudinary
        # ================================
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

        # ================================
        # Clean up temporary file
        # ================================
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        print("=== GENERATION COMPLETE ===")

        # ================================
        # Return Cloudinary URL and file_id
        # ================================
        return {
            "url": cloudinary_result["url"],
            "file_id": cloudinary_result["file_id"],
            "filename": cloudinary_result["filename"]
        }

    except Exception as e:
        # ================================
        # Error handling and cleanup
        # ================================
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        print(f"Error in generate_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))