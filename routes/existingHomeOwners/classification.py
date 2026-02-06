"""
Style Classification API for Existing Home Owners
This file handles the style classification of room images using a pre-trained model 
that is deployed on Hugging Face.
Workflow:
1. It accepts an image file upload via a POST request.
2. The image is uploaded to Cloudinary for permanent storage.
3. The image is temporarily saved to disk for HuggingFace processing.
4. The image file is sent to the Hugging Face model for style classification.
5. Returns the detected style, confidence score, and Cloudinary image URL in the response.

This file is part of the backend of the application and is used 
by styleAnalysis component in the existingHomeOwners frontend page
"""

from fastapi import APIRouter, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from middleware.auth import _verify_api_key
from Services.FileManager import FileManager
import tempfile
import os

router = APIRouter(prefix="/existingHomeOwners/styleClassification", tags=["Existing Home Owners Style Classification"], dependencies=[Depends(_verify_api_key)])

# ================================
# Style classification endpoint
# ================================
@router.post("/styleAnalysis")
async def analyze_room_style(file: UploadFile = File(...)):
    """
    Analyze the style of a room from an uploaded image file.
    Args: 
        file (UploadFile): The uploaded image file of the room from frontend via post request.
    Returns:
        JSONResponse: A JSON response containing detected style, confidence score, 
                      Cloudinary image URL, file_id, and success status.

    """

    tmp_file_path = None
    try:
        # ================================
        # Upload to Cloudinary first
        # ================================
        # Store the uploaded image in Cloudinary for permanent access
        cloudinary_result = FileManager.store_file(
            file=file,
            subfolder="Uploaded Room Images"
        )
        
        # Reset file pointer after FileManager reads it
        await file.seek(0)

        # ================================
        # Save uploaded image temporarily for HuggingFace
        # ================================
        # HuggingFace needs a local file path, so we create a temp file
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # ================================
        # Calls Hugging Face model for inference
        # ================================
        client = Client("jiaxinnnnn/Interior-Style-Classification-Deployment")

        result = client.predict(
            handle_file(tmp_file_path),
            api_name="/predict"
        )

        # ================================
        # Parse model output
        # ================================
        # The model output is expected to be a dict

        if isinstance(result, dict):
            detected_style = result.get("style", "Unknown")
            confidence = result.get("confidence", 0.0)
        else:
            raise ValueError(f"Unexpected model output: {result}")

        # ================================
        # Clean up temporary file
        # ================================
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        # ================================
        # Return successful response with Cloudinary info
        # ================================
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "detected_style": detected_style,
                    "confidence": float(confidence) if confidence else 0.0,
                    "image_url": cloudinary_result["url"],
                    "file_id": cloudinary_result["file_id"],
                    "filename": cloudinary_result["filename"],
                    "message": "Style classification completed successfully"
                }
            }
        )
        

    except Exception as e:
        # ================================
        # Error handling and cleanup
        # ================================
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": f"ERROR: Failed to classify room style. Error: {str(e)}"
            }
        )