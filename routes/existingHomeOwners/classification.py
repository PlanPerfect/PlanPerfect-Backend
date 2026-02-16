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
6. Saves the analysis results to Firebase database.

This file is part of the backend of the application and is used
by styleAnalysis component in the existingHomeOwners frontend page
"""

from fastapi import APIRouter, File, UploadFile, Depends, Request, Form
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from middleware.auth import _verify_api_key
from Services.FileManager import FileManager
from Services import DatabaseManager as DM
from Services import Logger
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import traceback
import tempfile
import asyncio
import httpx
import json
import os

router = APIRouter(prefix="/existingHomeOwners/styleClassification", tags=["Existing Home Owners Style Classification"], dependencies=[Depends(_verify_api_key)])

executor = ThreadPoolExecutor(max_workers=8)

def run_style_classification(file_path: str):
    client = Client(
        "jiaxinnnnn/Interior-Style-Classification-Deployment",
        httpx_kwargs={
            "timeout": httpx.Timeout(
                timeout=300.0,
                connect=60.0,
                read=240.0,
                write=60.0
            )
        }
    )

    return client.predict(
        handle_file(file_path),
        api_name="/predict"
    )

# Style classification endpoint
@router.post("/styleAnalysis")
async def analyze_room_style(file: UploadFile = File(...), request: Request = None):
    """
    Analyze the style of a room from an uploaded image file.
    Args:
        file (UploadFile): The uploaded image file of the room from frontend via post request.
        request (Request): FastAPI request object to extract user information.
    Returns:
        JSONResponse: A JSON response containing detected style, confidence score,
                      Cloudinary image URL, file_id, and success status.

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
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "UERROR: One or more required fields are invalid / missing."
                        }
                    )

                user = DM.peek(["Users", user_id])
                if user is None:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": "UERROR: Please login again."
                        }
                    )

        # Generate unique filename to avoid duplicates
        # Use timestamp + user_id to ensure uniqueness
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1] if original_filename else '.png'
        unique_filename = f"room_style_{user_id}_{timestamp}{file_extension}"

        # Temporarily modify the file's filename attribute
        file.filename = unique_filename

        # Upload to Cloudinary first
        # Store the uploaded image in Cloudinary for permanent storage
        cloudinary_result = FileManager.store_file(
            file=file,
            subfolder="Uploaded Room Images"
        )

        # Reset file pointer after FileManager reads it
        await file.seek(0)

        # Save uploaded image temporarily for HuggingFace
        # HuggingFace needs a local file path, so we create a temp file
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Calls Hugging Face model for inference via thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            run_style_classification,
            tmp_file_path
        )

        # Parse model output
        # The model output is expected to be a dict

        if isinstance(result, dict):
            detected_style = result.get("style", "Unknown")
            confidence = result.get("confidence", 0.0)
        else:
            raise ValueError(f"Unexpected model output: {result}")

        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        # ================================
        # Return successful response with Cloudinary info
        # NOTE: We do NOT save to database here - that only happens
        # in the savePreferences endpoint when the user completes the preview page
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


    except asyncio.TimeoutError:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(
            status_code=504,
            content={
                "error": "ERROR: Service timeout. Please try again with a smaller image."
            }
        )

    except Exception as e:
        # Error handling and cleanup
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(status_code=500, content={ "error": str(e) })


# Save Preferences endpoint
@router.post("/savePreferences")
async def save_preferences(
    preferences: str = Form(...),
    selected_styles: str = Form(...),
    analysis_results: str = Form(...),
    user_id: str = Form(...)
):
    """
    Save user preferences from the preview page to the database.

    This endpoint is called when the user completes the preview page
    and clicks "Get Started". It stores all their selections including
    property preferences, detected style (from Style Analysis), and selected design themes.

    Args:
        preferences (str): JSON string of preferences data
        selected_styles (str): JSON string of selected styles array
        analysis_results (str): JSON string of style analysis results (detected_style, image_url, file_id, filename)
        user_id (str): User's unique identifier

    Returns:
        JSONResponse: Success confirmation with timestamp
    """
    try:
        if not user_id or not user_id.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: Please login again."
                }
            )

        try:
            preferences_dict = json.loads(preferences) if isinstance(preferences, str) else preferences
            selected_styles_list = json.loads(selected_styles) if isinstance(selected_styles, str) else selected_styles
            analysis_dict = json.loads(analysis_results) if isinstance(analysis_results, str) else analysis_results
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={ "error": str(e) })

        # Prepare data for database
        timestamp = datetime.utcnow().isoformat()

        # Save Style Analysis data
        analysis_data = {
            "image_url": analysis_dict.get('image_url'),
            "file_id": analysis_dict.get('file_id'),
            "filename": analysis_dict.get('filename'),
            "detected_style": analysis_dict.get('detected_style')
        }

        DM.data["Users"][user_id]["Existing Homeowner"]["Style Analysis"] = analysis_data
        DM.save()

        # Save Preferences data
        preferences_data = {
            "property_type": preferences_dict.get('propertyType', 'Not specified'),
            "unit_type": preferences_dict.get('unitType', 'Not specified'),
            "budget_min": preferences_dict.get('budgetMin', 0),
            "budget_max": preferences_dict.get('budgetMax', 0),
            "detected_style": analysis_dict.get('detected_style', 'Unknown'),
            "selected_styles": selected_styles_list,
        }

        # Save preferences to Firebase Database
        # Path: Users/{userId}/Existing Homeowner/Preferences

        DM.data["Users"][user_id]["Existing Homeowner"]["Preferences"] = preferences_data
        DM.data["Users"][user_id]["flow"] = "existingHomeOwner"
        DM.save()

        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "message": "Preferences saved successfully",
                    "saved_at": timestamp,
                    "user_id": user_id
                }
            }
        )

    except Exception as e:
        error_details = traceback.format_exc()
        Logger.log(f"[CLASSIFICATION] - Error saving preferences: {error_details}")

        return JSONResponse(status_code=500, content={ "error": str(e) })


# Get Preferences endpoint
@router.get("/getPreferences/{user_id}")
async def get_preferences(user_id: str):
    """
    Retrieve user preferences from the database.

    This endpoint is called by the imageGeneration page to fetch
    the user's saved preferences, analysis results, and selected styles.
    It also fetches the original image URL from the Style Analysis node.

    Args:
        user_id (str): User's unique identifier

    Returns:
        JSONResponse: User's preferences data including original image URL
    """
    try:
        if not user_id or not user_id.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: Please login again."
                }
            )

        # Path: Users/{userId}/Existing Homeowner/Preferences
        path = ["Users", user_id, "Existing Homeowner", "Preferences"]

        preferences_data = DM.peek(path)

        if not preferences_data:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "result": "No preferences found for this user"
                }
            )

        # Also fetch the original image URL from Style Analysis
        analysis_path = ["Users", user_id, "Existing Homeowner", "Style Analysis"]
        analysis_data = DM.peek(analysis_path)

        # Add original image URL to the response if available
        if analysis_data and analysis_data.get('image_url'):
            preferences_data['original_image_url'] = analysis_data.get('image_url')

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": preferences_data
            }
        )

    except Exception as e:
        error_details = traceback.format_exc()
        Logger.log(f"[CLASSIFICATION] - Error retrieving preferences: {error_details}")

        return JSONResponse(status_code=500, content={ "error": str(e) })