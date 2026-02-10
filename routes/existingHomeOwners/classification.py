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
from datetime import datetime
import tempfile
import os

router = APIRouter(prefix="/existingHomeOwners/styleClassification", tags=["Existing Home Owners Style Classification"], dependencies=[Depends(_verify_api_key)])

# ================================
# Style classification endpoint
# ================================
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
        # ================================
        # Extract user ID from request
        # ================================
        user_id = None
        if request:
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                user_id = request.headers.get('X-User-ID')
        
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
        # Save to Firebase Database
        # ================================
        if user_id:
            try:
                # Prepare analysis data
                analysis_data = {
                    "detected_style": detected_style,
                    "confidence": float(confidence) if confidence else 0.0,
                    "image_url": cloudinary_result["url"],
                    "file_id": cloudinary_result["file_id"],
                    "filename": cloudinary_result["filename"],
                    "analyzed_at": datetime.utcnow().isoformat()
                }
                
                # Path: Users/{userId}/Existing Homeowner/Style Analysis
                path = ["Users", user_id, "Existing Homeowner", "Style Analysis"]
                
                # Save to database
                DM.set_value(path, analysis_data)
                DM.save()
                
            except Exception as db_error:
                # Log error but don't fail the request
                print(f"Warning: Failed to save to database: {str(db_error)}")

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


# ================================
# Save Preferences endpoint
# ================================
@router.post("/savePreferences")
async def save_preferences(
    preferences: str = Form(...),
    analysis: str = Form(...),
    selected_styles: str = Form(...),
    user_id: str = Form(...)
):
    """
    Save user preferences from the preview page to the database.
    
    This endpoint is called when the user completes the preview page
    and clicks "Get Started". It stores all their selections including
    property preferences, detected style, and selected design themes.
    
    Args:
        preferences (str): JSON string of preferences data
        analysis (str): JSON string of analysis data
        selected_styles (str): JSON string of selected styles array
        user_id (str): User's unique identifier
        
    Returns:
        JSONResponse: Success confirmation with timestamp
    """
    try:
        # ================================
        # Parse JSON strings from form data
        # ================================
        import json
        
        try:
            preferences_dict = json.loads(preferences) if isinstance(preferences, str) else preferences
            analysis_dict = json.loads(analysis) if isinstance(analysis, str) else analysis
            selected_styles_list = json.loads(selected_styles) if isinstance(selected_styles, str) else selected_styles
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "result": f"ERROR: Invalid JSON format - {str(e)}"
                }
            )
        
        # ================================
        # Prepare data for database
        # ================================
        timestamp = datetime.utcnow().isoformat()
        
        preferences_data = {
            "property_type": preferences_dict.get('propertyType', 'Not specified'),
            "unit_type": preferences_dict.get('unitType', 'Not specified'),
            "budget": preferences_dict.get('budget', 'Not specified'),
            "detected_style": analysis_dict.get('detected_style', 'Unknown'),
            "confidence": analysis_dict.get('confidence', 0.0),
            "original_image_url": analysis_dict.get('original_image_url', ''),
            "file_id": analysis_dict.get('file_id', ''),
            "filename": analysis_dict.get('filename', ''),
            "selected_styles": selected_styles_list,
            "saved_at": timestamp
        }
        
        # ================================
        # Save to Firebase Database
        # ================================
        # Path: Users/{userId}/Existing Homeowner/Preferences
        path = ["Users", user_id, "Existing Homeowner", "Preferences"]
        
        DM.set_value(path, preferences_data)
        DM.save()
        
        # ================================
        # Return success response
        # ================================
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
        import traceback
        error_details = traceback.format_exc()
        print(f"Error saving preferences: {error_details}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": f"ERROR: Failed to save preferences. Error: {str(e)}"
            }
        )


# ================================
# Get Preferences endpoint
# ================================
@router.get("/getPreferences/{user_id}")
async def get_preferences(user_id: str):
    """
    Retrieve user preferences from the database.
    
    This endpoint is called by the imageGeneration page to fetch
    the user's saved preferences, analysis results, and selected styles.
    
    Args:
        user_id (str): User's unique identifier
        
    Returns:
        JSONResponse: User's preferences data or error message
    """
    try:
        # ================================
        # Fetch from Firebase Database
        # ================================
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
        
        # ================================
        # Return preferences data
        # ================================
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": preferences_data
            }
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error retrieving preferences: {error_details}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": f"ERROR: Failed to retrieve preferences. Error: {str(e)}"
            }
        )