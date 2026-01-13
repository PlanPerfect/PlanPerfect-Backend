from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from middleware.auth import checkHeaders
import tempfile
import os

router = APIRouter(prefix="/existingHomeOwners/styleClassification", tags=["Existing Home Owners Style Classification"])

@router.post("/styleAnalysis")
@checkHeaders
async def analyze_room_style(file: UploadFile = File(...)):
    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        client = Client("jiaxinnnnn/Interior-Style-Classification-Deployment")
        
        result = client.predict(
            handle_file(tmp_file_path),
            api_name="/predict"
        )

<<<<<<< HEAD
        if isinstance(result, (list, tuple)) and len(result) == 2:
            detected_style, confidence = result
        elif isinstance(result, dict):
            detected_style = result.get("style", "Unknown")
            confidence = result.get("confidence", 0.0)
        else:
            raise ValueError(f"Unexpected model output: {result}")

        
=======
        # Parse result - model returns (style, confidence)
        if isinstance(result, dict):
            # If model returns a dictionary
            detected_style = result.get("style", "Unknown")
            confidence = result.get("confidence", 0.0)
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            # If model returns a tuple/list [style, confidence]
            detected_style = result[0]
            confidence = result[1]
        else:
            # Fallback
            detected_style = str(result)
            confidence = 0.0
        
        # Clean up temporary file
>>>>>>> jesh
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "detected_style": detected_style,
                    "confidence": float(confidence) if confidence else 0.0,
<<<<<<< HEAD
=======
                    # "detected_furniture": detected_furniture if isinstance(detected_furniture, list) else [],
>>>>>>> jesh
                    "message": "Style classification completed successfully"
                }
            }
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
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