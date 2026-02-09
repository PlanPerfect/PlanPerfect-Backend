from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from Services import ServiceOrchestra

router = APIRouter(prefix="/services", tags=["AI Services"])

# ================================
# Image Generation Endpoint
# ================================
@router.post("/generateImage")
async def generate_image(prompt: str = Form(...)):
    try:
        # Call ServiceOrchestra to generate and upload image
        result = ServiceOrchestra.generate_image(prompt=prompt)

        if not result:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to generate image. Please check logs for details."
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "image_url": result["url"],
                    "file_id": result["file_id"],
                    "filename": result["filename"],
                    "message": "Image generated successfully"
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"ERROR: Failed to generate image. Error: {str(e)}"
            }
        )


# ================================
# Style Classification Endpoint
# ================================
@router.post("/classifyStyle")
async def classify_style(file: UploadFile = File(...)):
    try:
        # Call ServiceOrchestra to classify style
        result = await ServiceOrchestra.classify_style(file=file)

        if not result:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to classify room style. Please check logs for details."
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "detected_style": result["detected_style"],
                    "confidence": result["confidence"],
                    "message": "Style classification completed successfully"
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"ERROR: Failed to classify room style. Error: {str(e)}"
            }
        )

# ================================
# Furniture Detection Endpoint
# ================================
@router.post("/detectFurniture")
async def detect_furniture(file: UploadFile = File(...)):
    try:
        # Call ServiceOrchestra to detect furniture
        result = await ServiceOrchestra.detect_furniture(file=file)

        if not result:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to detect furniture. Please check logs for details."
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "detections": result["detections"],
                    "total_items": result["total_items"],
                    "message": "Furniture detection completed successfully"
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"ERROR: Failed to detect furniture. Error: {str(e)}"
            }
        )