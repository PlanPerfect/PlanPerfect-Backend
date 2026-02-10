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

# ================================
# Furniture Recommendations Endpoint
# ================================
@router.post("/getRecommendations")
async def get_recommendations(
    style: str = Form(...),
    furniture_name: str = Form(...)
):
    try:
        result = await ServiceOrchestra.get_recommendations(
            style=style,
            furniture_name=furniture_name
        )

        if not result:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to get recommendations. Please check logs for details."
                }
            )

        if "error" in result:
            if result["error"] == "rate_limit":
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "message": result["message"]
                    }
                )
            elif result["error"] == "forbidden":
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": result["message"]
                    }
                )
            elif result["error"] == "timeout":
                return JSONResponse(
                    status_code=504,
                    content={
                        "success": False,
                        "message": result["message"]
                    }
                )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "recommendations": result["recommendations"],
                    "message": "Recommendations retrieved successfully"
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"ERROR: Failed to get recommendations. Error: {str(e)}"
            }
        )