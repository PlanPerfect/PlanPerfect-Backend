from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import JSONResponse
from middleware.auth import _verify_api_key
from gradio_client import Client, handle_file
from PIL import Image
import tempfile
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from Services import RAGManager as RAG
from Services import LLMManager as LLM

router = APIRouter(prefix="/stylematch/detection", tags=["Furniture Detection"], dependencies=[Depends(_verify_api_key)])

@router.post("/detect-furniture")
async def detect_furniture(file: UploadFile):
    """
    Detect furniture in the uploaded room image.

    Args:
        file: Uploaded room image file

    Returns:
        JSON response with detected furniture items and their confidence scores
    """
    tmp_file_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        client = Client("Joshua-is-tired/PlanPerfect-Furniture-Detection")

        result = client.predict(
            pilimg=handle_file(tmp_file_path),
            api_name="/predict"
        )

        annotated_image_path, detection_data = result

        now = datetime.now()
        hour = now.strftime("%I").lstrip("0")  # Remove leading zero from hour
        timestamp = f"{now.strftime('%d%b')}{hour}{now.strftime('%M%p')}"
        base_output_dir = Path("static/predictions/furniture-detection")
        output_dir = base_output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        original_image = Image.open(tmp_file_path)
        img_width, img_height = original_image.size

        padding_percent = 0.3

        class_counts = defaultdict(int)
        cropped_images = []

        for item in detection_data.get("detected_items", []):
            class_name = item["class_name"]
            bbox = item["bbox"]

            class_counts[class_name] += 1

            filename = f"{class_name}_{class_counts[class_name]}.png"

            bbox_width = bbox["x2"] - bbox["x1"]
            bbox_height = bbox["y2"] - bbox["y1"]

            padding_x = bbox_width * padding_percent
            padding_y = bbox_height * padding_percent

            x1_padded = max(0, bbox["x1"] - padding_x)
            y1_padded = max(0, bbox["y1"] - padding_y)
            x2_padded = min(img_width, bbox["x2"] + padding_x)
            y2_padded = min(img_height, bbox["y2"] + padding_y)

            cropped = original_image.crop((
                x1_padded,
                y1_padded,
                x2_padded,
                y2_padded
            ))

            output_path = output_dir / filename
            cropped.save(output_path)

            cropped_images.append({
                "class": class_name,
                "filename": filename,
                "confidence": round(item["confidence"])
            })

        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "detections": detection_data,
                "images": cropped_images
            }
        )

    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )