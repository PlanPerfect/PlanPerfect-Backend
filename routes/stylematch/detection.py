from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from middleware.auth import _verify_api_key
from gradio_client import Client, handle_file
from PIL import Image
import pillow_avif
import tempfile
import os
import asyncio
import httpx
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

router = APIRouter(
    prefix="/stylematch/detection",
    tags=["Furniture Detection"],
    dependencies=[Depends(_verify_api_key)]
)

executor = ThreadPoolExecutor(max_workers=8)


def run_furniture_detection(file_path: str):
    """Blocking function to run furniture detection with extended timeout"""
    client = Client(
        "Joshua-is-tired/PlanPerfect-Furniture-Detection",
        httpx_kwargs={
            "timeout": httpx.Timeout(
                timeout=300.0,  # 5 minutes total
                connect=60.0,   # 1 minute to connect
                read=240.0,     # 4 minutes to read
                write=60.0      # 1 minute to write
            )
        }
    )

    return client.predict(
        pilimg=handle_file(file_path),
        api_name="/predict"
    )


@router.post("/detect-furniture")
async def detect_furniture(file: UploadFile = File(...)):
    """
    Detect furniture in the uploaded room image.

    Args:
        file: Uploaded room image file

    Returns:
        JSON response with detected furniture items and their confidence scores
    """
    tmp_file_path = None
    png_file_path = None

    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        image = Image.open(tmp_file_path)

        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if 'A' in image.mode:
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        png_fd, png_file_path = tempfile.mkstemp(suffix=".png")
        os.close(png_fd)

        image.save(png_file_path, format="PNG", optimize=False)
        image.close()

        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
            tmp_file_path = None

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            run_furniture_detection,
            png_file_path
        )

        annotated_image_path, detection_data = result

        now = datetime.now()
        hour = now.strftime("%I").lstrip("0")
        timestamp = f"{now.strftime('%d%b')}{hour}{now.strftime('%M%p')}"
        output_dir = Path("static/predictions/furniture-detection") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        original_image = Image.open(png_file_path)
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

            x1 = max(0, bbox["x1"] - padding_x)
            y1 = max(0, bbox["y1"] - padding_y)
            x2 = min(img_width, bbox["x2"] + padding_x)
            y2 = min(img_height, bbox["y2"] + padding_y)

            cropped = original_image.crop((x1, y1, x2, y2))
            output_path = output_dir / filename
            cropped.save(output_path, format="PNG")

            cropped_images.append({
                "class": class_name,
                "filename": filename,
                "timestamp": timestamp,
                "confidence": f"{round(item['confidence'] * 100)}%"
            })

        original_image.close()

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "detections": detection_data,
                "images": cropped_images
            }
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Detection service timed out. Please try with a smaller image."
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

        if png_file_path and os.path.exists(png_file_path):
            try:
                os.unlink(png_file_path)
            except Exception:
                pass