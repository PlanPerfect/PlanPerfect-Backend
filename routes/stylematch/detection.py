from fastapi import APIRouter, Depends, UploadFile, File, Header
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from Services import FileManager as FM, DatabaseManager as DM
import io

router = APIRouter(
    prefix="/stylematch/detection",
    tags=["Furniture Detection"],
    dependencies=[Depends(_verify_api_key)]
)

executor = ThreadPoolExecutor(max_workers=8)

def run_furniture_detection(file_path: str):
    client = Client(
        "Joshua-is-tired/PlanPerfect-Furniture-Detection",
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
        pilimg=handle_file(file_path),
        api_name="/predict"
    )


@router.post("/detect-furniture")
async def detect_furniture(
    file: UploadFile = File(...),
    x_user_id: str = Header(..., alias="X-User-ID")
):
    tmp_file_path = None
    png_file_path = None

    if not x_user_id:
        return JSONResponse(
            status_code=400,
            content={
                "error": "UERROR: One or more required fields are invalid / missing."
            }
        )

    try:
        user = DM.peek(["Users", x_user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

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

        original_image = Image.open(png_file_path)
        img_width, img_height = original_image.size
        padding_percent = 0.3

        class_counts = defaultdict(int)
        cropped_images = []

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        for item in detection_data.get("detected_items", []):
            class_name = item["class_name"]
            bbox = item["bbox"]
            class_counts[class_name] += 1

            bbox_width = bbox["x2"] - bbox["x1"]
            bbox_height = bbox["y2"] - bbox["y1"]
            padding_x = bbox_width * padding_percent
            padding_y = bbox_height * padding_percent

            x1 = max(0, bbox["x1"] - padding_x)
            y1 = max(0, bbox["y1"] - padding_y)
            x2 = min(img_width, bbox["x2"] + padding_x)
            y2 = min(img_height, bbox["y2"] + padding_y)

            cropped = original_image.crop((x1, y1, x2, y2))

            img_byte_arr = io.BytesIO()
            cropped.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            filename = f"{class_name}_{class_counts[class_name]}_{timestamp}.png"

            class MockUploadFile:
                def __init__(self, file_bytes, filename):
                    self.file = io.BytesIO(file_bytes)
                    self.filename = filename

            mock_file = MockUploadFile(img_byte_arr.getvalue(), filename)

            upload_result = FM.store_file(
                file=mock_file,
                subfolder=f"Detected Furniture/{x_user_id}"
            )

            detection_key = f"{timestamp}_{class_name}_{class_counts[class_name]}"
            DM.set_value(
                path=["Users", x_user_id, "Existing Homeowner", "Detected Furniture", "furniture", detection_key],
                value={
                    "file_id": upload_result["file_id"],
                    "url": upload_result["url"]
                }
            )

            cropped_images.append({
                "class": class_name,
                "url": upload_result["url"],
                "confidence": f"{round(item['confidence'] * 100)}%"
            })

            cropped.close()

        original_image.close()

        DM.save()

        return JSONResponse(
            status_code=200,
            content={
                "detections": cropped_images
            }
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": "ERROR: Service timeout. Please try again with a smaller image."
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
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