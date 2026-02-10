from google import genai
from google.genai import types
from typing import Optional, Dict
import os
import base64
import tempfile
from io import BytesIO
from PIL import Image
import pillow_avif  # Required for AVIF support
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime
from Services import Logger
from Services import FileManager as FM
from gradio_client import Client, handle_file

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_MODEL = "gemini-3-pro-image-preview"

class ServiceOrchestraClass:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        try:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._initialized = True
            print("SERVICE ORCHESTRA INITIALIZED. SERVICES READY.\n")
        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Failed to initialize. Error: {e}")
            raise

    def generate_image(
        self,
        prompt: str,
    ) -> Optional[Dict[str, str]]:
        if not self._initialized:
            raise RuntimeError("ServiceOrchestra not initialized. Call initialize() first.")

        if not self._client:
            Logger.log("[SERVICE ORCHESTRA] - ERROR: Client not available.")
            return None

        try:
            response = self._client.models.generate_content(
                model=IMAGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                        image_size="4K"
                    ),
                )
            )

            image_bytes = None

            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    pass

                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    if image_data:
                        if isinstance(image_data, bytes):
                            image_bytes = image_data
                        else:
                            image_bytes = base64.b64decode(image_data)
                        break

            if not image_bytes:
                Logger.log("[SERVICE ORCHESTRA] - WARNING: No image data found in response.")
                return None

            filename = f"{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
            file_obj = BytesIO(image_bytes)
            file_obj.seek(0)

            class MockUploadFile:
                def __init__(self, filename: str, file_obj: BytesIO):
                    self.filename = filename
                    self.file = file_obj
                    self.content_type = "image/png"

            mock_file = MockUploadFile(filename, file_obj)

            upload_result = FM.store_file(
                file=mock_file,
                subfolder="Generated Images"
            )

            return upload_result

        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Image generation/upload failed. {str(e)}")
            return None

    async def classify_style(
        self,
        file
    ) -> Optional[Dict[str, any]]:
        if not self._initialized:
            raise RuntimeError("ServiceOrchestra not initialized. Call initialize() first.")

        if not self._client:
            Logger.log("[SERVICE ORCHESTRA] - ERROR: Client not available.")
            return None

        tmp_file_path = None

        try:
            suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                if hasattr(file, 'read'):
                    content = await file.read()
                else:
                    content = file.file.read()

                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            client = Client("jiaxinnnnn/Interior-Style-Classification-Deployment")

            result = client.predict(
                handle_file(tmp_file_path),
                api_name="/predict"
            )

            if isinstance(result, dict):
                detected_style = result.get("style", "Unknown")
                confidence = result.get("confidence", 0.0)
            else:
                Logger.log(f"[SERVICE ORCHESTRA] - WARNING: Unexpected model output: {result}")
                return None

            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

            return {
                "detected_style": detected_style,
                "confidence": float(confidence) if confidence else 0.0
            }

        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Style classification failed. {str(e)}")

            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

            return None

    async def detect_furniture(
        self,
        file
    ) -> Optional[Dict[str, any]]:
        if not self._initialized:
            raise RuntimeError("ServiceOrchestra not initialized. Call initialize() first.")

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
            detected_items = []

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

                img_byte_arr = BytesIO()
                cropped.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                filename = f"{class_name}_{class_counts[class_name]}_{timestamp}.png"

                class MockUploadFile:
                    def __init__(self, file_bytes, filename):
                        self.file = BytesIO(file_bytes)
                        self.filename = filename

                mock_file = MockUploadFile(img_byte_arr.getvalue(), filename)

                upload_result = FM.store_file(
                    file=mock_file,
                    subfolder="Detected Furniture"
                )

                detected_items.append({
                    "class": class_name,
                    "url": upload_result["url"],
                    "file_id": upload_result["file_id"],
                    "confidence": round(item['confidence'] * 100)
                })

                cropped.close()

            original_image.close()

            if png_file_path and os.path.exists(png_file_path):
                os.unlink(png_file_path)

            return {
                "detections": detected_items,
                "total_items": len(detected_items)
            }

        except asyncio.TimeoutError:
            Logger.log("[SERVICE ORCHESTRA] - ERROR: Furniture detection timeout.")

            if png_file_path and os.path.exists(png_file_path):
                try:
                    os.unlink(png_file_path)
                except:
                    pass

            return None

        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Furniture detection failed. {str(e)}")

            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

            if png_file_path and os.path.exists(png_file_path):
                try:
                    os.unlink(png_file_path)
                except:
                    pass

            return None

    async def get_recommendations(
        self,
        style: str,
        furniture_name: str
    ) -> Optional[Dict[str, any]]:
        if not self._initialized:
            raise RuntimeError("ServiceOrchestra not initialized. Call initialize() first.")

        try:
            import httpx
            import random

            search_query = f"{style} styled {furniture_name}"

            params = {
                "query": search_query,
                "per_page": 5,
                "orientation": "landscape"
            }

            PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

            if not PEXELS_API_KEY:
                Logger.log("[SERVICE ORCHESTRA] - ERROR: PEXELS_API_KEY not found in environment.")
                return None

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.pexels.com/v1/search",
                    params=params,
                    headers={"Authorization": PEXELS_API_KEY},
                    timeout=10.0
                )

                if response.status_code == 429:
                    Logger.log("[SERVICE ORCHESTRA] - WARNING: Pexels API rate limit reached.")
                    return {"error": "rate_limit", "message": "Too many requests. Please try again later."}

                if response.status_code == 403:
                    Logger.log("[SERVICE ORCHESTRA] - WARNING: Pexels API access forbidden.")
                    return {"error": "forbidden", "message": "Too many requests. Please try again later."}

                if response.status_code != 200:
                    Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Pexels API returned status {response.status_code}")
                    return None

                data = response.json()

                recommendations = []
                for photo in data.get("photos", []):
                    description = (
                        photo.get("alt") or
                        f"A beautiful {style} styled {furniture_name}"
                    )

                    def truncate_description(desc: str, max_length: int = 100) -> str:
                        if len(desc) <= max_length:
                            return desc
                        return desc[:max_length].rsplit(' ', 1)[0] + "..."

                    recommendations.append({
                        "name": f"{style}-themed {furniture_name}",
                        "image": photo["src"]["large"],
                        "description": truncate_description(description),
                        "match": random.randint(85, 99),
                    })

                return {
                    "recommendations": recommendations
                }

        except httpx.TimeoutException:
            Logger.log("[SERVICE ORCHESTRA] - ERROR: Pexels API timeout.")
            return {"error": "timeout", "message": "Service timeout. Please try again."}

        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Failed to get recommendations. {str(e)}")
            return None


ServiceOrchestra = ServiceOrchestraClass()