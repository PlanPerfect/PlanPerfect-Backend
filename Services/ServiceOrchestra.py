from google import genai
from google.genai import types
from typing import Optional, Dict
import os
import base64
import tempfile
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


ServiceOrchestra = ServiceOrchestraClass()