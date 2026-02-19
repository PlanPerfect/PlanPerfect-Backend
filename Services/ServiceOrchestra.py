from google import genai
from google.genai import types
from typing import Optional, Dict, List, Any
import os
import base64
import inspect
import json
import tempfile
from io import BytesIO
from PIL import Image
import pillow_avif  # Required for AVIF support
import asyncio
import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from io import BytesIO
from sklearn.cluster import KMeans
from datetime import datetime
from Services import Logger
from Services import FileManager as FM
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from sklearn.cluster import KMeans
from linkup import LinkupClient

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

    async def _read_input_file_bytes(self, file: Any) -> bytes:
        if hasattr(file, "_content") and isinstance(getattr(file, "_content"), (bytes, bytearray)):
            return bytes(getattr(file, "_content"))

        read_obj = getattr(file, "read", None)
        tell_obj = getattr(file, "tell", None)
        seek_obj = getattr(file, "seek", None)

        inner_file = getattr(file, "file", None)
        inner_read_obj = getattr(inner_file, "read", None)
        inner_tell_obj = getattr(inner_file, "tell", None)
        inner_seek_obj = getattr(inner_file, "seek", None)

        start_pos = None
        if callable(tell_obj):
            try:
                current_pos = tell_obj()
                start_pos = await current_pos if inspect.isawaitable(current_pos) else current_pos
            except Exception:
                start_pos = None
        elif callable(inner_tell_obj):
            try:
                start_pos = inner_tell_obj()
            except Exception:
                start_pos = None

        data: bytes = b""
        if callable(read_obj):
            raw = read_obj()
            raw = await raw if inspect.isawaitable(raw) else raw
            if isinstance(raw, (bytes, bytearray)):
                data = bytes(raw)
            elif isinstance(raw, str):
                data = raw.encode("utf-8")
        elif callable(inner_read_obj):
            raw = inner_read_obj()
            if isinstance(raw, (bytes, bytearray)):
                data = bytes(raw)
            elif isinstance(raw, str):
                data = raw.encode("utf-8")

        if start_pos is not None:
            if callable(seek_obj):
                try:
                    result = seek_obj(start_pos)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
            elif callable(inner_seek_obj):
                try:
                    inner_seek_obj(start_pos)
                except Exception:
                    pass

        return data

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
                        image_size="2K"
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

            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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
                content = await self._read_input_file_bytes(file)
                if not content:
                    raise ValueError("Empty file content")

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
                content = await self._read_input_file_bytes(file)
                if not content:
                    raise ValueError("Empty file content")
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

    @staticmethod
    def _to_plain_dict(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload

        if hasattr(payload, "model_dump"):
            try:
                dumped = payload.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass

        if hasattr(payload, "dict"):
            try:
                dumped = payload.dict()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass

        return {}

    @staticmethod
    def _extract_linkup_answer(payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()

        data = ServiceOrchestraClass._to_plain_dict(payload)
        if not data:
            return ""

        for key in ("answer", "result", "response", "message"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        nested_data = data.get("data")
        if isinstance(nested_data, dict):
            nested_answer = ServiceOrchestraClass._extract_linkup_answer(nested_data)
            if nested_answer:
                return nested_answer

        results = data.get("results")
        if isinstance(results, list):
            snippets: List[str] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                snippet = item.get("content") or item.get("snippet") or item.get("name")
                if isinstance(snippet, str):
                    cleaned = snippet.strip()
                    if cleaned:
                        snippets.append(cleaned)
                if len(snippets) >= 3:
                    break
            if snippets:
                return " ".join(snippets)

        return ""

    @staticmethod
    async def _direct_linkup_search(query: str, api_key: str) -> Optional[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PlanPerfect-ServiceOrchestra/1.0",
        }
        payload = {
            "q": query,
            "depth": "deep",
            "outputType": "sourcedAnswer",
            "includeImages": False,
        }

        try:
            async with httpx.AsyncClient(
                base_url="https://api.linkup.so/v1",
                headers=headers,
                timeout=httpx.Timeout(timeout=45.0, connect=10.0, read=35.0, write=10.0),
            ) as client:
                response = await client.post("/search", json=payload)
        except httpx.HTTPError as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Web search HTTP fallback failed. {str(e)}")
            return {
                "error": "network_error",
                "message": "Web search is temporarily unavailable. Please try again in a moment.",
            }

        response_text = response.text.strip()
        if response.status_code != 200:
            detail = ""
            if response_text:
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        error_obj = error_data.get("error")
                        if isinstance(error_obj, dict):
                            detail = str(error_obj.get("message") or "").strip()
                        elif isinstance(error_data.get("message"), str):
                            detail = error_data.get("message", "").strip()
                except Exception:
                    detail = response_text[:300]
            Logger.log(
                f"[SERVICE ORCHESTRA] - ERROR: Linkup returned status {response.status_code}. "
                f"Details: {detail or 'No error payload'}"
            )

            if response.status_code in {401, 403}:
                return {
                    "error": "auth_error",
                    "message": "Web search service credentials are invalid or expired.",
                }
            if response.status_code == 429:
                return {
                    "error": "rate_limited",
                    "message": "Web search is rate-limited right now. Please try again later.",
                }
            return {
                "error": "upstream_error",
                "message": "Web search failed to return a valid result. Please try again.",
            }

        if not response_text:
            Logger.log("[SERVICE ORCHESTRA] - ERROR: Linkup returned HTTP 200 with empty body.")
            return {
                "error": "empty_response",
                "message": "Web search returned an empty response. Please try again.",
            }

        try:
            data = response.json()
        except ValueError:
            Logger.log(
                "[SERVICE ORCHESTRA] - ERROR: Linkup returned non-JSON response body. "
                f"Body preview: {response_text[:300]}"
            )
            return {
                "error": "invalid_response",
                "message": "Web search returned an invalid response format. Please try again.",
            }

        answer = ServiceOrchestraClass._extract_linkup_answer(data)
        if not answer:
            return {
                "error": "no_answer",
                "message": "I couldn't extract a useful answer from web search results.",
            }

        result: Dict[str, Any] = {"answer": answer}
        if isinstance(data, dict) and isinstance(data.get("sources"), list):
            result["sources"] = data.get("sources", [])
        return result

    @staticmethod
    async def web_search(query: str) -> Optional[Dict[str, any]]:
        clean_query = str(query or "").strip()
        if not clean_query:
            return {
                "error": "empty_query",
                "message": "Please provide a search query.",
            }

        api_key = str(os.getenv("LINKUP_API_KEY") or "").strip()
        if not api_key:
            Logger.log("[SERVICE ORCHESTRA] - ERROR: LINKUP_API_KEY not found in environment.")
            return {
                "error": "missing_api_key",
                "message": "Web search is not configured right now.",
            }

        try:
            client = LinkupClient(api_key=api_key)
            response = await client.async_search(
                query=clean_query,
                depth="deep",
                output_type="sourcedAnswer",
                include_images=False,
            )

            payload = ServiceOrchestraClass._to_plain_dict(response)
            answer = ServiceOrchestraClass._extract_linkup_answer(payload or response)
            if answer:
                result: Dict[str, Any] = {"answer": answer}
                if isinstance(payload.get("sources"), list):
                    result["sources"] = payload.get("sources", [])
                return result

            Logger.log("[SERVICE ORCHESTRA] - WARNING: Linkup SDK returned no answer; trying HTTP fallback.")
        except json.JSONDecodeError as e:
            Logger.log(
                "[SERVICE ORCHESTRA] - WARNING: Linkup SDK returned non-JSON/empty payload. "
                f"Error: {str(e)}. Trying HTTP fallback."
            )
        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - WARNING: Linkup SDK search failed. {str(e)}. Trying HTTP fallback.")

        fallback_result = await ServiceOrchestraClass._direct_linkup_search(
            query=clean_query,
            api_key=api_key,
        )
        if fallback_result is not None:
            return fallback_result

        return {
            "error": "unknown_error",
            "message": "Web search failed. Please try again in a moment.",
        }

    async def extract_colors(
        self,
        file
    ) -> Optional[Dict[str, any]]:
        if not self._initialized:
            raise RuntimeError("ServiceOrchestra not initialized. Call initialize() first.")

        tmp_file_path = None

        try:
            suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await self._read_input_file_bytes(file)
                if not content:
                    raise ValueError("Empty file content")
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

            img_array = np.array(image)

            pixels = img_array[::4, ::4].reshape(-1, 3)

            if len(pixels) == 0:
                Logger.log("[SERVICE ORCHESTRA] - WARNING: No pixels found in image.")
                return None

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=15)
            kmeans.fit(pixels)

            colors = []
            for center in kmeans.cluster_centers_:
                r, g, b = int(center[0]), int(center[1]), int(center[2])
                hex_color = f"#{r:02x}{g:02x}{b:02x}"

                colors.append({
                    "r": r,
                    "g": g,
                    "b": b,
                    "hex": hex_color.upper()
                })

            image.close()

            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

            return {
                "colors": colors,
                "total_colors": len(colors)
            }

        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Color extraction failed. {str(e)}")

            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

            return None


    async def generate_floor_plan(
        self,
        file,
        furniture_list: list,
        furniture_counts: Optional[Dict[str, int]] = None,
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
                content = await self._read_input_file_bytes(file)
                if not content:
                    raise ValueError("Empty file content")
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            with open(tmp_file_path, 'rb') as img_file:
                image_bytes = img_file.read()

            mime_type = "image/png"
            if suffix.lower() in ['.jpg', '.jpeg']:
                mime_type = "image/jpeg"
            elif suffix.lower() == '.webp':
                mime_type = "image/webp"

            normalized_counts: Dict[str, int] = {}
            if isinstance(furniture_counts, dict):
                for raw_name, raw_count in furniture_counts.items():
                    name = str(raw_name).strip()
                    if not name:
                        continue
                    try:
                        count = int(raw_count)
                    except Exception:
                        count = 1
                    normalized_counts[name] = max(1, count)

            normalized_furniture: List[Dict[str, Any]] = []
            seen_names = set()
            if isinstance(furniture_list, list):
                for item in furniture_list:
                    name = str(item).strip()
                    if not name:
                        continue
                    count = normalized_counts.get(name, 1)
                    normalized_furniture.append({"name": name, "count": count})
                    seen_names.add(name)

            for name, count in normalized_counts.items():
                if name in seen_names:
                    continue
                normalized_furniture.append({"name": name, "count": count})

            furniture_lines = "\n".join(
                f"   - {entry['name']}: {entry['count']} item(s)"
                for entry in normalized_furniture
            )
            if not furniture_lines:
                furniture_lines = "   - No furniture provided"

            prompt = f"""CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. FIRST, analyze if the uploaded image is a valid floor plan or architectural drawing.
   - Valid floor plan indicators: walls, rooms, architectural symbols, top-down view, measurements
   - If NOT a valid floor plan, respond ONLY with the text: "Sorry, but no valid floor plan was detected"

2. IF it IS a valid floor plan, generate a new image that:
   - Is an exact copy of the original floor plan
   - Has simple furniture drawings placed on it to represent furniture placement
   - Each furniture item should be drawn as a recognizable symbol/icon in top-down view
   - Drawings should be appropriately sized and positioned in logical locations on the floor plan

3. Furniture to place on the floor plan:
{furniture_lines}
   - Draw exactly the requested count for each furniture type

4. Layout guidelines:
   - Place sofas along walls in living areas
   - Place dining tables in dining areas with chairs around them
   - Place beds in bedrooms
   - Place desks in study/office areas
   - Ensure furniture placement makes logical sense for room flow and usage
   - Do NOT overlap furniture items
   - Leave adequate walking space between furniture

5. Visual requirements for the furniture drawings:
   - Draw simple, recognizable top-down furniture symbols
   - Use black outlines with minimal detail
   - Examples:
     * Sofa: L-shaped or rectangular shape with cushion segments indicated
     * Chair: Simple seat with backrest indicated
     * Table: Rectangular or circular outline
     * Bed: Rectangle with pillow area indicated at one end
     * Desk: Rectangle with drawer indicators on one side
   - Scale furniture appropriately to the floor plan
   - Use standard architectural furniture symbols

6. IMPORTANT:
   - Maintain the exact same floor plan layout and dimensions
   - Keep all architectural details visible
   - Only ADD the furniture drawings, do not modify the floor plan itself
   - Generate a high-quality image output
   - Furniture should look like professional architectural floor plan symbols

Remember: If this is NOT a floor plan, output ONLY the text "Sorry, but no valid floor plan was detected" and nothing else."""

            response = self._client.models.generate_content(
                model=IMAGE_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                        image_size="2K"
                    ),
                )
            )

            response_text = None
            image_bytes_result = None

            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    response_text = part.text.strip()

                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    if image_data:
                        if isinstance(image_data, bytes):
                            image_bytes_result = image_data
                        else:
                            image_bytes_result = base64.b64decode(image_data)

            if response_text and "no valid floor plan was detected" in response_text.lower():
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

                return {
                    "error": "invalid_floor_plan",
                    "message": "Sorry, but no valid floor plan was detected"
                }

            if not image_bytes_result:
                Logger.log("[SERVICE ORCHESTRA] - WARNING: No floor plan image generated.")

                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

                return {
                    "error": "no_image",
                    "message": "Failed to generate floor plan with furniture"
                }

            filename = f"floor_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            file_obj = BytesIO(image_bytes_result)
            file_obj.seek(0)

            class MockUploadFile:
                def __init__(self, filename: str, file_obj: BytesIO):
                    self.filename = filename
                    self.file = file_obj
                    self.content_type = "image/png"

            mock_file = MockUploadFile(filename, file_obj)

            upload_result = FM.store_file(
                file=mock_file,
                subfolder="Generated Floor Plans"
            )

            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

            return {
                "floor_plan_url": upload_result["url"],
                "file_id": upload_result["file_id"],
                "filename": upload_result["filename"],
                "furniture_placed": normalized_furniture,
            }

        except Exception as e:
            Logger.log(f"[SERVICE ORCHESTRA] - ERROR: Floor plan generation failed. {str(e)}")

            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

            return None

ServiceOrchestra = ServiceOrchestraClass()
