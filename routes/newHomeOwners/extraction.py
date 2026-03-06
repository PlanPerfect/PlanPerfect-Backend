from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Depends, Form
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from middleware.auth import _verify_api_key
from concurrent.futures import ThreadPoolExecutor
import cv2
import re
import tempfile
import os
import base64
import json
import traceback
import asyncio
import httpx
from collections import Counter

from typing import Dict, List, Optional
from pydantic import BaseModel
from groq import Groq
from Services import DatabaseManager as DM
from Services import FileManager as FM
from Services import Logger
from Services import ServiceOrchestra as SO

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_alt_client = Groq(api_key=os.getenv("GROQ_ALT_API_KEY"))

router = APIRouter(prefix="/newHomeOwners/extraction", tags=["New Home Owners AI Extraction"], dependencies=[Depends(_verify_api_key)])

executor = ThreadPoolExecutor(max_workers=8)
ocr_executor = ThreadPoolExecutor(max_workers=1)

MAX_OCR_UPLOAD_BYTES = int(os.getenv("MAX_OCR_UPLOAD_BYTES", str(8 * 1024 * 1024)))
OCR_TIMEOUT_SECONDS = float(os.getenv("OCR_TIMEOUT_SECONDS", "60"))

# Unicode cleaning
UNICODE_REPLACEMENTS = str.maketrans({
    "\u202f": " ",
    "\u00a0": " ",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2011": "-",
    "\u2012": "-",
    "\u00d7": "x",
    "\u25a0": "",
    "\u00ab": '"',
    "\u00bb": '"',
})

def _clean_text(text: str) -> str:
    return text.translate(UNICODE_REPLACEMENTS)

# JSON extraction & repair
def _extract_json_str(raw: str) -> str:
    raw = raw.strip()
    if "```json" in raw:
        return raw.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()
    return raw

def _attempt_repair(bad_json: str) -> dict:
    attempts = [bad_json]

    # Fix unquoted string values
    repaired = re.sub(
        r':\s*([A-Za-z][^",\}\]\n]{0,120})"',
        lambda m: ': "' + m.group(1).replace('"', "'") + '"',
        bad_json,
    )
    attempts.append(repaired)

    # Remove trailing commas before } or ]
    no_trailing = re.sub(r',\s*([}\]])', r'\1', bad_json)
    attempts.append(no_trailing)

    # Both repairs combined
    both = re.sub(r',\s*([}\]])', r'\1', repaired)
    attempts.append(both)

    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

def _parse_llm_response(response_text: str) -> dict:
    cleaned = _clean_text(response_text)
    json_str = _extract_json_str(cleaned)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        Logger.log("[EXISTING DOCUMENT LLM] - ERROR: Initial JSON parse failed, attempting repair...")
        return _attempt_repair(json_str)


def _call_groq(system_prompt: str, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    params = dict(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=7000,
        response_format={"type": "json_object"},
    )

    # Primary key attempt
    try:
        completion = groq_client.chat.completions.create(**params)
        return _parse_llm_response(completion.choices[0].message.content)
    except Exception as primary_err:
        Logger.log(f"[DOCUMENT LLM] - WARNING: Primary Groq key failed ({primary_err}), retrying with alt key...")

    # Alt key attempt
    try:
        completion = groq_alt_client.chat.completions.create(**params)
        return _parse_llm_response(completion.choices[0].message.content)
    except Exception as groq_err:
        err_str = str(groq_err)
        failed_gen = None
        try:
            match = re.search(r"'failed_generation':\s*'(.*?)'(?:\s*\})", err_str, re.DOTALL)
            if match:
                failed_gen = match.group(1).encode("utf-8").decode("unicode_escape")
            else:
                body_match = re.search(r"\{.*\}", err_str, re.DOTALL)
                if body_match:
                    body = json.loads(body_match.group(0).replace("'", '"'))
                    failed_gen = body.get("error", {}).get("failed_generation")
        except Exception:
            pass

        if failed_gen:
            Logger.log("[DOCUMENT LLM] - ERROR: Groq json_validate_failed - attempting to parse failed_generation...")
            try:
                return _parse_llm_response(failed_gen)
            except Exception as repair_err:
                Logger.log(f"[DOCUMENT LLM] - ERROR: Repair also failed: {repair_err}")

        raise groq_err

def run_room_segmentation(file_path: str):
    client = Client(
        "https://tallmanager267-sg-room-segmentation.hf.space/",
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
        pil_img=handle_file(file_path),
        api_name="/predict"
    )

# Lazy load OCR model only when needed
_ocr_model = None

# Function to get or load the OCR model (Doctr)
def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        from doctr.models import ocr_predictor
        _ocr_model = ocr_predictor(pretrained=True)
    return _ocr_model

# New endpoint for saving user input
@router.post("/saveUserInput")
async def save_user_input(
    floor_plan: UploadFile = File(...),
    segmented_floor_plan: UploadFile = File(None),
    preferences: str = Form(...),
    budget: str = Form(None),
    unit_info: str = Form(None),
    furniture_selections: str = Form(None),
    user_id: str = Form(...),
):
    """
    Saves new home owner user inputs to Firebase RTDB and Cloud Storage.
    Stores floor plan and segmented floor plan images, along with preferences, budget, and unit information.
    """
    try:
        if not user_id or not user_id.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "result": "ERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "result": "ERROR: Please login again."
                }
            )

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Parse preferences JSON
        try:
            preferences_data = json.loads(preferences) if isinstance(preferences, str) else preferences

            # Handle if preferences is a list or dict
            if isinstance(preferences_data, list):
                styles_list = preferences_data
            elif isinstance(preferences_data, dict):
                styles_list = preferences_data.get("styles", [])
            else:
                styles_list = []

        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={ "error": str(e) })

        # Parse unit info JSON if provided
        unit_info_dict = None
        if unit_info:
            try:
                unit_info_dict = json.loads(unit_info) if isinstance(unit_info, str) else unit_info
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={ "error": str(e) })

        # Upload floor plan
        original_floor_plan_name = floor_plan.filename
        name, ext = os.path.splitext(original_floor_plan_name)
        floor_plan.filename = f"{name}_{timestamp}{ext}"

        floor_plan_result = FM.store_file(
            file=floor_plan,
            subfolder=f"newHomeOwners/{user_id}"
        )
        floor_plan_url = floor_plan_result["url"]
        floor_plan_file_id = floor_plan_result["file_id"]

        # Upload segmented floor plan if provided
        segmented_floor_plan_url = None
        segmented_floor_plan_file_id = None

        if segmented_floor_plan:
            original_segmented_name = segmented_floor_plan.filename
            seg_name, seg_ext = os.path.splitext(original_segmented_name)
            segmented_floor_plan.filename = f"{seg_name}_{timestamp}{seg_ext}"

            segmented_result = FM.store_file(
                file=segmented_floor_plan,
                subfolder=f"newHomeOwners/{user_id}"
            )
            segmented_floor_plan_url = segmented_result["url"]
            segmented_floor_plan_file_id = segmented_result["file_id"]

        # Set flow to "newHomeOwner"
        DM.data["Users"][user_id]["flow"] = "newHomeOwner"

        # Set Preferences
        DM.data["Users"][user_id]["New Home Owner"]["Preferences"]["Preferred Styles"]["styles"] = styles_list
        DM.data["Users"][user_id]["New Home Owner"]["Preferences"]["budget"] = budget if budget else None

        # Set Uploaded Floor Plan
        DM.data["Users"][user_id]["New Home Owner"]["Uploaded Floor Plan"]["url"] = floor_plan_url

        # Set Segmented Floor Plan
        DM.data["Users"][user_id]["New Home Owner"]["Segmented Floor Plan"]["url"] = (
            segmented_floor_plan_url if segmented_floor_plan_url else None
        )

        # Set Unit Information
        if unit_info_dict:
            # Set unit rooms (e.g., "2-BEDROOM")
            unit_value = unit_info_dict.get("unit_rooms") if unit_info_dict.get("unit_rooms") else None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unit"] = unit_value

            # Set unit types
            unit_types = unit_info_dict.get("unit_types", [])
            if isinstance(unit_types, list):
                DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitType"] = unit_types
            elif isinstance(unit_types, str):
                DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitType"] = [
                    t.strip() for t in unit_types.split(",") if t.strip()
                ]
            else:
                DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitType"] = None

            # Set unit sizes
            unit_sizes = unit_info_dict.get("unit_sizes", [])
            if isinstance(unit_sizes, list):
                DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitSize"] = unit_sizes
            elif isinstance(unit_sizes, str):
                DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitSize"] = [
                    s.strip() for s in unit_sizes.split(",") if s.strip()
                ]
            else:
                DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitSize"] = None

            # Set room counts
            room_counts = unit_info_dict.get("room_counts", {})
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["balcony"] = room_counts.get("BALCONY", 0)
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["bathroom"] = room_counts.get("BATH", 0)
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["bedroom"] = room_counts.get("BEDROOM", 0)
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["kitchen"] = room_counts.get("KITCHEN", 0)
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["ledge"] = room_counts.get("LEDGE", 0)
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["livingRoom"] = room_counts.get("LIVING", 0)
        else:
            # Set default None values if no unit info
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unit"] = None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitType"] = None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["unitSize"] = None

            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["balcony"] = None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["bathroom"] = None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["bedroom"] = None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["kitchen"] = None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["ledge"] = None
            DM.data["Users"][user_id]["New Home Owner"]["Unit Information"]["Number Of Rooms"]["livingRoom"] = None

        # Save furniture selections if provided
        if furniture_selections:
            try:
                DM.data["Users"][user_id]["New Home Owner"]["Furniture Selections"] = json.loads(furniture_selections)
            except json.JSONDecodeError:
                pass

        # Save to Firebase RTDB
        DM.save()

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "message": "User input saved successfully",
                    "floor_plan_url": floor_plan_url,
                    "floor_plan_file_id": floor_plan_file_id,
                    "segmented_floor_plan_url": segmented_floor_plan_url,
                    "segmented_floor_plan_file_id": segmented_floor_plan_file_id,
                    "user_id": user_id
                }
            }
        )

    except Exception as e:
        error_details = traceback.format_exc()
        Logger.log(f"[EXTRACTION] - ERROR: Error saving user input: {error_details}")

        return JSONResponse(status_code=500, content={ "error": str(e) })

# Endpoint for room segmentation
@router.post("/roomSegmentation")
async def room_segmentation(file: UploadFile = File(...)):
    """
    Performs room segmentation on uploaded floor plan image using Hugging Face API.
    """
    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Perform room segmentation via thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            run_room_segmentation,
            tmp_file_path
        )

        #handle result
        if isinstance(result, str):
            segmented_path = result
        elif isinstance(result, dict):
            segmented_path = result.get("path") or result.get("url")
        else:
            raise ValueError("Unexpected response from segmentation API")

        if not segmented_path or not os.path.exists(segmented_path):
            raise ValueError(f"Segmented image not found at {segmented_path}")

        # Convert to base64 for frontend
        with open(segmented_path, "rb") as f:
            img_bytes = f.read()
        data_url = f"data:image/webp;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "segmented_image": data_url,
                    "message": "Room segmentation completed successfully"
                }
            }
        )

    except asyncio.TimeoutError:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(
            status_code=504,
            content={
                "error": "ERROR: Service timeout. Please try again with a smaller image."
            }
        )

    except Exception as e:
        # Clean up temporary file if it exists before returning error
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(status_code=500, content={ "error": str(e) })

# Endpoint for unit information extraction
@router.post("/unitInformationExtraction")
async def unit_information_extraction(file: UploadFile = File(...)):
    """
    Extracts unit information from floor plan using OCR.
    """
    tmp_file_path = None
    try:
        # Read and validate upload before OCR to avoid proxy-level failures on large payloads.
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={ "error": "ERROR: Uploaded file is empty." })

        if len(content) > MAX_OCR_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error": f"ERROR: Uploaded file too large. Max size is {MAX_OCR_UPLOAD_BYTES // (1024 * 1024)}MB."
                }
            )

        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Process OCR in a dedicated small worker pool to reduce memory spikes.
        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(ocr_executor, process_floorplan_image, tmp_file_path),
            timeout=OCR_TIMEOUT_SECONDS
        )

        # Clean up temporary file
        os.unlink(tmp_file_path)

        # Format the response
        response_data = {
            "unit_rooms": results["unit_rooms"],
            "unit_types": results["unit_types"],
            "unit_sizes": results["unit_sizes"],
            "room_counts": dict(results["room_counts"]),
            "detected_lines": results.get("detected_lines", [])
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": response_data
            }
        )

    except asyncio.TimeoutError:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(
            status_code=504,
            content={ "error": "ERROR: OCR timeout. Please try a smaller or clearer floor plan image." }
        )

    except Exception as e:
        # Clean up temporary file if it exists before returning error
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        return JSONResponse(status_code=500, content={ "error": str(e) })


def process_floorplan_image(img_path):
    """
    Process a floor plan image:
    - OCR text detection
    - Extract unit info (rooms, type, sizes)
    - Count room occurrences (filtered)
    """
    # Read image
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise ValueError(f"Image not found: {img_path}")

    # Downscale large images to keep OCR latency and memory bounded on Render.
    max_dim = 1800
    h, w = image_bgr.shape[:2]
    largest_side = max(h, w)
    if largest_side > max_dim:
        scale = max_dim / float(largest_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Conver BGR image to RGB image for OCR
    image_rgb = image_bgr[:, :, ::-1]

    # Get OCR model and perform OCR
    ocr_model = get_ocr_model()
    result = ocr_model([image_rgb])

    # Collect detected lines from OCR result
    detected_lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([w.value for w in line.words])
                detected_lines.append(line_text.strip())

    normalized_lines = []
    i = 0
    while i < len(detected_lines):
        if detected_lines[i].strip().lower() == 'type' and i + 1 < len(detected_lines):
            normalized_lines.append(f"Type {detected_lines[i+1].strip()}")
            i += 2
        else:
            normalized_lines.append(detected_lines[i])
            i += 1
    detected_lines = normalized_lines

    # Extract unit info
    unit_rooms = "" # e.g., "3-BEDROOM"
    unit_types = [] # e.g., ["(TYPE A)"]
    unit_sizes = [] # e.g., ["1200 sq ft"]

    # Define patterns for key information
    rooms_pattern = re.compile(r'^\d+\s*-?\s*(BEDROOM|ROOM)\b', re.IGNORECASE)
    reject_room_pattern_words = re.compile(r'\s+FLOOR PLAN$', re.IGNORECASE)
    unit_type_pattern = re.compile(
        r'\bTYPE\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\([A-Za-z0-9]+\)|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)',
        re.IGNORECASE
    )
    unit_size_pattern = re.compile(
        r'(\d+)\s*sq[.\s]?m\s*[/\(]\s*([\d,]+)\s*sq[.\s]?ft',
        re.IGNORECASE
    )

    # Ignore lines if they include these words
    ignore_patterns = [
        re.compile(r'.*includes.*', re.IGNORECASE),
        re.compile(r'.*ceiling.*', re.IGNORECASE),
        re.compile(r'.*strata void.*', re.IGNORECASE),
        re.compile(r'.*WITHSTRATA VOIDI.*', re.IGNORECASE),
        re.compile(r'strata', re.IGNORECASE),
        re.compile(r'VOIDOF', re.IGNORECASE),
        re.compile(r'.*LIVING, DINING.*DRY KITCHEN.*', re.IGNORECASE)
    ]

    # Extract unit information from lines
    skip_next = False
    for line in detected_lines:
        # If previous line was a strata/void/includes line, skip this one too
        if skip_next:
            skip_next = False
            continue

        # Skip ignored lines and flag the next line to skip as well
        if any(pat.search(line) for pat in ignore_patterns):
            skip_next = True
            continue

        if rooms_pattern.match(line):
            cleaned_line = reject_room_pattern_words.sub('', line).strip()
            unit_rooms = cleaned_line

        match_type = unit_type_pattern.search(line)
        if match_type:
            unit_types.append(match_type.group(1))

        match_size = unit_size_pattern.search(line)
        if match_size:
            sqm = match_size.group(1)
            sqft = match_size.group(2).replace(',', '')
            unit_sizes.append(f"{sqm} sqm / {sqft} sqft")

    # Count room names
    room_keywords = ["WC", "BATH", "BALCONY", "BEDROOM", "KITCHEN", "LIVING", "LEDGE"]
    room_counter = Counter()

    for line in detected_lines:
        line_upper = line.upper()

        # Skip unit info and ignoring descriptive lines
        if rooms_pattern.match(line) or unit_type_pattern.match(line) or unit_size_pattern.match(line):
            continue
        if any(pat.match(line) for pat in ignore_patterns):
            continue

        for keyword in room_keywords:
            if keyword in line_upper:
                room_counter[keyword] += 1

    joined_text = " ".join(detected_lines)
    if not unit_sizes:
        for pat in [
            re.compile(r'(\d+)\s*sq\s*[.\s]?m\s*[/\(]\s*([\d,]+)\s*sq\s*[.\s]?ft', re.IGNORECASE),
            re.compile(r'(\d+)\s*sqm\s*\(([\d,]+)\s*sq\s*ft\)', re.IGNORECASE),
        ]:
            m = pat.search(joined_text)
            if m:
                sqft = m.group(2).replace(',', '')
                unit_sizes.append(f"{m.group(1)} sqm / {sqft} sqft")
                break
    seen = []
    for type in unit_types:
        if type not in seen:
            seen.append(type)
    unit_types = seen

    if not unit_rooms:
        unit_rooms = "N/A"

    if not unit_types:
        unit_types = ["N/A"]

    if not unit_sizes:
        unit_sizes = ["N/A"]

    return {
        "img_path": img_path,
        "unit_rooms": unit_rooms,
        "unit_types": unit_types,
        "unit_sizes": unit_sizes,
        "room_counts": room_counter,
        "detected_lines": detected_lines,
        "raw_result": result
    }

def _get_furniture_quotation(budget: str, unit_info: dict, furniture_list: list, furniture_counts: dict) -> dict:
    unit = unit_info.get("unit", "N/A") if unit_info else "N/A"
    unit_size = unit_info.get("unitSize", ["N/A"])[0] if unit_info and unit_info.get("unitSize") else "N/A"
    furniture_summary = ", ".join([f"{v}x {k}" for k, v in (furniture_counts or {}).items()]) or "Not specified"

    system_prompt = (
        "You are an interior design cost estimator specialising in Singapore (2026 market). "
        "You MUST respond with ONLY a valid JSON object, no other text. "
        "Use ONLY plain ASCII characters - no special dashes, no curly quotes. "
        "The JSON object must contain exactly these keys: "
        "quotation_range, recommended_quote, quote_basis."
    )

    user_prompt = (
        f"Estimate the furniture cost for this unit:\n"
        f"- Unit type: {unit}\n"
        f"- Unit size: {unit_size}\n"
        f"- Client stated budget: {budget or 'Not specified'}\n"
        f"- Furniture selected: {furniture_summary}\n\n"
        "Singapore furniture cost reference (2026):\n"
        "- Basic (HDB): S$8,000 - S$15,000\n"
        "- Mid-range (HDB/Condo): S$15,000 - S$35,000\n"
        "- Premium (Condo/Landed): S$35,000 - S$80,000+\n"
        "- Bed frame: S$500-S$3,000 | Sofa: S$800-S$5,000 | Dining set: S$600-S$3,000\n\n"
        "Respond with ONLY this JSON structure:\n"
        '{"quotation_range": "S$XX,000 - S$XX,000", "recommended_quote": "S$XX,000", "quote_basis": "Brief explanation."}'
    )

    return _call_groq(system_prompt, user_prompt)

class FurnitureFloorPlanRequest(BaseModel):
    furniture_list: List[str]
    furniture_counts: Optional[Dict[str, int]] = None

@router.post("/generateFurnitureFloorPlan/{user_id}")
async def generate_furniture_floor_plan(
    user_id: str,
    body: FurnitureFloorPlanRequest,
):
    """
    Generates a new floor plan image with the user's selected furniture placed on it.
    Fetches the user's original uploaded floor plan from the database, then calls the AI
    service to overlay the chosen furniture symbols.

    The AI will replace existing furniture if the floor plan already contains any,
    or add new furniture symbols to an empty floor plan.
    """
    try:
        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "result": "UERROR: No user found. Please login again."
                }
            )

        floor_plan_url = DM.peek(["Users", user_id, "New Home Owner", "Uploaded Floor Plan", "url"])
        if not floor_plan_url:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "result": "UERROR: No floor plan image found. Please upload a floor plan first."
                }
            )

        async with httpx.AsyncClient(timeout=60.0) as client:
            img_response = await client.get(floor_plan_url)
            img_response.raise_for_status()
            image_bytes = img_response.content

        filename = floor_plan_url.split("/")[-1].split("?")[0] or "floor_plan.png"

        result = await SO.generate_furniture_floor_plan(
            image_bytes=image_bytes,
            filename=filename,
            furniture_list=body.furniture_list,
            furniture_counts=body.furniture_counts or {},
        )

        if result is None:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "result": "ERROR: Floor plan generation failed. Please try again."
                }
            )

        if result.get("error") == "invalid_floor_plan":
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "result": f"UERROR: {result['message']}"
                }
            )

        if result.get("error") == "no_image":
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "result": "ERROR: AI did not return a floor plan image. Please try again."
                }
            )

        DM.data["Users"][user_id]["New Home Owner"]["Furniture Floor Plan"] = {
            "url": result["floor_plan_url"],
            "furniture_list": body.furniture_list,
            "furniture_counts": body.furniture_counts or {},
        }
        DM.save()

        quotation_range = None
        try:
            unit_info = DM.peek(["Users", user_id, "New Home Owner", "Unit Information"])
            budget = DM.peek(["Users", user_id, "New Home Owner", "Preferences", "budget"])
            quotation_data = _get_furniture_quotation(
                budget=budget,
                unit_info=unit_info,
                furniture_list=body.furniture_list,
                furniture_counts=body.furniture_counts or {},
            )
            quotation_range = quotation_data.get("quotation_range")
            DM.data["Users"][user_id]["New Home Owner"]["Quotation"] = quotation_data
            DM.save()
        except Exception as e:
            Logger.log(f"[GENERATE FURNITURE FLOOR PLAN] - WARNING: Quotation generation failed: {str(e)}")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "floor_plan_url": result["floor_plan_url"],
                    "furniture_placed": result.get("furniture_placed", []),
                    "quotation_range": quotation_range,
                }
            }
        )

    except Exception as e:
        Logger.log(f"[GENERATE FURNITURE FLOOR PLAN] - ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": f"ERROR: {str(e)}"
            }
        )
