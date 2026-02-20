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

from Services import DatabaseManager as DM
from Services import FileManager as FM
from Services import Logger

router = APIRouter(prefix="/newHomeOwners/extraction", tags=["New Home Owners AI Extraction"], dependencies=[Depends(_verify_api_key)])

executor = ThreadPoolExecutor(max_workers=8)

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
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Process the floor plan
        results = process_floorplan_image(tmp_file_path)

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

    # Extract unit info
    unit_rooms = "" # e.g., "3-BEDROOM"
    unit_types = [] # e.g., ["(TYPE A)"]
    unit_sizes = [] # e.g., ["1200 sq ft"]

    # Define patterns for key information
    rooms_pattern = re.compile(r'^\d+\-(BEDROOM|ROOM)\b', re.IGNORECASE)
    reject_room_pattern_words = re.compile(r'\s+FLOOR PLAN$', re.IGNORECASE)
    unit_type_pattern = re.compile(r'\bType\s*[:\-]?\s*('r'\([A-Za-z0-9]+\)'r'|'r'[A-Za-z0-9]+\([A-Za-z0-9]+\)'r'|'r'[A-Za-z0-9]+'r')\b',re.IGNORECASE)
    unit_size_pattern = re.compile(r'\b(\d+)\s*sq\s*m\s*/\s*(\d+)\s*sq\s*ft\b',re.IGNORECASE)

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
    for line in detected_lines:
        # Skip ignored lines
        if any(pat.search(line) for pat in ignore_patterns):
            continue

        if rooms_pattern.match(line):
            cleaned_line = reject_room_pattern_words.sub('', line).strip()
            unit_rooms = cleaned_line

        match_type = unit_type_pattern.search(line)
        if match_type:
            raw = match_type.group(0)

            # Remove "Type" prefix
            cleaned = re.sub(r'Type\s*', '', raw, flags=re.IGNORECASE).strip()

            # Case 1 — (B)
            bracket_only = re.fullmatch(r'\(([A-Za-z0-9]+)\)', cleaned)
            if bracket_only:
                unit_types.append(bracket_only.group(1))

            # Case 2 — B2 or B2(D)
            else:
                unit_types.append(cleaned)

        match_size = unit_size_pattern.search(line)
        if match_size:
            cleaned = re.sub(r'\s+', ' ', match_size.group(0)).strip()
            unit_sizes.append(cleaned.lower())

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