from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
import cv2
import numpy as np
import re
import tempfile
import os
from PIL import Image
from collections import Counter

router = APIRouter(prefix="/newHomeOwners/extraction", tags=["New Home Owners AI Extraction"])

# Lazy load OCR model only when needed
_ocr_model = None

def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        from doctr.models import ocr_predictor
        _ocr_model = ocr_predictor(pretrained=True)
    return _ocr_model

@router.post("/roomSegmentation")
async def room_segmentation(file: UploadFile = File(...)):
    """
    Performs room segmentation on uploaded floor plan image using Hugging Face API.
    
    Args:
        file: Uploaded floor plan image file
        
    Returns:
        JSON response with segmented image URL and metadata
    """
    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Connect to Hugging Face API
        client = Client("https://tallmanager267-sg-room-segmentation.hf.space/")
        
        # Perform room segmentation
        result = client.predict(
            pil_img=handle_file(tmp_file_path),
            api_name="/predict"
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": {
                    "segmented_image": result,
                    "message": "Room segmentation completed successfully"
                }
            }
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": f"ERROR: Failed to perform room segmentation. Error: {str(e)}"
            }
        )

@router.post("/unitInformationExtraction")
async def unit_information_extraction(file: UploadFile = File(...)):
    """
    Extracts unit information from floor plan using OCR.
    
    Args:
        file: Uploaded floor plan image file
        
    Returns:
        JSON response with extracted unit info (rooms, type, sizes, room counts)
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
        # Clean up temporary file if it exists
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": f"ERROR: Failed to extract unit information. Error: {str(e)}"
            }
        )


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
    
    image_rgb = image_bgr[:, :, ::-1]  # BGR -> RGB

    # Get OCR model and perform OCR
    ocr_model = get_ocr_model()
    result = ocr_model([image_rgb])

    # Collect detected lines
    detected_lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([w.value for w in line.words])
                detected_lines.append(line_text.strip())

    # Extract unit info
    unit_rooms = None
    unit_types = []
    unit_sizes = []

    # Define patterns
    rooms_pattern = re.compile(r'^\d+\-(BEDROOM|ROOM)\b', re.IGNORECASE)
    reject_room_pattern_words = re.compile(r'\s+FLOOR PLAN$', re.IGNORECASE)
    unit_type_pattern = re.compile(r'\(TYPE\s+\d+\)|Type\s+\w+', re.IGNORECASE)
    unit_size_pattern = re.compile(r'(\d+)\s*(sq\s*m|sq\s*ft)', re.IGNORECASE)

    # Descriptive lines to ignore in room counts
    ignore_patterns = [
        re.compile(r'.*includes.*', re.IGNORECASE),
        re.compile(r'.*ceiling.*', re.IGNORECASE),
        re.compile(r'.*strata void.*', re.IGNORECASE),
        re.compile(r'.*WITHSTRATA VOIDI.*', re.IGNORECASE),
        re.compile(r'.*LIVING, DINING.*DRY KITCHEN.*', re.IGNORECASE)
    ]

    # Extract unit information from lines
    for line in detected_lines:
        if rooms_pattern.match(line):
            cleaned_line = reject_room_pattern_words.sub('', line).strip()
            unit_rooms = cleaned_line
            
        match_type = unit_type_pattern.search(line)
        if match_type:
            unit_types.append(match_type.group(0)) 
            
        match_size = unit_size_pattern.search(line)
        if match_size:
            num, unit = match_size.groups()
            unit_sizes.append(f"{num} {unit}")
            
    # Count room names
    room_keywords = ["WC", "BATH", "BALCONY", "BEDROOM", "KITCHEN", "LIVING", "LEDGE"]
    room_counter = Counter()

    for line in detected_lines:
        line_upper = line.upper()

        # Skip unit info and descriptive lines
        if rooms_pattern.match(line) or unit_type_pattern.match(line) or unit_size_pattern.match(line):
            continue
        if any(pat.match(line) for pat in ignore_patterns):
            continue

        for keyword in room_keywords:
            if keyword in line_upper:
                room_counter[keyword] += 1

    # Return results
    return {
        "img_path": img_path,
        "unit_rooms": unit_rooms,
        "unit_types": unit_types,
        "unit_sizes": unit_sizes,
        "room_counts": room_counter,
        "detected_lines": detected_lines,
        "raw_result": result
    }