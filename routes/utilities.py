from fastapi import APIRouter
from Services import Utilities

router = APIRouter(prefix="/utilities", tags=["Tools & Services"])

@router.post("/generate-random-id")
def generate_id(length: int = 32):
    random_id = Utilities.GenerateRandomID(length)
    return {"random_id": random_id}

@router.post("/generate-random-int")
def generate_int(min_value: int = 0, max_value: int = 100):
    random_int = Utilities.GenerateRandomInt(min_value, max_value)
    return {"random_int": random_int}

@router.post("/hash-string")
def hash_string(input_string: str):
    hashed = Utilities.HashString(input_string)
    return {"hashed_string": hashed}

@router.post("/encode-base64")
def encode_base64(input_string: str):
    encoded = Utilities.EncodeToBase64(input_string)
    return {"encoded_string": encoded}

@router.post("/decode-base64")
def decode_base64(encoded_string: str):
    decoded = Utilities.DecodeFromBase64(encoded_string)
    return {"decoded_string": decoded}