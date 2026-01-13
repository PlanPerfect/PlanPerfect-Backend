from fastapi import Header, HTTPException, status
from dotenv import load_dotenv
from functools import wraps
import os

load_dotenv()

async def _verify_api_key(vite_api_key: str = Header(None, alias="vite_api_key")):
    expected_api_key = os.getenv("API_KEY")

    if not vite_api_key or vite_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ERROR: Unauthorized - Invalid / Missing API Key",
        )

    return vite_api_key