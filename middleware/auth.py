from fastapi import Header, HTTPException, status, Depends
from dotenv import load_dotenv
from functools import wraps
import os

load_dotenv()

async def _verify_api_key(api_key: str = Header(None, alias="API_KEY")):
    expected_api_key = os.getenv("API_KEY")

    if not api_key or api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ERROR: Unauthorized - Invalid / Missing API Key",
        )

    return api_key

def checkHeaders(func):
    @wraps(func)
    async def wrapper(*args, _api_key: str = Depends(_verify_api_key), **kwargs):
        return await func(*args, **kwargs)

    return wrapper