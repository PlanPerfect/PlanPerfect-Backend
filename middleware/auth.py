from fastapi import Header, HTTPException, status, Depends
from dotenv import load_dotenv
import os
import inspect

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
    if inspect.iscoroutinefunction(func):
        async def wrapper(*args, _api_key: str = Depends(_verify_api_key), **kwargs):
            return await func(*args, **kwargs)
    else:
        def wrapper(*args, _api_key: str = Depends(_verify_api_key), **kwargs):
            return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    return wrapper