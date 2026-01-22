from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from middleware.auth import _verify_api_key
import httpx
import os
from typing import Optional
import random

router = APIRouter(
    prefix="/stylematch/recommendations",
    tags=["Get Recommendations"],
    dependencies=[Depends(_verify_api_key)]
)

class RecommendationRequest(BaseModel):
    style: str
    furniture_name: str
    per_page: int = 5
    page: Optional[int] = None

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

def truncate_description(text: str, max_length: int = 135) -> str:
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

@router.post("/get-recommendations")
async def get_recommendations(request: RecommendationRequest):
    try:
        search_query = f"{request.style} styled {request.furniture_name}"

        params = {
            "query": search_query,
            "per_page": request.per_page,
            "orientation": "landscape"
        }

        if request.page:
            params["page"] = request.page

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pexels.com/v1/search",
                params=params,
                headers={"Authorization": PEXELS_API_KEY},
                timeout=10.0
            )

            if response.status_code == 429:
                return JSONResponse(
                    status_code=429,
                    content={ "error": "UERROR: Too many requests. Please try again later." }
                )

            if response.status_code == 403:
                return JSONResponse(
                    status_code=403,
                    content={ "error": "UERROR: Too many requests. Please try again later." }
                )

            data = response.json()

            recommendations = []
            for photo in data.get("photos", []):
                description = (
                    photo.get("alt") or
                    f"A beautiful {request.style} styled {request.furniture_name}"
                )

                recommendations.append({
                    "name": f"{request.style}-themed {request.furniture_name}",
                    "image": photo["src"]["large"],
                    "description": truncate_description(description),
                    "match": random.randint(85, 99),
                })

            return JSONResponse(status_code=200, content={"recommendations": recommendations})

    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={"error": "ERROR: Service timeout. Please try again."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})