from fastapi import APIRouter, Depends, HTTPException
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

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

def truncate_description(text: str, max_length: int = 135) -> str:
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

@router.post("/get-recommendations")
async def get_recommendations(request: RecommendationRequest):
    try:
        search_query = f"{request.style}-style {request.furniture_name}"

        params = {
            "query": search_query,
            "per_page": request.per_page,
            "orientation": "landscape"
        }

        if request.page:
            params["page"] = request.page

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.unsplash.com/search/photos",
                params=params,
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                timeout=10.0
            )

            if response.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )

            if response.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="Rate limit exceeded. Please try again later."
                )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch recommendations from Unsplash"
                )

            data = response.json()

            recommendations = []
            for result in data.get("results", []):
                description = (
                    result.get("description") or
                    result.get("alt_description") or
                    f"A beautiful {request.style} style {request.furniture_name}"
                )

                recommendations.append({
                    "unsplashId": result["id"],
                    "name": f"{request.style}-themed {request.furniture_name}",
                    "image": result["urls"]["regular"],
                    "description": truncate_description(description),
                    "match": random.randint(85, 99)
                })

            return JSONResponse(content={
                "success": True,
                "recommendations": recommendations
            })

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Unsplash timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")