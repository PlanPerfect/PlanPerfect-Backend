from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from middleware.auth import _verify_api_key
import httpx
import os
from typing import Optional
import random
from urllib.parse import unquote
from Services import DatabaseManager as DM

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

class SaveRecommendationRequest(BaseModel):
    name: str
    image: str
    description: str
    match: int

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

@router.post("/save-recommendation")
async def save_recommendation(
    request: SaveRecommendationRequest,
    x_user_id: str = Header(..., alias="X-User-ID")
):
    try:
        if not x_user_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", x_user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        rec_id = request.image

        existing = DM.peek(["Users", x_user_id, "Saved Reccomendations", rec_id])
        if existing:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "Recommendation already saved",
                    "recommendation_id": rec_id
                }
            )

        recommendation_data = {
            "name": request.name,
            "image": request.image,
            "description": request.description,
            "match": request.match,
        }

        success = DM.set_value(
            ["Users", x_user_id, "Saved Reccomendations", rec_id],
            recommendation_data
        )

        if not success:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to save recommendation to database"}
            )

        DM.save()

        return JSONResponse(
            status_code=200,
            content={
                "message": "Recommendation saved successfully",
                "recommendation_id": rec_id
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error saving recommendation: {str(e)}"}
        )

@router.delete("/delete-recommendation/{rec_id:path}")
async def delete_recommendation(
    rec_id: str,
    x_user_id: str = Header(..., alias="X-User-ID")
):
    try:
        if not x_user_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", x_user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        decoded_rec_id = unquote(rec_id)

        success = DM.destroy(
            ["Users", x_user_id, "Saved Reccomendations", decoded_rec_id]
        )

        if not success:
            return JSONResponse(
                status_code=404,
                content={"error": "Recommendation not found or already deleted"}
            )

        DM.save()

        return JSONResponse(
            status_code=200,
            content={"message": "Recommendation deleted successfully"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error deleting recommendation: {str(e)}"}
        )

@router.get("/get-saved-recommendations")
async def get_saved_recommendations(
    x_user_id: str = Header(..., alias="X-User-ID")
):
    try:
        saved_recs = DM.peek(["Users", x_user_id, "Saved Reccomendations"])

        if not saved_recs:
            return JSONResponse(
                status_code=200,
                content={"recommendations": []}
            )

        recommendations = [
            {"id": rec_id, **rec_data}
            for rec_id, rec_data in saved_recs.items()
        ]

        return JSONResponse(
            status_code=200,
            content={"recommendations": recommendations}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching saved recommendations: {str(e)}"}
        )