from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
from middleware.auth import _verify_api_key
from Services import DatabaseManager as DM

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    dependencies=[Depends(_verify_api_key)]
)

class UserData(BaseModel):
    uid: str
    email: str
    displayName: str

@router.post("/register")
async def register_user(user_data: UserData):
    try:
        existing_user = DM.peek(["Users", user_data.uid])

        if existing_user is None:
            default_agent = {
                "session_id": str(uuid.uuid4()),
                "status": "idle",
                "current_step": "",
                "steps": [],
                "Outputs": {
                    "Generated Images": [],
                    "Generated Floor Plans": [],
                    "Classified Style": [],
                    "Detected Furniture": [],
                    "Reccomendations": [],
                    "Web Searches": [],
                    "Extracted Colors": []
                }
            }

            user_info = {
                "uid": user_data.uid,
                "email": user_data.email,
                "displayName": user_data.displayName,
                "Agent": default_agent
            }

            DM.data["Users"][user_data.uid] = user_info

            DM.save()

            return JSONResponse(
                status_code=200,
                content={ "response": "SUCCESS: User registered successfully" }
            )

        else:
            existing_agent = existing_user.get("Agent") if isinstance(existing_user, dict) else None
            if not isinstance(existing_agent, dict):
                DM.data["Users"][user_data.uid]["Agent"] = {
                    "session_id": str(uuid.uuid4()),
                    "status": "idle",
                    "current_step": "",
                    "steps": [],
                    "Outputs": {
                        "Generated Images": [],
                        "Generated Floor Plans": [],
                        "Classified Style": [],
                        "Detected Furniture": [],
                        "Reccomendations": [],
                        "Web Searches": [],
                        "Extracted Colors": []
                    }
                }
                DM.save()

            return JSONResponse(
                status_code=200,
                content={ "response": "SUCCESS: User already exists" }
            )

    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })
