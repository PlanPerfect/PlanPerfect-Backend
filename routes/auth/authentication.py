from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
            user_info = {
                "uid": user_data.uid,
                "email": user_data.email,
                "displayName": user_data.displayName,
                "RAG Conversations": {
                    "history": []
                },
                "Generated Document": {
                    "url": None
                },
                "Existing Homeowner": {
                    "Detected Furniture": {
                        "furniture": []
                    },
                    "Saved Reccomendations": {
                        "recommendations": []
                    },
                    "Detected Style": {
                        "style": None
                    },
                    "Generated Photo": {
                        "url": None
                    },
                    "Preferences": {
                        "budget": None,
                        "propertyType": None,
                        "unitType": None
                    },
                    "Preferred Style": {
                        "style": None
                    },
                    "Uploaded Photo": {
                        "url": None
                    }
                },
                "New Homeowner": {
                    "Preferences": {
                        "budget": None,
                        "Preferred Styles": {
                            "styles": []
                        }
                    },
                    "Segmented Floor Plan": {
                        "url": None
                    },
                    "Unit Information": {
                        "Number Of Rooms": {
                            "balcony": None,
                            "bathroom": None,
                            "bedroom": None,
                            "kitchen": None,
                            "ledge": None,
                            "livingRoom": None
                        },
                        "unit": None,
                        "unitSize": None,
                        "unitType": None
                    },
                    "Uploaded Floor Plan": {
                        "url": None
                    }
                }
            }

            success = DM.set_value(["Users", user_data.uid], user_info)

            if success:
                DM.save()
                return JSONResponse(
                    status_code=200,
                    content={ "response": "SUCCESS: User registered successfully" }
                )
            else:
                return JSONResponse(status_code=500, content={ "ERROR: Failed to register user" })
        else:
            return JSONResponse(
                status_code=200,
                content={ "response": "SUCCESS: User already exists" }
            )

    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })