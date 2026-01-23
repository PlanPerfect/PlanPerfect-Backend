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
                "displayName": user_data.displayName
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