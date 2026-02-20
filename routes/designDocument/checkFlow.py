from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from middleware.auth import _verify_api_key
from Services import DatabaseManager as DM
from Services import Logger

router = APIRouter(prefix="/designDocument/checkFlow", tags=["User"], dependencies=[Depends(_verify_api_key)])

@router.get("/{user_id}")
async def get_user_flow(user_id: str):
    """
    Returns the flow value stored on the user record in the database.
    Possible values: "newHomeOwner", "existingHomeOwner", or null if not set.
    """
    try:
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "UERROR: User ID is required."})

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(status_code=404, content={"error": "UERROR: Please login again."})

        flow = user.get("flow", None)

        return {
            "success": True,
            "result": {
                "flow": flow
            }
        }

    except Exception as e:
        Logger.log(f"[USER] - ERROR: Error fetching user flow: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"ERROR: Failed to fetch user flow. {str(e)}"}
        )