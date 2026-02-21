from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from middleware.auth import _verify_api_key
from Services import DatabaseManager as DM
from Services import Logger
from Services.Emailer import Emailer

router = APIRouter(
    prefix="/designDocument/clientConfirmationEmail",
    tags=["Design Document"],
    dependencies=[Depends(_verify_api_key)]
)

emailer = Emailer()


@router.post("/{user_id}")
async def send_client_confirmation_email(user_id: str):
    """
    Sends confirmation email to client after design document download.
    """

    try:
        if not user_id:
            return JSONResponse(
                status_code=400,
                content={"error": "UERROR: User ID is required."}
            )

        user = DM.peek(["Users", user_id])

        if user is None:
            return JSONResponse(
                status_code=404,
                content={"error": "UERROR: Please login again."}
            )

        email = user.get("email")
        username = user.get("displayName", "Valued Client")

        if not email:
            return JSONResponse(
                status_code=400,
                content={"error": "UERROR: Email not found for this user."}
            )

        # Send Email
        result = emailer.send_email(
            to=email,
            subject="Your Design Document is Ready",
            template="design_document_confirmation",
            variables={
                "username": username
            }
        )

        if result.startswith("ERROR"):
            Logger.log(f"[EMAIL] - ERROR: {result}")
            return JSONResponse(
                status_code=500,
                content={"error": f"ERROR: Failed to send email. {result}"}
            )

        return {
            "success": True,
            "message": "Confirmation email sent successfully."
        }

    except Exception as e:
        Logger.log(f"[EMAIL] - ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"ERROR: Failed to send confirmation email. {str(e)}"}
        )