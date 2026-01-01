from fastapi import APIRouter
from fastapi.responses import JSONResponse
from Services import Emailer

router = APIRouter(prefix="/emailer", tags=["Tools & Services"])

@router.post("/send-email")
def send_email(to: str, subject: str, template: str):
    try:
        emailer = Emailer()

        variables = {
            "username": "John Appleseed"
        }

        result = emailer.send_email(to=to, subject=subject, template=template, variables=variables)

        return JSONResponse(status_code=200, content={ "success": True, "result": f"SUCCESS: Email sent to {to}" })
    except Exception as e:
        return JSONResponse(status_code=500, content={ "success": False, "result": f"ERROR: Failed to send email. Error: {e}" })