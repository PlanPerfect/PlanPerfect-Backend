from fastapi import APIRouter
from Services import Emailer

router = APIRouter(prefix="/emailer", tags=["Tools & Services"])

@router.post("/send-email")
def send_email(to: str, subject: str, template: str):
    emailer = Emailer()

    variables = {
        "username": "John Appleseed"
    }

    result = emailer.send_email(to=to, subject=subject, template=template, variables=variables)

    return {"result": result}