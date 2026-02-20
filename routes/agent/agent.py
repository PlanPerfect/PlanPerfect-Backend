from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import uuid
import io
from PIL import Image as PILImage
from middleware.auth import _verify_api_key
from Services import Logger
from Services import AgentSynthesizer as AGS
from Services import LLMManager as LLM

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(_verify_api_key)])

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class ConvertedFile:
    def __init__(self, content: bytes, filename: str, content_type: str = "image/jpeg"):
        self._content = content
        self.filename = filename
        self.content_type = content_type


async def _validate_and_convert_to_jpg(file: UploadFile) -> ConvertedFile:
    filename = file.filename or "image"
    content_type = (file.content_type or "").lower().strip()
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if content_type not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={ "error": f"UERROR: Only PNG and PNG/JPG/JPEG images are allowed." },
        )

    try:
        raw_bytes = await file.read()
        img = PILImage.open(io.BytesIO(raw_bytes))

        if img.mode in ("RGBA", "LA"):
            bg = PILImage.new("RGB", img.size, (255, 255, 255))
            alpha = img.convert("RGBA").split()[3]
            bg.paste(img, mask=alpha)
            img = bg
        elif img.mode == "P":
            img = img.convert("RGBA")
            bg = PILImage.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, optimize=True)
        jpg_bytes = buf.getvalue()

        base = filename.rsplit(".", 1)[0] if "." in filename else filename

        return ConvertedFile(
            content=jpg_bytes,
            filename=f"{base}.jpg",
            content_type="image/jpeg",
        )

    except Exception as e:
        Logger.log(f"[AGENT] - ERROR: Image conversion failed for '{filename}': {e}")
        return JSONResponse(status_code=500, content={ "error": str(e) })


class AgentQueryRequest(BaseModel):
    uid: str
    query: str
    session_id: Optional[str] = None


class AgentSessionRequest(BaseModel):
    uid: str
    session_id: Optional[str] = None


@router.post("/execute")
async def execute_agent(
    uid: str = Form(...),
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    try:
        if not query or not query.strip():
            return JSONResponse(status_code=400, content={ "error": "UERROR: Message cannot be empty." })

        uploaded_files = []

        if files:
            for file in files:
                converted = await _validate_and_convert_to_jpg(file)
                file_id = str(uuid.uuid4())

                AGS.register_file(
                    file_id=file_id,
                    file_obj=converted,
                    user_id=uid,
                    source="request",
                )

                uploaded_files.append(
                    {
                        "file_id": file_id,
                        "filename": converted.filename,
                        "content_type": converted.content_type,
                        "file_obj": converted,
                    }
                )

        result = await AGS.execute(
            user_id=uid,
            query=query.strip(),
            session_id=session_id,
            uploaded_files=uploaded_files if uploaded_files else None,
        )

        return result

    except Exception as e:
        Logger.log(f"[AGENT] - ERROR: Agent execution failed. Error: {str(e)}")
        return JSONResponse(status_code=500, content={ "error": str(e) })


@router.post("/get-session")
async def get_session(request: AgentSessionRequest):
    try:
        session = AGS.get_session(
            user_id=request.uid,
            session_id=request.session_id,
        )

        if not session:
            return {
                "session_id": None,
                "status": "idle",
                "current_step": "Thinking...",
                "steps": [],
                "Outputs": AGS._default_outputs(),
                "Uploaded Files": [],
            }

        return session

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to get session. Error: {str(e)}")
        return JSONResponse(status_code=500, content={ "error": str(e) })


@router.post("/list-sessions")
async def list_sessions(request: AgentSessionRequest):
    try:
        sessions = AGS.list_sessions(user_id=request.uid)
        return {"sessions": sessions or {}}

    except Exception as e:
        Logger.log(f"[AGENT] - ERROR: Failed to list sessions. Error: {str(e)}")
        return JSONResponse(status_code=500, content={ "error": str(e) })


@router.post("/clear-session")
async def clear_session(request: AgentSessionRequest):
    try:
        success = AGS.clear_session(user_id=request.uid)
        return {"success": success}

    except Exception as e:
        Logger.log(f"[AGENT] - ERROR: Failed to clear session. Error: {str(e)}")
        return JSONResponse(status_code=500, content={ "error": str(e) })


@router.post("/clear-files")
async def clear_files(request: AgentSessionRequest):
    try:
        AGS.clear_file_registry(user_id=request.uid)
        return {"success": True}

    except Exception as e:
        Logger.log(f"[AGENT] - ERROR: Failed to clear files. Error: {str(e)}")
        return JSONResponse(status_code=500, content={ "error": str(e) })

@router.get("/current-agent-model")
async def get_current_agent_model():
    try:
        current_model = LLM.get_current_agent_model()
        return JSONResponse(status_code=200, content={ "model": current_model })
    except Exception as e:
        Logger.log(f"[AGENT] - ERROR: Failed to get current agent model. Error: {str(e)}")
        return JSONResponse(status_code=500, content={ "error": str(e) })