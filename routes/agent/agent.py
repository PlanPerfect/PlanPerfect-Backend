from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import uuid
import io
from PIL import Image as PILImage
from middleware.auth import _verify_api_key
from Services import AgentSynthesizer
from Services import Logger

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(_verify_api_key)])

# ── File validation constants ──────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class ConvertedFile:
    """
    Thin wrapper that holds a JPG-converted image in memory.
    _content is read directly by AgentSynthesizer._read_file_bytes.
    """

    def __init__(self, content: bytes, filename: str, content_type: str = "image/jpeg"):
        self._content = content
        self.filename = filename
        self.content_type = content_type


async def _validate_and_convert_to_jpg(file: UploadFile) -> ConvertedFile:
    """
    Validate that the upload is a PNG or JPG/JPEG, then convert it to a
    high-quality JPEG and return a ConvertedFile wrapper.
    Raises HTTPException(400) for invalid types or unreadable images.
    """
    filename = file.filename or "image"
    content_type = (file.content_type or "").lower().strip()
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if content_type not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"UERROR: Only PNG and JPG/JPEG images are allowed. "
                f"Received '{filename}'."
            ),
        )

    try:
        raw_bytes = await file.read()
        img = PILImage.open(io.BytesIO(raw_bytes))

        # Normalise colour mode to RGB before saving as JPEG
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

    except HTTPException:
        raise
    except Exception as e:
        Logger.log(
            f"[AGENT ROUTES] - ERROR: Image conversion failed for '{filename}': {e}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"UERROR: Could not process image '{filename}'. Please upload a valid PNG or JPG.",
        )


# ── Request models ─────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    uid: str
    query: str
    session_id: Optional[str] = None


class AgentSessionRequest(BaseModel):
    uid: str
    session_id: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/execute")
async def execute_agent(
    uid: str = Form(...),
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Execute the agent with an optional list of image uploads.
    A non-empty query is always required — files cannot be sent alone.
    All uploaded images are converted to JPG before processing.
    """
    try:
        # Change 1: Require a text prompt — files alone are not accepted.
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="UERROR: Please enter a message before sending.",
            )

        uploaded_files = []

        if files:
            for file in files:
                # Change 5 & 10: Validate type (PNG/JPG only) and convert to JPG.
                converted = await _validate_and_convert_to_jpg(file)
                file_id = str(uuid.uuid4())

                AgentSynthesizer.register_file(
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

        result = await AgentSynthesizer.execute(
            user_id=uid,
            query=query.strip(),
            session_id=session_id,
            uploaded_files=uploaded_files if uploaded_files else None,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Agent execution failed. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/get-session")
async def get_session(request: AgentSessionRequest):
    """Get agent session data for a user."""
    try:
        session = AgentSynthesizer.get_session(
            user_id=request.uid,
            session_id=request.session_id,
        )

        if not session:
            return {
                "session_id": None,
                "status": "idle",
                "current_step": "Thinking...",
                "steps": [],
                "Outputs": AgentSynthesizer._default_outputs(),
                "Uploaded Files": [],
            }

        return session

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to get session. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/list-sessions")
async def list_sessions(request: AgentSessionRequest):
    """List all sessions for a user."""
    try:
        sessions = AgentSynthesizer.list_sessions(user_id=request.uid)
        return {"sessions": sessions or {}}

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to list sessions. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/clear-session")
async def clear_session(request: AgentSessionRequest):
    """Clear agent session for a user."""
    try:
        success = AgentSynthesizer.clear_session(user_id=request.uid)
        return {"success": success}

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to clear session. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/clear-files")
async def clear_files(request: AgentSessionRequest):
    """Clear all uploaded files for a user."""
    try:
        AgentSynthesizer.clear_file_registry(user_id=request.uid)
        return {"success": True}

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to clear files. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")

@router.get("/current-agent-model")
async def get_current_agent_model():
    try:
        from Services import LLMManager as LLM
        current_model = LLM.get_current_agent_model()
        return JSONResponse(status_code=200, content={ "model": current_model })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")