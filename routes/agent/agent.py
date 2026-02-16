from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel
import uuid
from middleware.auth import _verify_api_key
from Services import AgentSynthesizer
from Services import Logger

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(_verify_api_key)])

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
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Execute agent with optional file uploads.
    Files are uploaded alongside the query.
    """
    try:
        uploaded_files = []

        if files:
            for file in files:
                # Generate unique file ID
                file_id = str(uuid.uuid4())

                # Register file with agent
                AgentSynthesizer.register_file(
                    file_id=file_id,
                    file_obj=file,
                    user_id=uid,
                    source="request"
                )

                uploaded_files.append({
                    "file_id": file_id,
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "file_obj": file
                })

        # Execute agent
        result = await AgentSynthesizer.execute(
            user_id=uid,
            query=query,
            session_id=session_id,
            uploaded_files=uploaded_files if uploaded_files else None
        )

        return result

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Agent execution failed. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/get-session")
async def get_session(request: AgentSessionRequest):
    """
    Get agent session data for a user.
    """
    try:
        session = AgentSynthesizer.get_session(
            user_id=request.uid,
            session_id=request.session_id
        )

        if not session:
            return {
                "session_id": None,
                "status": "idle",
                "current_step": "Thinking...",
                "steps": [],
                "Outputs": AgentSynthesizer._default_outputs(),
                "Uploaded Files": []
            }

        return session

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to get session. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/list-sessions")
async def list_sessions(request: AgentSessionRequest):
    """
    List all sessions for a user.
    """
    try:
        sessions = AgentSynthesizer.list_sessions(user_id=request.uid)
        return {"sessions": sessions or {}}

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to list sessions. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/clear-session")
async def clear_session(request: AgentSessionRequest):
    """
    Clear agent session for a user.
    """
    try:
        success = AgentSynthesizer.clear_session(user_id=request.uid)
        return {"success": success}

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to clear session. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")


@router.post("/clear-files")
async def clear_files(request: AgentSessionRequest):
    """
    Clear all uploaded files for a user.
    """
    try:
        AgentSynthesizer.clear_file_registry(user_id=request.uid)
        return {"success": True}

    except Exception as e:
        Logger.log(f"[AGENT ROUTES] - ERROR: Failed to clear files. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ERROR: {str(e)}")