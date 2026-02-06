from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from middleware.auth import _verify_api_key

from Services import RAGManager as RAG
from Services import LLMManager as LLM
from Services import DatabaseManager as DM

router = APIRouter(prefix="/chatbot", tags=["Chatbot"], dependencies=[Depends(_verify_api_key)])

class ChatRequest(BaseModel):
    uid: str
    query: str

class ClearHistoryRequest(BaseModel):
    uid: str

@router.get("/current-model")
async def get_current_model():
    try:
        current_model = LLM.get_current_model()
        return JSONResponse(status_code=200, content={ "model": current_model })
    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })

@router.post("/chat-completion")
async def chat_completion(request: ChatRequest):
    try:
        if not request.uid or not request.uid.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", request.uid])

        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        if not request.query or not request.query.strip():
            return JSONResponse(status_code=400, content={ "error": "UERROR: Query cannot be empty." })

        RAG.add_to_history(request.uid, "user", request.query)

        context = RAG.retrieve_query(request.uid, request.query)

        response = LLM.chat(context)

        RAG.add_to_history(request.uid, "assistant", response)

        current_model = LLM.get_current_model()

        return JSONResponse(status_code=200, content={
            "response": response,
            "model": current_model
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })

@router.post("/clear-history")
async def clear_history(request: ClearHistoryRequest):
    try:
        if not request.uid or not request.uid.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", request.uid])

        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        RAG.clear_history(request.uid)
        return JSONResponse(status_code=200, content={ "response": "SUCCESS: Conversation history cleared" })
    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })

@router.get("/history/{uid}")
async def get_history(uid: str):
    try:
        if not uid or not uid.strip():
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", uid])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        history = RAG.get_history(uid)
        return JSONResponse(status_code=200, content={ "history": history })
    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })