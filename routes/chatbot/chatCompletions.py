from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from middleware.auth import _verify_api_key

from Services import RAGManager as RAG
from Services import LLMManager as LLM

router = APIRouter(prefix="/chatbot", tags=["Chatbot"], dependencies=[Depends(_verify_api_key)])

class ChatRequest(BaseModel):
    query: str

@router.post("/chat-completion")
async def chat_completion(request: ChatRequest):
    try:
        if not request.query or not request.query.strip():
            return JSONResponse(status_code=400, content={ "error": "UERROR: Query cannot be empty." })

        RAG.add_to_history("user", request.query)

        context = RAG.retrieve_query(request.query)

        response = LLM.chat(context)

        RAG.add_to_history("assistant", response)

        return JSONResponse(content={ "response": response })

    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })

@router.post("/clear-history")
async def clear_history():
    try:
        RAG.clear_history()
        return JSONResponse(status_code=200, content={ "response": "Conversation history cleared" })
    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })