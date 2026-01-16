from fastapi import APIRouter, Depends, HTTPException
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
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        RAG.add_to_history("user", request.query)

        context = RAG.retrieve_query(request.query)

        response = LLM.chat(context)

        RAG.add_to_history("assistant", response)

        return JSONResponse(content={"response": response})

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in chat_completion: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/clear-history")
async def clear_history():
    try:
        RAG.clear_history()
        return JSONResponse(content={"message": "Conversation history cleared"})
    except Exception as e:
        print(f"ERROR in clear_history: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)