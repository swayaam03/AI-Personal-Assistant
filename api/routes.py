from fastapi import APIRouter, HTTPException
from config import settings
from services import assistant_service
from api.schemas import ChatRequest, ChatResponse, HistoryResponse, HealthResponse

router = APIRouter(prefix="/api", tags=["Assistant"])


@router.post("/chat", response_model=ChatResponse, summary="Send message to AI Personal Assistant")
async def chat_endpoint(request: ChatRequest):
    """
    Sends a message to the LangGraph AI Assistant.
    Maintains multi-turn conversation memory using `thread_id`.
    """
    try:
        result = assistant_service.process_user_message(
            message=request.message,
            thread_id=request.thread_id
        )
        return ChatResponse(
            response=result["response"],
            tool_calls=result["tool_calls"],
            thread_id=result["thread_id"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing assistant request: {str(e)}")


@router.get("/history/{thread_id}", response_model=HistoryResponse, summary="Retrieve conversation history for thread")
async def history_endpoint(thread_id: str):
    """
    Returns the chronological conversation history for a specific thread_id.
    """
    try:
        history = assistant_service.get_conversation_history(thread_id=thread_id)
        return HistoryResponse(
            thread_id=thread_id,
            history=history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching thread history: {str(e)}")


@router.get("/health", response_model=HealthResponse, summary="System health status check")
async def health_endpoint():
    """
    Returns application health status and model configuration state.
    """
    has_key = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here")
    return HealthResponse(
        status="healthy",
        model=settings.GEMINI_MODEL,
        api_key_configured=has_key
    )
