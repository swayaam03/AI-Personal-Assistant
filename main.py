import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from api import router as api_router

# Initialize FastAPI Application
app = FastAPI(
    title="LangGraph AI Personal Assistant API",
    description="A stateful, multi-turn AI Personal Assistant API built with LangGraph, LangChain, and Gemini.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(api_router)


@app.get("/", summary="Root Endpoint")
async def root():
    return {
        "message": "Welcome to LangGraph AI Personal Assistant API",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
