from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# Import your PetPal chatbot
from chat import PetPalChatbot  # Update with your actual filename

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User's message to PetPal")
    session_id: Optional[str] = Field(default="default", description="Session ID for conversation continuity")

class ChatResponse(BaseModel):
    response: str = Field(..., description="PetPal's response")
    session_id: str = Field(..., description="Session ID used")
    conversation_stage: str = Field(..., description="Current conversation stage")
    message_count: int = Field(..., description="Number of messages in this session")

class UserProfileResponse(BaseModel):
    name: Optional[str] = None
    pet_preference: str = "unknown"
    engagement_level: int = 5
    stories_heard: List[str] = []
    compliments_received: List[str] = []

class SessionInfoResponse(BaseModel):
    session_id: str
    user_profile: UserProfileResponse
    conversation_stage: str
    message_count: int
    is_active: bool

# Global chatbot instance
chatbot_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot_instance
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    chatbot_instance = PetPalChatbot(groq_api_key=groq_api_key)
    print("PetPal chatbot initialized successfully!")
    
    yield
    
    # Shutdown
    print("PetPal chatbot shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="PetPal Chatbot API",
    description="A charming AI companion that loves pets and making people feel special",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get chatbot instance
async def get_chatbot() -> PetPalChatbot:
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot_instance

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to PetPal Chatbot API! 🐾",
        "description": "A charming AI companion who loves pets and making people feel special",
        "endpoints": {
            "/chat": "Send a message to PetPal",
            "/session/{session_id}": "Get session information",
            "/sessions": "List all active sessions",
            "/health": "Health check"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_petpal(
    request: ChatRequest,
    chatbot: PetPalChatbot = Depends(get_chatbot)
):
    """
    Send a message to PetPal and get a response
    
    - **message**: Your message to PetPal (1-1000 characters)
    - **session_id**: Optional session ID for conversation continuity (defaults to "default")
    """
    try:
        # Get response from chatbot
        response = await chatbot.chat(request.message, request.session_id)
        
        # Get session context for additional info
        context = await chatbot.get_or_create_session(request.session_id)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            conversation_stage=context.stage.value,
            message_count=context.messages_count
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Sorry, I'm having trouble right now. Please try again! 🐾"
        )

@app.get("/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(
    session_id: str,
    chatbot: PetPalChatbot = Depends(get_chatbot)
):
    """
    Get information about a specific conversation session
    
    - **session_id**: The session ID to get information about
    """
    try:
        if session_id not in chatbot.active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        context = chatbot.active_sessions[session_id]
        profile = context.user_profile
        
        return SessionInfoResponse(
            session_id=session_id,
            user_profile=UserProfileResponse(
                name=profile.name,
                pet_preference=profile.pet_preference.value,
                engagement_level=profile.engagement_level,
                stories_heard=profile.stories_heard,
                compliments_received=profile.compliments_received[-5:]  # Last 5 compliments
            ),
            conversation_stage=context.stage.value,
            message_count=context.messages_count,
            is_active=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving session information")

@app.get("/sessions")
async def list_active_sessions(chatbot: PetPalChatbot = Depends(get_chatbot)):
    """
    List all active conversation sessions
    """
    try:
        sessions = []
        for session_id, context in chatbot.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "user_name": context.user_profile.name,
                "message_count": context.messages_count,
                "conversation_stage": context.stage.value,
                "pet_preference": context.user_profile.pet_preference.value,
                "engagement_level": context.user_profile.engagement_level
            })
        
        return {
            "active_sessions": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        print(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving sessions")

@app.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    chatbot: PetPalChatbot = Depends(get_chatbot)
):
    """
    Clear/reset a specific conversation session
    
    - **session_id**: The session ID to clear
    """
    try:
        if session_id in chatbot.active_sessions:
            del chatbot.active_sessions[session_id]
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail="Error clearing session")

@app.post("/session/{session_id}/reset")
async def reset_session(
    session_id: str,
    chatbot: PetPalChatbot = Depends(get_chatbot)
):
    """
    Reset a session while keeping the session_id
    
    - **session_id**: The session ID to reset
    """
    try:
        # Clear existing session if it exists
        if session_id in chatbot.active_sessions:
            del chatbot.active_sessions[session_id]
        
        # Create fresh session
        new_context = chatbot.get_or_create_session(session_id)
        
        return {
            "message": f"Session {session_id} reset successfully",
            "session_id": session_id,
            "conversation_stage": new_context.stage.value
        }
        
    except Exception as e:
        print(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail="Error resetting session")

@app.get("/health")
async def health_check(chatbot: PetPalChatbot = Depends(get_chatbot)):
    """
    Health check endpoint
    """
    try:
        # Basic health check - ensure chatbot is responsive
        test_response = await chatbot.chat("Hi", f"health_check_{hash('health')}")
        
        # Clean up health check session
        health_session_id = f"health_check_{hash('health')}"
        if health_session_id in chatbot.active_sessions:
            del chatbot.active_sessions[health_session_id]
        
        return {
            "status": "healthy",
            "chatbot": "operational",
            "active_sessions": len(chatbot.active_sessions),
            "test_response_received": bool(test_response),
            "groq_api": "connected"
        }
        
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Example startup configuration
if __name__ == "__main__":
    import uvicorn
    
    # Make sure to set GROQ_API_KEY environment variable
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable not set!")
        print("Please set it with: export GROQ_API_KEY='your_api_key_here'")
        exit(1)
        
    print("🐾 Starting PetPal Chatbot API...")
    print("📝 API Documentation will be available at: http://localhost:8000/docs")
    print("🔄 Interactive API testing at: http://localhost:8000/redoc")
    
    PORT = int(os.getenv("PORT", 8000))  # Use PORT env var or default to 8000
    
    uvicorn.run(
        "main:app",  # Replace "main" with your filename
        host="127.0.0.1",
        port=PORT,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )