from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
import json
import os
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Robust environment variable loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check PROJECT ROOT (one level up from /api/) and WEB_APP directory
ENV_PATHS = [
    os.path.join(BASE_DIR, "..", "..", "web_app", ".env"), # Local dev sibling
    os.path.join(BASE_DIR, "..", ".env"),                 # Local in backend
    os.path.join(BASE_DIR, ".env"),                      # In /api/
]

env_found = False
for env_path in ENV_PATHS:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Backend: Loaded env from {env_path}")
        env_found = True
        break

if not env_found:
    print("Backend: No .env file found. Relying on system environment variables.")

# Environment Variables
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "loi_maroc_db")

# Models
class UserLogin(BaseModel):
    email: str
    password: str

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(int(datetime.now().timestamp() * 1000)))
    role: str # 'user' or 'omar'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSession(BaseModel):
    user_id: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    user_id: str
    message: str

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize MongoDB and RAG
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI not found.")
    else:
        app.mongodb_client = AsyncIOMotorClient(MONGODB_URI)
        app.db = app.mongodb_client[DATABASE_NAME]
        print(f"Connected to MongoDB: {DATABASE_NAME}")
    yield
    # Shutdown: Close connections
    if hasattr(app, "mongodb_client"):
        app.mongodb_client.close()

app = FastAPI(title="LoiMaroc AI Backend", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://loi-maroc-ai.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_origin_regex="https://.*\\.vercel\\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log incoming request details
    origin = request.headers.get("origin")
    method = request.method
    url = request.url
    print(f"DEBUG_CORS: {method} {url} | Origin: {origin}")
    
    response = await call_next(request)
    
    # Log response details
    status_code = response.status_code
    print(f"DEBUG_CORS: Response Status: {status_code}")
    return response

# Import engine after env is loaded
try:
    try:
        from .rag_engine import engine
    except (ImportError, ValueError):
        import api.rag_engine as rag_engine
        engine = rag_engine.engine
except Exception as e:
    print(f"Backend: ERROR importing rag_engine: {e}")
    engine = None

@app.get("/api/debug")
async def debug_headers(request: Request):
    return {
        "headers": dict(request.headers),
        "method": request.method,
        "url": str(request.url),
        "base_dir": BASE_DIR,
        "env_path": [p for p in ENV_PATHS if os.path.exists(p)]
    }

@app.get("/")
@app.get("/api/health")
async def root():
    return {"status": "LoiMaroc API is running"}

@app.get("/get")
@app.get("/api/test")
async def get_test():
    return {
        "status": "connected", 
        "message": "Backend is reachable!",
        "time": str(datetime.now())
    }

# --- Endpoints ---

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    user = await app.db.users.find_one({"email": credentials.email})
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "user_id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"]
    }

@app.post("/api/auth/register")
async def register(user: UserCreate):
    existing_user = await app.db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = {
        "name": user.name,
        "email": user.email,
        "password": user.password,
        "created_at": datetime.now()
    }
    result = await app.db.users.insert_one(new_user)
    return {"user_id": str(result.inserted_id), "status": "success"}

@app.post("/api/chat")
async def chat(request_data: ChatRequest):
    if not engine:
        raise HTTPException(status_code=500, detail="RAG engine not available")
    
    user_id = request_data.user_id
    message = request_data.message

    # Get AI response
    response_data = await engine.get_response(message)
    
    # Store in history
    chat_msg_user = ChatMessage(role="user", content=message)
    chat_msg_omar = ChatMessage(role="omar", content=response_data["answer"])
    
    await app.db.chats.update_one(
        {"user_id": user_id},
        {
            "$push": {"messages": {"$each": [chat_msg_user.dict(), chat_msg_omar.dict()]}},
            "$set": {"updated_at": datetime.now()},
            "$setOnInsert": {"created_at": datetime.now()}
        },
        upsert=True
    )
    
    return {
        "answer": response_data["answer"],
        "context": response_data["context"]
    }

@app.get("/api/chat/history/{user_id}")
async def get_history(user_id: str):
    session = await app.db.chats.find_one({"user_id": user_id})
    if not session:
        return {"messages": []}
    return {"messages": session["messages"]}
