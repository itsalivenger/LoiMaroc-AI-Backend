from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import time

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

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "loi_maroc_db")

# Default Config for app settings
DEFAULT_CONFIG = {
    "persistence_threshold": 5,
    "rag_k": 5,
    "about_title": "À Propos de LoiMaroc AI",
    "about_content": "LoiMaroc AI est votre assistant juridique intelligent dédié au droit marocain. Notre mission est de démocratiser l'accés à l'information juridique grâce à l'intelligence artificielle.",
    "contact_recipient": "aliho.venger@gmail.com",
    "contact_phone": "+212 600-000000",
    "linkedin_url": "https://linkedin.com/in/alivenger",
    "portfolio_url": "https://alivenger.com"
}

# Models
class UserLogin(BaseModel):
    email: str
    password: str

class AdminLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(int(datetime.now().timestamp() * 1000)))
    role: str
    content: str
    source: Optional[str] = None

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[Message]
    email: Optional[str] = None
    updatedAt: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_email: Optional[str] = None

class Review(BaseModel):
    id: str = Field(default_factory=lambda: str(int(datetime.now().timestamp() * 1000)))
    rating: int
    comment: str
    session_id: str
    user_email: Optional[str] = None
    createdAt: datetime = Field(default_factory=datetime.now)

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

@app.get("/api/admin/config")
async def get_config():
    config = await app.db.settings.find_one({"type": "app_config"})
    if not config:
        return DEFAULT_CONFIG
    return {k: v for k, v in config.items() if k != "_id" and k != "type"}

@app.post("/api/admin/config")
async def update_config(new_config: dict):
    await app.db.settings.update_one(
        {"type": "app_config"},
        {"$set": new_config},
        upsert=True
    )
    return new_config

@app.post("/api/contact")
async def contact_me(
    name: str = Body(..., embed=True),
    email: str = Body(..., embed=True),
    phone: str = Body(None, embed=True),
    message: str = Body(..., embed=True)
):
    try:
        config = await get_config()
        recipient = config.get("contact_recipient") or os.getenv("EMAIL_FROM")
        
        if not recipient:
            raise HTTPException(status_code=500, detail="Contact email not configured.")

        smtp_host = os.getenv("SMTP_HOST")
        smtp_port_str = os.getenv("SMTP_PORT", "587")
        smtp_port = int(smtp_port_str) if smtp_port_str.isdigit() else 587
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        mail_from = os.getenv("EMAIL_FROM", smtp_user)

        if smtp_host and smtp_user and smtp_pass:
            msg = MIMEMultipart()
            msg['From'] = formataddr(("Loi Maroc", str(mail_from)))
            msg['To'] = str(recipient)
            msg['Subject'] = f"Nouveau message de {name} (LoiMaroc AI)"
            
            body = f"Nom: {name}\nEmail: {email}\nTéléphone: {phone or 'Non fourni'}\n\nMessage:\n{message}"
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(str(smtp_host), smtp_port)
            server.starttls()
            server.login(str(smtp_user), str(smtp_pass))
            server.send_message(msg)
            server.quit()
        
        return {"status": "success", "message": "Message envoyé avec succès."}
    except Exception as e:
        print(f"Contact failed: {e}")
        raise HTTPException(status_code=500, detail=f"Échec de l'envoi: {str(e)}")

# --- Endpoints ---

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    user = await app.db.users.find_one({"email": credentials.email})
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "status": "success",
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"]
        }
    }

@app.post("/api/admin/login")
async def admin_login(credentials: AdminLogin):
    # Get admin credentials from env, or use defaults
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASS", "admin123")
    admin_email = os.getenv("ADMIN_EMAIL", "emailtrash226@gmail.com")
    
    # Allow login with either username or email as the 'username' field
    is_valid_user = (credentials.username == admin_user or credentials.username == admin_email)
    is_valid_pass = (credentials.password == admin_pass)
    
    if is_valid_user and is_valid_pass:
        return {"status": "success", "message": "Admin logged in"}
    
    print(f"FAILED_ADMIN_LOGIN: Attempt by {credentials.username}")
    raise HTTPException(status_code=401, detail="Invalid admin credentials")

# --- Admin Management Endpoints ---

@app.get("/api/admin/users")
async def get_all_users():
    cursor = app.db.users.find().sort("created_at", -1)
    users = await cursor.to_list(length=1000)
    # Map _id to id and remove password
    for user in users:
        user["id"] = str(user["_id"])
        user.pop("_id", None)
        user.pop("password", None)
        # Ensure createdAt exists for frontend
        if "created_at" in user:
            user["createdAt"] = int(user["created_at"].timestamp() * 1000)
        else:
            user["createdAt"] = int(datetime.now().timestamp() * 1000)
        # Default verified to false if not present
        if "verified" not in user:
            user["verified"] = False
    return users

@app.delete("/api/admin/users/{email}")
async def delete_user(email: str):
    # Delete user
    user_result = await app.db.users.delete_one({"email": email})
    # Delete all sessions associated with this email
    sessions_result = await app.db.sessions.delete_many({"email": email})
    
    if user_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"status": "success", "deleted_sessions": sessions_result.deleted_count}

@app.patch("/api/admin/users/{email}/verify")
async def toggle_user_verify(email: str):
    user = await app.db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_status = not user.get("verified", False)
    await app.db.users.update_one({"email": email}, {"$set": {"verified": new_status}})
    return {"status": "success", "verified": new_status}

@app.get("/api/admin/sessions", response_model=List[ChatSession])
async def get_all_sessions():
    cursor = app.db.sessions.find().sort("updatedAt", -1)
    sessions = await cursor.to_list(length=1000)
    return sessions

@app.delete("/api/admin/sessions/{session_id}")
async def admin_delete_session(session_id: str):
    result = await app.db.sessions.delete_one({"id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

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
    
    query = request_data.query
    session_id = request_data.session_id
    user_email = request_data.user_email

    # Get AI response
    response_data = await engine.get_response(query)
    
    # Store in history if session exists
    if session_id:
        chat_msg_user = Message(role="user", content=query)
        chat_msg_omar = Message(role="assistant", content=response_data["answer"], source=response_data["context"][0] if response_data["context"] else None)
        
        await app.db.sessions.update_one(
            {"id": session_id},
            {
                "$push": {"messages": {"$each": [chat_msg_user.dict(), chat_msg_omar.dict()]}},
                "$set": {
                    "updatedAt": int(datetime.now().timestamp() * 1000),
                    "email": user_email
                },
                "$setOnInsert": {
                    "title": query[:50] + "..." if len(query) > 50 else query
                }
            },
            upsert=True
        )
    
    return {
        "answer": response_data["answer"],
        "context": response_data["context"]
    }

@app.get("/api/sessions/user/{email}", response_model=List[ChatSession])
async def get_user_sessions(email: str):
    cursor = app.db.sessions.find({"email": email}).sort("updatedAt", -1)
    sessions = await cursor.to_list(length=100)
    return sessions

@app.get("/api/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    session = await app.db.sessions.find_one({"id": session_id})
    if session:
        return session
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/sessions")
async def save_session(session: ChatSession):
    try:
        await app.db.sessions.replace_one({"id": session.id}, session.dict(), upsert=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    result = await app.db.sessions.delete_one({"id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

# --- Review Endpoints ---

@app.post("/api/reviews")
async def create_review(review: Review):
    try:
        # Check if review already exists for this session
        existing = await app.db.reviews.find_one({"session_id": review.session_id})
        if existing:
            raise HTTPException(status_code=400, detail="Review already submitted for this session")
        
        review_dict = review.dict()
        # Convert datetime to string or ISO format for MongoDB if needed, 
        # but motor usually handles datetime objects fine.
        await app.db.reviews.insert_one(review_dict)
        return {"status": "success", "id": review.id}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/admin/reviews", response_model=List[Review])
async def get_reviews():
    cursor = app.db.reviews.find().sort("createdAt", -1)
    reviews = await cursor.to_list(length=200)
    return reviews
