from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import random

# Robust path handling for .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Try multiple possible env paths
ENV_PATHS = [
    os.path.join(BASE_DIR, "..", "web_app", ".env"), # Local dev sibling
    os.path.join(BASE_DIR, ".env"),                 # Local in backend
]
env_loaded = False
for path in ENV_PATHS:
    if os.path.exists(path):
        load_dotenv(dotenv_path=path)
        env_loaded = True
        break
if not env_loaded:
    load_dotenv() # Fallback to system environment variables (Vercel)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DATABASE_NAME", "loimaroc_ai")

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

class UserAuth(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class VerifyRequest(BaseModel):
    email: str
    code: str

class AdminAuth(BaseModel):
    username: str
    password: str

# Message and Session models
class Message(BaseModel):
    role: str
    content: str
    source: Optional[str] = None

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[Message]
    email: Optional[str] = None
    updatedAt: int = Field(default_factory=lambda: int(time.time() * 1000))

# Lifespan for DB connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        await client.admin.command('ping')
        print("Backend: Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Backend: Error connecting to MongoDB: {e}")
    
    # Initialize default admin if none exists
    admin_count = await db.admins.count_documents({})
    if admin_count == 0:
        await db.admins.insert_one({"username": "admin", "password": "admin123"})
        print("Backend: Default admin created (admin/admin123)")
    yield
    # Shutdown logic
    client.close()

app = FastAPI(title="LoiMaroc AI Backend", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://loi-maroc-ai.vercel.app",
        "https://loi-maroc-ai.vercel.app/",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import engine after env is loaded
try:
    try:
        from rag_engine import engine
    except ImportError:
        import rag_engine
        engine = rag_engine.engine
except Exception as e:
    print(f"Backend: ERROR importing rag_engine: {e}")
    engine = None

@app.get("/")
async def root():
    return {"status": "LoiMaroc API is running"}

@app.get("/api/health")
async def health_check():
    health = {
        "status": "healthy",
        "mongodb": "connected" if db is not None else "disconnected",
        "rag_engine": "ready" if engine and engine.chain else "error"
    }
    if db is None: health["status"] = "degraded"
    return health

# Load config from DB with fallback
DEFAULT_CONFIG = {
    "persistence_threshold": 5,
    "rag_k": 5,
    "about_title": "À Propos de LoiMaroc AI",
    "about_content": "LoiMaroc AI est votre assistant juridique intelligent dédié au droit marocain. Notre mission est de démocratiser l'accés à l'information juridique grâce à l'intelligence artificielle.",
    "contact_recipient": "",
    "contact_phone": "",
    "linkedin_url": "",
    "portfolio_url": ""
}

@app.get("/api/admin/config")
async def get_config():
    config = await db.settings.find_one({"type": "app_config"})
    if not config:
        return DEFAULT_CONFIG
    return {k: v for k, v in config.items() if k != "_id" and k != "type"}

@app.post("/api/admin/config")
async def update_config(new_config: dict):
    await db.settings.update_one(
        {"type": "app_config"},
        {"$set": new_config},
        upsert=True
    )
    return new_config

# Authentication Endpoints
@app.post("/api/auth/register")
async def register(user: UserAuth):
    # Check if user already exists in main users
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="L'email est déjà utilisé.")
    
    # Generate 6-digit code
    code = str(random.randint(100000, 999999))
    
    # Save to pending (overwrite if exists)
    user_data = user.model_dump()
    user_data["code"] = code
    user_data["expiresAt"] = int(time.time() * 1000) + (15 * 60 * 1000) # 15 mins
    
    await db.pending_registrations.update_one(
        {"email": user.email},
        {"$set": user_data},
        upsert=True
    )

    # EMAIL SENDING LOGIC (Placeholder for variables)
    try:
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port_str = os.getenv("SMTP_PORT", "587")
        smtp_port = int(smtp_port_str) if smtp_port_str.isdigit() else 587
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        mail_from = os.getenv("EMAIL_FROM", smtp_user)

        if smtp_host and smtp_user and smtp_pass:
            msg = MIMEMultipart()
            # Use formataddr for professional "LoiMaroc AI" sender identity
            msg['From'] = formataddr(("LoiMaroc AI", str(mail_from or smtp_user)))
            msg['To'] = str(user.email)
            msg['Subject'] = "Votre code de vérification LoiMaroc AI"
            
            body = f"Bonjour {user.name},\n\nVotre code de vérification est : {code}\n\nCe code expirera dans 15 minutes."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(str(smtp_host), smtp_port)
            server.starttls()
            server.login(str(smtp_user), str(smtp_pass))
            server.send_message(msg)
            server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")
        # In a real app, maybe don't fail registration if only mail fails, 
        # but here we want to ensure the code is sent.
    
    return {"status": "success", "message": "Code de vérification envoyé."}

@app.post("/api/auth/verify")
async def verify(request: VerifyRequest):
    pending = await db.pending_registrations.find_one({"email": request.email, "code": request.code})
    
    if not pending:
        raise HTTPException(status_code=400, detail="Code incorrect ou expiré.")
    
    # Move to users
    user_data = {k: v for k, v in pending.items() if k not in ["_id", "code", "expiresAt"]}
    # Use update to avoid lint issues with item assignment on Pydantic/Motor dicts
    user_data.update({
        "verified": True,
        "createdAt": int(time.time() * 1000)
    })
    
    await db.users.insert_one(user_data)
    await db.pending_registrations.delete_one({"email": request.email})
    
    return {"status": "success", "message": "Compte vérifié avec succès."}

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email, "password": credentials.password})
    if not user:
        raise HTTPException(status_code=401, detail="Identifiants incorrects.")
    
    return {"status": "success", "user": {"name": user["name"], "email": user["email"]}}

@app.get("/api/admin/users")
async def get_users():
    cursor = db.users.find({}).sort("createdAt", -1)
    users = await cursor.to_list(length=1000)
    # Remove password from response
    for user in users:
        user["id"] = str(user["_id"])
        del user["_id"]
        if "password" in user:
            del user["password"]
    return users

@app.delete("/api/admin/users/{email}")
async def delete_user(email: str):
    await db.users.delete_one({"email": email})
    # Also delete their sessions
    await db.sessions.delete_many({"email": email})
    return {"status": "success"}

@app.get("/api/admin/sessions")
async def get_all_sessions():
    cursor = db.sessions.find({}).sort("updatedAt", -1)
    sessions = await cursor.to_list(length=1000)
    for session in sessions:
        if "_id" in session:
            session["_id"] = str(session["_id"])
    return sessions

@app.delete("/api/admin/sessions/{session_id}")
async def delete_admin_session(session_id: str):
    result = await db.sessions.delete_one({"id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

@app.patch("/api/admin/users/{email}/verify")
async def toggle_verify(email: str):
    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_status = not user.get("verified", False)
    await db.users.update_one({"email": email}, {"$set": {"verified": new_status}})
    return {"verified": new_status}

@app.post("/api/admin/login")
async def admin_login(credentials: AdminAuth):
    admin = await db.admins.find_one({"username": credentials.username, "password": credentials.password})
    if not admin:
        raise HTTPException(status_code=401, detail="Accès admin refusé.")
    return {"status": "success", "username": admin["username"]}

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

        msg = MIMEMultipart()
        # Use formataddr for professional "Loi Maroc" sender identity
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
        raise HTTPException(status_code=500, detail="Échec de l'envoi du message.")

@app.post("/api/chat")
async def chat_endpoint(
    query: str = Body(..., embed=True), 
    session_id: Optional[str] = Body(None),
    user_email: Optional[str] = Body(None)
):
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized.")
    
    try:
        # 1. Get response from RAG Engine
        result = await engine.get_response(query)
        
        answer = result["answer"]
        context_sources = result["context"]
        
        # 2. Sync with MongoDB if session_id exists
        if session_id:
            user_msg = Message(role="user", content=query)
            ai_msg = Message(role="assistant", content=answer, source=context_sources[0] if context_sources else None)
            
            await db.sessions.update_one(
                {"id": session_id},
                {
                    "$push": {"messages": {"$each": [user_msg.model_dump(), ai_msg.model_dump()]}}, 
                    "$set": {
                        "updatedAt": int(time.time() * 1000),
                        "email": user_email
                    }
                },
                upsert=True
            )

        return {
            "answer": answer,
            "session_id": session_id or "new_session",
            "sources": context_sources
        }
    except Exception as e:
        print(f"Chat API Exception: {e}")
        return {
            "answer": "Désolé, une erreur de communication est survenue avec le serveur d'IA. Veuillez vérifier votre connexion et réessayer.",
            "session_id": session_id or "error_session",
            "sources": ["Erreur Technique"]
        }

# Sessions Endpoints
@app.get("/api/sessions/user/{email}", response_model=List[ChatSession])
async def get_user_sessions(email: str):
    cursor = db.sessions.find({"email": email}).sort("updatedAt", -1)
    sessions = await cursor.to_list(length=100)
    return sessions

@app.get("/api/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    session = await db.sessions.find_one({"id": session_id})
    if session:
        return session
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/sessions")
async def save_session(session: ChatSession):
    try:
        # Use model_dump for Pydantic v2
        await db.sessions.replace_one({"id": session.id}, session.model_dump(), upsert=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
