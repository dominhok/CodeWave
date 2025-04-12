import os
from fastapi import FastAPI, Form, Request, Response, Depends, HTTPException, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import pathlib
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
import time # time 모듈 임포트

from . import models
from .models import Base, engine, User, get_db

# --- Langchain Imports ---
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage # Use ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader # Using community loader

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv() # Load environment variables from .env file

app = FastAPI()

# --- Global Variables for RAG --- 
# Initialize vectorstore to None; will be loaded/created on startup
vectorstore = None
VECTORSTORE_PATH = "faiss_disaster_manuals" # Path to save/load FAISS index

# --- Register Startup Event --- 
@app.on_event("startup")
def on_startup():
    setup_database_and_rag(app) # Combined setup function

# --- Dependency to get DB session (Modified) ---
def get_db(request: Request) -> Session:
    SessionLocal = getattr(request.app.state, 'SessionLocal', None)
    if SessionLocal is None:
        # This might happen if startup failed critically
        raise HTTPException(status_code=500, detail="Database session factory not available.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CORS Middleware Configuration ---
# Allow requests from the React dev server (typically http://localhost:3000)
origins = [
    "http://localhost:3000", # React default dev port
    # Add other origins if needed, e.g., your frontend production URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"], # Allow relevant methods
    allow_headers=["*"], # Allow all headers
)

# Mount static files directory
static_dir = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configure Jinja2 templates
templates_dir = static_dir / "maps" # Templates are in static/maps
templates = Jinja2Templates(directory=templates_dir)

# Twilio credentials from .env file
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
kakao_map_app_key = os.getenv('KAKAO_MAP_APP_KEY')
safety_data_service_key = os.getenv('SAFETY_DATA_SERVICE_KEY')

logging.info(f"TWILIO_ACCOUNT_SID: {account_sid}")
logging.info(f"auth_token: {'*' * len(auth_token) if auth_token else None}") # 토큰은 로그에 직접 노출하지 않도록 처리
logging.info(f"TWILIO_PHONE_NUMBER: {twilio_phone_number}")

# Check if credentials are loaded
if not account_sid or not auth_token or not twilio_phone_number:
    # Optionally, you might want to handle this more gracefully
    # depending on whether Twilio functionality is always required.
    print("Warning: Twilio credentials not fully found in .env file. Twilio features may not work.")
    # raise ValueError("Twilio credentials not found in .env file")

# Initialize client only if credentials exist
client = None
if account_sid and auth_token:
    try:
        client = Client(account_sid, auth_token)
        logging.info("Twilio client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Twilio client: {e}")
else:
    logging.warning("Twilio credentials (account_sid or auth_token) not found in environment variables.")

# --- Pydantic Schemas ---
class UserBase(BaseModel):
    vulnerability_type: str
    address: str
    phone_number: str
    has_guardian: bool = False
    guardian_phone_number: Optional[str] = None
    wants_info_call: bool = True

class UserCreate(UserBase):
    pass # Inherits all fields from UserBase

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True # Changed from orm_mode = True for Pydantic v2

class CallRequest(BaseModel):
    to: str
    message: str

class SmsRequest(BaseModel):
    to: str
    message: str

# --- NEW: Pydantic model based on Real Disaster API Output ---
class DisasterAlertData(BaseModel):
    SN: str             # 일련번호
    CRT_DT: str         # 생성일시 (문자열로 받음, 필요시 파싱)
    MSG_CN: str         # 메시지내용
    RCPTN_RGN_NM: str   # 수신지역명
    EMRG_STEP_NM: Optional[str] = None # 긴급단계명 (선택적)
    DST_SE_NM: str      # 재해구분명
    REG_YMD: Optional[str] = None      # 등록일자 (선택적)
    MDFCN_YMD: Optional[str] = None    # 수정일자 (선택적)
    
# --- CRUD Operations (simplified within app.py) ---
def get_user_by_phone(db: Session, phone_number: str):
    return db.query(User).filter(User.phone_number == phone_number).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate):
    db_user = User(**user.model_dump()) # Use model_dump() for Pydantic v2
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Helper Functions for Disaster Simulation Logic ---

def filter_target_users(disaster_alert: DisasterAlertData, db: Session) -> List[User]:
    """Filters users based on the disaster reception region."""
    target_location_keyword = None
    if disaster_alert.RCPTN_RGN_NM:
        parts = disaster_alert.RCPTN_RGN_NM.split()
        if len(parts) > 1:
            target_location_keyword = parts[-1]
            if '전체' in target_location_keyword or ',' in disaster_alert.RCPTN_RGN_NM:
                target_location_keyword = parts[0]
        else:
            target_location_keyword = disaster_alert.RCPTN_RGN_NM
    
    filtered_users = []
    if target_location_keyword:
        print(f"Filtering users in location containing: '{target_location_keyword}'")
        try:
            filtered_users = db.query(User).filter(User.address.contains(target_location_keyword)).all()
            print(f"Found {len(filtered_users)} users to notify.")
        except Exception as e:
            print(f"Error filtering users: {e}")
            # Return empty list on error or handle differently
            return [] 
    else:
        print("No location keyword to filter by from RCPTN_RGN_NM.")
        
    return filtered_users

# --- RAG Based Message Generation (Target Length ~60, No URL, No Ellipsis) ---
def generate_notification_messages(disaster_alert: DisasterAlertData, users: List[User]) -> List[dict]:
    """Generates structured notification messages using RAG, targeting ~60 chars, without URL or ellipsis on truncation."""
    notifications = []
    # Base alert in English, formatted concisely
    base_alert_info = f"{disaster_alert.DST_SE_NM} in {disaster_alert.RCPTN_RGN_NM}"
    print(f"Base Alert Info for RAG query: {base_alert_info}")

    # Adjust target length for RAG - VERY Short target
    target_rag_len = 60 
    total_max_len = 70 # Ensure total length doesn't exceed this

    rag_response_content = "" # Initialize with empty string
    if vectorstore is None:
        print("  WARNING: Vectorstore not available. Cannot generate RAG content.")
        # Fallback message (no URL)
        fallback_base = f"ALERT: {base_alert_info}. Follow local guidance."
        if len(fallback_base) > total_max_len: # Check against total max (70)
             # Hard truncate, no ellipsis
             rag_response_content = fallback_base[:total_max_len]
        else:
             rag_response_content = fallback_base

    else:
        try:
            upstage_api_key = os.getenv("UPSTAGE_API_KEY")
            llm = ChatUpstage(api_key=upstage_api_key, model="solar-pro")
            retriever = vectorstore.as_retriever()

            # --- Updated English Prompt (Targeting ~60 chars, EXTREME brevity) --- 
            prompt_template = f"""
            Generate an EXTREMELY concise emergency alert message in ENGLISH, strictly aiming for {target_rag_len} characters or less.

            Required Information (must be included):
            1. Disaster Type: [from {{question}}]
            2. Location: [from {{question}}]
            3. 1 brief CRITICAL immediate action.

            Context Documents (for action guidance):
            {{context}}

            Output ONLY the SMS-ready alert text. ABSOLUTELY NO EXTRA TEXT. BE EXTREMELY BRIEF.
            Example: "ALERT: Earthquake Seoul Seongdong-gu. Drop, Cover, Hold On."

            Alert Message (target ~{target_rag_len} chars):
            """
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False
            )
            
            query = base_alert_info
            print(f"  RAG Query: {query}")
            
            result = qa_chain.invoke({"query": query})
            rag_summary_text = result.get('result', '').strip()
            
            print(f"  RAG Generated Response (raw):\n{rag_summary_text}")

            # --- Simplified Truncation Logic (NO ELLIPSIS) --- 
            if not rag_summary_text: # If RAG failed or returned empty
                print("  WARNING: RAG generation resulted in empty string. Using fallback.")
                fallback_base = f"ALERT: {base_alert_info}. Check official sources."
                if len(fallback_base) > total_max_len: # Check against total max (70)
                     # Hard truncate, no ellipsis
                     rag_response_content = fallback_base[:total_max_len]
                else:
                     rag_response_content = fallback_base
            else:
                # Use the generated text directly
                rag_response_content = rag_summary_text
                
                # Final hard check to ensure total length doesn't exceed absolute max (70)
                if len(rag_response_content) > total_max_len:
                     print(f"  WARNING: Final message still exceeds {total_max_len} chars. Force truncating (hard cut). Len: {len(rag_response_content)}")
                     rag_response_content = rag_response_content[:total_max_len]

        except Exception as e:
            print(f"  ERROR during RAG generation: {e}")
            # Fallback message on error (no URL)
            fallback_base = f"ALERT: {base_alert_info}. Error getting details. Follow local guidance."
            if len(fallback_base) > total_max_len: # Check against total max (70)
                 # Hard truncate, no ellipsis
                 rag_response_content = fallback_base[:total_max_len]
            else:
                 rag_response_content = fallback_base

    # --- End RAG Generation --- 

    # Log the final generated message content and its length 
    logging.info(f"Final generated message content (Alert only, ~60 chars, no ellipsis):\n{rag_response_content}")
    logging.info(f"---> Final Alert message length: {len(rag_response_content)} chars")

    for user in users:
        # Message content no longer includes the URL here
        notifications.append({"user": user, "message": rag_response_content})
        
    return notifications

def send_twilio_notifications(notifications_to_send: List[dict]) -> dict:
    """Sends notifications via Twilio SMS (map URL first, then alert with delay) and returns counts."""
    successful_notifications = 0
    failed_notifications = 0
    
    # Check Twilio config
    if not client or not twilio_phone_number:
        print("  ERROR: Twilio client not configured. Cannot send SMS.")
        logging.error("Twilio client not configured. Cannot send SMS.") 
        return {"success": successful_notifications, "failed": len(notifications_to_send)} 

    # Construct the map URL message (needs base_url)
    # TODO: Get base URL from environment variable or config for flexibility
    base_url = "https://ff74-121-160-208-235.ngrok-free.app" 
    full_map_url = f"{base_url}/map/map_api"
    map_message_body = f"Nearby shelters/safety info: {full_map_url}" # Slightly shortened prefix

    print("--- SENDING REAL TWILIO NOTIFICATIONS (2-Step: Map URL + Alert) --- ")
    logging.info("--- Starting to send real Twilio notifications (Map URL + Alert) ---") 
    for item in notifications_to_send:
        user = item["user"]
        alert_message_body = item["message"] # This is the shorter alert message generated by RAG
        
        try:
            # 1. Send the MAP URL SMS first
            logging.info(f"  1 -> Attempting to send MAP URL SMS to {user.phone_number} (User ID: {user.id})")
            message1 = client.messages.create(
                body=map_message_body,
                from_=twilio_phone_number,
                to=user.phone_number
            )
            logging.info(f"    Map URL SMS sent successfully! SID: {message1.sid} to {user.phone_number}")

            # 2. If map URL SMS was successful, wait briefly and send the main alert message
            try:
                logging.info(f"  ... Waiting 0.5 seconds before sending alert SMS ...")
                time.sleep(0.5) # 0.5초 지연 추가
                
                logging.info(f"  2 -> Attempting to send ALERT SMS to {user.phone_number} (User ID: {user.id}). Content:\n{alert_message_body}")
                message2 = client.messages.create(
                    body=alert_message_body,
                    from_=twilio_phone_number,
                    to=user.phone_number
                )
                logging.info(f"    Alert SMS sent successfully! SID: {message2.sid} to {user.phone_number}")
                successful_notifications += 1 # Count success only if both messages are sent
                
            except Exception as e2:
                logging.error(f"    ERROR sending ALERT SMS to {user.phone_number} after successful map URL: {e2}", exc_info=True) 
                failed_notifications += 1 # Map URL sent, but alert failed

        except Exception as e1:
            # This now represents failure sending the MAP URL
            logging.error(f"    ERROR sending MAP URL SMS to {user.phone_number}: {e1}", exc_info=True) 
            failed_notifications += 1 # Map URL failed, didn't attempt alert
            
    logging.info(f"--- Notification sending complete (Success pairs: {successful_notifications}, Failed attempts: {failed_notifications}) ---")
    return {"success": successful_notifications, "failed": failed_notifications}

# --- FastAPI Endpoints ---

# Prefixing API routes for better organization
api_router = APIRouter(prefix="/api")

@api_router.post("/users/", response_model=UserResponse, tags=["Users"])
def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_phone(db, phone_number=user.phone_number)
    if db_user:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    # DEBUG: Print received data before creating
    print("Received user data in endpoint:", user.model_dump())
    try:
        created_user = create_user(db=db, user=user)
        print(f"Successfully created user with ID: {created_user.id}")
        return created_user
    except Exception as e:
        print(f"Error during user creation or DB commit: {e}")
        # Rollback in case of error during commit
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error during user creation.")

@api_router.get("/users/", response_model=List[UserResponse], tags=["Users"])
def read_users_endpoint(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = get_users(db, skip=skip, limit=limit)
    return users

@api_router.get("/users/{phone_number}", response_model=UserResponse, tags=["Users"])
def read_user_endpoint(phone_number: str, db: Session = Depends(get_db)):
    db_user = get_user_by_phone(db, phone_number=phone_number)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@api_router.post("/send_sms", tags=["Twilio"])
async def send_sms(sms_request: SmsRequest):
    if not client or not twilio_phone_number:
        raise HTTPException(status_code=500, detail="Twilio client not configured")
    try:
        message = client.messages.create(
            body=sms_request.message,
            from_=twilio_phone_number,
            to=sms_request.to
        )
        return {"status": "SMS sent", "sid": message.sid}
    except Exception as e:
        print(f"Error sending SMS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/make_call", tags=["Twilio"])
async def make_call(call_request: CallRequest):
    if not client or not twilio_phone_number:
        raise HTTPException(status_code=500, detail="Twilio client not configured")
    try:
        call = client.calls.create(
                            twiml=f'<Response><Say language="ko-KR">{call_request.message}</Say></Response>',
                            to=call_request.to,
                            from_=twilio_phone_number
                        )
        return {"status": "Call initiated", "sid": call.sid}
    except Exception as e:
        print(f"Error making call: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint to proxy Safety Data API calls
@api_router.get("/shelters", tags=["External APIs"])
async def get_shelters(region: str, page: int = 1, rows: int = 10):
    if not safety_data_service_key:
        raise HTTPException(status_code=500, detail="Safety Data API key not configured")

    api_url = "https://www.safetydata.go.kr/V2/api/DSSP-IF-10941"
    params = {
        "serviceKey": safety_data_service_key,
        "region": region,
        "numOfRows": rows,
        "pageNo": page,
        "returnType": "json" # Ensure API returns JSON
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(api_url, params=params, timeout=10.0)
            response.raise_for_status() # Raise exception for bad status codes
            data = response.json()
            # Check for API-specific error structure if known
            if data.get("response", {}).get("header", {}).get("resultCode") != "00":
                 error_msg = data.get("response", {}).get("header", {}).get("resultMsg", "Unknown API error")
                 raise HTTPException(status_code=500, detail=f"Safety API Error: {error_msg}")
            return data # Return the successful JSON response from the external API
        except httpx.RequestError as exc:
            print(f"Error requesting Safety Data API: {exc}")
            raise HTTPException(status_code=502, detail="Failed to connect to Safety Data API")
        except httpx.HTTPStatusError as exc:
            print(f"Error response {exc.response.status_code} from Safety Data API: {exc.response.text}")
            raise HTTPException(status_code=exc.response.status_code, detail="Error received from Safety Data API")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred while fetching shelter data")

# --- Refactored endpoint for simulating disaster ---
@api_router.post("/simulate_disaster", tags=["Disaster Simulation"])
async def simulate_disaster(disaster_alert: DisasterAlertData, db: Session = Depends(get_db)):
    print("\n--- STARTING DISASTER SIMULATION (Refactored) ---")
    print(f"Received Alert SN: {disaster_alert.SN} for Region: {disaster_alert.RCPTN_RGN_NM}")
    
    # 1. Filter users
    users_to_notify = filter_target_users(disaster_alert, db)
    
    # 2. Generate messages
    notifications = generate_notification_messages(disaster_alert, users_to_notify)
    
    # 3. Send notifications
    send_results = send_twilio_notifications(notifications)
    
    print("--- SIMULATION PROCESSING COMPLETE ---")

    return {
        "status": "Disaster simulation processed (Refactored, Twilio SMS Attempted)", 
        "received_data": disaster_alert,
        "filtered_user_count": len(users_to_notify),
        "successful_notifications": send_results["success"],
        "failed_notifications": send_results["failed"]
    }

app.include_router(api_router)

# --- Twilio Webhook Endpoints ---
twilio_router = APIRouter(prefix="/twilio")

@twilio_router.post("/sms", tags=["Twilio Webhooks"])
async def sms_reply(Body: str = Form(...)):
    msg = Body
    print(f"\n**\n📩 받은 메시지: {msg}\n**\n")
    resp = MessagingResponse()
    resp.message(f"응답: '{msg}' 잘 받았어요!")
    return Response(content=str(resp), media_type="application/xml")

@twilio_router.post("/voice", tags=["Twilio Webhooks"])
async def voice(request: Request):
    resp = VoiceResponse()
    gather = Gather(input='speech', action='/twilio/handle-gather', method='POST', language='ko-KR', speechTimeout='auto')
    gather.say("안녕하세요. 재난 안전 시스템입니다. 신고하시려면 신고, 문의사항이 있으시면 문의 라고 말씀해주세요.", voice='Polly.Seoyeon')
    resp.append(gather)
    resp.redirect('/twilio/voice')
    return Response(content=str(resp), media_type="application/xml")

@twilio_router.post("/handle-gather", tags=["Twilio Webhooks"])
async def handle_gather(request: Request, db: Session = Depends(get_db)):
    resp = VoiceResponse()
    form = await request.form()
    speech_result = form.get('SpeechResult', '').strip()
    caller_phone_number = form.get('From')
    print(f"\n**\n📞 Incoming call from: {caller_phone_number}\n📢 Speech Result: {speech_result}\n**\n")
    if '신고' in speech_result:
        resp.say("신고 접수를 시작하겠습니다. 필요한 정보를 말씀해주세요.", voice='Polly.Seoyeon')
        resp.hangup()
    elif '문의' in speech_result:
        resp.say("문의사항 접수를 위해 잠시 후 상담원을 연결해 드리겠습니다.", voice='Polly.Seoyeon')
        resp.hangup()
    else:
        resp.say("죄송합니다, 이해하지 못했습니다. 다시 시도해주세요.", voice='Polly.Seoyeon')
        resp.redirect('/twilio/voice')
    return Response(content=str(resp), media_type="application/xml")

app.include_router(twilio_router)

# --- Frontend and Map Serving Endpoints ---

@app.get("/map/{map_file_name}", response_class=HTMLResponse, tags=["Serving"])
async def serve_map(request: Request, map_file_name: str):
    allowed_maps = ["map_api", "map_wide", "shelter_temp", "temp"]
    base_map_name = map_file_name.replace(".html", "")
    if base_map_name not in allowed_maps:
        raise HTTPException(status_code=404, detail="Map not found")

    template_name = f"{base_map_name}.html"

    # --- DEBUGGING: Print the key value being used ---
    print(f"DEBUG: Using KAKAO_MAP_APP_KEY: {kakao_map_app_key}")
    # -------------------------------------------------

    context = {
        "request": request,
        "kakao_map_app_key": kakao_map_app_key
    }
    try:
        return templates.TemplateResponse(template_name, context)
    except Exception as e:
        print(f"Error rendering template {template_name}: {e}")
        raise HTTPException(status_code=404, detail="Map template not found or error rendering")

@app.get("/", response_class=HTMLResponse, tags=["Serving"])
async def serve_frontend_app():
    index_path = static_dir / "app" / "index.html"
    if not index_path.is_file():
        return HTMLResponse(content="Frontend not built or index.html not found in static/app", status_code=404)
    return HTMLResponse(content=index_path.read_text(), status_code=200)

# --- Database and RAG Setup Function (called on startup) ---
def setup_database_and_rag(app: FastAPI):
    global vectorstore # Declare intent to modify the global variable
    print("Attempting to create database engine and tables...")
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./flask_twilio_demo/users.db") 
        if DATABASE_URL.startswith("sqlite:///./"): 
            project_root = pathlib.Path(__file__).parent.parent
            db_relative_path = DATABASE_URL.split("///./", 1)[1]
            db_path = project_root / db_relative_path
            db_path.parent.mkdir(parents=True, exist_ok=True) 
            DATABASE_URL = f"sqlite:///{db_path.resolve()}"
        print(f"Using DATABASE_URL: {DATABASE_URL}")
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        app.state.db_engine = engine
        app.state.SessionLocal = SessionLocal
        print(f"Database Engine created for: {DATABASE_URL}")
        Base.metadata.create_all(bind=engine)
        print("Database tables checked/created successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR during database setup: {e}")
        # Exit or handle failure appropriately
        return # Stop further setup if DB fails

    # --- Build/Load FAISS Vectorstore --- 
    print("Attempting to load or build FAISS vectorstore...")
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    if not upstage_api_key or upstage_api_key == 'YOUR_UPSTAGE_API_KEY':
        print("  WARNING: UPSTAGE_API_KEY not found or not set in .env. RAG will not work.")
        return # Cannot proceed without API key
        
    project_root = pathlib.Path(__file__).parent.parent
    vectorstore_full_path = project_root / VECTORSTORE_PATH
    how_to_dir = project_root / "how-to-s"

    try:
        # --- Specify the embedding model --- 
        embeddings = UpstageEmbeddings(api_key=upstage_api_key, model="solar-embedding-1-large")
        # -----------------------------------

        if vectorstore_full_path.exists():
            print(f"Loading existing vectorstore from: {vectorstore_full_path}")
            vectorstore = FAISS.load_local(str(vectorstore_full_path), embeddings, allow_dangerous_deserialization=True)
            print("Vectorstore loaded successfully.")
        else:
            print(f"Building new vectorstore from .txt files in: {how_to_dir}")
            if not how_to_dir.is_dir():
                 print(f"  ERROR: Directory not found: {how_to_dir}")
                 return
                 
            documents = []
            for txt_file in how_to_dir.glob("*.txt"):
                try:
                    print(f"  Reading: {txt_file.name}")
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read()
                        # Add filename as metadata for potential future use
                        documents.append(Document(page_content=text, metadata={"source": txt_file.name})) 
                except Exception as e:
                    print(f"  Error reading file {txt_file.name}: {e}")
            
            if not documents:
                print("  ERROR: No documents found or read from how-to-s directory.")
                return
                
            print(f"Splitting {len(documents)} documents...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks.")
            
            print("Embedding documents and creating FAISS index...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            print("Saving vectorstore...")
            vectorstore.save_local(str(vectorstore_full_path))
            print(f"Vectorstore built and saved to: {vectorstore_full_path}")
            
    except Exception as e:
        print(f"CRITICAL ERROR during vectorstore setup: {e}")
        vectorstore = None # Ensure vectorstore is None on error

# --- Main Execution Block (No table creation here anymore) ---
if __name__ == "__main__":
    # create_db_tables() # Removed from here
    print("Starting Uvicorn server (directly running script)...") 
    # Note: Running script directly might not use reload properly
    # It's better to run with 'uvicorn flask_twilio_demo.app:app --reload'
    uvicorn.run(app, host='0.0.0.0', port=30000)