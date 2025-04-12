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
import urllib.parse # URL 인코딩을 위해 임포트

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

# --- Explicitly load .env from the SAME directory as app.py --- 
current_dir = pathlib.Path(__file__).parent
dotenv_path = current_dir / '.env' # Corrected path calculation
logging.info(f"Attempting to load environment variables from: {dotenv_path}")
# Make sure python-dotenv is installed: pip install python-dotenv
load_success = load_dotenv(dotenv_path=dotenv_path, override=True) # Use override=True just in case
logging.info(f".env file load success: {load_success} (Path Exists: {dotenv_path.exists()})") # Add existence check
# --------------------------------------------------------------

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
    is_visually_impaired: bool = False

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
            # Select only necessary columns explicitly to avoid schema mismatch errors
            query_result = db.query(
                User.id, 
                User.phone_number, 
                User.wants_info_call, 
                User.address # Keep address if needed elsewhere, or remove
                # We don't strictly NEED is_visually_impaired for the current logic flow after this point
            ).filter(User.address.contains(target_location_keyword)).all()
            
            # Reconstruct User objects (or adapt downstream code to use tuples)
            # For simplicity, let's reconstruct partial User objects needed for separation
            filtered_users = [
                User(id=row.id, phone_number=row.phone_number, wants_info_call=row.wants_info_call, address=row.address) 
                for row in query_result
            ]
            print(f"Found {len(filtered_users)} users to notify (selected specific columns).")
        except Exception as e:
            print(f"Error filtering users: {e}")
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

# --- RAG Based VOICE Message Generation (ENGLISH) ---
def generate_voice_alert_message(disaster_alert: DisasterAlertData) -> str:
    """
    Generates an English voice alert message using RAG, suitable for TTS delivery.
    Includes disaster info, key actions, and prompts for user response ('Report').
    """
    # Base alert info in English
    base_alert_info = f"{disaster_alert.DST_SE_NM} in {disaster_alert.RCPTN_RGN_NM}"
    print(f"Base Alert Info for VOICE RAG query: {base_alert_info}")

    # Target length for the main RAG content (excluding the final prompt)
    target_rag_len = 150 
    total_max_len = 200 
    # Fixed ENGLISH prompt to ask the user to say 'Report'
    report_prompt_text = " If you need to report something, please say 'Report'." 

    voice_message_content = ""

    if vectorstore is None:
        print("  WARNING: Vectorstore not available for VOICE RAG. Using fallback.")
        # English Fallback
        fallback_base = f"Emergency alert regarding {base_alert_info}. Please follow guidance from local authorities."
        if len(fallback_base) > (total_max_len - len(report_prompt_text)):
            # Truncate base if too long, keeping space for prompt
            fallback_base = fallback_base[:(total_max_len - len(report_prompt_text) - 3)] + "..."
        voice_message_content = fallback_base + report_prompt_text
        return voice_message_content

    try:
        upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        if not upstage_api_key:
             raise ValueError("UPSTAGE_API_KEY not found.")
             
        llm = ChatUpstage(api_key=upstage_api_key, model="solar-pro")
        retriever = vectorstore.as_retriever()

        # --- Voice-Optimized English Prompt (Remains the same) ---
        prompt_template = f"""
        Generate a clear and concise emergency alert message in ENGLISH, suitable for voice delivery (Text-to-Speech). 
        Aim for approximately {target_rag_len} characters for this main alert portion.

        Include:
        1. Disaster Type and Location: [from {{question}}]
        2. 1-2 CRITICAL immediate actions based on the context documents. Be direct.

        Context Documents (for action guidance):
        {{context}}

        Output ONLY the core alert message text. Do NOT include any introductory phrases like "Here is the alert".
        Example: "ALERT: Heavy rain advisory for Seoul Gangnam-gu. Avoid low-lying areas. Monitor updates closely."

        Core Alert Message (target ~{target_rag_len} chars):
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
        print(f"  Voice RAG Query: {query}")
        
        result = qa_chain.invoke({"query": query})
        rag_summary_text = result.get('result', '').strip()
        
        print(f"  Voice RAG Generated Response (raw core text):\n{rag_summary_text}")

        # --- Assemble Final Voice Message (with English prompt) ---
        if not rag_summary_text:
            print("  WARNING: Voice RAG generation resulted in empty string. Using fallback.")
            # English Fallback
            fallback_base = f"ALERT: {base_alert_info}. Check official sources for guidance."
            if len(fallback_base) > (total_max_len - len(report_prompt_text)):
                 fallback_base = fallback_base[:(total_max_len - len(report_prompt_text) - 3)] + "..."
            voice_message_content = fallback_base + report_prompt_text
        else:
            # Truncate the RAG part if it exceeds its target length
            if len(rag_summary_text) > target_rag_len:
                print(f"  WARNING: Voice RAG core text exceeded target {target_rag_len} chars ({len(rag_summary_text)}). Hard truncating.")
                rag_summary_text = rag_summary_text[:target_rag_len] 
            
            # Combine core message and the fixed English reporting prompt
            voice_message_content = rag_summary_text + report_prompt_text
            
            # Final check on total length
            if len(voice_message_content) > total_max_len:
                 print(f"  WARNING: Final VOICE message still exceeds {total_max_len} chars ({len(voice_message_content)}). Force truncating final.")
                 voice_message_content = voice_message_content[:total_max_len]

    except Exception as e:
        print(f"  ERROR during Voice RAG generation: {e}")
        # English Fallback on error
        fallback_base = f"ALERT: {base_alert_info}. Error retrieving details. Follow local guidance."
        if len(fallback_base) > (total_max_len - len(report_prompt_text)):
             fallback_base = fallback_base[:(total_max_len - len(report_prompt_text) - 3)] + "..."
        voice_message_content = fallback_base + report_prompt_text

    # Log the final generated message content and its length
    logging.info(f"Final generated ENGLISH VOICE message content:\n{voice_message_content}")
    logging.info(f"---> Final VOICE message length: {len(voice_message_content)} chars")
    
    return voice_message_content

# --- Function to send Twilio SMS (Map URL first, then alert with delay) --- 
# Revert signature, use hardcoded base_url inside
def send_twilio_notifications(notifications_to_send: List[dict]) -> dict:
    """Sends notifications via Twilio SMS (map URL first, then alert with delay) and returns counts."""
    successful_notifications = 0
    failed_notifications = 0
    
    # Check Twilio config
    if not client or not twilio_phone_number:
        print("  ERROR: Twilio client not configured. Cannot send SMS.")
        logging.error("Twilio client not configured. Cannot send SMS.") 
        return {"success": successful_notifications, "failed": len(notifications_to_send)} 

    # --- HARDCODED base_url --- 
    hardcoded_base_url = "https://5eaf-121-160-208-235.ngrok-free.app"
    # --------------------------
    full_map_url = f"{hardcoded_base_url}/map/map_api"
    map_message_body = f"Nearby shelters/safety info: {full_map_url}" 

    print(f"--- SENDING REAL TWILIO NOTIFICATIONS (2-Step: Map URL: {full_map_url} + Alert) --- ")
    logging.info(f"--- Starting to send real Twilio notifications (Map URL + Alert) using hardcoded base URL: {hardcoded_base_url} ---") 
    for item in notifications_to_send:
        user = item["user"]
        alert_message_body = item["message"]
        
        try:
            # 1. Send the MAP URL SMS first
            logging.info(f"  1 -> Attempting to send MAP URL SMS ({map_message_body}) to {user.phone_number} (User ID: {user.id})")
            message1 = client.messages.create(
                body=map_message_body,
                from_=twilio_phone_number,
                to=user.phone_number
            )
            logging.info(f"    Map URL SMS sent successfully! SID: {message1.sid} to {user.phone_number}")

            # 2. If map URL SMS was successful, wait briefly and send the main alert message
            try:
                logging.info(f"  ... Waiting 0.5 seconds before sending alert SMS ...")
                time.sleep(0.5)
                
                logging.info(f"  2 -> Attempting to send ALERT SMS to {user.phone_number} (User ID: {user.id}). Content:\n{alert_message_body}")
                message2 = client.messages.create(
                    body=alert_message_body,
                    from_=twilio_phone_number,
                    to=user.phone_number
                )
                logging.info(f"    Alert SMS sent successfully! SID: {message2.sid} to {user.phone_number}")
                successful_notifications += 1
                
            except Exception as e2:
                logging.error(f"    ERROR sending ALERT SMS to {user.phone_number} after successful map URL: {e2}", exc_info=True) 
                failed_notifications += 1

        except Exception as e1:
            logging.error(f"    ERROR sending MAP URL SMS to {user.phone_number}: {e1}", exc_info=True) 
            failed_notifications += 1
            
    logging.info(f"--- Notification sending complete (Success pairs: {successful_notifications}, Failed attempts: {failed_notifications}) ---")
    return {"success": successful_notifications, "failed": failed_notifications}

# --- Function to send Twilio VOICE alerts (ENGLISH) --- 
# Revert signature, use hardcoded base_url inside
def send_twilio_voice_alerts(notifications_to_send: List[dict]) -> dict:
    """Sends English voice alerts via Twilio Call API using generated TwiML, passing alert context."""
    successful_calls = 0
    failed_calls = 0
    
    if not client or not twilio_phone_number:
        print("  ERROR: Twilio client not configured. Cannot make calls.")
        logging.error("Twilio client not configured for making calls.") 
        return {"success": successful_calls, "failed": len(notifications_to_send)}

    # --- HARDCODED base_url --- 
    hardcoded_base_url = "https://5eaf-121-160-208-235.ngrok-free.app"
    # --------------------------

    print(f"--- SENDING ENGLISH VOICE CALL ALERTS using hardcoded base URL: {hardcoded_base_url} --- ")
    logging.info(f"--- Starting to send {len(notifications_to_send)} English voice call alerts ---") 
    for item in notifications_to_send:
        user = item["user"]
        message_text = item["message_text"] 
        
        encoded_message = urllib.parse.quote(message_text)
        # Construct full action URL using the hardcoded base_url
        action_url = f'{hardcoded_base_url}/twilio/handle-voice-alert-response?alert_message={encoded_message}'
        
        try:
            # Construct TwiML dynamically for English
            response = VoiceResponse()
            gather = Gather(input='speech', 
                            action=action_url, # Use the FULL action URL constructed with hardcoded base
                            method='POST', 
                            language='en-US', 
                            speechTimeout='auto',
                            actionOnEmptyResult=True,
                            hints="report")
            gather.say(message_text, voice='Polly.Joanna', language="en-US")
            response.append(gather)
            
            response.say("Processing your response or ending call if none given.", voice='Polly.Joanna', language="en-US")
            response.hangup()
            
            twiml_content = str(response)
            
            logging.info(f"  -> Attempting call to {user.phone_number} (User ID: {user.id}) Action URL: {action_url}")

            call = client.calls.create(
                twiml=twiml_content,
                to=user.phone_number,
                from_=twilio_phone_number
            )
            logging.info(f"    English Voice call initiated successfully! SID: {call.sid} to {user.phone_number}")
            successful_calls += 1
            
            time.sleep(0.2)

        except Exception as e:
            logging.error(f"    ERROR initiating English voice call to {user.phone_number}: {e}", exc_info=True) 
            failed_calls += 1
            
    logging.info(f"--- English Voice call initiation complete (Success: {successful_calls}, Failed: {failed_calls}) ---")
    return {"success": successful_calls, "failed": failed_calls}

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
                            twiml=f'<Response><Say language="en-EN">{call_request.message}</Say></Response>',
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
    
    # 1. Filter users based on location
    users_to_notify = filter_target_users(disaster_alert, db)
    
    # --- Separate users by notification type (Based on wants_info_call only) --- 
    sms_users = []
    voice_users = []
    print("Separating users based on wants_info_call preference:")
    for user in users_to_notify:
        # Voice call if user wants the call
        if user.wants_info_call:
            voice_users.append(user)
            print(f"  - User {user.id} ({user.phone_number}): Wants info call -> VOICE")
        # Otherwise, send SMS
        else:
            sms_users.append(user)
            print(f"  - User {user.id} ({user.phone_number}): Does NOT want info call -> SMS")
                
    print(f"Separation complete: {len(sms_users)} for SMS, {len(voice_users)} for Voice Call")
    # ----------------------------------------------------------------------------
    
    # 2. Generate messages (adapt for SMS/Voice)
    # Generate SMS messages for users who don't want calls
    sms_notifications = []
    if sms_users:
        print("Generating SMS messages...")
        sms_notifications = generate_notification_messages(disaster_alert, sms_users)
        
    # Generate Voice messages for users who want calls
    voice_notifications = [] # This will store user and the voice message text
    if voice_users:
        print("Generating Voice messages...")
        # Generate the voice message text once for the disaster alert
        voice_alert_text = generate_voice_alert_message(disaster_alert) 
        # Assign the same message text to all voice users for this alert
        voice_notifications = [{"user": user, "message_text": voice_alert_text} for user in voice_users]
        # logging.warning("Voice message generation not yet implemented.") # Placeholder Removed

    # 3. Send notifications (adapt for SMS/Voice)
    sms_send_results = {"success": 0, "failed": 0}
    if sms_notifications:
        print("Sending SMS notifications...")
        sms_send_results = send_twilio_notifications(sms_notifications)

    voice_send_results = {"success": 0, "failed": 0}
    if voice_notifications: # This check will now pass if there are voice users
        print("Sending Voice call notifications...")
        # Call the newly implemented function
        voice_send_results = send_twilio_voice_alerts(voice_notifications)
        # logging.warning("Voice call sending not yet implemented.") # Placeholder Removed
    
    print("--- SIMULATION PROCESSING COMPLETE ---")

    return {
        "status": "Disaster simulation processed", 
        "received_data": disaster_alert,
        "total_filtered_user_count": len(users_to_notify),
        "sms_user_count": len(sms_users),
        "voice_user_count": len(voice_users),
        "successful_sms_notifications": sms_send_results["success"],
        "failed_sms_notifications": sms_send_results["failed"],
        "successful_voice_calls": voice_send_results["success"],
        "failed_voice_calls": voice_send_results["failed"]
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

# --- Webhook Endpoint for Voice Alert Response (ENGLISH) --- 
@twilio_router.post("/handle-voice-alert-response", tags=["Twilio Webhooks"])
# Revert signature to only take request, parse query param manually
async def handle_voice_alert_response(request: Request):
    """Handles the user's English speech input, including the original alert message context."""
    # Log entry immediately
    logging.info(f"--- Entered /handle-voice-alert-response (Manual Query Param Parsing) --- ") 
    logging.info(f"Request URL: {request.url}")
    
    resp = VoiceResponse()
    form = None
    speech_result = "(SpeechResult not found or error)"
    speech_confidence = "(Confidence not found)"
    caller_phone_number = "(From not found)"
    call_sid = "(CallSid not found)"
    original_alert = "(Original alert message not available)"
    alert_message_raw = "(Query param not found)"

    # Manually extract query parameter
    try:
        alert_message_raw = request.query_params.get('alert_message')
        logging.info(f"Manually extracted alert_message query param (raw): {alert_message_raw}")
    except Exception as e:
        logging.error(f"Error extracting query parameters: {e}", exc_info=True)
        # Continue processing if possible, alert context might be missing
    
    try:
        logging.info("Attempting to read form data from request...")
        form = await request.form()
        logging.info(f"Raw Form data received: {form}")
        
        speech_result = form.get('SpeechResult', '').strip().lower() 
        speech_confidence = form.get('Confidence', 'N/A')
        logging.info(f"*** Initial Speech Result Extracted: '{speech_result}' (Confidence: {speech_confidence}) ***")

        caller_phone_number = form.get('From')
        call_sid = form.get('CallSid')
        logging.info(f"Extracted Form Data - From: {caller_phone_number}, CallSid: {call_sid}")

    except Exception as e:
        logging.error(f"Error reading form data: {e}", exc_info=True)
        resp.say("An error occurred processing your response.", voice='Polly.Joanna', language="en-US")
        resp.hangup()
        return Response(content=str(resp), media_type="application/xml")
    
    # Decode the alert message if it was found
    if alert_message_raw and alert_message_raw != "(Query param not found)":
        try:
            original_alert = urllib.parse.unquote(alert_message_raw)
            logging.info(f"Successfully decoded alert_message: {original_alert}")
        except Exception as e:
            logging.warning(f"Could not decode alert_message parameter: {e}")
            original_alert = "(Error decoding alert message)"
    else:
        logging.info("alert_message query parameter was not found or empty.")
            
    logging.info(f"--- Processing voice response --- ")
    logging.info(f"  Caller: {caller_phone_number}")
    logging.info(f"  Call SID: {call_sid}")
    logging.info(f"  Speech Result (before check): '{speech_result}' (Confidence: {speech_confidence})") 
    logging.info(f"  Original Alert Context: {original_alert}")
    
    # Check if the user said 'report' (case-insensitive)
    logging.info(f"Checking for 'report' keyword in speech: '{speech_result}'")
    if 'report' in speech_result:
        logging.info(f"--> 'report' DETECTED for {caller_phone_number}.")
        resp.say("Report request acknowledged. Ending call.", voice='Polly.Joanna', language="en-US")
        logging.info(f"ACTION NEEDED: User {caller_phone_number} requested report regarding alert: {original_alert}")
        print(f"ACTION NEEDED: User {caller_phone_number} requested report regarding alert: {original_alert}") 
    else:
        logging.info(f"--> 'report' NOT detected for {caller_phone_number}.")
        resp.say("Okay. Ending call.", voice='Polly.Joanna', language="en-US")
        
    logging.info("Appending Hangup TwiML.")
    resp.hangup()
    
    final_twiml = str(resp)
    logging.info(f"Generated final TwiML response: {final_twiml}")
    logging.info(f"--- Exiting /handle-voice-alert-response --- ")
    return Response(content=final_twiml, media_type="application/xml")

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