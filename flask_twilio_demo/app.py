import os
from fastapi import FastAPI, Form, Request, Response, Depends, HTTPException, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import pathlib
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
import time # time 모듈 임포트
import urllib.parse # URL 인코딩을 위해 임포트
from datetime import datetime
import re # Import regex module
import math
from math import radians, cos, sin, asin, sqrt # Import math functions for Haversine
import anthropic # Import Anthropic library

from . import models
from .models import Base, engine, User, get_db

# --- Langchain Imports ---
from langchain.schema import Document, HumanMessage
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
# Allow requests from the React dev server AND potentially ngrok or any origin for debugging
origins = [
    "http://localhost:3000", # React default dev port
    # Add other origins if needed, e.g., your frontend production URL
    "*" # Allow all origins for debugging - REMOVE/RESTRICT in production!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"], # Allow relevant methods
    allow_headers=["*"], # Allow all headers
)

# --- REVISED Static files mounting --- 
# Serve files from the 'static' sub-directory within the main static folder
root_static_dir = pathlib.Path(__file__).parent / "static"
build_static_dir = root_static_dir / "static" # Path to the nested static dir (e.g., static/static)

if build_static_dir.is_dir():
    # Mount the nested static directory (containing CSS, JS) to /static URL path
    app.mount("/static", StaticFiles(directory=build_static_dir), name="static_assets")
    logging.info(f"Mounted static assets from: {build_static_dir} at /static")
else:
    logging.warning(f"Build output's static directory not found at: {build_static_dir}. Static assets might not load.")
    # Fallback or alternative: Mount the root static dir if nested doesn't exist?
    # app.mount("/static", StaticFiles(directory=root_static_dir), name="static_root_fallback")

# Configure Jinja2 templates
# templates_dir = static_dir / "maps" # 경로 오류 수정: root_static_dir 사용
templates_dir = root_static_dir / "maps"
if not templates_dir.is_dir():
     logging.warning(f"Jinja2 templates directory not found at {templates_dir}")
templates = Jinja2Templates(directory=templates_dir)

# Twilio credentials from .env file
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
emergency_phone_number = os.getenv('EMERGENCY_PHONE_NUMBER') # Load emergency number
kakao_map_app_key = os.getenv('KAKAO_MAP_APP_KEY')
safety_data_service_key = os.getenv('SAFETY_DATA_SERVICE_KEY')
kakao_rest_api_key = os.getenv('KAKAO_REST_API_KEY') # Also load Kakao REST key
upstage_api_key = os.getenv('UPSTAGE_API_KEY') # Also load Upstage key
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY') # Also load Anthropic key

# Log loaded keys (mask sensitive ones)
logging.info(f"TWILIO_ACCOUNT_SID loaded: {'Yes' if account_sid else 'No'}")
logging.info(f"TWILIO_AUTH_TOKEN loaded: {'Yes' if auth_token else 'No'}")
logging.info(f"TWILIO_PHONE_NUMBER loaded: {'Yes' if twilio_phone_number else 'No'}")
logging.info(f"EMERGENCY_PHONE_NUMBER loaded: {'Yes' if emergency_phone_number else 'No'}") # Log emergency number load status
logging.info(f"KAKAO_MAP_APP_KEY loaded: {'Yes' if kakao_map_app_key else 'No'}")
logging.info(f"KAKAO_REST_API_KEY loaded: {'Yes' if kakao_rest_api_key else 'No'}")
logging.info(f"SAFETY_DATA_SERVICE_KEY loaded: {'Yes' if safety_data_service_key else 'No'}")
logging.info(f"UPSTAGE_API_KEY loaded: {'Yes' if upstage_api_key else 'No'}")
logging.info(f"ANTHROPIC_API_KEY loaded: {'Yes' if anthropic_api_key else 'No'}")

# Check if credentials are loaded
# Modify check to also include emergency_phone_number if critical
if not account_sid or not auth_token or not twilio_phone_number:
    # Optionally, you might want to handle this more gracefully
    # depending on whether Twilio functionality is always required.
    print("Warning: Twilio credentials not fully found in .env file. Twilio features may not work.")
    # raise ValueError("Twilio credentials not found in .env file")

# Check if emergency phone number is loaded - log warning if not
if not emergency_phone_number:
    logging.warning("EMERGENCY_PHONE_NUMBER not found in .env file. Emergency call feature will be disabled.")

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
    vulnerability_type: str = Field(..., alias='personType')
    address: Optional[str] = None
    phone_number: str = Field(..., alias='phone')
    has_guardian: bool = Field(False, alias='hasGuardian')
    guardian_phone_number: Optional[str] = Field(None, alias='guardianPhone')
    wants_info_call: bool = Field(True, alias='needCall')
    preferred_language: Optional[str] = Field(None, alias='preferredLanguage')

    class Config:
        populate_by_name = True
        from_attributes = True

class UserCreate(UserBase):
    pass # Inherits all fields from UserBase

class UserResponse(UserBase):
    id: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    last_voice_response: Optional[str] = None # Add field to response
    last_response_timestamp: Optional[datetime] = None # Add field to response
    # preferred_language is inherited from UserBase

    class Config:
        populate_by_name = True
        from_attributes = True # Changed from orm_mode = True for Pydantic v2

# New schema for detailed user response including last response
class UserResponseWithDetails(UserResponse):
    pass # Inherits all fields including the ones added to UserResponse

# New schema for dashboard summary
class RegionSummary(BaseModel):
    region_name: str
    summary: str

# --- Disaster Alert Data Schema (Re-added) ---
class DisasterAlertData(BaseModel):
    SN: str             # 일련번화
    CRT_DT: str         # 생성일시 (문자열로 받음, 필요시 파싱)
    MSG_CN: str         # 메시지내용
    RCPTN_RGN_NM: str   # 수신지역명
    EMRG_STEP_NM: Optional[str] = None # 긴급단계명 (선택적)
    DST_SE_NM: str      # 재해구분명
    REG_YMD: Optional[str] = None      # 등록일자 (선택적)
    MDFCN_YMD: Optional[str] = None    # 수정일자 (선택적)
    latitude: Optional[float] = None    # 위도 추가
    longitude: Optional[float] = None    # 경도 추가

class CallRequest(BaseModel):
    to: str

class SmsRequest(BaseModel):
    to: str
    message: str

# --- NEW Pydantic Schemas for Translation ---
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: Optional[str] = None
    error: Optional[str] = None
# -----------------------------------------

# --- CRUD Operations (simplified within app.py) ---
def get_user_by_phone(db: Session, phone_number: str):
    return db.query(User).filter(User.phone_number == phone_number).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()

# Modified create_user to accept and store coordinates
def create_user(db: Session, user_data: Dict):
    db_user = User(**user_data)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Haversine Calculation --- 
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the earth."""
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# --- User Filtering by Location --- 
def filter_target_users_by_location(db: Session, disaster_lat: float, disaster_lon: float, radius_km: int):
    """Filters users within a given radius from a disaster location."""
    users = db.query(User).all()
    target_users = []
    for user in users:
        if user.latitude is not None and user.longitude is not None:
            try:
                distance = haversine(disaster_lat, disaster_lon, user.latitude, user.longitude)
                if distance <= radius_km:
                    target_users.append(user)
            except ValueError as e:
                # Log potential math domain errors (e.g., from invalid coordinates)
                logging.error(f"Haversine calculation error for user {user.id}: {e}")
        else:
            # Log users skipped due to missing coordinates
            logging.warning(f"User {user.id} skipped due to missing coordinates (lat={user.latitude}, lon={user.longitude}).")
    return target_users

# --- Korean to English Disaster Type Mapping ---
DISASTER_TYPE_MAP_KO_EN = {
    "지진": "Earthquake",
    "태풍": "Typhoon",
    "호우": "Heavy Rain",
    "홍수": "Flood",
    "강풍": "Strong Wind",
    "풍랑": "High Waves",
    "대설": "Heavy Snow",
    "한파": "Cold Wave",
    "폭염": "Heat Wave",
    "가뭄": "Drought",
    "황사": "Yellow Dust",
    "화재": "Fire",
    "산불": "Wildfire",
    "미세먼지": "Fine Dust",
}

# --- Language Code Mappings and Helper ---
LANG_CODE_TO_NAME = {
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "hi": "Hindi",
    # Add other languages supported by your form/LLM
}
DEFAULT_LANG_NAME = "Korean" # Default language if preference is missing or unknown

def get_twilio_lang_code(lang_code: Optional[str]) -> str:
    """Maps our internal language code (e.g., 'ko') to Twilio standard codes (e.g., 'ko-KR')."""
    # Maps our language codes to Twilio standard voice/gather codes
    mapping = {
        "ko": "ko-KR",
        "en": "en-US", # Or en-GB, en-AU etc.
        "ja": "ja-JP",
        "zh": "zh-CN", # Mandarin - Simplified
        # Add more mappings as needed for Twilio supported voices
    }
    # Default to Korean if lang_code is None or not in mapping
    default_twilio_code = "ko-KR"
    if lang_code is None:
        return default_twilio_code
    return mapping.get(lang_code, default_twilio_code)

# --- Upstage Translation Helper --- 
def translate_ko_to_en(text: str) -> Optional[str]:
    """Translates Korean text to English using Upstage translation model."""
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    if not upstage_api_key:
        logging.error("UPSTAGE_API_KEY not found. Cannot perform translation.")
        return None

    try:
        # Initialize the specific translation chat model
        translation_chat = ChatUpstage(api_key=upstage_api_key, model="translation-koen")
        
        # Simple invocation (adjust if history/system prompt is needed)
        messages = [HumanMessage(content=text)]
        response = translation_chat.invoke(messages)
        
        translated_text = response.content.strip() if response and response.content else None
        if translated_text:
            logging.info(f"Successfully translated KO>EN: '{text[:30]}...' -> '{translated_text[:30]}...'")
            return translated_text
        else:
            logging.warning(f"Translation KO>EN resulted in empty content for input: {text[:50]}...")
            return None
            
    except Exception as e:
        logging.error(f"Error during Upstage translation KO>EN: {e}", exc_info=True)
        return None

# --- LLM Based Voice Response Interpretation (Using Anthropic Claude) --- 
def interpret_voice_response(speech_text: str, alert_context: str) -> int:
    """
    Analyzes the user's voice response using Anthropic Claude to determine if follow-up is needed.
    Returns 1 if report/assistance needed, 0 otherwise.
    """
    if not speech_text:
        logging.debug("Interpret request skipped: Empty speech text.")
        return 0
    
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logging.error("ANTHROPIC_API_KEY not found in environment variables. Cannot interpret voice response.")
        return 0
        
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)

        prompt = f"""Task: Analyze the user's voice response regarding a disaster alert. Determine if the user wants to report something specific, needs help, or is indicating an emergency situation requiring follow-up. Output ONLY '1' if a report or assistance is needed, otherwise output ONLY '0'.

Original Alert Context (for reference): {alert_context}
User's Voice Response: {speech_text}

Output (1 or 0):"""
        
        # Log the request being sent to Claude
        logging.info(f"Invoking Anthropic Claude to interpret voice response...")
        logging.info(f"  User Response Sent: '{speech_text}'")
        # logging.debug(f"  Full Prompt Sent:\n{prompt}") # Log full prompt only in debug

        message = client.messages.create(
            model="claude-3-haiku-20240307", # Use Haiku for speed and cost-effectiveness
            max_tokens=5, # Only need 1 token for '1' or '0'
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse the response
        interpretation_result = ""
        if message.content and isinstance(message.content, list) and hasattr(message.content[0], 'text'):
             interpretation_result = message.content[0].text.strip()
        
        logging.info(f"Anthropic Claude interpretation result: '{interpretation_result}'")
        
        if interpretation_result == "1":
            return 1
        else:
            return 0
            
    except Exception as e:
        logging.error(f"Error during Anthropic Claude voice response interpretation: {e}", exc_info=True)
        return 0 # Default to no action on error

# --- User Specific RAG Context Retrieval Helper ---
def _get_user_specific_rag_context(db_user: User, disaster_alert: DisasterAlertData) -> str:
    """Retrieves RAG context filtered by disaster type and user vulnerability type."""
    global vectorstore
    if not vectorstore:
        logging.warning("Vectorstore not available, cannot retrieve RAG context.")
        return "" # Return empty string if no vectorstore

    # 1. Determine disaster type key for filtering
    disaster_type_ko = disaster_alert.DST_SE_NM
    disaster_type_en_key = next((en.lower().replace(" ", "") for ko, en in DISASTER_TYPE_MAP_KO_EN.items() if ko == disaster_type_ko), None)
    if not disaster_type_en_key:
        logging.warning(f"Could not map disaster type '{disaster_type_ko}' for RAG filter. No context retrieved.")
        return ""

    # 2. Determine user vulnerability type key for filtering
    # Assuming user.vulnerability_type directly matches the metadata keys (e.g., 'visual', 'hearing', 'elderly')
    # If not, mapping might be needed here too.
    user_type_key = db_user.vulnerability_type
    # Include 'normal' type documents for everyone
    allowed_vulnerability_types = [user_type_key, 'normal'] 

    # 3. Prepare the query (using English disaster type and region)
    core_region_name = _extract_region_from_address(disaster_alert.RCPTN_RGN_NM) or disaster_alert.RCPTN_RGN_NM
    disaster_type_en = DISASTER_TYPE_MAP_KO_EN.get(disaster_type_ko, disaster_type_ko)
    base_query = f"{disaster_type_en} in {core_region_name}"

    # 4. Perform filtered similarity search
    relevant_docs = []
    try:
        # Use similarity search to get candidates first
        # Increase k slightly to ensure potential matches aren't missed before metadata filtering
        results_with_scores = vectorstore.similarity_search_with_score(base_query, k=7) 

        # Apply metadata filters in Python
        filtered_docs = []
        logging.info(f"Filtering RAG results for disaster='{disaster_type_en_key}', user_types={allowed_vulnerability_types}")
        for doc, score in results_with_scores:
            doc_meta_disaster = doc.metadata.get('disaster_type')
            doc_meta_vuln = doc.metadata.get('vulnerability_type')
            
            # Check if disaster type matches AND vulnerability type is allowed
            if doc_meta_disaster == disaster_type_en_key and doc_meta_vuln in allowed_vulnerability_types:
                filtered_docs.append(doc)
                logging.info(f"  - Matched Doc: disaster='{doc_meta_disaster}', vuln='{doc_meta_vuln}', source='{doc.metadata.get('source')}'")
                # Limit the number of documents to use (e.g., top 3)
                if len(filtered_docs) >= 3:
                    break 
            # else: 
            #     logging.debug(f"  - Skipped Doc: disaster='{doc_meta_disaster}', vuln='{doc_meta_vuln}'")
        
        relevant_docs = filtered_docs
        logging.info(f"Retrieved {len(relevant_docs)} specific relevant docs for user {db_user.id}.")

    except Exception as e:
        logging.error(f"Error during user-specific RAG search for user {db_user.id}: {e}", exc_info=True)
        relevant_docs = []

    # 5. Combine content into a single string
    context_string = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context_string
# --------------------------------------------------

# --- RAG Based Message Generation (Korean, takes user info for prompt) ---
def generate_notification_messages(disaster_alert: DisasterAlertData, user: User) -> str: # Takes user object
    """Generates a Korean SMS alert message considering user type via prompt."""
    # Base alert info (English for RAG query consistency)
    base_alert_info_en = f"{DISASTER_TYPE_MAP_KO_EN.get(disaster_alert.DST_SE_NM, disaster_alert.DST_SE_NM)} in {_extract_region_from_address(disaster_alert.RCPTN_RGN_NM) or disaster_alert.RCPTN_RGN_NM}"
    print(f"Base Alert Info for SMS RAG query: {base_alert_info_en}")

    target_rag_len = 60 
    total_max_len = 70
    rag_response_content = "" 
    global vectorstore

    if vectorstore is None:
        print("  WARNING: Vectorstore not available. Cannot generate RAG content.")
        fallback_base = f"[경보] {disaster_alert.DST_SE_NM} ({_extract_region_from_address(disaster_alert.RCPTN_RGN_NM) or disaster_alert.RCPTN_RGN_NM}) 발생. 관련 기관 안내 확인."
        rag_response_content = fallback_base[:total_max_len]
        return rag_response_content

    try:
        # Determine disaster type key for filtering RAG context
        disaster_type_ko = disaster_alert.DST_SE_NM
        disaster_type_en_key = next((en.lower().replace(" ", "") for ko, en in DISASTER_TYPE_MAP_KO_EN.items() if ko == disaster_type_ko), None)
        
        upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        llm = ChatUpstage(api_key=upstage_api_key, model="solar-pro")
        
        # Retriever setup: Filter ONLY by disaster type
        search_kwargs = {}
        if disaster_type_en_key:
            search_kwargs['filter'] = {'disaster_type': disaster_type_en_key}
            logging.info(f"Applying RAG metadata filter: disaster_type='{disaster_type_en_key}'")
        else:
            logging.warning("No disaster type filter applied for RAG retriever.")
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Get RAG context based on disaster type
        query = base_alert_info_en 
        relevant_docs = retriever.get_relevant_documents(query)
        rag_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        logging.info(f"Retrieved {len(relevant_docs)} docs for context (disaster filter only).")

        # --- Korean Prompt including User Type --- 
        user_type_info = f"(수신자 특성: {user.vulnerability_type})"
        prompt_template = f"""
        제공된 정보와 참고 문서를 바탕으로, 가장 중요한 핵심 대처 방안 1가지를 포함하여 한국어로 간결한 비상 알림 메시지를 생성하세요.
        {user_type_info} 이 정보를 참고하여 답변의 뉘앙스나 강조점을 조절할 수 있습니다.
        목표 길이: {target_rag_len}자 내외. 추가 설명 없이 알림 내용만 출력하세요.

        참고 문서:
        {{context}}

        입력 알림 정보 (재난 유형/지역):
        {{question}}

        알림 메시지 (목표 ~{target_rag_len}자, 한국어):
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        # -----------------------------------------

        # Construct the full prompt and invoke LLM
        full_prompt = PROMPT.format(context=rag_context, question=query)
        llm_response = llm.invoke(full_prompt)
        rag_summary_text = llm_response.content.strip() if llm_response and llm_response.content else ""
        print(f"  LLM Generated SMS Response (raw): {rag_summary_text}")

        # Truncation Logic
        if not rag_summary_text:
            fallback_base = f"[경보] {disaster_alert.DST_SE_NM} ({_extract_region_from_address(disaster_alert.RCPTN_RGN_NM) or disaster_alert.RCPTN_RGN_NM}). 공식 발표 확인."
            rag_response_content = fallback_base[:total_max_len]
        else:
            rag_response_content = rag_summary_text
            if len(rag_response_content) > total_max_len:
                 rag_response_content = rag_response_content[:total_max_len]

    except Exception as e:
        print(f"  ERROR during LLM generation for SMS for user {user.id}: {e}")
        fallback_base = f"[경보] {disaster_alert.DST_SE_NM} ({_extract_region_from_address(disaster_alert.RCPTN_RGN_NM) or disaster_alert.RCPTN_RGN_NM}). 오류 발생. 관련 기관 확인."
        rag_response_content = fallback_base[:total_max_len]

    logging.info(f"Final generated KOREAN SMS content for user {user.id}: {rag_response_content}")
    return rag_response_content

# --- RAG Based VOICE Message Generation (Korean, takes user info for prompt) ---
def generate_voice_alert_message(disaster_alert: DisasterAlertData, user: User) -> str: # Takes user object
    """
    Generates a Korean voice alert message considering user type via prompt.
    Does NOT include the trailing 'Report' prompt here.
    """
    # --- Base Info & RAG Setup --- 
    core_region_name = _extract_region_from_address(disaster_alert.RCPTN_RGN_NM) or disaster_alert.RCPTN_RGN_NM
    disaster_type_ko = disaster_alert.DST_SE_NM
    disaster_type_en = DISASTER_TYPE_MAP_KO_EN.get(disaster_type_ko, disaster_type_ko)
    base_alert_info_en = f"{disaster_type_en} in {core_region_name}"
    print(f"Base Alert Info for VOICE RAG query: {base_alert_info_en}")

    target_rag_len = 90
    voice_message_content = ""
    global vectorstore

    if vectorstore is None:
        print("  WARNING: Vectorstore not available. Cannot generate RAG content for Voice.")
        fallback_base = f"[경보] {disaster_alert.DST_SE_NM} ({core_region_name}) 발생. 관련 기관 안내를 확인하십시오."
        voice_message_content = fallback_base[:target_rag_len]
        return voice_message_content

    try:
        # Determine disaster type key for filtering RAG context
        disaster_type_en_key = next((en.lower().replace(" ", "") for ko, en in DISASTER_TYPE_MAP_KO_EN.items() if ko == disaster_type_ko), None)
        
        upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        if not upstage_api_key:
             raise ValueError("UPSTAGE_API_KEY not found.")
        llm = ChatUpstage(api_key=upstage_api_key, model="solar-pro")
        
        # Retriever setup: Filter ONLY by disaster type
        search_kwargs = {}
        if disaster_type_en_key:
            search_kwargs['filter'] = {'disaster_type': disaster_type_en_key}
            logging.info(f"Applying RAG metadata filter: disaster_type='{disaster_type_en_key}'")
        else:
            logging.warning("No disaster type filter applied for RAG retriever.")
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Get RAG context based on disaster type
        query = base_alert_info_en
        relevant_docs = retriever.get_relevant_documents(query)
        rag_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        logging.info(f"Retrieved {len(relevant_docs)} docs for context (disaster filter only).")

        # --- Korean Voice Prompt including User Type --- 
        user_type_info = f"(수신자 특성: {user.vulnerability_type})"
        prompt_template = f"""
        Task: 한국어로 간결한 음성 안내 메시지의 핵심 내용을 생성.
        Output Format: "[경보 종류] [지역] 상황. [핵심 행동 1-2가지]."
        Strict Rules:
        - 제공된 정보(Context, Input)와 수신자 특성({user_type_info})을 참고하여 생성.
        - 핵심 행동은 Context에서 가져올 것.
        - 지역명은 Input({base_alert_info_en}) 기반으로 하되 자연스럽게.
        - 추가 설명 절대 금지.
        - 목표 길이: {target_rag_len}자 내외.
        - 끝에 'Report' 관련 문구는 절대 포함하지 말 것.

        Context Documents (for action guidance):
        {{context}}

        Input Alert Details (for disaster type/location):
        {{question}}

        Core Alert Message (Max {target_rag_len} chars, 한국어):
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        # ---------------------------------------------
        
        # Construct the full prompt and invoke LLM
        full_prompt = PROMPT.format(context=rag_context, question=query)
        llm_response = llm.invoke(full_prompt)
        rag_summary_text = llm_response.content.strip() if llm_response and llm_response.content else ""
        print(f"  LLM Generated Voice Response (raw core text): {rag_summary_text}")

        # Truncation Logic
        if not rag_summary_text:
            print("  WARNING: LLM generation resulted in empty string. Using fallback.")
            fallback_base = f"[경보] {disaster_alert.DST_SE_NM} ({core_region_name}). 공식 발표 확인."
            voice_message_content = fallback_base[:target_rag_len]
        else:
            voice_message_content = rag_summary_text
            if len(voice_message_content) > target_rag_len:
                 voice_message_content = voice_message_content[:target_rag_len]

    except Exception as e:
        print(f"  ERROR during LLM generation for Voice for user {user.id}: {e}")
        fallback_base = f"[경보] {disaster_alert.DST_SE_NM} ({core_region_name}). 오류 발생. 관련 기관 확인."
        voice_message_content = fallback_base[:target_rag_len]

    logging.info(f"Final generated KOREAN VOICE core content for user {user.id}: {voice_message_content}")
    return voice_message_content

# --- Kakao Geocoding Helper --- 
async def _get_coordinates_from_address(address: str) -> Optional[Dict[str, float]]:
    """Calls Kakao Geocoding API to get latitude and longitude from an address."""
    kakao_rest_api_key = os.getenv('KAKAO_REST_API_KEY') # Use the correct env var name
    if not kakao_rest_api_key or not address:
        logging.warning("Kakao REST API Key not found or address is empty. Skipping geocoding.")
        return None

    geocoding_url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {kakao_rest_api_key}"}
    params = {"query": address}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(geocoding_url, headers=headers, params=params)
            response.raise_for_status() # Raise exception for bad status codes
            data = response.json()
            
            if data.get('documents'):
                # Get coordinates from the first result
                coords = data['documents'][0]['address']
                lat = float(coords.get('y'))
                lon = float(coords.get('x'))
                logging.info(f"Geocoding success for '{address}': lat={lat}, lon={lon}")
                return {"latitude": lat, "longitude": lon}
            else:
                logging.warning(f"Geocoding failed for '{address}': No results found.")
                return None
        except httpx.RequestError as exc:
            logging.error(f"An error occurred while requesting Kakao Geocoding API: {exc}")
            return None
        except httpx.HTTPStatusError as exc:
            logging.error(f"Kakao Geocoding API returned an error: {exc.response.status_code} - {exc.response.text}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            logging.error(f"Error parsing Kakao Geocoding API response: {e}")
            return None

# --- Helper function to extract region --- 
def _extract_region_from_address(address: str) -> Optional[str]:
    """Attempts to extract a '구' or '시' name from the address string."""
    if not address:
        return None
    # Simple extraction based on '구' or '시' - might need refinement
    parts = address.split()
    for part in parts:
        if part.endswith('구'):
            # Handle cases like '서울특별시' -> '서울시' (consistency)
            city_part = parts[0] if parts else ""
            if city_part.endswith('시') and city_part != part:
                 # e.g., 서울특별시 성동구 -> 서울시 성동구
                return f"{city_part} {part}" 
            return part # e.g., 성동구
        elif part.endswith('시') and len(parts) > 1 and parts[parts.index(part)+1].endswith('구'):
             # Handles cases where city name comes before gu directly, like '서울시 성동구' if already formatted
             continue # Let the '구' part handle this
        elif part.endswith('시'): 
            # If only city is available or it's the most specific part
            return part # e.g., 수원시 
    return None # Could not extract a region

# --- API Router Definition ---
api_router = APIRouter(prefix="/api")

# --- NEW Translation Endpoint ---
@api_router.post("/translate/ko-en", response_model=TranslationResponse, tags=["Utilities"])
async def translate_korean_to_english(request_data: TranslationRequest):
    """Translates Korean text to English using Upstage API."""
    korean_text = request_data.text
    if not korean_text:
        return TranslationResponse(error="Input text cannot be empty.")
    
    translated = translate_ko_to_en(korean_text)
    
    if translated:
        return TranslationResponse(translated_text=translated)
    else:
        # If translation failed (e.g., API key missing, Upstage error)
        logging.error(f"Translation failed for text: {korean_text[:50]}...")
        return TranslationResponse(error="Translation failed.")
# ----------------------------

# --- NEW Endpoint to retrieve simulation info ---
@api_router.get("/simulation_info/{simulation_id}", tags=["Disaster"])
# Add request: Request to the function signature
async def get_simulation_info(request: Request, simulation_id: str):
    """
    Retrieves the details of a specific disaster simulation using its ID (SN).
    Used by the disaster map page to get context.
    Returns data as a PlainTextResponse containing a JSON string.
    """
    try: 
        # ... (User-Agent logging remains the same) ...
        user_agent = request.headers.get("user-agent", "Unknown")
        logging.info(f"Request User-Agent for simulation {simulation_id}: {user_agent}")

        logging.info(f"Request received for simulation info with ID: {simulation_id}")
        simulation_data = request.app.state.active_simulations.get(simulation_id)
        
        if not simulation_data:
            logging.warning(f"Simulation data not found in app.state for ID: {simulation_id}")
            raise HTTPException(status_code=404, detail=f"Simulation data not found for ID {simulation_id}")
            
        logging.info(f"Returning simulation data from app.state for ID: {simulation_id}")
        
        # Manually convert Pydantic model to JSON string
        json_string = simulation_data.model_dump_json()
        
        # Return as PlainTextResponse with JSON media type
        return PlainTextResponse(content=json_string, media_type="application/json")
        
    except HTTPException as http_exc:
        # ... (HTTPException handling remains the same) ...
        raise http_exc
    except Exception as e:
        # ... (Generic exception handling remains the same) ...
        raise HTTPException(status_code=500, detail="Internal server error fetching simulation info.")
# -------------------------------------------------

# --- Dashboard Endpoint --- 
@api_router.get("/dashboard/summary", response_model=List[RegionSummary], tags=["Dashboard"])
def get_dashboard_summary(request: Request, db: Session = Depends(get_db)):
    """Provides a summary of user responses clustered by region."""
    users = get_users(db) # Get all users
    
    region_responses: Dict[str, List[str]] = {}
    
    # 1. Group responses by region
    for user in users:
        if user.address and user.last_voice_response:
            region = _extract_region_from_address(user.address)
            if region:
                # Add non-empty responses
                response_text = user.last_voice_response.strip()
                if response_text: # Only add if there's actual text
                    region_responses.setdefault(region, []).append(response_text)

    summaries: List[RegionSummary] = []
    
    # Access chat model from app state
    chat_model = getattr(request.app.state, 'chat_model', None)
    
    if not chat_model:
        logging.error("Chat model not available from app state for summarization.")
        # Return empty list or raise error, depending on desired behavior
        return [] 

    # 2. Summarize responses for each region
    for region, responses in region_responses.items():
        if not responses: # Skip if no valid responses for the region
            continue
            
        # Combine responses into a context string
        context = "\n".join([f"- {res}" for res in responses])
        
        # Create prompt for summarization
        prompt = f"다음은 [{region}] 지역 사용자들의 최근 보고 내용입니다. 이 내용을 바탕으로 해당 지역의 현재 상황을 한국어로 간결하게 요약해주세요.:\n\n{context}"
        
        try:
            # Call the Upstage LLM
            logging.info(f"Generating summary for region: {region}")
            llm_response = chat_model.invoke(prompt)
            # Ensure the response content is a string
            summary_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            logging.info(f"Generated summary: {summary_text}")
            
            summaries.append(RegionSummary(region_name=region, summary=summary_text))
            
        except Exception as e:
            logging.error(f"Error generating summary for region {region}: {e}", exc_info=True)
            # Optionally add a placeholder summary or skip the region
            summaries.append(RegionSummary(region_name=region, summary="요약 생성 중 오류 발생"))
            
    return summaries

# --- User Endpoints --- 
@api_router.post("/users/", response_model=UserResponse, tags=["Users"])
async def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_phone(db, phone_number=user.phone_number)
    if db_user:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    # --- Geocode Address --- 
    latitude = None
    longitude = None
    if user.address:
        coordinates = await _get_coordinates_from_address(user.address)
        if coordinates:
            latitude = coordinates.get('latitude')
            longitude = coordinates.get('longitude')
    # -----------------------

    # Prepare data for DB creation
    user_data_dict = user.model_dump()
    user_data_dict['latitude'] = latitude
    user_data_dict['longitude'] = longitude
    
    print("Creating user with data:", user_data_dict)
    
    try:
        # Pass the dictionary including coordinates to create_user
        created_user = create_user(db=db, user_data=user_data_dict)
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

# New endpoint to get all users with their last response details
@api_router.get("/users/with_details", response_model=List[UserResponseWithDetails], tags=["Users"])
def read_users_with_details_endpoint(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = get_users(db, skip=skip, limit=limit)
    return users

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

# Rename the endpoint and modify parameters to match disaster_map.html
@api_router.get("/shelters_proxy", tags=["External APIs"])
async def shelters_proxy(request: Request):
    # ... (parameter extraction and validation as before) ...
    start_lat = request.query_params.get("startLat")
    end_lat = request.query_params.get("endLat")
    start_lot = request.query_params.get("startLot")
    end_lot = request.query_params.get("endLot")

    service_key = safety_data_service_key # Use the key loaded earlier

    if not service_key:
        # ... (error handling for missing key)
        logging.error("SAFETY_DATA_SERVICE_KEY is not loaded from environment variables.")
        raise HTTPException(
            status_code=500, detail="Server configuration error: Safety API key missing."
        )
        
    if not all([start_lat, end_lat, start_lot, end_lot]):
        # ... (error handling for missing coordinates)
        logging.warning(
            f"Missing coordinate parameters: startLat={start_lat}, endLat={end_lat}, startLot={start_lot}, endLot={end_lot}"
        )
        raise HTTPException(
            status_code=400, detail="Missing coordinate parameters"
        )
        
    # Construct the external API URL - matching app_minsu.py structure
    external_api_url = (
        f"https://www.safetydata.go.kr/V2/api/DSSP-IF-10941?"
        f"serviceKey={service_key}&" # Removed urllib.parse.quote
        f"startLot={start_lot}&endLot={end_lot}&"
        f"startLat={start_lat}&endLat={end_lat}&"
        f"pageNo=1&numOfRows=100"  # Removed returnType=json
    )

    logging.info(f"Proxying shelter request to (adjusted): {external_api_url}")

    # Define a common browser User-Agent header (Keep this change for now, might still be relevant)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(external_api_url, headers=headers, timeout=15.0)
            
            # --- Log raw response BEFORE parsing --- 
            logging.info(f"External API response status: {response.status_code}")
            # Limit logging response text size to prevent flooding logs
            response_text_preview = response.text[:500] + ('...' if len(response.text) > 500 else '')
            logging.debug(f"External API response text (preview): {response_text_preview}") 
            # ----------------------------------------

            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
            
            # --- Attempt to parse JSON --- 
            try:
                data = response.json()
                logging.debug("Successfully parsed JSON response.")
            except Exception as json_err:
                logging.error(f"Failed to parse JSON from external API response. Error: {json_err}", exc_info=True)
                logging.error(f"Raw response text that failed parsing: {response.text}") # Log full text on error
                # Use 502 Bad Gateway when the upstream server sends something unparseable
                raise HTTPException(status_code=502, detail="Failed to parse response from external API") 
            # ----------------------------

            # --- Check for API-specific functional error structure ---
            # Added robust checking for nested keys before accessing
            # Corrected: Directly access 'header' from the root of the data dictionary
            api_header = data.get("header") if isinstance(data, dict) else None
            if not api_header or not isinstance(api_header, dict):
                logging.error(f"External Safety API response JSON structure is invalid or missing 'header'. Data: {data}")
                raise HTTPException(status_code=502, detail="Invalid response structure from external Safety API.")

            result_code = api_header.get("resultCode")
            
            if result_code != "00":
                 error_msg = api_header.get("resultMsg", "Unknown API error")
                 log_message = f"External Safety API functional error: {error_msg} (Code: {result_code}) - URL: {external_api_url}"
                 logging.error(log_message)
                 # Use 502 Bad Gateway as the upstream server responded, but with a functional error
                 raise HTTPException(status_code=502, detail=f"External Safety API Error: {error_msg}") 
            # ------------------------------------------------------

            # If resultCode is '00', return the data
            logging.info("External Safety API call successful (resultCode=00).")
            return data

        except httpx.HTTPStatusError as exc:
            # Log the error details from the external API
            # This catches the 4xx/5xx errors after raise_for_status()
            logging.error(f"HTTP error status from external API: {exc.response.status_code} - Response: {exc.response.text[:500]}... - URL: {external_api_url}")
            status_code_to_return = exc.response.status_code if exc.response.status_code >= 500 else 502
            raise HTTPException(
                status_code=status_code_to_return, 
                detail={"error": "External API returned HTTP error", "details": exc.response.text}
            )
        except httpx.TimeoutException:
            logging.error(f"Timeout error calling external API: {external_api_url}")
            raise HTTPException(status_code=504, detail="External API timed out")
        except httpx.RequestError as exc:
            logging.error(f"Request error calling external API: {exc} - URL: {external_api_url}")
            raise HTTPException(status_code=502, detail=f"Error connecting to external API: {exc}")
        except Exception as e:
            # Check if the caught exception is already an HTTPException
            if isinstance(e, HTTPException):
                # If it is, re-raise it directly to preserve the original status code and detail
                raise e 
            else:
                # Catch any other unexpected errors during the process
                logging.error(f"An unexpected internal error occurred in shelters_proxy: {e}", exc_info=True)
                # For truly unexpected errors, return a generic 500
                raise HTTPException(status_code=500, detail="An internal server error occurred")

# --- Refactored endpoint for simulating disaster ---
@api_router.post("/simulate_disaster/", tags=["Disaster"])
# Add request: Request to the function signature
async def simulate_disaster(request: Request, disaster_alert: DisasterAlertData, db: Session = Depends(get_db)):
    """
    Simulates a disaster alert scenario:
    1. Stores disaster info for later retrieval by map.
    2. Updates disaster coordinates if address provided.
    3. Filters users based on location proximity.
    4. Generates user-specific messages (SMS & Voice) considering vulnerability type.
    5. Handles translation based on user preference.
    6. Sends notifications directly via Twilio API (SMS map url, SMS alert, Voice call).
    """
    # REMOVE the global declaration
    # global active_simulations
    
    logging.info(f"--- STARTING DISASTER SIMULATION --- ")
    logging.info(f"Received Disaster Alert: SN={disaster_alert.SN}, Type={disaster_alert.DST_SE_NM}, Region={disaster_alert.RCPTN_RGN_NM}")

    # --- 1. Geocode Disaster Location (if address available) --- 
    disaster_lat = disaster_alert.latitude
    disaster_lon = disaster_alert.longitude
    if not disaster_lat or not disaster_lon:
        if disaster_alert.RCPTN_RGN_NM:
            logging.info(f"Attempting to geocode disaster location: {disaster_alert.RCPTN_RGN_NM}")
            coordinates = await _get_coordinates_from_address(disaster_alert.RCPTN_RGN_NM)
            if coordinates:
                disaster_lat = coordinates.get('latitude')
                disaster_lon = coordinates.get('longitude')
                logging.info(f"  > Geocoded disaster location: Lat={disaster_lat}, Lon={disaster_lon}")
            else:
                logging.warning(f"  > Could not geocode disaster location address: {disaster_alert.RCPTN_RGN_NM}. Cannot filter by location.")
        else:
             logging.warning("Disaster alert has no address and no explicit coordinates. Cannot filter by location.")
             
    # Update the disaster_alert object with coordinates
    disaster_alert.latitude = disaster_lat
    disaster_alert.longitude = disaster_lon
    
    # --- STORE Disaster Info for Map Retrieval using app.state ---
    # Use SN as the key. WARNING: Overwrites if same SN used again quickly.
    if disaster_alert.SN:
        # Access active_simulations via request.app.state
        request.app.state.active_simulations[disaster_alert.SN] = disaster_alert.model_copy(deep=True) 
        logging.info(f"Stored simulation data in app.state for SN: {disaster_alert.SN}")
    else:
        # Handle cases where SN might be missing (though it's required by schema)
        logging.error("Disaster Alert SN is missing. Cannot store simulation data for map.")
        # Decide how to proceed - maybe raise an error or generate a temporary ID?
        # For now, we'll log an error and proceed.
        # Map link functionality will likely fail later if SN is missing.
        pass 
    # -------------------------------------------

    # --- 2. Filter Target Users by Location --- 
    # Define the radius for notification (e.g., 10 km)
    notification_radius_km = 10 
    users_to_notify: List[User] = []

    if disaster_lat is not None and disaster_lon is not None:
        logging.info(f"Filtering users within {notification_radius_km}km of disaster location ({disaster_lat}, {disaster_lon})")
        users_to_notify = filter_target_users_by_location(db, disaster_lat, disaster_lon, notification_radius_km)
        logging.info(f"Found {len(users_to_notify)} users within the radius.")
    else:
        logging.warning("Disaster location coordinates unknown. Cannot filter users by location. Alerting ALL users.")
        users_to_notify = get_users(db, limit=500) # Be careful with large numbers
        logging.info(f"Alerting {len(users_to_notify)} users (fallback: no location filter).")

    if not users_to_notify:
        logging.warning("No users found to notify. Exiting simulation.")
        return {"status": "No users to notify", "sms_sent": 0, "sms_failed": 0, "calls_initiated": 0, "calls_failed": 0}

    if not client or not twilio_phone_number:
        logging.error("Twilio client not configured. Cannot send notifications.")
        raise HTTPException(status_code=500, detail="Twilio configuration error, cannot send notifications.")
        
    # Update the hardcoded base URL to the currently correct one
    hardcoded_base_url = "https://71d4c4523155.ngrok.app" 
    
    sms_sent_count = 0
    sms_failed_count = 0
    calls_initiated_count = 0
    calls_failed_count = 0

    logging.info(f"--- Starting Notification Loop for {len(users_to_notify)} users --- ")
    
    # --- Prepare Map URL with Simulation ID (SN) ---
    # Only generate the link if SN exists
    map_url_for_sms = None
    if disaster_alert.SN:
        # Use the simulation ID (SN) in the path
        map_url_for_sms = f"{hardcoded_base_url}/map/disaster_map/{disaster_alert.SN}"
        logging.info(f"Generated map URL with simulation ID (SN): {map_url_for_sms}")
    else:
         # Log warning if SN is missing, map link won't be generated
         logging.warning("Cannot generate map URL as disaster SN is missing.")
    # ------------------------------------------

    for user in users_to_notify:
        logging.info(f"Processing user: ID={user.id}, Phone={user.phone_number}, Lang={user.preferred_language}, Type={user.vulnerability_type}")
        
        # --- Generate Base Korean Messages --- 
        try:
            sms_content_ko = generate_notification_messages(disaster_alert, user)
        except Exception as e_sms_gen:
            logging.error(f"  ERROR generating SMS content for user {user.id}: {e_sms_gen}", exc_info=True)
            sms_content_ko = f"[경보] {disaster_alert.DST_SE_NM}. 시스템 오류. 공식 발표 확인." # Fallback
        
        try:
            voice_content_ko = generate_voice_alert_message(disaster_alert, user)
        except Exception as e_voice_gen:
            logging.error(f"  ERROR generating Voice content for user {user.id}: {e_voice_gen}", exc_info=True)
            voice_content_ko = f"[경보] {disaster_alert.DST_SE_NM}. 시스템 오류 발생. 공식 발표를 확인하십시오." # Fallback
            
        # --- Handle Translation if needed --- 
        sms_content_final = sms_content_ko
        voice_content_final = voice_content_ko
        tts_code = "ko-KR" # Default TTS code
        tts_voice = "Polly.Seoyeon"

        if user.preferred_language == 'en':
            logging.info(f"  User {user.id} prefers English. Translating messages...")
            sms_translation = translate_ko_to_en(sms_content_ko)
            voice_translation = translate_ko_to_en(voice_content_ko)
            
            if sms_translation:
                sms_content_final = sms_translation
            else:
                logging.warning(f"  Failed to translate SMS for user {user.id}. Sending in Korean.")
                
            if voice_translation:
                voice_content_final = voice_translation
                tts_code = "en-US" # Change TTS code for English
                tts_voice = "Polly.Joanna"
            else:
                logging.warning(f"  Failed to translate Voice for user {user.id}. Sending in Korean.")
        
        logging.info(f"  Final SMS Content ({user.preferred_language or 'ko'}): {sms_content_final}")
        logging.info(f"  Final Voice Content ({user.preferred_language or 'ko'}): {voice_content_final}")
        logging.info(f"  TTS Code: {tts_code}, Voice: {tts_voice}")
        
        # --- Send SMS Notifications (Map URL + Alert) --- 
        # Revert to sending two separate SMS messages

        # 1. Send Map URL SMS (only if URL exists)
        if map_url_for_sms:
            # Modify the message body to include ONLY the URL for testing
            map_message_body = map_url_for_sms # Directly use the URL
            
            try:
                logging.info(f"  1 -> Sending MAP URL ONLY SMS ({map_message_body}) to {user.phone_number}") # Log message changed
                message1 = client.messages.create(
                    body=map_message_body, # Send only the URL
                    from_=twilio_phone_number,
                    to=user.phone_number
                )
                logging.info(f"    Map URL SMS sent! SID: {message1.sid}")
                # REMOVE the delay between messages
                # time.sleep(0.5)

                # 2. Send Alert SMS (ONLY if map URL SMS was attempted)
                try:
                    logging.info(f"  2 -> Sending ALERT SMS ({sms_content_final}) to {user.phone_number}")
                    message2 = client.messages.create(
                        body=sms_content_final,
                        from_=twilio_phone_number,
                        to=user.phone_number
                    )
                    logging.info(f"    Alert SMS sent! SID: {message2.sid}")
                    sms_sent_count += 1 # Count success only if both are attempted/sent
                except Exception as e2:
                    logging.error(f"    ERROR sending ALERT SMS to {user.phone_number} after map URL: {e2}", exc_info=True)
                    sms_failed_count += 1 # Count failure if alert SMS fails

            except Exception as e1:
                # Log message updated for clarity
                logging.error(f"    ERROR sending MAP URL ONLY SMS to {user.phone_number}: {e1}", exc_info=True)
                sms_failed_count += 1

        else:
            # If map URL couldn't be generated (due to missing SN), send only the alert SMS
            logging.warning(f"  Skipping Map URL SMS for user {user.id} as map URL could not be generated (SN missing?).")
            try:
                 logging.info(f"  -> Sending ALERT SMS ONLY ({sms_content_final}) to {user.phone_number}")
                 message_alert_only = client.messages.create(
                     body=sms_content_final,
                     from_=twilio_phone_number,
                     to=user.phone_number
                 )
                 logging.info(f"    Alert SMS (only) sent! SID: {message_alert_only.sid}")
                 sms_sent_count += 1
            except Exception as e_alert_only:
                 logging.error(f"    ERROR sending ALERT SMS ONLY to {user.phone_number}: {e_alert_only}", exc_info=True)
                 sms_failed_count += 1
            
        # --- Send Voice Call Notification (if user wants it) ---
        if user.wants_info_call:
            logging.info(f"  User {user.id} wants info call. Initiating voice call...")
            
            # Construct TwiML dynamically for the voice call
            encoded_message_for_action = urllib.parse.quote(voice_content_final)
            action_url = f'{hardcoded_base_url}/twilio/handle-voice-alert-response?alert_message={encoded_message_for_action}&caller={urllib.parse.quote(user.phone_number)}'
            
            try:
                response = VoiceResponse()
                gather = Gather(input='speech', 
                                action=action_url, 
                                method='POST', 
                                language=tts_code, # Use determined TTS language for speech recognition
                                speechTimeout='auto',
                                actionOnEmptyResult=True,
                                hints="report") # Keep hint simple for now
                
                # Say the main alert message with correct voice/language
                gather.say(voice_content_final, voice=tts_voice, language=tts_code)
                # Append a more open-ended prompt for the LLM interpretation
                report_prompt = " 도움이 필요하시면 말씀해주세요." if tts_code == 'ko-KR' else " If you need assistance, please state your request now."
                gather.say(report_prompt, voice=tts_voice, language=tts_code)
                
                response.append(gather)
                
                # Fallback message if no input
                response.say("응답이 없어 통화를 종료합니다." if tts_code == 'ko-KR' else "No response received. Ending call.", voice=tts_voice, language=tts_code)
                response.hangup()
                
                twiml_content = str(response)
                
                logging.info(f"  -> Attempting call to {user.phone_number}. Action URL: {action_url}")
                # logging.debug(f"     Generated TwiML: {twiml_content}") # Log full TwiML only in debug

                call = client.calls.create(
                    twiml=twiml_content,
                    to=user.phone_number,
                    from_=twilio_phone_number
                )
                logging.info(f"    Voice call initiated! SID: {call.sid}")
                calls_initiated_count += 1
                
                time.sleep(0.2) # Brief delay between API calls

            except Exception as e_call:
                logging.error(f"    ERROR initiating voice call to {user.phone_number}: {e_call}", exc_info=True)
                calls_failed_count += 1
        else:
            logging.info(f"  User {user.id} does not want info calls. Skipping voice alert.")
            
        # --- End of User Loop --- 
        
    logging.info(f"--- Notification Loop COMPLETE --- ")
    logging.info(f"SMS Sent: {sms_sent_count}, SMS Failed: {sms_failed_count}")
    logging.info(f"Calls Initiated: {calls_initiated_count}, Calls Failed: {calls_failed_count}")
    logging.info(f"--- ENDING DISASTER SIMULATION --- ")
    
    return {
        "status": "Simulation complete",
        "users_targeted": len(users_to_notify),
        "sms_sent": sms_sent_count,
        "sms_failed": sms_failed_count,
        "calls_initiated": calls_initiated_count,
        "calls_failed": calls_failed_count
    }

# --- Dummy Disaster Data (keyed by ID) ---
DUMMY_DISASTERS = {
    "earthquake_gangnam": {
        "location": "Gangnam-gu, Seoul",
        "type": "Large Fire",
        "scale": "Magnitude 4.1",
        "time": "10:30",
        "coordinates": {"lat": 37.5503455, "lng": 127.0726955},
        "radius": 3000,
    },
    "fire_seongdong": {
        "location": "Seongdong-gu, Seoul",
        "type": "large fire",
        "scale": None,
        "time": "11:15",
        "coordinates": {"lat": 37.5637, "lng": 127.0368},
        "radius": 1500,
    },
    "flood_mapo": {
        "location": "Mapo-gu, Seoul",
        "type": "flood warning",
        "scale": "Heavy Rain Advisory",
        "time": "09:00",
        "coordinates": {"lat": 37.5665, "lng": 126.9011},  # Near Hongdae
        "radius": 5000,
    },
    "default": {  # A default if no ID is provided or matched
        "location": "Jung-gu, Seoul",
        "type": "unknown alert",
        "scale": "N/A",
        "time": "12:00",
        "coordinates": {"lat": 37.5665, "lng": 126.9780},  # City Hall area
        "radius": 2000,
    },
}

@api_router.get("/disaster/{disaster_id}", tags=["Disaster Info"])
async def get_disaster_data(disaster_id: str):
    """
    Retrieves dummy disaster information based on the provided ID.
    """
    disaster_info = DUMMY_DISASTERS.get(disaster_id)
    if not disaster_info:
        # Return default disaster if ID not found
        disaster_info = DUMMY_DISASTERS.get("default")
    return disaster_info

@api_router.get("/disaster/", tags=["Disaster Info"])
async def get_default_disaster_data():
    """
    Retrieves default dummy disaster information.
    """
    return DUMMY_DISASTERS.get("default")

# --- Dummy Emergency Info ---
DUMMY_EMERGENCY_INFO = {
    "earthquake_gangnam": {
        "emergency_contacts": {
            "police": "112",
            "fire_department": "119",
            "disaster_management": "02-2128-5800",
        },
        "safety_instructions": [
            "Drop to the ground and take cover under sturdy furniture",
            "Stay away from windows, exterior walls, and anything that could fall",
            "If you're outdoors, stay in open areas away from buildings and power lines",
            "After shaking stops, evacuate to open areas",
        ],
        "evacuation_centers": [
            "Gangnam Community Center - 23 Gangnam-daero, Gangnam-gu",
            "Yeoksam Elementary School - 146 Teheran-ro, Gangnam-gu",
            "COEX Convention Center - 513 Yeongdong-daero, Gangnam-gu",
        ],
    },
    "fire_seongdong": {
        "emergency_contacts": {
            "police": "112",
            "fire_department": "119",
            "disaster_management": "02-2286-5272",
        },
        "safety_instructions": [
            "Cover your mouth and nose with a wet cloth",
            "Stay close to the floor where air is less toxic",
            "Do not use elevators, use emergency staircases",
            "Feel doors before opening - if hot, find another exit route",
        ],
        "evacuation_centers": [
            "Seongdong-gu Office - 18 Wangsan-ro, Seongdong-gu",
            "Haengdang Public Park - 20 Haengdang-dong, Seongdong-gu",
            "Seoul Children's Grand Park - 216 Neungdong-ro, Seongdong-gu",
        ],
    },
    "flood_mapo": {
        "emergency_contacts": {
            "police": "112",
            "fire_department": "119",
            "disaster_management": "02-3153-8332",
        },
        "safety_instructions": [
            "Move to higher ground immediately",
            "Avoid walking or driving through flooded areas",
            "Stay away from power lines and electrical wires",
            "Be cautious of gas leaks and damaged infrastructure",
        ],
        "evacuation_centers": [
            "Mapo Arts Center - 31 World Cup buk-ro, Mapo-gu",
            "Sogang University Stadium - 35 Baekbeom-ro, Mapo-gu",
            "Hapjeong Middle School - 26 Wausan-ro, Mapo-gu",
        ],
    },
    "default": {
        "emergency_contacts": {
            "police": "112",
            "fire_department": "119",
            "disaster_management": "02-120",
        },
        "safety_instructions": [
            "Follow guidance from local authorities",
            "Turn on radio or TV for emergency information",
            "If evacuation is advised, do so immediately",
            "Help others who may require assistance",
        ],
        "evacuation_centers": [
            "Check local government announcements",
            "Contact 120 (Seoul city information) for nearest evacuation center",
        ],
    },
}

@api_router.get("/emergency_info/{disaster_id}", tags=["Disaster Info"])
async def get_emergency_info(disaster_id: str):
    """
    Returns dummy emergency information for a specific disaster
    """
    # Get emergency info for requested disaster or default if not found
    info = DUMMY_EMERGENCY_INFO.get(disaster_id, DUMMY_EMERGENCY_INFO["default"])

    # Add generic information that applies to all disasters
    info["general_advice"] = (
        "Call 119 for immediate emergencies. Follow official instructions."
    )

    return info

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
async def handle_voice_alert_response(request: Request, db: Session = Depends(get_db)):
    """Handles the user's English speech input, saves it to the User model, and includes original alert context."""
    # Log entry immediately
    logging.info(f"--- Entered /handle-voice-alert-response (Saving to User Model) --- ") 
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

        # Use 'To' field for the recipient's number in outbound calls
        caller_phone_number = form.get('To') 
        call_sid = form.get('CallSid')
        # Log both 'From' and 'To' for clarity
        logging.info(f"Extracted Form Data - From(Twilio): {form.get('From')}, To(Recipient): {caller_phone_number}, CallSid: {call_sid}")

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
    
    # --- Save response to Database --- 
    if caller_phone_number and caller_phone_number != "(From not found)" and speech_result != "(SpeechResult not found or error)": # Check caller_phone_number validity
        db_user = get_user_by_phone(db, phone_number=caller_phone_number)
        if db_user:
            try:
                db_user.last_voice_response = speech_result
                db_user.last_response_timestamp = datetime.utcnow() # Use UTC time 
                db.commit()
                logging.info(f"Successfully updated last response for user {caller_phone_number} in DB.")
            except Exception as e:
                db.rollback() # Rollback in case of commit error
                logging.error(f"Database error updating user {caller_phone_number}: {e}", exc_info=True)
        else:
            logging.warning(f"User with phone number {caller_phone_number} not found in DB. Cannot save response.")
    else:
        logging.warning("Cannot save to DB: Caller phone number or speech result was invalid.")
    # --------------------------------

    # --- LLM Based TwiML Response --- 
    # Access chat model from app state
    # chat_model = getattr(request.app.state, 'chat_model', None) # No longer needed here
    
    # Interpret the speech result using LLM (now Claude)
    report_needed_flag = interpret_voice_response(speech_result, original_alert) # Pass only required args
    
    if report_needed_flag == 1:
        logging.info(f"--> Claude Interpretation: REPORT/ASSISTANCE NEEDED for {caller_phone_number}.")
        resp.say("Your request has been noted. Relevant authorities will be contacted.", voice='Polly.Joanna', language="en-US") # Modified voice response
        logging.info(f"ACTION NEEDED: User {caller_phone_number} requested report/assistance regarding alert: {original_alert} | Response: {speech_result}")
        print(f"ACTION NEEDED: User {caller_phone_number} requested report/assistance regarding alert: {original_alert} | Response: {speech_result}") 

        # --- Initiate Call to Emergency Number --- 
        if emergency_phone_number and client and twilio_phone_number:
            try:
                # Prepare the message to be spoken on the call
                spoken_message = f"[코드웨이브 긴급 알림] 사용자 {caller_phone_number} 로부터 지원 요청 또는 신고가 접수되었습니다. 사용자 음성 메시지: {speech_result}"
                # Create TwiML for the outbound call
                emergency_call_twiml = VoiceResponse()
                emergency_call_twiml.say(spoken_message, voice='Polly.Seoyeon', language='ko-KR')
                emergency_call_twiml.hangup()
                
                logging.info(f"  -> Initiating call to EMERGENCY_PHONE_NUMBER ({emergency_phone_number}) from {twilio_phone_number}")
                emergency_call = client.calls.create(
                    twiml=str(emergency_call_twiml),
                    to=emergency_phone_number,
                    from_=twilio_phone_number
                )
                logging.info(f"    Emergency call initiated! SID: {emergency_call.sid}")
            except Exception as call_err:
                logging.error(f"    ERROR initiating emergency call to {emergency_phone_number}: {call_err}", exc_info=True)
        else:
            logging.warning("Could not initiate emergency call: EMERGENCY_PHONE_NUMBER not configured or Twilio client issue.")
        # --------------------------------------

    else:
        logging.info(f"--> Claude Interpretation: Report/Assistance NOT needed for {caller_phone_number}.")
        resp.say("Okay. Ending call.", voice='Polly.Joanna', language="en-US")
        
    logging.info("Appending Hangup TwiML.")
    resp.hangup()
    
    final_twiml = str(resp)
    logging.info(f"Generated final TwiML response: {final_twiml}")
    logging.info(f"--- Exiting /handle-voice-alert-response --- ")
    return Response(content=final_twiml, media_type="application/xml")

app.include_router(twilio_router)

# --- Frontend and Map Serving Endpoints ---

# Modify map serving endpoint to accept the ID in the path
@app.get("/map/{map_file_path:path}", response_class=HTMLResponse, tags=["Serving"])
async def serve_map_or_html(request: Request, map_file_path: str):
    """Serves map HTML files, extracting ID if path matches disaster_map pattern."""
    
    # Check if the request is for the disaster map with an ID using regex
    # Pattern: starts with 'disaster_map/', followed by one or more non-slash characters (the ID)
    match = re.match(r"^disaster_map/([^/]+)$", map_file_path)
    if match:
        simulation_id = match.group(1) # Extracted simulation ID (SN)
        template_name = "disaster_map.html"
        context = {
            "request": request, 
            "kakao_map_app_key": kakao_map_app_key, # Still needed for Kakao map script
            # Pass the extracted ID to the template context as 'simulation_id'
            "simulation_id": simulation_id 
        }
        logging.info(f"Serving disaster map for simulation ID: {simulation_id}")
    elif map_file_path.endswith(".html"):
        # Serve other potential HTML files directly
        template_name = map_file_path
        # For other HTML files, simulation_id is not relevant
        context = {"request": request, "kakao_map_app_key": kakao_map_app_key} 
        logging.info(f"Serving general map/HTML file: {template_name}")
    else:
        # If it's not disaster_map/{id} or an .html file, return 404
        logging.warning(f"Map file path not recognized: {map_file_path}")
        raise HTTPException(status_code=404, detail="Map file not found")

    try:
        # Check if the template exists before trying to render
        template_path = templates_dir / template_name
        if not template_path.is_file():
            logging.warning(f"Map template file not found at: {template_path}")
            raise HTTPException(status_code=404, detail=f"Map template '{template_name}' not found.")
            
        # Render the template with the determined context
        return templates.TemplateResponse(template_name, context)
    except Exception as e:
        # Log the specific error for debugging
        logging.error(f"Error rendering template {template_name}: {e}", exc_info=True)
        # Re-raise a generic 404 if the template wasn't found, otherwise 500
        if isinstance(e, HTTPException) and e.status_code == 404:
             raise e
        raise HTTPException(status_code=500, detail=f"Error rendering map template '{template_name}'")

# --- Frontend Serving (Catch-all for React Router) --- 
@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
async def serve_react_app(request: Request, full_path: str):
    index_path = root_static_dir / "index.html" # root_static_dir 사용 확인
    if not index_path.is_file():
        logging.error(f"Frontend index.html not found at: {index_path}")
        return HTMLResponse(content="Frontend not built or index.html not found.", status_code=503)
    return HTMLResponse(content=index_path.read_text())
# ----------------------------------------------------

# --- Database and RAG Setup Function (called on startup) ---
def setup_database_and_rag(app: FastAPI):
    global vectorstore # Declare intent to modify the global variable
    global chat_model # Declare intent to modify the global variable
    
    # Initialize active_simulations dictionary in app.state
    print("Initializing active_simulations dictionary in app.state...")
    app.state.active_simulations = {}
    print("active_simulations initialized.")

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

    # --- Initialize Chat Model --- 
    print("Attempting to initialize ChatUpstage model...")
    if upstage_api_key:
        try:
            chat_model = ChatUpstage(api_key=upstage_api_key, model="solar-pro") # Or use a different model if needed
            app.state.chat_model = chat_model # Store in app state
            print("ChatUpstage model initialized successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR during ChatUpstage model initialization: {e}")
            chat_model = None
            app.state.chat_model = None
    else:
        print("  WARNING: UPSTAGE_API_KEY not found. Chat model cannot be initialized.")
        chat_model = None
        app.state.chat_model = None

# --- Main Execution Block (No table creation here anymore) ---
if __name__ == "__main__":
    # create_db_tables() # Removed from here
    print("Starting Uvicorn server (directly running script)...") 
    # Note: Running script directly might not use reload properly
    # It's better to run with 'uvicorn flask_twilio_demo.app:app --reload'
    uvicorn.run(app, host='0.0.0.0', port=30000)