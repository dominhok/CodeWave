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
import time  # time 모듈 임포트
import urllib.parse
import re

from . import models
from .models import Base, engine, User, get_db

# --- Langchain Imports ---
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage  # Use ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    DirectoryLoader,
)  # Using community loader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# --- Global Variables for RAG ---
# Initialize vectorstore to None; will be loaded/created on startup
vectorstore = None
VECTORSTORE_PATH = "faiss_disaster_manuals"  # Path to save/load FAISS index


# --- Register Startup Event ---
@app.on_event("startup")
def on_startup():
    setup_database_and_rag(app)  # Combined setup function


# --- Dependency to get DB session (Modified) ---
def get_db(request: Request) -> Session:
    SessionLocal = getattr(request.app.state, "SessionLocal", None)
    if SessionLocal is None:
        # This might happen if startup failed critically
        raise HTTPException(
            status_code=500, detail="Database session factory not available."
        )
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- CORS Middleware Configuration ---
# Allow requests from the React dev server (typically http://localhost:3000)
origins = [
    "http://localhost:3000",  # React default dev port
    # Add other origins if needed, e.g., your frontend production URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allow relevant methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files directory
static_dir = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configure Jinja2 templates
templates_dir = static_dir / "maps"  # Templates are in static/maps
templates = Jinja2Templates(directory=templates_dir)

# Twilio credentials from .env file
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
kakao_map_app_key = os.getenv("KAKAO_MAP_APP_KEY")
safety_data_service_key = os.getenv("SAFETY_DATA_SERVICE_KEY")

logging.info(f"TWILIO_ACCOUNT_SID: {account_sid}")
logging.info(
    f"auth_token: {'*' * len(auth_token) if auth_token else None}"
)  # 토큰은 로그에 직접 노출하지 않도록 처리
logging.info(f"TWILIO_PHONE_NUMBER: {twilio_phone_number}")

# Check if credentials are loaded
if not account_sid or not auth_token or not twilio_phone_number:
    # Optionally, you might want to handle this more gracefully
    # depending on whether Twilio functionality is always required.
    print(
        "Warning: Twilio credentials not fully found in .env file. Twilio features may not work."
    )
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
    logging.warning(
        "Twilio credentials (account_sid or auth_token) not found in environment variables."
    )


# --- Pydantic Schemas ---
class UserBase(BaseModel):
    vulnerability_type: str
    address: str
    phone_number: str
    has_guardian: bool = False
    guardian_phone_number: Optional[str] = None
    wants_info_call: bool = True


class UserCreate(UserBase):
    pass  # Inherits all fields from UserBase


class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True  # Changed from orm_mode = True for Pydantic v2


class CallRequest(BaseModel):
    to: str
    message: str


class SmsRequest(BaseModel):
    to: str
    message: str


# --- NEW: Pydantic model based on Real Disaster API Output ---
class DisasterAlertData(BaseModel):
    SN: str  # 일련번호
    CRT_DT: str  # 생성일시 (문자열로 받음, 필요시 파싱)
    MSG_CN: str  # 메시지내용
    RCPTN_RGN_NM: str  # 수신지역명
    EMRG_STEP_NM: Optional[str] = None  # 긴급단계명 (선택적)
    DST_SE_NM: str  # 재해구분명
    REG_YMD: Optional[str] = None  # 등록일자 (선택적)
    MDFCN_YMD: Optional[str] = None  # 수정일자 (선택적)


class DisasterAlertTestRequest(BaseModel):
    disaster_alert: DisasterAlertData
    user_type: str  # 사용자 유형 (e.g., "elderly", "disabled", "general")


# --- Specialized evacuation manual based on user types ---
class SpecializedAlertRequest(BaseModel):
    disaster_alert: DisasterAlertData
    user_type: str  # 'normal', 'visually_impaired', or 'hearing_impaired'


# --- CRUD Operations (simplified within app.py) ---
def get_user_by_phone(db: Session, phone_number: str):
    return db.query(User).filter(User.phone_number == phone_number).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate):
    db_user = User(**user.model_dump())  # Use model_dump() for Pydantic v2
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# --- Helper Functions for Disaster Simulation Logic ---


def filter_target_users(disaster_alert: DisasterAlertData, db: Session) -> List[User]:
    """Filters users based on the disaster reception region."""
    target_location_keyword = "성동구"
    # if disaster_alert.RCPTN_RGN_NM:
    #     parts = disaster_alert.RCPTN_RGN_NM.split()
    #     if len(parts) > 1:
    #         target_location_keyword = parts[-1]
    #         if "전체" in target_location_keyword or "," in disaster_alert.RCPTN_RGN_NM:
    #             target_location_keyword = parts[0]
    #     else:
    #         target_location_keyword = disaster_alert.RCPTN_RGN_NM

    filtered_users = []
    if target_location_keyword:
        print(f"Filtering users in location containing: '{target_location_keyword}'")
        try:
            filtered_users = (
                db.query(User)
                .filter(User.address.contains(target_location_keyword))
                .all()
            )
            print(f"Found {len(filtered_users)} users to notify.")
        except Exception as e:
            print(f"Error filtering users: {e}")
            # Return empty list on error or handle differently
            return []
    else:
        print("No location keyword to filter by from RCPTN_RGN_NM.")

    return filtered_users


# --- RAG Based Message Generation (Target Length  No URL, No Ellipsis) ---
def generate_notification_messages(
    disaster_alert: DisasterAlertData, users: List[User]
) -> List[dict]:
    """Generates structured notification messages using RAG, without URL or ellipsis on truncation."""
    notifications = []
    # Base alert in English, formatted concisely
    base_alert_info = f"{disaster_alert.DST_SE_NM} in {disaster_alert.RCPTN_RGN_NM}"
    print(f"Base Alert Info for RAG query: {base_alert_info}")

    # Group users by vulnerability type to generate type-specific messages
    user_types = {}
    for user in users:
        vulnerability_type = user.vulnerability_type
        if vulnerability_type not in user_types:
            user_types[vulnerability_type] = []
        user_types[vulnerability_type].append(user)
    print(user_types)
    # Map vulnerability_type values to recognized user types for RAG
    vulnerability_type_mapping = {
        "visually_impaired": "visually_impaired",
        "hearing_impaired": "hearing_impaired",
        "시각장애": "visually_impaired",
        "청각장애": "hearing_impaired",
        "normal": "normal",
        "일반": "normal",
        # Add more mappings as needed
    }

    # Process each user type separately
    for vulnerability_type, type_users in user_types.items():
        # Map to normalized type for RAG
        mapped_type = vulnerability_type_mapping.get(
            vulnerability_type.lower(), "normal"
        )
        print(
            f"Processing {len(type_users)} users with vulnerability type: {vulnerability_type} (mapped to: {mapped_type})"
        )

        # Adjust target length for RAG - VERY Short target
        target_rag_len = 70

        rag_response_content = ""  # Initialize with empty string
        if vectorstore is None:
            print("  WARNING: Vectorstore not available. Cannot generate RAG content.")
            # Fallback message (no URL)
            fallback_base = f"ALERT: {base_alert_info}. Follow local guidance."
            rag_response_content = fallback_base

        else:
            try:
                upstage_api_key = os.getenv("UPSTAGE_API_KEY")
                llm = ChatUpstage(api_key=upstage_api_key, model="solar-pro")
                retriever = vectorstore.as_retriever()

                # Customize prompt template based on user type
                if mapped_type == "visually_impaired":
                    user_type_guidance = "For visually impaired individuals. Focus on audio cues and tactile guidance."
                    action_guidance = (
                        "Include audio cues and non-visual navigation instructions."
                    )
                elif mapped_type == "hearing_impaired":
                    user_type_guidance = (
                        "For hearing impaired individuals. Emphasize visual alerts."
                    )
                    action_guidance = "Include visual cues and text-based alerts."
                else:
                    user_type_guidance = (
                        "For general population. Use clear instructions."
                    )
                    action_guidance = "Focus on simplicity and clarity."

                # --- Updated English Prompt  ---
                prompt_template = f"""
                You MUST generate an emergency alert message in ENGLISH language ONLY.
                
                Your response must be in ENGLISH, not Korean. Alert should be concise.
                This alert is for {mapped_type} individuals.

                Required Information (must be included):
                1. Disaster Type: [from {{question}}]
                2. Location: [from {{question}}]
                3. 1 brief CRITICAL immediate action suitable for {mapped_type} people.

                USER TYPE GUIDANCE:
                {user_type_guidance}
                {action_guidance}

                Context Documents (for action guidance):
                {{context}}

                OUTPUT ONLY THE ENGLISH SMS-ready alert text. NO KOREAN TEXT. ABSOLUTELY NO EXTRA TEXT.
                Example: "ALERT: Earthquake Seoul. Drop, Cover, Hold On. Listen for audio instructions."

                ENGLISH ONLY Alert Message:
                """
                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=False,
                )

                # Add user type to query for more relevant results
                query = f"{mapped_type} evacuation during {base_alert_info}"
                print(f"  RAG Query: {query}")

                result = qa_chain.invoke({"query": query})
                rag_summary_text = result.get("result", "").strip()

                print(
                    f"  RAG Generated Response (raw) for {mapped_type}:\n{rag_summary_text}"
                )

                # --- Simplified Truncation Logic (NO ELLIPSIS) ---
                if not rag_summary_text:  # If RAG failed or returned empty
                    print(
                        f"  WARNING: RAG generation resulted in empty string for {mapped_type}. Using fallback."
                    )
                    fallback_base = f"ALERT: {base_alert_info}. Check official sources."

                    rag_response_content = fallback_base
                else:
                    # Use the generated text directly
                    rag_response_content = rag_summary_text

            except Exception as e:
                print(f"  ERROR during RAG generation for {mapped_type}: {e}")
                # Fallback message on error (no URL)
                fallback_base = f"ALERT: {base_alert_info}. Error getting details. Follow local guidance."
                rag_response_content = fallback_base

        # --- End RAG Generation ---

        # Log the final generated message content and its length
        logging.info(
            f"Final generated message content for {mapped_type} (Alert only,no ellipsis):\n{rag_response_content}"
        )
        logging.info(
            f"---> Final Alert message length: {len(rag_response_content)} chars"
        )

        # Add the message for each user of this type
        for user in type_users:
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
        return {
            "success": successful_notifications,
            "failed": len(notifications_to_send),
        }

    # Construct the map URL message (needs base_url)
    # TODO: Get base URL from environment variable or config for flexibility
    base_url = "https://ff74-121-160-208-235.ngrok-free.app"
    full_map_url = f"{base_url}/map/map_api"
    map_message_body = (
        f"Nearby shelters/safety info: {full_map_url}"  # Slightly shortened prefix
    )

    print("--- SENDING REAL TWILIO NOTIFICATIONS (2-Step: Map URL + Alert) --- ")
    logging.info("--- Starting to send real Twilio notifications (Map URL + Alert) ---")
    for item in notifications_to_send:
        user = item["user"]
        alert_message_body = item[
            "message"
        ]  # This is the shorter alert message generated by RAG

        try:
            # 1. Send the MAP URL SMS first
            logging.info(
                f"  1 -> Attempting to send MAP URL SMS to {user.phone_number} (User ID: {user.id})"
            )
            message1 = client.messages.create(
                body=map_message_body, from_=twilio_phone_number, to=user.phone_number
            )
            logging.info(
                f"    Map URL SMS sent successfully! SID: {message1.sid} to {user.phone_number}"
            )

            # 2. If map URL SMS was successful, wait briefly and send the main alert message
            try:
                logging.info(f"  ... Waiting 0.5 seconds before sending alert SMS ...")
                time.sleep(0.5)  # 0.5초 지연 추가

                logging.info(
                    f"  2 -> Attempting to send ALERT SMS to {user.phone_number} (User ID: {user.id}). Content:\n{alert_message_body}"
                )
                message2 = client.messages.create(
                    body=alert_message_body,
                    from_=twilio_phone_number,
                    to=user.phone_number,
                )
                logging.info(
                    f"    Alert SMS sent successfully! SID: {message2.sid} to {user.phone_number}"
                )
                successful_notifications += (
                    1  # Count success only if both messages are sent
                )

            except Exception as e2:
                logging.error(
                    f"    ERROR sending ALERT SMS to {user.phone_number} after successful map URL: {e2}",
                    exc_info=True,
                )
                failed_notifications += 1  # Map URL sent, but alert failed

        except Exception as e1:
            # This now represents failure sending the MAP URL
            logging.error(
                f"    ERROR sending MAP URL SMS to {user.phone_number}: {e1}",
                exc_info=True,
            )
            failed_notifications += 1  # Map URL failed, didn't attempt alert

    logging.info(
        f"--- Notification sending complete (Success pairs: {successful_notifications}, Failed attempts: {failed_notifications}) ---"
    )
    return {"success": successful_notifications, "failed": failed_notifications}


# --- FastAPI Endpoints ---

# Prefixing API routes for better organization
api_router = APIRouter(prefix="/api")


@api_router.delete("/users/{phone_number}", tags=["Users"])
def delete_user_endpoint(phone_number: str, db: Session = Depends(get_db)):
    """Delete a user by their phone number"""
    db_user = get_user_by_phone(db, phone_number=phone_number)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Delete the user
        db.delete(db_user)
        db.commit()
        return {
            "status": "success",
            "message": f"User with phone number {phone_number} has been deleted",
        }
    except Exception as e:
        db.rollback()
        print(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")


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
        raise HTTPException(
            status_code=500, detail="Database error during user creation."
        )


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
            body=sms_request.message, from_=twilio_phone_number, to=sms_request.to
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
            from_=twilio_phone_number,
        )
        return {"status": "Call initiated", "sid": call.sid}
    except Exception as e:
        print(f"Error making call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint to proxy Safety Data API calls
@api_router.get("/shelters_proxy", tags=["External APIs"])
async def shelters_proxy(request: Request):
    # Get coordinates from the query string sent by JavaScript
    start_lat = request.query_params.get("startLat")
    end_lat = request.query_params.get("endLat")
    start_lot = request.query_params.get("startLot")
    end_lot = request.query_params.get("endLot")

    # Get service key securely (using the one defined earlier or from env)
    # service_key = os.getenv('SAFETY_DATA_SERVICE_KEY')
    service_key = os.getenv("SAFETY_DATA_SERVICE_KEY")  # Fallback to placeholder

    if not all([start_lat, end_lat, start_lot, end_lot, service_key]):
        print(
            f"Missing parameters: startLat={start_lat}, endLat={end_lat}, startLot={start_lot}, endLot={end_lot}, service_key_present={bool(service_key)}"
        )
        raise HTTPException(
            status_code=400, detail="Missing coordinate parameters or service key"
        )

    # Construct the external API URL
    external_api_url = (
        f"https://www.safetydata.go.kr/V2/api/DSSP-IF-10941?"
        f"serviceKey={service_key}&"
        f"startLot={start_lot}&endLot={end_lot}&"
        f"startLat={start_lat}&endLat={end_lat}&"
        f"pageNo=1&numOfRows=100"  # Fetch up to 100
    )

    print(f"Proxying request to: {external_api_url}")  # Log the external URL call

    async with httpx.AsyncClient() as client:
        try:
            # Make the async request from the server
            response = await client.get(external_api_url, timeout=10.0)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Return the JSON response from the external API
            # FastAPI automatically handles JSON response conversion
            return response.json()

        except httpx.TimeoutException:
            print(f"Timeout error calling external API: {external_api_url}")
            raise HTTPException(
                status_code=504, detail="External API timed out"
            )  # Gateway Timeout
        except httpx.HTTPStatusError as exc:
            print(
                f"HTTP error calling external API: {exc.response.status_code} - {exc.response.text}"
            )
            # Try to return the actual error from the external API if possible
            try:
                error_detail = exc.response.json()
            except:
                error_detail = exc.response.text
            raise HTTPException(
                status_code=exc.response.status_code,
                detail={
                    "error": "Failed to fetch data from external API",
                    "details": error_detail,
                },
            )
        except httpx.RequestError as exc:
            print(f"Request error calling external API: {exc}")
            raise HTTPException(
                status_code=502, detail=f"Error connecting to external API: {exc}"
            )  # Bad Gateway
        except Exception as e:
            print(f"An unexpected error occurred in proxy: {e}")
            raise HTTPException(
                status_code=500, detail="An internal server error occurred"
            )


# --- Refactored endpoint for simulating disaster ---
@api_router.post("/simulate_disaster", tags=["Disaster Simulation"])
async def simulate_disaster(
    disaster_alert: DisasterAlertData, db: Session = Depends(get_db)
):
    print("\n--- STARTING DISASTER SIMULATION (Refactored) ---")
    print(
        f"Received Alert SN: {disaster_alert.SN} for Region: {disaster_alert.RCPTN_RGN_NM}"
    )

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
        "failed_notifications": send_results["failed"],
    }


# --- Specialized evacuation manual based on user types ---
@api_router.post("/specialized_evacuation_manual", tags=["Testing"])
async def specialized_evacuation_manual(request: SpecializedAlertRequest):
    """
    Generates specialized evacuation instructions based on disaster type
    and user vulnerability type using type-specific manual content.

    This endpoint loads content from type-specific text files in how-to-s/type/
    and uses RAG to generate appropriate evacuation guidance.
    """
    # Validate and map user_type to the correct file
    user_type_map = {
        "normal": "normal.txt",
        "visually_impaired": "visually_impaired.txt",  # Note the typo in the filename
        "hearing_impaired": "hearing_impaired.txt",  # Note the typo in the filename
    }

    if request.user_type.lower() not in user_type_map:
        valid_types = ", ".join(user_type_map.keys())
        return {"error": f"Invalid user_type. Must be one of: {valid_types}"}

    # Extract disaster information
    disaster_type = request.disaster_alert.DST_SE_NM
    location = request.disaster_alert.RCPTN_RGN_NM
    base_alert_info = f"{disaster_type} in {location}"

    print(f"\n=== SPECIALIZED EVACUATION MANUAL FOR USER TYPE: {request.user_type} ===")
    print(f"Disaster Type: {disaster_type}")
    print(f"Region: {location}")

    # Step 1: Load the appropriate type-specific manual content for guidance
    manual_content = ""
    try:
        # Get the project root and file path
        project_root = pathlib.Path(__file__).parent.parent
        type_file = user_type_map[request.user_type.lower()]
        manual_path = project_root / "how-to-s" / "type" / type_file

        print(f"Loading manual from: {manual_path}")

        with open(manual_path, "r", encoding="utf-8") as f:
            manual_content = f.read()
            print(f"Successfully loaded {len(manual_content)} characters from manual")
    except Exception as e:
        print(f"Error loading manual file: {e}")
        return {
            "error": f"Failed to load manual for user type '{request.user_type}': {str(e)}"
        }

    # Step 2: Configure the RAG prompt based on user type
    if request.user_type.lower() == "normal":
        user_prompt = "For general population. Use clear, concise instructions."
        user_specific_guidance = (
            "Focus on simplicity and clarity. Include evacuation route information."
        )
    elif request.user_type.lower() == "visually_impaired":
        user_prompt = "For visually impaired individuals. Focus on audio cues and tactile guidance."
        user_specific_guidance = "Include audio cues, tactile landmarks, and non-visual navigation instructions."
    elif request.user_type.lower() == "hearing_impaired":
        user_prompt = "For hearing impaired individuals. Emphasize visual alerts and text-based communication."
        user_specific_guidance = "Include visual cues, text-based alerts, and non-audio communication methods."

    # Step 3: Generate the evacuation manual using RAG with the global vectorstore
    evacuation_manual = ""

    if vectorstore is None:
        print("WARNING: Vectorstore not available. Using manual excerpt directly.")
        # Simple keyword-based fallback without RAG
        relevant_sections = []
        manual_paragraphs = manual_content.split("\n\n")

        # Look for paragraphs mentioning the disaster type
        for para in manual_paragraphs:
            if disaster_type.lower() in para.lower():
                relevant_sections.append(para)

        if relevant_sections:
            evacuation_manual = "\n\n".join(
                relevant_sections[:2]
            )  # Take first two relevant paragraphs
        else:
            evacuation_manual = f"ALERT: {base_alert_info}. Seek immediate safety according to local guidance."
    else:
        try:
            print(
                "Using global vectorstore for RAG to generate specialized evacuation manual..."
            )
            upstage_api_key = os.getenv("UPSTAGE_API_KEY")
            llm = ChatUpstage(api_key=upstage_api_key, model="solar-pro")

            # Use the global vectorstore's retriever directly
            retriever = vectorstore.as_retriever()

            # Extract key user type-specific guidance from the manual
            # Create a more focused query to find relevant advice for this user type
            user_type_query = f"{request.user_type} evacuation during {disaster_type}"

            # Create a prompt template that includes both the user type guidance and specific instructions
            prompt_template = f"""
            Generate a brief, focused evacuation manual for a {request.user_type} person facing a {disaster_type} disaster in {location}.
            
            {user_prompt}
            
            USER TYPE SPECIFIC NEEDS:
            {user_specific_guidance}
            
            CRITICAL GUIDANCE FROM MANUAL:
            {manual_content[:1000]}  # Include the first 1000 chars from the specific manual
            
            Format your response in these sections:
            1. IMMEDIATE ACTIONS (2-3 critical steps)
            2. EVACUATION INSTRUCTIONS (specific to this disaster type)
            3. SAFETY TIPS (focused on the needs of {request.user_type} individuals)
            
            Use the general disaster guidelines from:
            {{context}}
            
            Based on the query: {{question}}
            
            Keep your response under 400 words. Focus on practical, specific guidance.
            """

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False,
            )

            # Generate detailed evacuation instructions for this specific disaster and user type
            query = f"How should a {request.user_type} person evacuate during a {disaster_type} in {location}? Include specific needs for {request.user_type} individuals."
            result = qa_chain.invoke({"query": query})
            evacuation_manual = result.get("result", "").strip()

            if not evacuation_manual:
                evacuation_manual = f"ALERT: {base_alert_info}. Unable to generate specialized instructions."

        except Exception as e:
            print(f"Error during RAG generation: {e}")
            evacuation_manual = f"ALERT: {base_alert_info}. Error retrieving specialized instructions: {str(e)}"

    # Print the generated manual to console
    print(f"\n=== GENERATED EVACUATION MANUAL ===\n{evacuation_manual}\n")
    print("=" * 50)

    # Step 4: Also generate a short SMS alert message (under 300 chars)
    short_alert = ""
    try:
        if vectorstore is not None:
            sms_prompt_template = f"""
            Generate a concise emergency alert SMS for a {request.user_type} person during a {disaster_type}.
            Keep it UNDER 300 characters.
            Include location ({location}) and critical actions.
            For {request.user_type}: {user_prompt}
            
            USER TYPE SPECIFIC GUIDANCE:
            {user_specific_guidance}
            """

            sms_PROMPT = PromptTemplate(
                template=sms_prompt_template, input_variables=["context", "question"]
            )

            sms_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": sms_PROMPT},
                return_source_documents=False,
            )

            sms_result = sms_chain.invoke({"query": base_alert_info})
            short_alert = sms_result.get("result", "").strip()

            if len(short_alert) > 300:
                short_alert = short_alert[:300]

        if not short_alert:
            short_alert = f"ALERT: {base_alert_info}. Seek safety immediately."
            if len(short_alert) > 300:
                short_alert = short_alert[:300]

    except Exception as e:
        print(f"Error generating short alert: {e}")
        short_alert = f"ALERT: {base_alert_info}"
        if len(short_alert) > 300:
            short_alert = short_alert[:300]

    return {
        "user_type": request.user_type,
        "disaster_info": {
            "type": request.disaster_alert.DST_SE_NM,
            "region": request.disaster_alert.RCPTN_RGN_NM,
            "message": request.disaster_alert.MSG_CN,
        },
        "short_alert": short_alert,
        "evacuation_manual": evacuation_manual,
    }


@api_router.get("/news", tags=["External APIs"])
async def get_news(location: str, disaster_type: str, items_per_page: int = 20):
    """
    네이버 검색 API를 사용하여 위치와 재난 유형에 관련된 뉴스를 검색합니다.

    - location: 검색할 지역 이름 (예: 서울, 부산, 성동구)
    - disaster_type: 재난 유형 (예: 지진, 홍수, 화재)
    - items_per_page: 반환할 뉴스 아이템 수 (기본값: 20)
    """
    # 네이버 API 키 확인
    naver_client_id = os.getenv("NAVER_CLIENT_ID")
    naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

    if not naver_client_id or not naver_client_secret:
        raise HTTPException(
            status_code=500,
            detail="Naver API credentials not configured. Please set NAVER_CLIENT_ID and NAVER_CLIENT_SECRET in .env file.",
        )

    # 검색 쿼리 생성 (지역 + 재난 유형)
    query = f"{location} {disaster_type}"
    encoded_query = urllib.parse.quote(query)

    # 네이버 뉴스 검색 API URL
    url = f"https://openapi.naver.com/v1/search/news.json?query={encoded_query}&display={items_per_page}&sort=date"

    # 요청 헤더 설정
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()  # HTTP 오류 발생시 예외 발생

            data = response.json()

            # 응답 데이터 가공 (HTML 태그 제거)
            processed_items = []
            for item in data.get("items", []):
                processed_item = {
                    "title": re.sub(
                        r"<[^>]+>", "", item.get("title", "")
                    ),  # HTML 태그 제거
                    "link": item.get("link", ""),
                    "description": re.sub(
                        r"<[^>]+>", "", item.get("description", "")
                    ),  # HTML 태그 제거
                    "pubDate": item.get("pubDate", ""),
                }
                processed_items.append(processed_item)

            return {
                "query": query,
                "total": data.get("total", 0),
                "start": data.get("start", 1),
                "display": data.get("display", len(processed_items)),
                "items": processed_items,
            }

    except httpx.HTTPStatusError as exc:
        error_msg = f"Error response {exc.response.status_code} from Naver API: {exc.response.text}"
        print(error_msg)
        raise HTTPException(
            status_code=exc.response.status_code, detail="Error from Naver API"
        )

    except httpx.RequestError as exc:
        error_msg = f"Error requesting Naver API: {exc}"
        print(error_msg)
        raise HTTPException(status_code=502, detail="Failed to connect to Naver API")

    except Exception as e:
        error_msg = f"Unexpected error processing news request: {e}"
        print(error_msg)
        raise HTTPException(
            status_code=500, detail="Internal server error processing news request"
        )


# Dummy disaster data keyed by ID
DUMMY_DISASTERS = {
    "earthquake_gangnam": {
        "location": "Gangnam-gu, Seoul",
        "type": "earthquake",
        "scale": "Magnitude 4.1",
        "time": "10:30",
        "coordinates": {"lat": 37.5513455, "lng": 127.0726955},
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
    # Add more dummy data as needed
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


@api_router.get("/emergency_info/{disaster_id}", tags=["Disaster Info"])
async def get_emergency_info(disaster_id: str):
    """
    Returns dummy emergency information for a specific disaster
    """
    # Dummy emergency information based on disaster type
    emergency_info = {
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

    # Get emergency info for requested disaster or default if not found
    info = emergency_info.get(disaster_id, emergency_info["default"])

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
    gather = Gather(
        input="speech",
        action="/twilio/handle-gather",
        method="POST",
        language="ko-KR",
        speechTimeout="auto",
    )
    gather.say(
        "안녕하세요. 재난 안전 시스템입니다. 신고하시려면 신고, 문의사항이 있으시면 문의 라고 말씀해주세요.",
        voice="Polly.Seoyeon",
    )
    resp.append(gather)
    resp.redirect("/twilio/voice")
    return Response(content=str(resp), media_type="application/xml")


@twilio_router.post("/handle-gather", tags=["Twilio Webhooks"])
async def handle_gather(request: Request, db: Session = Depends(get_db)):
    resp = VoiceResponse()
    form = await request.form()
    speech_result = form.get("SpeechResult", "").strip()
    caller_phone_number = form.get("From")
    print(
        f"\n**\n📞 Incoming call from: {caller_phone_number}\n📢 Speech Result: {speech_result}\n**\n"
    )
    if "신고" in speech_result:
        resp.say(
            "신고 접수를 시작하겠습니다. 필요한 정보를 말씀해주세요.",
            voice="Polly.Seoyeon",
        )
        resp.hangup()
    elif "문의" in speech_result:
        resp.say(
            "문의사항 접수를 위해 잠시 후 상담원을 연결해 드리겠습니다.",
            voice="Polly.Seoyeon",
        )
        resp.hangup()
    else:
        resp.say(
            "죄송합니다, 이해하지 못했습니다. 다시 시도해주세요.", voice="Polly.Seoyeon"
        )
        resp.redirect("/twilio/voice")
    return Response(content=str(resp), media_type="application/xml")


app.include_router(twilio_router)

# --- Frontend and Map Serving Endpoints ---


@app.get("/map/{map_file_name}", response_class=HTMLResponse, tags=["Serving"])
async def serve_map(request: Request, map_file_name: str):
    allowed_maps = [
        "minsoo",
        "disaster_map",
        "map_api",
        "map_wide",
        "shelter_temp",
        "temp",
    ]
    base_map_name = map_file_name.replace(".html", "")
    if base_map_name not in allowed_maps:
        raise HTTPException(status_code=404, detail="Map not found")

    template_name = f"{base_map_name}.html"

    # --- DEBUGGING: Print the key value being used ---
    print(f"DEBUG: Using KAKAO_MAP_APP_KEY: {kakao_map_app_key}")
    # -------------------------------------------------

    context = {"request": request, "kakao_map_app_key": kakao_map_app_key}
    try:
        return templates.TemplateResponse(template_name, context)
    except Exception as e:
        print(f"Error rendering template {template_name}: {e}")
        raise HTTPException(
            status_code=404, detail="Map template not found or error rendering"
        )


# New route that handles disaster_map with disaster_id as path parameter
@app.get(
    "/map/disaster_map/{disaster_id}", response_class=HTMLResponse, tags=["Serving"]
)
async def serve_disaster_map_with_id(request: Request, disaster_id: str):
    template_name = "disaster_map.html"

    # Add the disaster_id to the context, which can be used in the template if needed
    context = {
        "request": request,
        "kakao_map_app_key": kakao_map_app_key,
        "disaster_id": disaster_id,
    }

    try:
        return templates.TemplateResponse(template_name, context)
    except Exception as e:
        print(
            f"Error rendering template {template_name} with disaster_id {disaster_id}: {e}"
        )
        raise HTTPException(
            status_code=404, detail="Map template not found or error rendering"
        )


@app.get("/", response_class=HTMLResponse, tags=["Serving"])
async def serve_frontend_app():
    index_path = static_dir / "app" / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            content="Frontend not built or index.html not found in static/app",
            status_code=404,
        )
    return HTMLResponse(content=index_path.read_text(), status_code=200)


# --- Database and RAG Setup Function (called on startup) ---
def setup_database_and_rag(app: FastAPI):
    global vectorstore  # Declare intent to modify the global variable
    print("Attempting to create database engine and tables...")
    try:
        DATABASE_URL = os.getenv(
            "DATABASE_URL", "sqlite:///./flask_twilio_demo/users.db"
        )
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
        return  # Stop further setup if DB fails

    # --- Build/Load FAISS Vectorstore ---
    print("Attempting to load or build FAISS vectorstore...")
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    if not upstage_api_key or upstage_api_key == "YOUR_UPSTAGE_API_KEY":
        print(
            "  WARNING: UPSTAGE_API_KEY not found or not set in .env. RAG will not work."
        )
        return  # Cannot proceed without API key

    project_root = pathlib.Path(__file__).parent.parent
    vectorstore_full_path = project_root / VECTORSTORE_PATH
    how_to_dir = project_root / "how-to-s"
    how_to_type_dir = project_root / "how-to-s" / "type"

    try:
        # --- Specify the embedding model ---
        embeddings = UpstageEmbeddings(
            api_key=upstage_api_key, model="solar-embedding-1-large"
        )
        # -----------------------------------

        if vectorstore_full_path.exists():
            print(f"Loading existing vectorstore from: {vectorstore_full_path}")
            vectorstore = FAISS.load_local(
                str(vectorstore_full_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print("Vectorstore loaded successfully.")
        else:
            print(
                f"Building new vectorstore from .txt files in: {how_to_dir} and {how_to_type_dir}"
            )
            if not how_to_dir.is_dir():
                print(f"  ERROR: Directory not found: {how_to_dir}")
                return

            if not how_to_type_dir.is_dir():
                print(f"  ERROR: Directory not found: {how_to_type_dir}")
                return

            documents = []
            # Load regular how-to documents
            for txt_file in how_to_dir.glob("*.txt"):
                try:
                    print(f"  Reading: {txt_file.name}")
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read()
                        # Add filename as metadata for potential future use
                        documents.append(
                            Document(
                                page_content=text, metadata={"source": txt_file.name}
                            )
                        )
                except Exception as e:
                    print(f"  Error reading file {txt_file.name}: {e}")

            # Load type-specific how-to documents
            for txt_file in how_to_type_dir.glob("*.txt"):
                try:
                    print(f"  Reading type-specific file: {txt_file.name}")
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read()
                        # Add filename and type metadata for future use
                        type_name = txt_file.stem  # e.g., "visually_impaired"
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": f"type/{txt_file.name}",
                                    "user_type": type_name,
                                },
                            )
                        )
                except Exception as e:
                    print(f"  Error reading type file {txt_file.name}: {e}")

            if not documents:
                print("  ERROR: No documents found or read from how-to-s directories.")
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
        vectorstore = None  # Ensure vectorstore is None on error


# --- Main Execution Block (No table creation here anymore) ---
if __name__ == "__main__":
    # create_db_tables() # Removed from here
    print("Starting Uvicorn server (directly running script)...")
    # Note: Running script directly might not use reload properly
    # It's better to run with 'uvicorn flask_twilio_demo.app:app --reload'
    uvicorn.run(app, host="0.0.0.0", port=30000)
