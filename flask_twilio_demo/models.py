from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    vulnerability_type = Column(String, index=True) # 어떤 취약계층인가
    address = Column(String, nullable=True) # 어디에 사는가
    phone_number = Column(String, unique=True, index=True) # 번호는 무엇인가
    has_guardian = Column(Boolean, default=False) # 보호자 연결 여부
    guardian_phone_number = Column(String, nullable=True) # 보호자 번호
    wants_info_call = Column(Boolean, default=True) # 안내 전화 여부
    
    # New columns for last voice response
    last_voice_response = Column(String, nullable=True) # Store the last recognized speech
    last_response_timestamp = Column(DateTime(timezone=True), nullable=True) # Timestamp of the last response

    # --- 위도/경도 컬럼 추가 ---
    latitude = Column(Float, nullable=True) # 위도
    longitude = Column(Float, nullable=True) # 경도
    # -------------------------

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 