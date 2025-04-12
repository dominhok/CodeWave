from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
    address = Column(String) # 어디에 사는가
    phone_number = Column(String, unique=True, index=True) # 번호는 무엇인가
    has_guardian = Column(Boolean, default=False) # 보호자 연결 여부
    guardian_phone_number = Column(String, nullable=True) # 보호자 번호
    wants_info_call = Column(Boolean, default=True) # 안내 전화 여부
    is_visually_impaired = Column(Boolean, default=False, nullable=False) # 시각장애 여부 플래그 추가

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 