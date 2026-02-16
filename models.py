from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://helmet_user:1234@localhost:5433/helmet_db")

class Base(DeclarativeBase):
    pass

class Violation(Base):
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True)
    video_name = Column(String, index=True)
    track_id = Column(Integer, index=True)
    frame_idx = Column(Integer)
    bbox = Column(String)  # "x1,y1,x2,y2"
    ratio_no_helmet = Column(Float)
    image_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print(" Таблица violations готова!")
