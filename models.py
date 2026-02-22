from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

class Base(DeclarativeBase):
    pass

class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True)
    video_name = Column(String, index=True)
    track_id = Column(Integer, index=True)
    frame_idx = Column(Integer)
    bbox = Column(String)
    ratio_no_helmet = Column(Float)
    image_path = Column(String)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

engine = create_engine(settings.sqlalchemy_url)
SessionLocal = sessionmaker(bind=engine)

def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print("Таблица violations готова!")
