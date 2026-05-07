from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

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


def _make_engine(url: str):
    # SQLite :memory: с StaticPool используется в тестах — одна shared-конекция
    # между фикстурами и TestClient'ом FastAPI.
    if ":memory:" in url:
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url)


engine = _make_engine(settings.sqlalchemy_url)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print("Таблица violations готова!")
