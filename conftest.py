"""Глобальный conftest: переопределяем БД и Redis ДО импорта модулей приложения."""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest

from models import Base, SessionLocal, engine


@pytest.fixture(autouse=True)
def _reset_db():
    """Создаём таблицы перед каждым тестом и чистим их после."""
    Base.metadata.create_all(engine)
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
