"""Integration tests против реального Postgres.

Проверяем, что ORM-модель и SQL-запросы корректно работают на Postgres
(а не только на in-memory sqlite, как в unit-тестах).
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import app as app_module
from models import SessionLocal, Violation


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def session():
    s = SessionLocal()
    try:
        s.execute(text("TRUNCATE TABLE violations RESTART IDENTITY CASCADE"))
        s.commit()
        yield s
    finally:
        s.close()


@pytest.mark.integration
def test_violation_round_trip_on_postgres(session):
    v = Violation(
        video_name="real.mp4",
        track_id=10,
        frame_idx=200,
        bbox="0,0,1,1",
        ratio_no_helmet=0.95,
        image_path="/tmp/real.jpg",
    )
    session.add(v)
    session.commit()

    stored = session.query(Violation).filter_by(track_id=10).first()
    assert stored is not None
    assert stored.video_name == "real.mp4"
    assert stored.created_at is not None


@pytest.mark.integration
def test_violations_endpoint_returns_recent_first(client, session):
    for i in range(3):
        session.add(
            Violation(
                video_name=f"v{i}.mp4",
                track_id=i,
                frame_idx=i * 10,
                bbox="0,0,1,1",
                ratio_no_helmet=0.9,
                image_path=f"/tmp/{i}.jpg",
            )
        )
    session.commit()

    response = client.get("/violations")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # ORDER BY created_at DESC — последняя вставленная сверху
    assert data[0]["video_name"] == "v2.mp4"
