"""API-тесты через FastAPI TestClient. Celery замокан, YOLO не грузится."""

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app as app_module
from models import Violation


@pytest.fixture
def client():
    return TestClient(app_module.app)


class TestAnalyzeVideo:
    def test_rejects_non_video_extension(self, client):
        fake = io.BytesIO(b"hello")
        response = client.post(
            "/analyze_video",
            files={"file": ("notes.txt", fake, "text/plain")},
        )
        assert response.status_code == 400

    def test_queues_task_and_returns_id(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module.settings, "videos_dir", str(tmp_path))

        fake_task = MagicMock()
        fake_task.id = "fake-task-123"
        monkeypatch.setattr(
            app_module.celery_app,
            "send_task",
            MagicMock(return_value=fake_task),
        )

        fake_video = io.BytesIO(b"fake bytes")
        response = client.post(
            "/analyze_video",
            files={"file": ("video.mp4", fake_video, "video/mp4")},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "fake-task-123"
        assert data["status_url"] == "/tasks/fake-task-123"
        assert data["video_name"] == "video.mp4"


class TestTaskStatus:
    def test_pending_state(self, client, monkeypatch):
        fake_result = MagicMock(state="PENDING")
        monkeypatch.setattr(app_module, "AsyncResult", lambda *a, **kw: fake_result)

        response = client.get("/tasks/xyz")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "PENDING"
        assert data["task_id"] == "xyz"
        assert "detail" in data

    def test_progress_state(self, client, monkeypatch):
        fake_result = MagicMock(state="PROGRESS", info={"current": 100, "total": 500})
        monkeypatch.setattr(app_module, "AsyncResult", lambda *a, **kw: fake_result)

        response = client.get("/tasks/xyz")
        assert response.json()["progress"] == {"current": 100, "total": 500}

    def test_success_state_returns_result(self, client, monkeypatch):
        fake_result = MagicMock(state="SUCCESS", result={"violations_count": 3, "violations": []})
        monkeypatch.setattr(app_module, "AsyncResult", lambda *a, **kw: fake_result)

        response = client.get("/tasks/xyz")
        data = response.json()
        assert data["state"] == "SUCCESS"
        assert data["result"]["violations_count"] == 3

    def test_failure_state_exposes_error(self, client, monkeypatch):
        fake_result = MagicMock(state="FAILURE", info=RuntimeError("boom"))
        monkeypatch.setattr(app_module, "AsyncResult", lambda *a, **kw: fake_result)

        response = client.get("/tasks/xyz")
        data = response.json()
        assert data["state"] == "FAILURE"
        assert "boom" in data["error"]


class TestViolationsEndpoint:
    def test_empty_list(self, client):
        response = client.get("/violations")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_recent_violations(self, client, db_session):
        db_session.add(
            Violation(
                video_name="v.mp4",
                track_id=1,
                frame_idx=1,
                bbox="1,1,1,1",
                ratio_no_helmet=0.9,
                image_path="/tmp/viol.jpg",
            )
        )
        db_session.commit()

        response = client.get("/violations")
        data = response.json()
        assert len(data) == 1
        assert data[0]["video_name"] == "v.mp4"
        assert data[0]["ratio_no_helmet"] == 0.9
        assert data[0]["image_url"] == "/violations/viol.jpg"


class TestMetricsEndpoint:
    def test_metrics_endpoint_exposes_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus exposition format: текстовые HELP/TYPE директивы
        assert "# HELP" in response.text


class TestApiKeyAuth:
    def test_no_key_required_when_api_key_unset(self, client, monkeypatch):
        # По умолчанию api_key=None → auth выключен, существующие тесты это и проверяют.
        # Здесь явно подтверждаем, что без header'а POST проходит.
        monkeypatch.setattr(app_module.settings, "api_key", None)

        fake_task = MagicMock(id="t-1")
        monkeypatch.setattr(app_module.celery_app, "send_task", MagicMock(return_value=fake_task))
        fake_video = io.BytesIO(b"data")
        response = client.post("/analyze_video", files={"file": ("a.mp4", fake_video, "video/mp4")})
        assert response.status_code == 202

    def test_missing_key_returns_401(self, client, monkeypatch):
        monkeypatch.setattr(app_module.settings, "api_key", "secret")
        fake_video = io.BytesIO(b"data")
        response = client.post("/analyze_video", files={"file": ("a.mp4", fake_video, "video/mp4")})
        assert response.status_code == 401

    def test_wrong_key_returns_403(self, client, monkeypatch):
        monkeypatch.setattr(app_module.settings, "api_key", "secret")
        fake_video = io.BytesIO(b"data")
        response = client.post(
            "/analyze_video",
            files={"file": ("a.mp4", fake_video, "video/mp4")},
            headers={"X-API-Key": "wrong"},
        )
        assert response.status_code == 403

    def test_correct_key_passes(self, client, monkeypatch):
        monkeypatch.setattr(app_module.settings, "api_key", "secret")
        fake_task = MagicMock(id="t-2")
        monkeypatch.setattr(app_module.celery_app, "send_task", MagicMock(return_value=fake_task))
        fake_video = io.BytesIO(b"data")
        response = client.post(
            "/analyze_video",
            files={"file": ("a.mp4", fake_video, "video/mp4")},
            headers={"X-API-Key": "secret"},
        )
        assert response.status_code == 202


class TestHealthEndpoint:
    def test_healthy_when_all_deps_ok(self, client, monkeypatch, tmp_path):
        # Подкладываем фейковую модель, чтобы health-check её нашёл
        fake_model = tmp_path / "best.pt"
        fake_model.write_bytes(b"fake")
        monkeypatch.setattr(app_module.settings, "model_path", str(fake_model))

        # Мокаем Redis.ping
        fake_redis = MagicMock()
        fake_redis.ping.return_value = True
        monkeypatch.setattr(app_module.redis_lib.Redis, "from_url", lambda *a, **kw: fake_redis)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["checks"]["database"]["ok"] is True
        assert data["checks"]["redis"]["ok"] is True
        assert data["checks"]["model"]["ok"] is True

    def test_degraded_when_redis_down(self, client, monkeypatch, tmp_path):
        fake_model = tmp_path / "best.pt"
        fake_model.write_bytes(b"fake")
        monkeypatch.setattr(app_module.settings, "model_path", str(fake_model))

        broken_redis = MagicMock()
        broken_redis.ping.side_effect = ConnectionError("redis down")
        monkeypatch.setattr(app_module.redis_lib.Redis, "from_url", lambda *a, **kw: broken_redis)

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["checks"]["redis"]["ok"] is False

    def test_degraded_when_model_missing(self, client, monkeypatch, tmp_path):
        # Указываем заведомо несуществующий путь
        monkeypatch.setattr(app_module.settings, "model_path", str(tmp_path / "nope.pt"))

        fake_redis = MagicMock()
        fake_redis.ping.return_value = True
        monkeypatch.setattr(app_module.redis_lib.Redis, "from_url", lambda *a, **kw: fake_redis)

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["checks"]["model"]["ok"] is False
