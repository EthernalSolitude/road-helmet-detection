"""Integration test для /health: проверяем с реальными Postgres + Redis."""

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.mark.integration
def test_health_returns_ok_with_real_deps(client, tmp_path, monkeypatch):
    # Подкладываем фейковую модель — реальный best.pt в CI не нужен
    fake_model = tmp_path / "best.pt"
    fake_model.write_bytes(b"fake")
    monkeypatch.setattr(app_module.settings, "model_path", str(fake_model))

    response = client.get("/health")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    # Реальные Postgres и Redis из GHA services должны отвечать
    assert data["checks"]["database"]["ok"] is True
    assert data["checks"]["redis"]["ok"] is True
    assert data["checks"]["model"]["ok"] is True


@pytest.mark.integration
def test_health_503_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(app_module.settings, "model_path", "/nonexistent/path.pt")

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["checks"]["model"]["ok"] is False
