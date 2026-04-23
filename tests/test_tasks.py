"""Тесты Celery-задачи analyze_video_task в eager-режиме без брокера и YOLO."""
import pytest

from celery_app import celery_app
from tasks import analyze_video_task


@pytest.fixture(autouse=True)
def _eager_celery():
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


def test_task_raises_if_video_missing():
    with pytest.raises(FileNotFoundError):
        analyze_video_task.apply(args=["/no/such/file.mp4"]).get()


def test_task_returns_expected_payload_shape(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a real video but file exists")

    mock_payload = {"violations": [{"track_id": 1}], "frames": 123}
    monkeypatch.setattr("tasks.analyze_video", lambda *a, **kw: mock_payload)

    result = analyze_video_task.apply(args=[str(video)]).get()

    assert result["video_name"] == "clip.mp4"
    assert result["violations_count"] == 1
    assert result["frames_processed"] == 123
    assert "download_url" in result
    assert result["download_url"].endswith("/download_video/out_clip.mp4")
