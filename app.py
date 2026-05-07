import glob
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import redis as redis_lib
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.orm import Session

from auth import require_api_key
from celery_app import celery_app
from config import settings
from logging_setup import configure_logging, get_logger, new_request_id, set_request_id
from models import SessionLocal, engine, init_db
from tracing import (
    instrument_fastapi,
    instrument_redis,
    instrument_sqlalchemy,
    setup_tracing,
)

configure_logging(json=True, level="INFO")
log = get_logger("api")
setup_tracing("helmet-api")


def _apply_migrations() -> None:
    # In-memory sqlite (тесты) не дружит с alembic — обходимся create_all.
    alembic_ini = Path(__file__).parent / "alembic.ini"
    if not alembic_ini.exists() or ":memory:" in settings.sqlalchemy_url:
        init_db()
        return

    from alembic.config import Config

    from alembic import command

    cfg = Config(str(alembic_ini))
    cfg.set_main_option("sqlalchemy.url", settings.sqlalchemy_url)
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _apply_migrations()
    yield


app = FastAPI(title="Helmet Violation Service", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

instrument_fastapi(app)
instrument_sqlalchemy(engine)
instrument_redis()

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or new_request_id()
    set_request_id(rid)
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


app.mount("/violations", StaticFiles(directory=settings.violations_dir), name="violations")

os.makedirs(settings.videos_dir, exist_ok=True)
os.makedirs(settings.outputs_dir, exist_ok=True)
os.makedirs(settings.violations_dir, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
async def health(db: Session = Depends(get_db)):
    """Liveness/readiness probe для оркестраторов и мониторинга.

    Проверяет: доступность БД, Redis, наличие файла модели. Если хоть
    одна зависимость лежит — возвращаем HTTP 503, чтобы load balancer мог
    исключить инстанс из ротации.
    """
    checks: dict[str, dict] = {}
    healthy = True

    # 1. Postgres
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as e:
        checks["database"] = {"ok": False, "error": str(e)}
        healthy = False

    # 2. Redis (broker для Celery)
    try:
        r = redis_lib.Redis.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        checks["redis"] = {"ok": True}
    except Exception as e:
        checks["redis"] = {"ok": False, "error": str(e)}
        healthy = False

    # 3. Модель — должна быть на диске
    model_path = Path(settings.model_path)
    if model_path.exists():
        checks["model"] = {"ok": True, "path": str(model_path)}
    else:
        # Для .onnx допустим вариант, что есть .pt и она будет авто-экспортирована
        pt_fallback = model_path.with_suffix(".pt")
        if pt_fallback.exists():
            checks["model"] = {"ok": True, "path": str(pt_fallback), "note": "auto-export pending"}
        else:
            checks["model"] = {"ok": False, "error": f"{model_path} not found"}
            healthy = False

    payload = {"status": "ok" if healthy else "degraded", "checks": checks}
    return JSONResponse(payload, status_code=200 if healthy else 503)


@app.post("/analyze_video", status_code=202, dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit_analyze)
async def analyze_endpoint(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Только видео!")

    video_path = os.path.join(settings.videos_dir, file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Пробрасываем request_id в воркер через task headers — там его подхватит
    # before_task_publish/task_prerun сигнал и положит в structlog context.
    task = celery_app.send_task(
        "analyze_video",
        args=[video_path],
        headers={"request_id": request.headers.get("x-request-id") or ""},
    )

    log.info("video_queued", filename=file.filename, task_id=task.id)
    return {
        "task_id": task.id,
        "status_url": f"/tasks/{task.id}",
        "video_name": file.filename,
    }


@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    payload: dict = {"task_id": task_id, "state": res.state}

    if res.state == "PENDING":
        payload["detail"] = "Задача ещё не взята в работу"
    elif res.state == "PROGRESS":
        payload["progress"] = res.info or {}
    elif res.state == "SUCCESS":
        payload["result"] = res.result
    elif res.state == "FAILURE":
        payload["error"] = str(res.info)

    return payload


@app.get("/violations")
async def get_violations(db: Session = Depends(get_db)):
    result = db.execute(
        text("SELECT * FROM violations ORDER BY created_at DESC LIMIT 50")
    ).fetchall()
    return [
        {
            "id": r[0],
            "video_name": r[1],
            "track_id": r[2],
            "frame_idx": r[3],
            "ratio_no_helmet": float(r[5]),
            "image_url": f"/violations/{os.path.basename(r[6])}" if r[6] else None,
        }
        for r in result
    ]


@app.delete("/clear_history", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit_clear)
async def clear_history(request: Request, db: Session = Depends(get_db)):
    try:
        db.execute(text("TRUNCATE TABLE violations RESTART IDENTITY CASCADE"))
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Ошибка БД: {e}") from e

    deleted_count = 0
    folders_to_clean = [
        f"{settings.violations_dir}/*",
        f"{settings.outputs_dir}/*",
        f"{settings.videos_dir}/*",
    ]

    for folder_pattern in folders_to_clean:
        for f in glob.glob(folder_pattern):
            try:
                os.remove(f)
                deleted_count += 1
            except Exception:
                pass

    return {"message": f"История очищена. Удалено {deleted_count} файлов."}


@app.get("/download_video/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(settings.outputs_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename=filename,
            headers={"Accept-Ranges": "bytes"},
        )
    raise HTTPException(404, "Видео не найдено")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
