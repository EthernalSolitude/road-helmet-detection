import glob
import os
import shutil
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text
from sqlalchemy.orm import Session

from celery_app import celery_app
from config import settings
from models import SessionLocal, init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Helmet Violation Service", lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

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


@app.post("/analyze_video", status_code=202)
async def analyze_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Только видео!")

    video_path = os.path.join(settings.videos_dir, file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task = celery_app.send_task("analyze_video", args=[video_path])

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


@app.delete("/clear_history")
async def clear_history(db: Session = Depends(get_db)):
    try:
        db.execute(text("TRUNCATE TABLE violations RESTART IDENTITY CASCADE"))
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Ошибка БД: {e}")

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
