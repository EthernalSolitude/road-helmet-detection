from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import shutil
import os
import glob
from detection import analyze_video
from models import SessionLocal, Violation, init_db
from sqlalchemy import text
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Helmet Violation Service", lifespan=lifespan)

app.mount("/violations", StaticFiles(directory="violations_frames"), name="violations")

UPLOAD_DIR = "videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("violations_frames", exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/analyze_video")
async def analyze_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(400, "Только видео!")
    
    video_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    result = analyze_video(video_path)
    
    return {
        "video_name": file.filename,
        "download_url": f"http://localhost:8000/download_video/out_{file.filename}",
        "violations_count": len(result.get("violations", [])),
        "violations": result.get("violations", [])
    }

@app.get("/violations")
async def get_violations(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM violations ORDER BY created_at DESC LIMIT 50")).fetchall()
    return [
        {
            "id": r[0], 
            "video_name": r[1], 
            "track_id": r[2], 
            "frame_idx": r[3],
            "ratio_no_helmet": float(r[5]),
            "image_url": f"/violations/{os.path.basename(r[6])}" if r[6] else None
        } for r in result
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
    folders_to_clean = ["violations_frames/*", "outputs/*", "videos/*"] 
    
    for folder_pattern in folders_to_clean:
        files = glob.glob(folder_pattern)
        for f in files:
            try:
                os.remove(f)
                deleted_count += 1
            except Exception:
                pass 

    return {"message": f"История очищена. Удалено {deleted_count} файлов."}

@app.get("/download_video/{filename}")
async def download_video(filename: str):
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="video/mp4", 
            filename=filename,
            headers={"Accept-Ranges": "bytes"}
        )
    raise HTTPException(404, "Видео не найдено")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
