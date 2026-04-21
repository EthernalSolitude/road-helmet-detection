import os

from celery_app import celery_app
from config import settings
from detection import analyze_video


@celery_app.task(bind=True, name="analyze_video")
def analyze_video_task(self, video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    def progress(done: int, total: int):
        self.update_state(
            state="PROGRESS",
            meta={"current": done, "total": total},
        )

    result = analyze_video(video_path, progress_cb=progress)

    video_name = os.path.basename(video_path)
    return {
        "video_name": video_name,
        "download_url": f"{settings.public_base_url}/download_video/out_{video_name}",
        "violations_count": len(result.get("violations", [])),
        "violations": result.get("violations", []),
        "frames_processed": result.get("frames", 0),
    }
