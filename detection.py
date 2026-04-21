import os
import time
from collections import defaultdict
from typing import Callable, Optional

import cv2

from config import settings
from metrics import (
    active_video_tasks,
    frames_processed_total,
    frames_skipped_total,
    inference_seconds,
    video_processing_seconds,
    videos_processed_total,
    violations_detected_total,
)
from models import SessionLocal, Violation

_model = None


def get_model():
    """Ленивая загрузка YOLO — один раз на процесс (после fork воркера)."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(settings.model_path)
        print("Модель загружена, классы:", _model.names)
    return _model


ProgressFn = Optional[Callable[[int, int], None]]


def analyze_video(video_path: str, progress_cb: ProgressFn = None):
    model = get_model()
    session = SessionLocal()
    cap = cv2.VideoCapture(video_path)

    video_name = os.path.basename(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_skip = settings.frame_skip
    out_fps = int(fps / frame_skip) if fps > 0 else 15

    out_path = os.path.join(settings.outputs_dir, f"out_{video_name}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

    tracks_stats = defaultdict(lambda: {"helmet": 0, "no_helmet": 0, "violator": False})
    violators: list[dict] = []
    pending_violations: list[Violation] = []
    frame_count = 0

    active_video_tasks.inc()
    t_start = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                frames_skipped_total.inc()
                continue

            # BoT-SORT компенсирует пропуски кадров через ReID + CMC.
            t_inf = time.perf_counter()
            results = model.track(
                frame,
                conf=settings.conf_threshold,
                imgsz=settings.img_size,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
            )
            inference_seconds.observe(time.perf_counter() - t_inf)
            frames_processed_total.inc()

            annotated = results[0].plot()
            out.write(annotated)

            if progress_cb and frame_count % 30 == 0:
                progress_cb(frame_count, total_frames)

            if results[0].boxes.id is None:
                continue

            for box in results[0].boxes:
                track_id = int(box.id)
                cls_name = model.names[int(box.cls)].lower()

                if "helmet" in cls_name and "no" not in cls_name:
                    tracks_stats[track_id]["helmet"] += 1
                elif "no" in cls_name or "without" in cls_name:
                    tracks_stats[track_id]["no_helmet"] += 1

                stats = tracks_stats[track_id]
                total = stats["helmet"] + stats["no_helmet"]

                if total < settings.min_track_total or stats["violator"]:
                    continue

                ratio = stats["no_helmet"] / total
                if ratio <= settings.violator_ratio:
                    continue

                stats["violator"] = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                img_path = os.path.join(
                    settings.violations_dir,
                    f"viol_track{track_id}_f{frame_count}.jpg",
                )
                if crop.size > 0:
                    cv2.imwrite(img_path, crop)

                pending_violations.append(
                    Violation(
                        video_name=video_name,
                        track_id=track_id,
                        frame_idx=frame_count,
                        bbox=f"{x1},{y1},{x2},{y2}",
                        ratio_no_helmet=float(ratio),
                        image_path=img_path,
                    )
                )
                violations_detected_total.inc()
                violators.append(
                    {
                        "track_id": track_id,
                        "frame": frame_count,
                        "ratio_no_helmet": round(ratio, 3),
                        "image_url": f"/violations/{os.path.basename(img_path)}",
                    }
                )
                print(f"НАРУШИТЕЛЬ! Track {track_id}, ratio={ratio:.2%}")

                # Батчим коммиты, чтобы не гонять round-trip по каждому нарушителю.
                if len(pending_violations) >= 10:
                    session.add_all(pending_violations)
                    session.commit()
                    pending_violations.clear()

        if pending_violations:
            session.add_all(pending_violations)
            session.commit()

        videos_processed_total.labels(status="success").inc()
        return {"violations": violators, "frames": frame_count}

    except Exception:
        session.rollback()
        videos_processed_total.labels(status="error").inc()
        raise
    finally:
        cap.release()
        out.release()
        session.close()
        video_processing_seconds.observe(time.perf_counter() - t_start)
        active_video_tasks.dec()
        print(f"Обработано {frame_count} кадров, нарушителей: {len(violators)}")
