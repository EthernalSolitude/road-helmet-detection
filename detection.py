import os
import cv2
from collections import defaultdict
from ultralytics import YOLO
from models import SessionLocal, Violation

model = YOLO("best.pt")
print("Классы модели:", model.names)

def analyze_video(video_path: str):
    session = SessionLocal()
    cap = cv2.VideoCapture(video_path)
    
    video_name = os.path.basename(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = f"outputs/out_{video_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    tracks_stats = defaultdict(lambda: {"helmet": 0, "no_helmet": 0, "violator": False})
    violators = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, conf=0.5, imgsz = 1280, persist=True, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
        
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                track_id = int(box.id)
                cls_name = model.names[int(box.cls)]
                
                # Считаем шлемы/без шлемов
                if "helmet" in cls_name.lower() and "no" not in cls_name.lower():
                    tracks_stats[track_id]["helmet"] += 1
                elif "no" in cls_name.lower() or "without" in cls_name.lower():
                    tracks_stats[track_id]["no_helmet"] += 1
                
                # Проверка на нарушителя
                h_cnt = tracks_stats[track_id]["helmet"]
                nh_cnt = tracks_stats[track_id]["no_helmet"]
                total = h_cnt + nh_cnt
                
                if total >= 30 and not tracks_stats[track_id]["violator"]:
                    ratio = nh_cnt / total
                    if ratio > 0.8:
                        tracks_stats[track_id]["violator"] = True
                        
                        # Сохраняем кроп
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        img_path = f"violations_frames/viol_track{track_id}_f{frame_count}.jpg"
                        
                        if crop.size > 0:
                            cv2.imwrite(img_path, crop)
                        
                        # Сохраняем в БД
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        violation = Violation(
                            video_name=video_name,
                            track_id=track_id,
                            frame_idx=frame_count,
                            bbox=f"{x1},{y1},{x2},{y2}",
                            ratio_no_helmet=float(ratio),
                            image_path=img_path
                        )
                        session.add(violation)
                        session.commit()
                        
                        violators.append({
                            "track_id": track_id,
                            "frame": frame_count,
                            "ratio_no_helmet": round(ratio, 3),
                            "image_url": f"/violations/{os.path.basename(img_path)}"
                        })
                        
                        print(f"НАРУШИТЕЛЬ! Track {track_id}, ratio={ratio:.2%}")
        
        frame_count += 1
    
    cap.release()
    out.release()
    session.close()
    
    print(f"Обработано {frame_count} кадров, нарушителей: {len(violators)}")
    return {"violations": violators}
