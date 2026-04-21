from prometheus_client import Counter, Histogram, Gauge


video_processing_seconds = Histogram(
    "helmet_video_processing_seconds",
    "Полное время обработки одного видео воркером",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200),
)

inference_seconds = Histogram(
    "helmet_inference_seconds",
    "Время инференса модели на один кадр",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

frames_processed_total = Counter(
    "helmet_frames_processed_total",
    "Суммарное количество обработанных кадров",
)

frames_skipped_total = Counter(
    "helmet_frames_skipped_total",
    "Суммарное количество пропущенных кадров (frame_skip)",
)

violations_detected_total = Counter(
    "helmet_violations_detected_total",
    "Суммарное количество зафиксированных нарушений",
)

videos_processed_total = Counter(
    "helmet_videos_processed_total",
    "Количество обработанных видео",
    labelnames=("status",),
)

active_video_tasks = Gauge(
    "helmet_active_video_tasks",
    "Количество видео, обрабатываемых прямо сейчас",
)
