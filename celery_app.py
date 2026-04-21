from celery import Celery
from celery.signals import worker_process_init

from config import settings


celery_app = Celery(
    "helmet",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60,
    result_expires=60 * 60 * 24,
)


@worker_process_init.connect
def _on_worker_start(**_kwargs):
    """После fork: греем модель и поднимаем /metrics воркера на 9100."""
    from prometheus_client import start_http_server

    from detection import get_model

    get_model()

    try:
        start_http_server(9100)
        print("Prometheus worker metrics: http://0.0.0.0:9100/metrics")
    except OSError as exc:
        # Если воркеров несколько в одном контейнере — порт займёт первый.
        print(f"Не удалось поднять metrics HTTP сервер: {exc}")
