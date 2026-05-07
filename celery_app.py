from celery import Celery
from celery.signals import task_postrun, task_prerun, worker_process_init

from config import settings
from logging_setup import configure_logging, set_request_id
from tracing import instrument_celery, setup_tracing

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
    """После fork: настраиваем логирование, греем модель, поднимаем /metrics."""
    from prometheus_client import start_http_server

    from detection import get_model

    configure_logging(json=True, level="INFO")
    setup_tracing("helmet-worker")
    instrument_celery()
    get_model()

    try:
        start_http_server(9100)
        print("Prometheus worker metrics: http://0.0.0.0:9100/metrics")
    except OSError as exc:
        # Если воркеров несколько в одном контейнере — порт займёт первый.
        print(f"Не удалось поднять metrics HTTP сервер: {exc}")


@task_prerun.connect
def _on_task_start(task=None, task_id=None, **_kwargs):
    """Подхватываем request_id из task headers, чтобы логи воркера и API
    можно было связать по grep'у одного значения."""
    rid = ""
    request = getattr(task, "request", None)
    if request is not None:
        headers = getattr(request, "headers", None) or {}
        rid = headers.get("request_id", "")
    # Если нет входящего — используем task_id как fallback.
    set_request_id(rid or (task_id or ""))


@task_postrun.connect
def _on_task_end(**_kwargs):
    set_request_id("")
