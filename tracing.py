"""OpenTelemetry tracing setup.

If OTEL_EXPORTER_OTLP_ENDPOINT is empty, all calls become no-ops — so this
module can be imported in test environments without the opentelemetry
packages installed.
"""

from __future__ import annotations

import os

_initialized = False


def _otlp_endpoint() -> str | None:
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None


def _enabled() -> bool:
    return bool(_otlp_endpoint())


def setup_tracing(service_name: str) -> None:
    global _initialized
    if _initialized or not _enabled():
        _initialized = True
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    endpoint = _otlp_endpoint()
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _initialized = True


def instrument_fastapi(app) -> None:
    if not _enabled():
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


def instrument_celery() -> None:
    if not _enabled():
        return
    from opentelemetry.instrumentation.celery import CeleryInstrumentor

    CeleryInstrumentor().instrument()


def instrument_sqlalchemy(engine) -> None:
    if not _enabled():
        return
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument(engine=engine)


def instrument_redis() -> None:
    if not _enabled():
        return
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    RedisInstrumentor().instrument()
