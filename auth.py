from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(api_key_header)) -> None:
    expected = settings.api_key
    if not expected:
        return

    if not provided:
        raise HTTPException(HTTP_401_UNAUTHORIZED, "Отсутствует header X-API-Key")
    if provided != expected:
        raise HTTPException(HTTP_403_FORBIDDEN, "Неверный API-ключ")
