"""Конфигурация для integration-тестов.

Запускается с реальными Postgres и Redis из GHA services. DATABASE_URL и
REDIS_URL берутся из окружения CI; локально интеграционные тесты по
умолчанию не запускаются (см. addopts в pyproject.toml).
"""

import os

import pytest

# Если DATABASE_URL не задан — этот pytest-сценарий неработоспособен.
# Лучше явно скипнуть, чем падать с обскюрной ошибкой при подключении к sqlite.
if "sqlite" in os.environ.get("DATABASE_URL", ""):
    pytest.skip(
        "Integration tests require real Postgres via DATABASE_URL env var",
        allow_module_level=True,
    )
