from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", protected_namespaces=())

    base_dir: Path = Path(__file__).parent

    database_url: str | None = None

    db_user: str = "helmet_user"
    db_password: str = "1234"
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "helmet_db"

    videos_dir: str = "videos"
    outputs_dir: str = "outputs"
    violations_dir: str = "violations_frames"

    model_path: str = "best.pt"
    conf_threshold: float = 0.5
    img_size: int = 1280
    min_track_total: int = 15
    violator_ratio: float = 0.8
    frame_skip: int = 2

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Если задан — write/delete ручки требуют header X-API-Key.
    # Пустой/None → auth выключен (dev-режим).
    api_key: str | None = None

    # Rate limit для дорогих ручек (POST /analyze_video, DELETE /clear_history)
    rate_limit_analyze: str = "10/minute"
    rate_limit_clear: str = "5/minute"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def public_base_url(self) -> str:
        return f"http://localhost:{self.app_port}"


settings = Settings()
