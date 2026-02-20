from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_dir: Path = Path(__file__).parent

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
    min_track_total: int = 30
    violator_ratio: float = 0.8

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def public_base_url(self) -> str:
        return f"http://localhost:{self.app_port}"


settings = Settings()
