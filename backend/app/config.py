import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass
class Settings:
    checkpoint: Path = field(
        default_factory=lambda: Path(
            os.getenv("MODEL_CHECKPOINT", str(PROJECT_ROOT / "artifacts/model/model.pt"))
        )
    )
    reports: Path = field(
        default_factory=lambda: Path(
            os.getenv("REPORT_DIR", str(PROJECT_ROOT / "artifacts/reports"))
        )
    )
    history_enabled: bool = field(
        default_factory=lambda: os.getenv("HISTORY_ENABLED", "false").lower() == "true"
    )
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'artifacts/history.sqlite3'}"
        )
    )
    origins: list[str] = field(
        default_factory=lambda: os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
    )
    max_upload_bytes: int = 8 * 1024 * 1024
    max_image_dimension: int = 4096
    max_image_pixels: int = 12_000_000
