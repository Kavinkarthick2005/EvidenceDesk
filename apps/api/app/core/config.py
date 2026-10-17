"""Application configuration using Pydantic Settings."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_env_file() -> Path | None:
    """Resolve the .env file path deterministically based on repository structure.

    Order of precedence for file resolution:
    1. An explicit file path provided via the ENV_FILE environment variable, if set.
    2. The repository root `.env` file (located 4 parent directories above this module).
    3. The service root `.env` file (`apps/api/.env`, located 2 parent directories above this module).

    Returns None if no candidate .env file exists on disk.
    """
    explicit_path = os.getenv("ENV_FILE")
    if explicit_path:
        candidate = Path(explicit_path).resolve()
        return candidate if candidate.is_file() else None

    try:
        repo_root = Path(__file__).resolve().parents[4]
        repo_env = repo_root / ".env"
        if repo_env.is_file():
            return repo_env

        api_env = Path(__file__).resolve().parents[2] / ".env"
        if api_env.is_file():
            return api_env
    except IndexError:
        pass

    return None


class Settings(BaseSettings):
    """Core settings for EvidenceDesk API service.

    Reads from environment variables and an optional .env file.
    Resolution of .env files is deterministic and independent of the working directory.
    Extra variables from future milestones are safely ignored.
    """

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "EvidenceDesk"
    APP_ENV: str = "development"
    APP_LOG_LEVEL: str = "INFO"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    VERSION: str = "0.1.0"

    def __init__(self, *args: Any, _env_file: Any = ..., **kwargs: Any) -> None:
        if _env_file is ...:
            _env_file = get_default_env_file()
        super().__init__(*args, _env_file=_env_file, **kwargs)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()

