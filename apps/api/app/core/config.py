"""Application configuration using Pydantic Settings."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url




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

    # PostgreSQL Configuration
    POSTGRES_DB: str = "evidencedesk"
    POSTGRES_USER: str = "evidencedesk"
    POSTGRES_PASSWORD: str = "replace_with_a_local_secret"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str | None = None

    # Document storage configuration
    DOCUMENT_STORAGE_PATH: str = "./data/documents"
    MAX_UPLOAD_SIZE_MB: int = 25

    # Embedding configuration
    EMBEDDING_PROVIDER: str = "mock"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NUM_THREADS: int | None = None

    @property
    def resolved_document_storage_path(self) -> Path:
        """Resolve DOCUMENT_STORAGE_PATH to an absolute path."""
        path = Path(self.DOCUMENT_STORAGE_PATH)
        if path.is_absolute():
            return path
        try:
            repo_root = Path(__file__).resolve().parents[4]
            return (repo_root / path).resolve()
        except IndexError:
            return path.resolve()

    @model_validator(mode="after")
    def assemble_database_url(self) -> Self:
        """Ensure DATABASE_URL is constructed safely with URL encoding if not provided."""
        if not self.DATABASE_URL:
            url = URL.create(
                drivername="postgresql+psycopg",
                username=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD,
                host=self.POSTGRES_HOST,
                port=self.POSTGRES_PORT,
                database=self.POSTGRES_DB,
            )
            self.DATABASE_URL = url.render_as_string(hide_password=False)
        return self

    @property
    def safe_database_url(self) -> str:
        """Return a sanitized version of DATABASE_URL with the password masked."""
        try:
            return make_url(self.DATABASE_URL).render_as_string(hide_password=True)
        except Exception:
            return "[REDACTED]"


    def __init__(self, *args: Any, _env_file: Any = ..., **kwargs: Any) -> None:
        if _env_file is ...:
            _env_file = get_default_env_file()
        super().__init__(*args, _env_file=_env_file, **kwargs)



@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()

