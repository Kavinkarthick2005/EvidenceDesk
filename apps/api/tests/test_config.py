"""Tests for configuration resolution and precedence."""

import os
from pathlib import Path

import pytest

from app.core.config import Settings, get_default_env_file, get_settings


def test_settings_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that default settings load with expected values when no env or .env is present."""
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENV_FILE", raising=False)

    settings = Settings(_env_file=None)
    assert settings.APP_NAME == "EvidenceDesk"
    assert settings.APP_ENV == "development"
    assert settings.APP_LOG_LEVEL == "INFO"
    assert settings.API_HOST == "127.0.0.1"
    assert settings.API_PORT == 8000
    assert settings.VERSION == "0.1.0"


def test_env_file_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that settings load values from a specified .env file."""
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=FromTestEnvFile\nAPP_ENV=testing\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)
    assert settings.APP_NAME == "FromTestEnvFile"
    assert settings.APP_ENV == "testing"


def test_env_var_precedence_over_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that OS environment variables take precedence over .env file values."""
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=FromFile\nAPP_ENV=FromFileEnv\n", encoding="utf-8")

    monkeypatch.setenv("APP_NAME", "FromOSEnviron")
    monkeypatch.delenv("APP_ENV", raising=False)

    settings = Settings(_env_file=env_file)
    assert settings.APP_NAME == "FromOSEnviron"
    assert settings.APP_ENV == "FromFileEnv"


def test_constructor_precedence_over_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that direct constructor arguments take precedence over environment variables."""
    monkeypatch.setenv("APP_NAME", "FromOSEnviron")

    settings = Settings(APP_NAME="ExplicitConstructor", _env_file=None)
    assert settings.APP_NAME == "ExplicitConstructor"


def test_env_file_resolution_via_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that ENV_FILE environment variable directs resolution to an explicit file."""
    custom_env = tmp_path / "custom.env"
    custom_env.write_text("APP_NAME=FromCustomEnvVar\n", encoding="utf-8")

    monkeypatch.setenv("ENV_FILE", str(custom_env))
    monkeypatch.delenv("APP_NAME", raising=False)

    resolved_path = get_default_env_file()
    assert resolved_path == custom_env.resolve()

    settings = Settings()
    assert settings.APP_NAME == "FromCustomEnvVar"


def test_env_resolution_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that default env file resolution targets repo root rather than CWD."""
    # Place a decoy .env inside the temporary directory and switch CWD into it
    dummy_env = tmp_path / ".env"
    dummy_env.write_text("APP_NAME=ShouldNotBeLoadedFromCwd\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENV_FILE", raising=False)
    monkeypatch.delenv("APP_NAME", raising=False)

    from app.core import config

    expected_repo_env = Path(config.__file__).resolve().parents[4] / ".env"
    expected_api_env = Path(config.__file__).resolve().parents[2] / ".env"
    orig_is_file = Path.is_file

    # Case 1: When no repo or api .env file exists on disk
    def mock_no_repo_env(path_obj: Path) -> bool:
        if path_obj in (expected_repo_env, expected_api_env):
            return False
        return orig_is_file(path_obj)

    monkeypatch.setattr(Path, "is_file", mock_no_repo_env)
    assert get_default_env_file() is None
    # Settings must use defaults and not load from CWD
    assert Settings().APP_NAME == "EvidenceDesk"

    # Case 2: When repo root .env exists on disk
    def mock_repo_env_exists(path_obj: Path) -> bool:
        if path_obj == expected_repo_env:
            return True
        if path_obj == expected_api_env:
            return False
        return orig_is_file(path_obj)

    monkeypatch.setattr(Path, "is_file", mock_repo_env_exists)
    resolved = get_default_env_file()
    assert resolved == expected_repo_env
    assert resolved != dummy_env

