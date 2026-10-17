"""Focused unit tests for database configuration, engine, and session factories.

These tests run entirely in-memory and do not require a live PostgreSQL server.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, MetaData
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.session import (
    create_db_engine,
    create_session_factory,
    get_db,
    get_engine,
    get_session_factory,
    reset_db_state,
)


def test_default_database_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that database settings defaults assemble a valid postgresql+psycopg URL."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)

    settings = Settings(_env_file=None)
    assert settings.POSTGRES_DB == "evidencedesk"
    assert settings.POSTGRES_USER == "evidencedesk"
    assert settings.POSTGRES_PASSWORD == "replace_with_a_local_secret"
    assert settings.POSTGRES_HOST == "127.0.0.1"
    assert settings.POSTGRES_PORT == 5432
    assert settings.DATABASE_URL == (
        "postgresql+psycopg://evidencedesk:replace_with_a_local_secret@127.0.0.1:5432/evidencedesk"
    )


def test_custom_database_components_assemble_url() -> None:
    """Verify that individual POSTGRES_* settings assemble into DATABASE_URL."""
    settings = Settings(
        POSTGRES_USER="myuser",
        POSTGRES_PASSWORD="mypassword",
        POSTGRES_HOST="dbserver",
        POSTGRES_PORT=5433,
        POSTGRES_DB="custom_evidence",
        _env_file=None,
    )
    assert settings.DATABASE_URL == (
        "postgresql+psycopg://myuser:mypassword@dbserver:5433/custom_evidence"
    )


def test_special_characters_encoded_in_database_url() -> None:
    """Verify special characters in database credentials are safely percent-encoded."""
    special_user = "user@domain"
    special_pass = "p@ss:word#123%/test"

    settings = Settings(
        POSTGRES_USER=special_user,
        POSTGRES_PASSWORD=special_pass,
        POSTGRES_HOST="127.0.0.1",
        POSTGRES_PORT=5432,
        POSTGRES_DB="evidencedesk",
        _env_file=None,
    )

    # Ensure special characters are percent-encoded in the generated URL
    assert "user%40domain" in settings.DATABASE_URL
    assert "p%40ss%3Aword%23123%25%2Ftest" in settings.DATABASE_URL

    # Ensure SQLAlchemy can parse back the exact credentials
    parsed = make_url(settings.DATABASE_URL)
    assert parsed.username == special_user
    assert parsed.password == special_pass

    # Ensure safe_database_url masks the secret and contains no unmasked password
    assert special_pass not in settings.safe_database_url
    assert "***" in settings.safe_database_url


def test_explicit_database_url_takes_precedence() -> None:
    """Verify that explicitly provided DATABASE_URL is preserved."""
    explicit_url = "postgresql+psycopg://otheruser:otherpass@remotehost:5432/remotedb"
    settings = Settings(DATABASE_URL=explicit_url, _env_file=None)
    assert settings.DATABASE_URL == explicit_url


def test_create_db_engine_lazy_configuration() -> None:
    """Verify engine creation without opening a socket connection."""
    test_settings = Settings(
        DATABASE_URL="postgresql+psycopg://test:test@127.0.0.1:5432/testdb",
        _env_file=None,
    )
    engine = create_db_engine(settings=test_settings)

    assert isinstance(engine, Engine)
    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.url.database == "testdb"
    assert engine.url.host == "127.0.0.1"
    assert engine.url.port == 5432
    assert engine.pool is not None


def test_create_session_factory_binding() -> None:
    """Verify sessionmaker factory binds to the engine with appropriate transaction options."""
    test_settings = Settings(
        DATABASE_URL="postgresql+psycopg://test:test@127.0.0.1:5432/testdb",
        _env_file=None,
    )
    engine = create_db_engine(settings=test_settings)
    session_factory = create_session_factory(engine=engine)

    assert isinstance(session_factory, sessionmaker)
    assert session_factory.kw.get("autocommit") is False
    assert session_factory.kw.get("autoflush") is False
    assert session_factory.kw.get("expire_on_commit") is False


def test_declarative_base_metadata() -> None:
    """Verify declarative Base has a valid MetaData instance for model registrations."""
    assert hasattr(Base, "metadata")
    assert isinstance(Base.metadata, MetaData)


def test_engine_and_session_factory_reuse() -> None:
    """Verify get_engine and get_session_factory reuse singleton instances across calls."""
    reset_db_state()
    try:
        e1 = get_engine()
        e2 = get_engine()
        assert e1 is e2

        s1 = get_session_factory()
        s2 = get_session_factory()
        assert s1 is s2
    finally:
        reset_db_state()


def test_custom_settings_produce_dedicated_engine() -> None:
    """Verify custom settings produce a dedicated, non-cached engine instance."""
    reset_db_state()
    try:
        shared = get_engine()
        custom_settings = Settings(
            DATABASE_URL="postgresql+psycopg://custom:custom@127.0.0.1:5432/customdb",
            _env_file=None,
        )
        custom_engine = get_engine(settings=custom_settings)
        assert custom_engine is not shared
        assert custom_engine.url.database == "customdb"
    finally:
        reset_db_state()


def test_get_db_generator_lifecycle() -> None:
    """Verify get_db dependency yields a session and ensures close is called upon completion."""
    mock_session = MagicMock(spec=Session)
    mock_factory = MagicMock(return_value=mock_session)

    with patch("app.db.session.get_session_factory", return_value=mock_factory):
        db_gen = get_db()
        session = next(db_gen)
        assert session is mock_session
        assert mock_session.close.call_count == 0

        # Terminate generator normally
        with pytest.raises(StopIteration):
            next(db_gen)

        assert mock_session.close.call_count == 1


def test_get_db_closes_session_on_exception() -> None:
    """Verify get_db session is closed even if an exception occurs during request handling."""
    mock_session = MagicMock(spec=Session)
    mock_factory = MagicMock(return_value=mock_session)

    with patch("app.db.session.get_session_factory", return_value=mock_factory):
        db_gen = get_db()
        session = next(db_gen)
        assert session is mock_session
        assert mock_session.close.call_count == 0

        # Simulate exception raised within route handler
        with pytest.raises(RuntimeError, match="Handler failure"):
            db_gen.throw(RuntimeError("Handler failure"))

        assert mock_session.close.call_count == 1
