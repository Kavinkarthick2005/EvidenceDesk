"""SQLAlchemy engine and session factory configuration.

Connections are lazy: importing this module or creating engines does not
establish an immediate socket connection to PostgreSQL.
"""

from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


# Module-level holders for shared lazy engine and session factory
_shared_engine: Engine | None = None
_shared_session_factory: sessionmaker[Session] | None = None


def create_db_engine(
    settings: Settings | None = None,
    echo: bool = False,
    **engine_kwargs: Any,
) -> Engine:
    """Create a new SQLAlchemy engine configured with application settings.

    Connections are created lazily upon first checkout by the connection pool.
    """
    if settings is None:
        settings = get_settings()

    return create_engine(
        settings.DATABASE_URL,
        echo=echo,
        pool_pre_ping=True,
        **engine_kwargs,
    )


def create_session_factory(
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> sessionmaker[Session]:
    """Create a sessionmaker factory bound to an engine."""
    if engine is None:
        engine = create_db_engine(settings=settings)

    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def get_engine(settings: Settings | None = None) -> Engine:
    """Retrieve or lazily initialize the shared application engine.

    If custom settings are supplied, a dedicated engine is returned.
    """
    global _shared_engine
    if settings is not None:
        return create_db_engine(settings=settings)

    if _shared_engine is None:
        _shared_engine = create_db_engine()
    return _shared_engine


def get_session_factory(
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> sessionmaker[Session]:
    """Retrieve or lazily initialize the shared session factory.

    If a custom engine or settings are supplied, a dedicated session factory is returned.
    """
    global _shared_session_factory
    if engine is not None or settings is not None:
        return create_session_factory(engine=engine, settings=settings)

    if _shared_session_factory is None:
        shared_engine = get_engine()
        _shared_session_factory = create_session_factory(engine=shared_engine)
    return _shared_session_factory


def reset_db_state() -> None:
    """Dispose the shared engine and reset cached session factories.

    Useful for test isolation and clean application shutdown.
    """
    global _shared_engine, _shared_session_factory
    if _shared_engine is not None:
        _shared_engine.dispose()
        _shared_engine = None
    _shared_session_factory = None


def get_db(settings: Settings | None = None) -> Generator[Session, None, None]:
    """FastAPI dependency yielding an isolated SQLAlchemy database session.

    Reuses the shared session factory for standard requests to avoid recreating
    engine connection pools, while ensuring sessions are always closed upon exit
    even if exceptions are raised during request handling.
    """
    session_factory = get_session_factory(settings=settings)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()

