import logging
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings

logger = logging.getLogger("uvicorn.error")


class Base(DeclarativeBase):
    pass


engine: Engine | None = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def _get_runtime_database_url() -> str:
    direct_url = settings.database_url_direct.strip()
    if direct_url:
        return direct_url
    return settings.database_url.strip()


def _uses_postgresql(database_url: str) -> bool:
    return database_url.startswith("postgresql://") or database_url.startswith("postgresql+")


def get_engine() -> Engine:
    global engine

    if engine is None:
        runtime_database_url = _get_runtime_database_url()
        if not runtime_database_url:
            raise RuntimeError("DATABASE_URL is not configured.")

        is_pooler_url = "pooler." in runtime_database_url
        engine_options: dict[str, Any] = {
            "pool_pre_ping": True,
            "pool_recycle": 1800,
        }
        if _uses_postgresql(runtime_database_url):
            engine_options["connect_args"] = {
                "options": "-c timezone=Asia/Seoul",
            }
        if is_pooler_url:
            engine_options["poolclass"] = NullPool

        engine = create_engine(runtime_database_url, **engine_options)
        SessionLocal.configure(bind=engine)
        logger.info(
            "Configured database engine. direct_url=%s pooler_url=%s sqlalchemy_pool=%s",
            bool(settings.database_url_direct.strip()),
            is_pooler_url,
            "NullPool" if is_pooler_url else "QueuePool",
        )

    return engine


def get_db():
    get_engine()
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except SQLAlchemyError:
            logger.exception("Failed to close DB session cleanly. Invalidating session.")
            try:
                db.invalidate()
            except Exception:
                logger.exception("Failed to invalidate DB session after close failure.")
