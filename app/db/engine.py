from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import AppBase
from app.db.migrations import run_migrations


@lru_cache
def get_engine() -> Engine:
    url = (settings.database_url or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Add it to your .env file.")
    return create_engine(url, connect_args={"connect_timeout": 5})


@lru_cache
def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def ensure_schema() -> None:
    engine = get_engine()
    AppBase.metadata.create_all(engine)
    run_migrations(engine)
