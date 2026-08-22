"""Lightweight schema migrations for additive changes beyond create_all()."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.domain.entities import LeadRecordStatus


def run_migrations(engine: Engine) -> None:
    inspector = inspect(engine)
    if inspector.has_table("leads"):
        columns = {column["name"] for column in inspector.get_columns("leads")}
        if "status" not in columns:
            default = LeadRecordStatus.NEW.value
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE leads ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT '{default}'"
                    )
                )
