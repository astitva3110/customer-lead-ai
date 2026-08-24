"""Lightweight schema migrations for additive changes beyond create_all()."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.domain.entities import RecordStatus


def run_migrations(engine: Engine) -> None:
    inspector = inspect(engine)
    if inspector.has_table("leads"):
        columns = {column["name"] for column in inspector.get_columns("leads")}
        if "status" not in columns:
            default = RecordStatus.OPEN.value
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE leads ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT '{default}'"
                    )
                )
        if "updated_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE leads ADD COLUMN updated_at TIMESTAMPTZ "
                        "NOT NULL DEFAULT NOW()"
                    )
                )
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE leads SET status = 'open' "
                    "WHERE status IN ('new', 'contacted', 'qualified')"
                )
            )
            connection.execute(
                text("UPDATE leads SET status = 'closed' WHERE status = 'closed'")
            )

    if inspector.has_table("support_tickets"):
        columns = {column["name"] for column in inspector.get_columns("support_tickets")}
        if "status" not in columns:
            default = RecordStatus.OPEN.value
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE support_tickets ADD COLUMN status VARCHAR(32) "
                        f"NOT NULL DEFAULT '{default}'"
                    )
                )
        if "updated_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE support_tickets ADD COLUMN updated_at TIMESTAMPTZ "
                        "NOT NULL DEFAULT NOW()"
                    )
                )
