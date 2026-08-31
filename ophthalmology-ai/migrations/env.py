"""Alembic environment.

Reads the connection URL from application settings (i.e. from the
environment), and takes the target metadata from the ORM models so
``--autogenerate`` sees every table.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.database.models import Base  # noqa: F401 - imports every model

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().DATABASE_URL)
target_metadata = Base.metadata


def render_item(type_: str, obj: object, autogen_context: object) -> str | bool:
    """Render application-specific column types with plain SQLAlchemy types.

    * :class:`StrEnumType` becomes ``sa.String``: the enum decorator only
      matters on the Python side, the column really is a string.
    * The JSON/JSONB variant is rendered explicitly so migration files never
      import from the application package - a migration must keep working
      after the models it was generated from have changed.
    """
    from sqlalchemy.types import JSON

    from app.database.base import StrEnumType

    if type_ != "type":
        return False
    if isinstance(obj, StrEnumType):
        return f"sa.String(length={obj.impl.length})"
    if isinstance(obj, JSON):
        return 'sa.JSON().with_variant(postgresql.JSONB(), "postgresql")'
    return False


def run_migrations_offline() -> None:
    """Emit SQL without a live connection (``alembic upgrade head --sql``)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_item=render_item,
            # SQLite cannot ALTER most things in place; batch mode rewrites the
            # table instead, so the same migrations run in tests and production.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
