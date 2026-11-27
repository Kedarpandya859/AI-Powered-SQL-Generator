"""Database access layer.

Works with any SQLAlchemy-supported database (MySQL, PostgreSQL, SQLite, ...).
Schema discovery uses SQLAlchemy's Inspector instead of dialect-specific
commands such as ``SHOW TABLES``, so the same code path serves every backend.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

load_dotenv()
logger = logging.getLogger(__name__)

# Schemas that hold server metadata, never user data. They are hidden from the
# UI and never sent to the LLM.
SYSTEM_SCHEMAS = {
    "information_schema",
    "mysql",
    "performance_schema",
    "sys",
    "pg_catalog",
    "pg_toast",
}


def _database_url() -> str:
    """DATABASE_URL wins; otherwise fall back to the original MYSQL_* variables."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("MYSQL_HOST", "localhost")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "")
    port = os.getenv("MYSQL_PORT", "3306")
    return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"


_engine: Optional[Engine] = None
_schema_cache: dict = {}


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(_database_url(), pool_pre_ping=True)
    return _engine


def set_engine(engine: Engine) -> None:
    """Swap the engine (used by tests and for switching connections at runtime)."""
    global _engine
    _engine = engine
    _schema_cache.clear()


def dialect_name() -> str:
    return get_engine().dialect.name


@dataclass
class TableInfo:
    schema: Optional[str]
    name: str
    columns: list = field(default_factory=list)       # [(name, type_str)]
    primary_key: list = field(default_factory=list)   # [column]
    foreign_keys: list = field(default_factory=list)  # [(column, ref_table, ref_column)]

    @property
    def qualified_name(self) -> str:
        # SQLite has a single implicit schema, so names stay unqualified there.
        if self.schema and dialect_name() != "sqlite":
            return f"{self.schema}.{self.name}"
        return self.name

    def to_prompt_line(self) -> str:
        pk = set(self.primary_key)
        fk = {c: f"{t}.{rc}" for c, t, rc in self.foreign_keys}
        parts = []
        for col, col_type in self.columns:
            desc = f"{col} {col_type}"
            if col in pk:
                desc += " PK"
            if col in fk:
                desc += f" -> {fk[col]}"
            parts.append(desc)
        return f"{self.qualified_name}({', '.join(parts)})"


def list_databases() -> list:
    try:
        names = inspect(get_engine()).get_schema_names()
    except Exception as e:  # noqa: BLE001 - surface any driver error as empty list
        logger.error("Error listing databases: %s", e)
        return []
    return [n for n in names if n.lower() not in SYSTEM_SCHEMAS]


def _check_database(database_name: str) -> None:
    if database_name not in list_databases():
        raise ValueError(f"Unknown database: {database_name!r}")


def list_tables(database_name: str) -> list:
    _check_database(database_name)
    return inspect(get_engine()).get_table_names(schema=database_name)


def list_columns(database_name: str, table_name: str) -> list:
    _check_database(database_name)
    cols = inspect(get_engine()).get_columns(table_name, schema=database_name)
    return [c["name"] for c in cols]


def get_schema(database_name: str, refresh: bool = False) -> list:
    """Full schema (columns, types, keys) for every table in a database, cached."""
    if not refresh and database_name in _schema_cache:
        return _schema_cache[database_name]
    _check_database(database_name)
    insp = inspect(get_engine())
    tables = []
    for name in insp.get_table_names(schema=database_name):
        cols = [(c["name"], str(c["type"])) for c in insp.get_columns(name, schema=database_name)]
        pk = insp.get_pk_constraint(name, schema=database_name).get("constrained_columns") or []
        fks = []
        for fk in insp.get_foreign_keys(name, schema=database_name):
            for col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                fks.append((col, fk["referred_table"], ref_col))
        tables.append(TableInfo(database_name, name, cols, pk, fks))
    _schema_cache[database_name] = tables
    return tables


def default_database() -> Optional[str]:
    """The database the connection URL points at, or the first user database."""
    engine = get_engine()
    configured = engine.url.database
    dbs = list_databases()
    if engine.dialect.name == "sqlite":
        return "main" if "main" in dbs else (dbs[0] if dbs else None)
    if engine.dialect.name == "postgresql":
        return "public" if "public" in dbs else (dbs[0] if dbs else None)
    if configured and configured in dbs:
        return configured
    return dbs[0] if dbs else None
