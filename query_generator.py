"""Natural language -> SQL with schema-aware prompting, plus safe execution."""
import io
import json
import logging
import os
import re
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from databases import default_database, dialect_name, get_engine, get_schema
from schema_retriever import select_relevant_tables
from sql_guard import UnsafeSQLError, validate_read_only

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K_TABLES = int(os.getenv("TOP_K_TABLES", "8"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "1000"))

_client = None


def get_client():
    """Lazily create the OpenAI client (openai>=1.0). None if no API key is set."""
    global _client
    if _client is None and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        _client = OpenAI()
    return _client


SYSTEM_PROMPT = """You are an expert data analyst who writes {dialect} SQL.
Rules:
- Write exactly ONE read-only SELECT statement (CTEs with WITH are fine).
- Use ONLY the tables and columns listed in the schema. Never invent names.
- Use the table names exactly as written (including any schema prefix).
- Join tables using the foreign keys shown as "column -> table.column".
- Add LIMIT 100 unless the user asks for all rows or an aggregate returns few rows.
- If the question cannot be answered from the schema, return an empty "sql"
  and say why in "explanation".
Respond with a JSON object: {{"sql": "...", "explanation": "one or two plain-English sentences"}}"""

USER_PROMPT = """Schema (table(column type [PK] [-> referenced_table.column])):
{schema}

Question: {question}"""


def clean_sql_output(raw: str) -> str:
    """Strip ```sql ... ``` fences and trailing semicolons the model may add."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    sql = match.group(1) if match else raw
    return sql.strip().rstrip(";").strip()


def build_prompt(question: str, database: Optional[str] = None, client=None):
    """Return (messages, tables_used) for a question against one database."""
    database = database or default_database()
    if database is None:
        raise ValueError("No database available.")
    tables = get_schema(database)
    relevant = select_relevant_tables(question, tables, top_k=TOP_K_TABLES, client=client)
    schema_text = "\n".join(t.to_prompt_line() for t in relevant)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(dialect=dialect_name())},
        {"role": "user", "content": USER_PROMPT.format(schema=schema_text, question=question)},
    ]
    return messages, [t.qualified_name for t in relevant]


def _call_llm(client, messages) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,  # deterministic SQL for the same question
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Some models ignore JSON mode; treat the whole reply as SQL.
        return {"sql": content, "explanation": ""}


def generate_sql_query(question: str, database: Optional[str] = None, client=None) -> dict:
    """Generate SQL and an explanation in a single LLM call.

    Returns {"sql", "explanation", "tables_used", "is_safe", "safety_error"}.
    """
    if not question or not question.strip():
        raise ValueError("Question is empty.")
    client = client or get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    messages, tables_used = build_prompt(question, database, client=client)
    data = _call_llm(client, messages)
    sql = clean_sql_output(str(data.get("sql", "")))
    explanation = str(data.get("explanation", "")).strip()

    is_safe, safety_error = True, None
    if sql:
        try:
            sql = validate_read_only(sql)
        except UnsafeSQLError as e:
            is_safe, safety_error = False, str(e)
    else:
        is_safe, safety_error = False, "The model could not answer this from the schema."

    return {
        "sql": sql,
        "explanation": explanation,
        "tables_used": tables_used,
        "is_safe": is_safe,
        "safety_error": safety_error,
    }


def _set_read_only(conn) -> None:
    """Second layer of defence: ask the database itself to refuse writes."""
    dialect = conn.dialect.name
    if dialect == "sqlite":
        conn.exec_driver_sql("PRAGMA query_only = ON")
    elif dialect == "postgresql":
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
    elif dialect in ("mysql", "mariadb"):
        conn.exec_driver_sql("START TRANSACTION READ ONLY")


def execute_query(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """Validate and run a SELECT. Raises UnsafeSQLError or SQLAlchemyError.

    Returns {"columns", "rows", "truncated"}; at most ``max_rows`` rows are fetched.
    """
    sql = validate_read_only(sql)
    with get_engine().connect() as conn:
        try:
            _set_read_only(conn)
            result = conn.execute(text(sql))
            columns = list(result.keys())
            fetched = result.fetchmany(max_rows + 1)
        finally:
            conn.rollback()
            if conn.dialect.name == "sqlite":
                conn.exec_driver_sql("PRAGMA query_only = OFF")
    truncated = len(fetched) > max_rows
    rows = [dict(r._mapping) for r in fetched[:max_rows]]
    return {"columns": columns, "rows": rows, "truncated": truncated}


def results_to_csv(rows: list, columns: Optional[list] = None) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    pd.DataFrame(rows, columns=columns).to_csv(buf, index=False)
    return buf.getvalue()
