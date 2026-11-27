# AI-Powered SQL Query Generator

Ask a question in plain English and get a SQL query, a plain-English explanation, and the results, which you can download as CSV. Works with MySQL, PostgreSQL and SQLite, and with schemas of 100+ tables.

## How it works

```
Streamlit UI (ui.py) ──HTTP──▶ FastAPI (app.py)
                                 ├─ databases.py        schema discovery via SQLAlchemy Inspector
                                 ├─ schema_retriever.py picks the relevant tables for each question
                                 ├─ query_generator.py  prompt → OpenAI → SQL + explanation (one call)
                                 └─ sql_guard.py        rejects anything but a single SELECT
```

1. **Schema discovery.** SQLAlchemy's Inspector reads tables, column types, primary keys and foreign keys. It works the same on every supported database. System schemas (`information_schema`, `mysql`, `pg_catalog`, ...) are hidden.
2. **Table retrieval.** Sending a 100-table schema to an LLM wastes tokens and confuses the model. Each table is ranked against the question using OpenAI embeddings, or keyword matching with business synonyms when no API key is set. Only the top `TOP_K_TABLES` (default 8) are sent, plus the tables they join to via foreign keys.
3. **Prompting.** The prompt names the SQL dialect, lists the tables as `table(col TYPE PK, col -> other.col)` so the model can see join paths, and forbids inventing names. It runs at temperature 0 and asks for JSON, so the SQL and the explanation come back from one call.
4. **Safety, in three layers:**
   - `sql_guard` allows only a single `SELECT`/`WITH` statement and blocks write keywords even inside CTEs.
   - The connection is opened read-only (`PRAGMA query_only`, `SET TRANSACTION READ ONLY`, `START TRANSACTION READ ONLY`).
   - You should also connect with a database user that has only `SELECT` privileges.
5. **Execution.** At most `MAX_ROWS` rows are fetched. Results can be downloaded as CSV. The UI lets you edit the SQL before running it.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # set DATABASE_URL (or MYSQL_*) and OPENAI_API_KEY
uvicorn app:app --reload
streamlit run ui.py    # in a second terminal, then open http://localhost:8501
```

## API

| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | dialect and default database |
| GET | `/databases` | user databases (system schemas hidden) |
| GET | `/tables/{database}` | table names |
| GET | `/columns/{database}/{table}` | column names |
| POST | `/schema/refresh/{database}` | reloads the cached schema |
| POST | `/generate_sql` | `{"question", "database"?}` → `{"sql", "explanation", "tables_used", "is_safe", "safety_error"}` |
| POST | `/execute_sql` | `{"sql"}` → `{"columns", "rows", "truncated", "csv"}` (400 for unsafe or invalid SQL) |

## Tests

```bash
pytest -q
```

The tests build a 100-table SQLite database and stub out the LLM, so they need no API key or network. They cover schema discovery, table retrieval, the SQL guard (including attempts to bypass it), read-only execution, and the API end to end.
