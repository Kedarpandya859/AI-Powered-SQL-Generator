from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from databases import default_database, dialect_name, get_schema, list_columns, list_databases, list_tables
from query_generator import execute_query, generate_sql_query, results_to_csv
from sql_guard import UnsafeSQLError

app = FastAPI(title="AI-Powered SQL Query Generator")


class GenerateRequest(BaseModel):
    question: str
    database: Optional[str] = None


class ExecuteRequest(BaseModel):
    sql: str


@app.get("/health")
def health():
    return {"status": "ok", "dialect": dialect_name(), "default_database": default_database()}


@app.get("/databases")
def get_databases():
    return list_databases()


@app.get("/tables/{database_name}")
def get_tables(database_name: str):
    try:
        return list_tables(database_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/columns/{database_name}/{table_name}")
def get_columns(database_name: str, table_name: str):
    try:
        return list_columns(database_name, table_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=404, detail=str(e.__cause__ or e))


@app.post("/schema/refresh/{database_name}")
def refresh_schema(database_name: str):
    try:
        return {"tables": len(get_schema(database_name, refresh=True))}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/generate_sql")
def generate_sql_endpoint(req: GenerateRequest):
    try:
        return generate_sql_query(req.question, req.database)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/execute_sql")
def execute_sql_endpoint(req: ExecuteRequest):
    try:
        result = execute_query(req.sql)
    except UnsafeSQLError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e.__cause__ or e}")
    result["csv"] = results_to_csv(result["rows"], result["columns"])
    return jsonable_encoder(result)
