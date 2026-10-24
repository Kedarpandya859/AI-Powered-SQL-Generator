import os
import openai
import re
import pandas as pd
from dotenv import load_dotenv
from databases import engine, list_databases, list_tables, list_columns
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_sql_output(text):
    pattern = r"``````"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def generate_sql_query(natural_language_query, max_tables=5, max_columns=5):
    # Smart schema context
    databases = list_databases()
    schema_info = ""
    for db in databases[:max_tables]:
        tables = list_tables(db)
        for tbl in tables[:max_tables]:
            columns = list_columns(db, tbl)
            schema_info += f"Table {db}.{tbl} with columns: {', '.join(columns[:max_columns])}\n"

    prompt = (
        f"You are an AI that converts natural language queries into optimized SQL queries.\n"
        f"Database schema info:\n{schema_info}\n"
        f"Convert the following request into a valid SQL query: \"{natural_language_query}\""
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=256,
        n=1,
        stop=None,
        temperature=0,
    )
    raw_sql = response.choices[0].text
    sql = clean_sql_output(raw_sql)
    return sql

def explain_sql_query(sql):
    prompt = (
        f"You are an AI SQL expert. Explain the purpose and logic of the following SQL query in plain English:\nSQL:\n{sql}"
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        temperature=0.4,
    )
    return response.choices[0].text.strip()

def execute_query(sql_query):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = [dict(row) for row in result]
        return rows
    except SQLAlchemyError as e:
        return f"SQL Execution Error: {str(e)}"

def results_to_csv(results):
    if not isinstance(results, list) or len(results) == 0:
        return ""
    df = pd.DataFrame(results)
    return df.to_csv(index=False)
