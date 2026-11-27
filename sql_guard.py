"""Reject anything that is not a single read-only SELECT.

LLM output is untrusted input: a prompt like "delete old orders" or a
hallucination can produce destructive SQL. This check runs before execution;
execute_query() additionally opens the connection in read-only mode, and the
README recommends a read-only database user as the final layer.
"""
import sqlparse
from sqlparse import tokens as T

FORBIDDEN = {
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE",
    "DROP", "ALTER", "CREATE", "TRUNCATE", "RENAME",
    "GRANT", "REVOKE", "CALL", "EXEC", "EXECUTE", "DO", "HANDLER",
    "INTO",  # SELECT ... INTO OUTFILE / INTO new_table
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "COPY",
    "LOCK", "UNLOCK", "SET", "USE", "LOAD",
}


class UnsafeSQLError(ValueError):
    pass


def validate_read_only(sql: str) -> str:
    """Return the cleaned statement, or raise UnsafeSQLError explaining why."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL.")

    statements = [s for s in sqlparse.parse(sql) if s.value.strip().strip(";").strip()]
    if len(statements) != 1:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")
    stmt = statements[0]

    if stmt.get_type() != "SELECT":
        raise UnsafeSQLError(f"Only SELECT queries are allowed (got {stmt.get_type()}).")

    # Walk every token so keywords hidden in CTEs or subqueries are caught,
    # while words inside string literals and identifiers are ignored.
    for tok in stmt.flatten():
        if tok.ttype in T.Keyword or tok.ttype in T.Name.Builtin:
            if tok.normalized.upper() in FORBIDDEN:
                raise UnsafeSQLError(f"Forbidden keyword in query: {tok.normalized.upper()}")

    return stmt.value.strip().rstrip(";").strip()
