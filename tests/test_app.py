import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import databases
import query_generator
from app import app
from conftest import FakeLLM
from schema_retriever import select_relevant_tables
from sql_guard import UnsafeSQLError, validate_read_only


# --- Schema discovery -------------------------------------------------------

def test_schema_discovery_on_100_tables(db):
    assert databases.list_databases() == ["main"]
    assert len(databases.list_tables("main")) == 100
    assert databases.list_columns("main", "orders") == ["id", "customer_id", "order_date", "status"]
    orders = next(t for t in databases.get_schema("main") if t.name == "orders")
    assert orders.primary_key == ["id"]
    assert ("customer_id", "customers", "id") in orders.foreign_keys
    assert "customer_id INTEGER -> customers.id" in orders.to_prompt_line()


def test_unknown_database_rejected(db):
    with pytest.raises(ValueError):
        databases.list_tables("main; DROP TABLE customers")


def test_system_schemas_hidden(monkeypatch, db):
    monkeypatch.setattr(databases, "inspect", lambda e: type("I", (), {
        "get_schema_names": lambda self: ["information_schema", "mysql", "shop", "sys"]})())
    assert databases.list_databases() == ["shop"]


# --- Retrieval --------------------------------------------------------------

@pytest.mark.parametrize("question, must_include", [
    ("total revenue per customer", {"customers", "order_items"}),
    ("which products sold the most units", {"products", "order_items"}),
    ("average salary by department", {"employees"}),
])
def test_retrieval_picks_relevant_tables(db, question, must_include):
    chosen = select_relevant_tables(question, databases.get_schema("main"), top_k=8)
    names = {t.name for t in chosen}
    assert len(chosen) == 8
    assert must_include <= names, names


def test_retrieval_adds_foreign_key_neighbours(db):
    chosen = {t.name for t in select_relevant_tables("order items quantity", databases.get_schema("main"), top_k=6)}
    # order_items references orders and products; both should come along for joins.
    assert {"order_items", "orders", "products"} <= chosen


# --- SQL guard --------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * FROM customers",
    "select name from customers where note = 'please drop table'",
    "WITH t AS (SELECT customer_id, COUNT(*) c FROM orders GROUP BY 1) SELECT * FROM t;",
    "SELECT REPLACE(name, 'a', 'b') FROM customers",
])
def test_guard_allows_reads(sql):
    assert validate_read_only(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers",
    "DELETE FROM orders",
    "UPDATE customers SET name = 'x'",
    "SELECT 1; DROP TABLE customers",
    "SELECT * INTO backup FROM customers",
    "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
    "PRAGMA writable_schema = 1",
    "",
])
def test_guard_blocks_writes(sql):
    with pytest.raises(UnsafeSQLError):
        validate_read_only(sql)


# --- Generation (LLM mocked) ------------------------------------------------

def test_generate_sends_only_relevant_tables(db):
    llm = FakeLLM(sql="```sql\nSELECT c.name FROM customers c;\n```", explanation="Lists names.")
    out = query_generator.generate_sql_query("list customer names", "main", client=llm)
    assert out["sql"] == "SELECT c.name FROM customers c"
    assert out["is_safe"] and out["explanation"] == "Lists names."
    prompt = llm.last_user_prompt
    assert "customers(" in prompt
    assert prompt.count("\n") <= query_generator.TOP_K_TABLES + 3  # not all 100 tables
    assert llm.calls[-1]["temperature"] == 0
    assert "sqlite" in llm.calls[-1]["messages"][0]["content"]


def test_generate_flags_unsafe_sql(db):
    out = query_generator.generate_sql_query("remove old orders", "main", client=FakeLLM(sql="DELETE FROM orders"))
    assert not out["is_safe"] and "SELECT" in out["safety_error"]


def test_generate_handles_non_json_reply(db):
    out = query_generator.generate_sql_query("count orders", "main", client=FakeLLM(raw="SELECT COUNT(*) FROM orders"))
    assert out["sql"] == "SELECT COUNT(*) FROM orders"


# --- Execution --------------------------------------------------------------

def test_execute_returns_rows_and_truncates(db):
    res = query_generator.execute_query(
        "SELECT c.name, SUM(oi.quantity * oi.unit_price) AS revenue FROM customers c "
        "JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id "
        "GROUP BY c.name ORDER BY revenue DESC")
    assert res["columns"] == ["name", "revenue"]
    assert res["rows"][0] == {"name": "Asha", "revenue": 65.0}
    assert query_generator.execute_query("SELECT * FROM order_items", max_rows=2)["truncated"] is True


def test_connection_is_read_only_even_if_guard_bypassed(db, monkeypatch):
    monkeypatch.setattr(query_generator, "validate_read_only", lambda s: s)
    with pytest.raises(Exception, match="readonly"):
        query_generator.execute_query("DELETE FROM orders")
    with db.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM orders")).scalar() == 3
        conn.execute(text("INSERT INTO employees VALUES (1, 'Z', 'ops', 1)"))  # normal conns still writable
        conn.commit()


# --- API --------------------------------------------------------------------

def test_api_end_to_end(db, monkeypatch):
    monkeypatch.setattr(query_generator, "get_client", lambda: FakeLLM(sql="SELECT name FROM customers ORDER BY id"))
    client = TestClient(app)
    assert client.get("/health").json()["dialect"] == "sqlite"
    assert client.get("/tables/nope").status_code == 404

    gen = client.post("/generate_sql", json={"question": "customer names", "database": "main"}).json()
    assert gen["is_safe"] and gen["sql"].startswith("SELECT")

    res = client.post("/execute_sql", json={"sql": gen["sql"]}).json()
    assert [r["name"] for r in res["rows"]] == ["Asha", "Ben"]
    assert res["csv"].splitlines() == ["name", "Asha", "Ben"]

    bad = client.post("/execute_sql", json={"sql": "DROP TABLE customers"})
    assert bad.status_code == 400
    err = client.post("/execute_sql", json={"sql": "SELECT nope FROM customers"})
    assert err.status_code == 400 and "SQL error" in err.json()["detail"]


def test_api_without_key_returns_503(db, monkeypatch):
    monkeypatch.setattr(query_generator, "get_client", lambda: None)
    r = TestClient(app).post("/generate_sql", json={"question": "anything"})
    assert r.status_code == 503
