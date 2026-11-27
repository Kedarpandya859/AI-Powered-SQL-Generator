"""Shared fixtures: a 100-table SQLite database and a fake OpenAI client."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import databases  # noqa: E402

CORE_SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, country TEXT);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price NUMERIC);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id),
                     order_date DATE, status TEXT);
CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id),
                          product_id INTEGER REFERENCES products(id), quantity INTEGER,
                          unit_price NUMERIC);
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary NUMERIC);
"""

FILLER_TOPICS = ["audit_log", "warehouse_stock", "shipment", "invoice_line", "marketing_campaign",
                 "support_ticket", "payroll_run", "supplier", "web_session", "sensor_reading"]


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    raw = engine.raw_connection()
    try:
        raw.executescript(CORE_SCHEMA)
        # 95 filler tables so the schema totals 100 tables.
        for i in range(95):
            topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
            raw.execute(f"CREATE TABLE {topic}_{i:02d} (id INTEGER PRIMARY KEY, ref_code TEXT, "
                        f"created_at TIMESTAMP, value NUMERIC)")
        raw.executescript("""
            INSERT INTO customers VALUES (1,'Asha','a@x.com','IN'),(2,'Ben','b@x.com','US');
            INSERT INTO products VALUES (1,'Pen','stationery',2.5),(2,'Lamp','home',30);
            INSERT INTO orders VALUES (1,1,'2026-01-05','paid'),(2,2,'2026-01-06','paid'),
                                      (3,1,'2026-02-01','refunded');
            INSERT INTO order_items VALUES (1,1,1,10,2.5),(2,1,2,1,30),(3,2,2,2,30),(4,3,1,4,2.5);
        """)
        raw.commit()
    finally:
        raw.close()
    databases.set_engine(engine)
    yield engine
    engine.dispose()


class FakeLLM:
    """Mimics the parts of the OpenAI client the app uses and records prompts."""

    def __init__(self, sql="SELECT 1", explanation="test", raw=None):
        self.calls = []
        reply = raw if raw is not None else json.dumps({"sql": sql, "explanation": explanation})

        def create(**kwargs):
            self.calls.append(kwargs)
            msg = SimpleNamespace(content=reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def embed(**kwargs):
            raise RuntimeError("no network in tests")

        self.embeddings = SimpleNamespace(create=embed)

    @property
    def last_user_prompt(self):
        return self.calls[-1]["messages"][-1]["content"]
