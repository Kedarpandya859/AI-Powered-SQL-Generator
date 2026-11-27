import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI SQL Query Generator", layout="wide")
st.title("AI-Powered SQL Query Generator")

for key, default in {"history": [], "sql": "", "generation": None}.items():
    st.session_state.setdefault(key, default)


def api_get(path):
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Backend unavailable at {API_URL}: {e}")
        st.stop()


def api_post(path, payload):
    try:
        resp = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    except requests.RequestException as e:
        st.error(f"Backend unavailable at {API_URL}: {e}")
        return None
    if resp.status_code != 200:
        st.error(resp.json().get("detail", resp.text))
        return None
    return resp.json()


# --- Sidebar: database selection and schema browser -------------------------
with st.sidebar:
    health = api_get("/health")
    st.caption(f"Connected to **{health['dialect']}**")
    databases = api_get("/databases")
    if not databases:
        st.warning("No user databases found.")
        st.stop()
    default = health.get("default_database")
    index = databases.index(default) if default in databases else 0
    database = st.selectbox("Database", databases, index=index)

    tables = api_get(f"/tables/{database}")
    st.caption(f"{len(tables)} tables. The most relevant ones are sent to the model automatically.")
    with st.expander("Browse schema"):
        table = st.selectbox("Table", tables) if tables else None
        if table:
            st.write(api_get(f"/columns/{database}/{table}"))
    if st.button("Refresh schema cache"):
        api_post(f"/schema/refresh/{database}", {})

# --- Generate ---------------------------------------------------------------
question = st.text_area("Ask a question about your data", placeholder="Top 10 customers by total order value")

if st.button("Generate SQL", type="primary", disabled=not question.strip()):
    with st.spinner("Generating..."):
        gen = api_post("/generate_sql", {"question": question, "database": database})
    if gen:
        st.session_state["generation"] = gen
        st.session_state["sql"] = gen["sql"]
        st.session_state["history"].append({"question": question, **gen})

gen = st.session_state["generation"]
if gen:
    if gen.get("explanation"):
        st.info(gen["explanation"])
    with st.expander("Tables sent to the model"):
        st.write(", ".join(gen.get("tables_used", [])))
    if not gen.get("is_safe"):
        st.warning(f"This query will not run: {gen.get('safety_error')}")

# Editable, so an analyst can tweak the generated SQL before running it.
sql = st.text_area("SQL", key="sql", height=160)

if st.button("Execute SQL", disabled=not sql.strip()):
    with st.spinner("Running..."):
        res = api_post("/execute_sql", {"sql": sql})
    if res is not None:
        if res["rows"]:
            st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]), use_container_width=True)
            if res.get("truncated"):
                st.caption("Showing the first rows only. Results were truncated.")
            st.download_button("Download results as CSV", res["csv"], file_name="results.csv")
        else:
            st.info("Query returned no rows.")

# --- History ----------------------------------------------------------------
if st.session_state["history"]:
    st.subheader("Query history")
    for i, h in enumerate(reversed(st.session_state["history"]), 1):
        st.markdown(f"**{i}. {h['question']}**")
        st.code(h["sql"] or "-- no SQL generated", language="sql")
        if h.get("explanation"):
            st.caption(h["explanation"])
