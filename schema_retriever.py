"""Pick the tables relevant to a question so large schemas fit in the prompt.

Sending every table to the LLM stops working past a few dozen tables (token
limits, cost, and the model gets distracted). Instead we rank tables against the
question and send only the top few, plus the tables they join to via foreign keys.

Ranking uses OpenAI embeddings when an API key is available (semantic match:
"revenue" finds ``order_items.unit_price``) and falls back to keyword overlap
otherwise, so the app and tests also work offline.
"""
import hashlib
import logging
import math
import os
import re
from collections import Counter

from databases import TableInfo

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
_embedding_cache: dict = {}  # sha1(doc text) -> vector


def _tokens(text: str) -> list:
    """Lowercase words, split snake_case/camelCase, crude singularisation."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-zA-Z]+", text.lower())
    out = []
    for w in words:
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


# Business words analysts use that rarely appear verbatim in column names.
# Only used by the keyword fallback; embeddings learn these relations themselves.
SYNONYMS = {
    "revenue": ["price", "amount", "total", "quantity", "order"],
    "sale": ["order", "quantity", "price", "amount"],
    "sold": ["order", "quantity"],
    "spend": ["amount", "price", "total"],
    "buyer": ["customer"],
    "client": ["customer"],
    "user": ["customer", "account"],
    "staff": ["employee"],
    "pay": ["salary", "payroll"],
    "item": ["product"],
    "unit": ["quantity"],
}


def _expand(tokens: list) -> Counter:
    out = Counter(tokens)
    for tok in tokens:
        for syn in SYNONYMS.get(tok, []):
            out[syn] += 0.5
    return out


def table_document(table: TableInfo) -> str:
    cols = " ".join(c for c, _ in table.columns)
    return f"table {table.name} columns {cols}"


def _keyword_scores(question: str, tables: list) -> list:
    q = _expand(_tokens(question))
    n = len(tables)
    docs = [Counter(_tokens(table_document(t))) for t in tables]
    df = Counter(tok for d in docs for tok in d)
    scores = []
    for t, d in zip(tables, docs):
        score = 0.0
        name_tokens = set(_tokens(t.name))
        for tok, weight in q.items():
            if tok in d:
                idf = math.log(1 + n / df[tok])
                # A match on the table name counts more than a column match.
                score += weight * idf * (3.0 if tok in name_tokens else 1.0)
        scores.append(score)
    return scores


def _embed(client, texts: list) -> list:
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    missing = [t for t, k in zip(texts, keys) if k not in _embedding_cache]
    for i in range(0, len(missing), 256):
        batch = missing[i:i + 256]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for text, item in zip(batch, resp.data):
            _embedding_cache[hashlib.sha1(text.encode()).hexdigest()] = item.embedding
    return [_embedding_cache[k] for k in keys]


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embedding_scores(client, question: str, tables: list) -> list:
    vectors = _embed(client, [table_document(t) for t in tables] + [question])
    q = vectors[-1]
    return [_cosine(q, v) for v in vectors[:-1]]


def select_relevant_tables(question: str, tables: list, top_k: int = 8, client=None) -> list:
    """Return at most ``top_k`` tables: best matches first, then FK neighbours."""
    if len(tables) <= top_k:
        return list(tables)

    scores = None
    if client is not None:
        try:
            scores = _embedding_scores(client, question, tables)
        except Exception as e:  # noqa: BLE001 - network/API errors fall back to keywords
            logger.warning("Embedding retrieval failed, using keyword match: %s", e)
    if scores is None:
        scores = _keyword_scores(question, tables)

    ranked = [t for _, t in sorted(zip(scores, tables), key=lambda p: -p[0])]
    by_name = {t.name: t for t in tables}

    # Take the strongest matches, leaving room for the tables they join to.
    seeds = ranked[: max(1, top_k // 2)]
    chosen = list(seeds)
    for t in seeds:
        for _, ref_table, _ in t.foreign_keys:
            ref = by_name.get(ref_table)
            if ref is not None and ref not in chosen and len(chosen) < top_k:
                chosen.append(ref)
    for t in ranked:
        if len(chosen) >= top_k:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen
