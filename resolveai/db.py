"""SQLite persistence: the mock 'company systems' plus the agent's case, audit and approval tables."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT, email TEXT,
    created_at TEXT,
    chargebacks INTEGER DEFAULT 0,
    address_mismatch INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    item TEXT,
    amount REAL,
    final_sale INTEGER DEFAULT 0,
    ordered_at TEXT, promised_at TEXT, delivered_at TEXT,
    tracking_status TEXT,
    signature INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT, customer_id TEXT,
    amount REAL, reason TEXT,
    created_at TEXT,
    source TEXT,
    idempotency_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    customer_id TEXT, order_id TEXT, message TEXT,
    scenario TEXT, expected_decision TEXT, expected_amount REAL,
    agent_decision TEXT, agent_amount REAL,
    final_outcome TEXT, final_amount REAL, resolved_by TEXT,
    category TEXT, summary TEXT, customer_reply TEXT,
    steps INTEGER, tool_calls INTEGER,
    tokens_in INTEGER, tokens_out INTEGER, latency_ms REAL,
    created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT, ts TEXT,
    event TEXT, name TEXT,
    payload TEXT,
    ok INTEGER, latency_ms REAL, attempt INTEGER
);
CREATE TABLE IF NOT EXISTS approvals (
    case_id TEXT PRIMARY KEY,
    status TEXT,
    recommendation TEXT, reasons TEXT, evidence TEXT,
    created_at TEXT, decided_at TEXT,
    operator TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_log(case_id);
CREATE INDEX IF NOT EXISTS idx_refunds_customer ON refunds(customer_id);
"""

_conns = {}


def get_conn(path=None):
    path = path or config.DB_PATH
    if path not in _conns:
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        _conns[path] = conn
    return _conns[path]


def reset(path=None):
    """Drop everything and recreate the schema."""
    conn = get_conn(path)
    for t in ["customers", "orders", "refunds", "cases", "audit_log", "approvals"]:
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

def log_event(conn, case_id, event, name=None, payload=None, ok=True, latency_ms=None, attempt=None):
    conn.execute(
        "INSERT INTO audit_log(case_id, ts, event, name, payload, ok, latency_ms, attempt) VALUES (?,?,?,?,?,?,?,?)",
        (case_id, now_iso(), event, name, json.dumps(payload, default=str), int(bool(ok)), latency_ms, attempt),
    )
    conn.commit()


def row_to_dict(row):
    return dict(row) if row is not None else None
