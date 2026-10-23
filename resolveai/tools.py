"""Mock company systems exposed as agent tools.

Every call goes through ToolRuntime.call(), which adds:
  * simulated latency and deterministic fault injection (transient errors + per-case outages)
  * bounded retries with exponential backoff
  * an audit-log row for every attempt and for the final result

Note: create_refund is NOT in PLANNER_TOOLS. The LLM can only *gather evidence*; money only
moves from the execute node (after deterministic guardrails) or from a human approval.
"""
import hashlib
import os
import re
import time
from datetime import datetime, timedelta

from . import config
from .db import log_event, row_to_dict

# ---------------------------------------------------------------------------
# Tool schemas (JSON Schema) shown to the LLM planner
# ---------------------------------------------------------------------------
PLANNER_TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order: item, amount, dates, tracking status, signature confirmation, final-sale flag and whether it was already refunded. Also verifies the order belongs to the customer.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "customer_id": {"type": "string"}},
            "required": ["order_id", "customer_id"],
        },
    },
    {
        "name": "get_customer_history",
        "description": "Customer account age, order count, lifetime spend, refunds in the last 30 days, total refunds and chargebacks.",
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "run_fraud_check",
        "description": "Risk score from 0 to 1 for this customer/order, with the signals that drove it.",
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "order_id": {"type": "string"}},
            "required": ["customer_id", "order_id"],
        },
    },
    {
        "name": "search_refund_policy",
        "description": "Search the refund policy. Returns the most relevant policy sections for a query such as 'damaged item' or 'late delivery'.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

EVIDENCE_KEY = {
    "lookup_order": "order",
    "get_customer_history": "history",
    "run_fraud_check": "fraud",
    "search_refund_policy": "policy",
}

BASE_LATENCY_MS = {
    "lookup_order": 40,
    "get_customer_history": 60,
    "run_fraud_check": 120,
    "search_refund_policy": 30,
    "create_refund": 80,
}


def _u(*parts):
    """Deterministic pseudo-random float in [0,1) from arbitrary parts."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _parse(ts):
    return datetime.fromisoformat(ts) if ts else None


# ---------------------------------------------------------------------------
# Policy search (tiny keyword retriever over policy.md)
# ---------------------------------------------------------------------------
_POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy.md")
_SYNONYMS = {
    "not_delivered": "lost missing package delivered never arrived transit",
    "damaged": "damaged broken defective cracked",
    "wrong_item": "wrong item size color",
    "late_delivery": "late delivery promised date goodwill credit",
    "changed_mind": "change of mind no longer want return final-sale",
}


def load_policy_sections():
    text = open(_POLICY_PATH, encoding="utf-8").read()
    sections = []
    for block in re.split(r"\n## ", text)[1:]:
        title, _, body = block.partition("\n")
        m = re.match(r"(§\d+)\s+(.*)", title.strip())
        sections.append({"section": m.group(1), "title": m.group(2), "text": body.strip()})
    return sections


def _tokens(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def policy_search(query, k=2):
    q = query
    for key, syn in _SYNONYMS.items():
        if key in query:
            q += " " + syn
    qt = _tokens(q)
    scored = []
    for s in load_policy_sections():
        st = _tokens(s["title"] + " " + s["text"])
        title_hits = len(qt & _tokens(s["title"]))
        scored.append((len(qt & st) + 2 * title_hits, s))
    scored.sort(key=lambda x: -x[0])
    out = [s for score, s in scored[:k] if score > 0]
    # Risk limits are always relevant to a refund decision.
    if not any(s["section"] == "§8" for s in out):
        out.append(next(s for s in load_policy_sections() if s["section"] == "§8"))
    return out


# ---------------------------------------------------------------------------
# Raw tool implementations (pure reads/writes against SQLite)
# ---------------------------------------------------------------------------
def _lookup_order(conn, order_id, customer_id):
    row = row_to_dict(conn.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone())
    if not row:
        return {"found": False, "order_id": order_id}
    if row["customer_id"] != customer_id:
        # Don't leak another customer's order details.
        return {"found": True, "owner_match": False, "order_id": order_id}
    refunded = conn.execute(
        "SELECT COALESCE(SUM(amount),0) s, COUNT(*) n FROM refunds WHERE order_id=?", (order_id,)
    ).fetchone()
    now = config.SIM_NOW
    delivered = _parse(row["delivered_at"])
    promised = _parse(row["promised_at"])
    return {
        "found": True,
        "owner_match": True,
        "order_id": order_id,
        "item": row["item"],
        "amount": row["amount"],
        "final_sale": bool(row["final_sale"]),
        "ordered_at": row["ordered_at"],
        "promised_at": row["promised_at"],
        "delivered_at": row["delivered_at"],
        "tracking_status": row["tracking_status"],
        "signature_confirmed": bool(row["signature"]),
        "days_since_delivery": (now - delivered).days if delivered else None,
        "days_late": (delivered - promised).days if delivered and promised else None,
        "days_past_promised": (now - promised).days if promised and not delivered else None,
        "already_refunded": refunded["n"] > 0,
        "refunded_amount": refunded["s"],
    }


def _customer_history(conn, customer_id):
    c = row_to_dict(conn.execute("SELECT * FROM customers WHERE customer_id=?", (customer_id,)).fetchone())
    if not c:
        return {"found": False, "customer_id": customer_id}
    now = config.SIM_NOW
    orders = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(amount),0) s FROM orders WHERE customer_id=?", (customer_id,)
    ).fetchone()
    since = (now - timedelta(days=30)).isoformat()
    r30 = conn.execute(
        "SELECT COUNT(*) FROM refunds WHERE customer_id=? AND created_at>=?",
        (customer_id, since),
    ).fetchone()[0]
    rall = conn.execute("SELECT COUNT(*) FROM refunds WHERE customer_id=?", (customer_id,)).fetchone()[0]
    return {
        "found": True,
        "customer_id": customer_id,
        "account_age_days": (now - _parse(c["created_at"])).days,
        "total_orders": orders["n"],
        "lifetime_spend": round(orders["s"], 2),
        "refunds_last_30d": r30,
        "refunds_total": rall,
        "chargebacks": c["chargebacks"],
    }


def _fraud_check(conn, customer_id, order_id):
    hist = _customer_history(conn, customer_id)
    c = row_to_dict(conn.execute("SELECT * FROM customers WHERE customer_id=?", (customer_id,)).fetchone()) or {}
    o = row_to_dict(conn.execute("SELECT amount FROM orders WHERE order_id=?", (order_id,)).fetchone()) or {}
    signals, score = [], 0.05
    if hist.get("found") and hist["account_age_days"] < config.NEW_ACCOUNT_DAYS:
        score += 0.25; signals.append("new_account")
    if hist.get("refunds_last_30d", 0):
        score += 0.12 * min(hist["refunds_last_30d"], 4); signals.append("recent_refunds")
    if c.get("chargebacks"):
        score += 0.30 * min(c["chargebacks"], 2); signals.append("prior_chargebacks")
    if c.get("address_mismatch"):
        score += 0.20; signals.append("billing_shipping_mismatch")
    if (o.get("amount") or 0) > 300:
        score += 0.10; signals.append("high_value_order")
    score += (_u("fraud-noise", customer_id, order_id) - 0.5) * 0.08
    return {"score": round(min(max(score, 0.0), 1.0), 3), "signals": signals}


def _create_refund(conn, order_id, customer_id, amount, reason, idempotency_key, source):
    existing = conn.execute("SELECT * FROM refunds WHERE idempotency_key=?", (idempotency_key,)).fetchone()
    if existing:
        return {**row_to_dict(existing), "idempotent_replay": True}
    refund_id = "R" + hashlib.md5(idempotency_key.encode()).hexdigest()[:10].upper()
    conn.execute(
        "INSERT INTO refunds(refund_id, order_id, customer_id, amount, reason, created_at, source, idempotency_key) VALUES (?,?,?,?,?,?,?,?)",
        (refund_id, order_id, customer_id, round(float(amount), 2), reason, config.SIM_NOW.isoformat(), source, idempotency_key),
    )
    conn.commit()
    return {"refund_id": refund_id, "order_id": order_id, "amount": round(float(amount), 2), "idempotent_replay": False}


# ---------------------------------------------------------------------------
# Runtime: retries, fault injection, latency, audit logging
# ---------------------------------------------------------------------------
class ToolError(Exception):
    pass


class ToolRuntime:
    def __init__(self, conn, case_id, failure_rate=None, outage_rate=None, latency_scale=None, sleep=True):
        self.conn = conn
        self.case_id = case_id
        self.failure_rate = config.TOOL_FAILURE_RATE if failure_rate is None else failure_rate
        self.outage_rate = config.OUTAGE_RATE if outage_rate is None else outage_rate
        self.latency_scale = config.LATENCY_SCALE if latency_scale is None else latency_scale
        self.sleep = sleep
        # One evidence tool may be "down" for the whole case.
        self.outage_tool = None
        if _u(case_id, "outage") < self.outage_rate:
            self.outage_tool = ["get_customer_history", "run_fraud_check"][int(_u(case_id, "which") * 2)]

    def _impl(self, name, args):
        c = self.conn
        if name == "lookup_order":
            return _lookup_order(c, args["order_id"], args["customer_id"])
        if name == "get_customer_history":
            return _customer_history(c, args["customer_id"])
        if name == "run_fraud_check":
            return _fraud_check(c, args["customer_id"], args["order_id"])
        if name == "search_refund_policy":
            return {"sections": policy_search(args["query"])}
        if name == "create_refund":
            return _create_refund(c, args["order_id"], args["customer_id"], args["amount"], args["reason"],
                                  args["idempotency_key"], args.get("source", "agent"))
        raise ToolError(f"unknown tool {name}")

    def _maybe_fail(self, name, args, attempt):
        if self.outage_tool == name:
            raise ConnectionError(f"{name}: 503 service unavailable")
        if _u(self.case_id, name, sorted(args.items()), attempt, "fail") < self.failure_rate:
            raise TimeoutError(f"{name}: upstream timeout")

    def call(self, name, args):
        """Run a tool with retries. Returns {ok, data, error, attempts, latency_ms}."""
        t0 = time.perf_counter()
        last_err = None
        attempts = 0
        for attempt in range(1, config.MAX_TOOL_RETRIES + 2):
            attempts = attempt
            a0 = time.perf_counter()
            try:
                if self.sleep and self.latency_scale > 0:
                    jitter = 0.7 + 0.6 * _u(self.case_id, name, attempt, "lat")
                    time.sleep(BASE_LATENCY_MS.get(name, 50) * jitter * self.latency_scale / 1000)
                self._maybe_fail(name, args, attempt)
                data = self._impl(name, args)
                log_event(self.conn, self.case_id, "tool_attempt", name, {"args": args}, True,
                          (time.perf_counter() - a0) * 1000, attempt)
                total = (time.perf_counter() - t0) * 1000
                log_event(self.conn, self.case_id, "tool_call", name, {"args": args, "result": data}, True, total, attempts)
                return {"ok": True, "data": data, "error": None, "attempts": attempts, "latency_ms": round(total, 1)}
            except ToolError:
                raise
            except Exception as e:  # transient / outage
                last_err = f"{type(e).__name__}: {e}"
                log_event(self.conn, self.case_id, "tool_attempt", name, {"args": args, "error": last_err}, False,
                          (time.perf_counter() - a0) * 1000, attempt)
                if attempt <= config.MAX_TOOL_RETRIES and self.sleep and self.latency_scale > 0:
                    time.sleep(0.1 * (2 ** (attempt - 1)) * self.latency_scale)
        total = (time.perf_counter() - t0) * 1000
        log_event(self.conn, self.case_id, "tool_call", name, {"args": args, "error": last_err}, False, total, attempts)
        return {"ok": False, "data": None, "error": last_err, "attempts": attempts, "latency_ms": round(total, 1)}
