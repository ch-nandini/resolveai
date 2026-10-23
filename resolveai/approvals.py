"""Human-in-the-loop: operators approve or reject escalated cases."""
import json

from .db import log_event, now_iso, row_to_dict
from .tools import ToolRuntime


def pending(conn):
    rows = conn.execute(
        """SELECT a.*, c.customer_id, c.order_id, c.message, c.category, c.summary
           FROM approvals a JOIN cases c USING(case_id) WHERE a.status='pending' ORDER BY a.created_at"""
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("recommendation", "reasons", "evidence"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out


def _get(conn, case_id):
    a = row_to_dict(conn.execute("SELECT * FROM approvals WHERE case_id=?", (case_id,)).fetchone())
    if not a:
        raise KeyError(f"No approval for {case_id}")
    if a["status"] != "pending":
        raise ValueError(f"{case_id} already {a['status']}")
    return a


def approve(conn, case_id, operator, amount, note=""):
    """Operator approves a refund. Uses the same idempotent refund tool the agent uses."""
    _get(conn, case_id)
    case = row_to_dict(conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())
    rt = ToolRuntime(conn, case_id, failure_rate=0, outage_rate=0, latency_scale=0)
    res = rt.call("create_refund", {
        "order_id": case["order_id"], "customer_id": case["customer_id"], "amount": float(amount),
        "reason": f"operator approval: {note}".strip(), "idempotency_key": f"{case_id}:refund", "source": "operator",
    })
    if not res["ok"]:
        raise RuntimeError(res["error"])
    conn.execute("UPDATE approvals SET status='approved', decided_at=?, operator=?, note=? WHERE case_id=?",
                 (now_iso(), operator, note, case_id))
    conn.execute("UPDATE cases SET final_outcome='refunded', final_amount=?, resolved_by=?, updated_at=? WHERE case_id=?",
                 (float(amount), f"human:{operator}", now_iso(), case_id))
    conn.commit()
    log_event(conn, case_id, "human_decision", "approve", {"operator": operator, "amount": amount, "note": note,
                                                           "refund_id": res["data"]["refund_id"]})
    return res["data"]


def reject(conn, case_id, operator, note=""):
    _get(conn, case_id)
    conn.execute("UPDATE approvals SET status='rejected', decided_at=?, operator=?, note=? WHERE case_id=?",
                 (now_iso(), operator, note, case_id))
    conn.execute("UPDATE cases SET final_outcome='denied', final_amount=NULL, resolved_by=?, updated_at=? WHERE case_id=?",
                 (f"human:{operator}", now_iso(), case_id))
    conn.commit()
    log_event(conn, case_id, "human_decision", "reject", {"operator": operator, "note": note})
