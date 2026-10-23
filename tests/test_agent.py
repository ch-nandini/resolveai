"""Run with: python -m pytest -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from resolveai import approvals, config, datagen, db, guardrails  # noqa: E402
from resolveai.graph import ResolveAgent  # noqa: E402
from resolveai.llm import MockLLM  # noqa: E402

FAST = {"latency_scale": 0}


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "t.db")
    c = db.get_conn(path)
    db.reset(path)
    datagen.generate(c, seed=7)
    return c


def first_case(conn, scenario):
    return conn.execute("SELECT case_id FROM cases WHERE scenario=? ORDER BY case_id LIMIT 1", (scenario,)).fetchone()[0]


# ---------------------------------------------------------------- guardrails
def _state(action="refund", amount=50.0, fraud=0.1, r30=0, age=400, conf=0.9, order_amt=100.0):
    return {
        "intent_confidence": conf,
        "evidence": {"order": {"amount": order_amt, "already_refunded": False}, "history": {"refunds_last_30d": r30, "account_age_days": age},
                     "fraud": {"score": fraud}, "policy": {"sections": []}},
        "recommendation": {"action": action, "amount": amount, "confidence": 0.9},
    }


def test_clean_refund_auto_approves():
    assert guardrails.evaluate(_state())["decision"] == "auto_approve"


@pytest.mark.parametrize("kw", [
    {"amount": 250.0, "order_amt": 300.0},     # over limit
    {"fraud": 0.75},                           # fraud
    {"r30": 3},                                # velocity
    {"age": 5, "amount": 80.0},                # new account
    {"conf": 0.4},                             # low confidence
    {"amount": 150.0, "order_amt": 100.0},     # more than paid
])
def test_risk_rules_escalate(kw):
    g = guardrails.evaluate(_state(**kw))
    assert g["decision"] == "escalate" and g["reasons"]


def test_llm_cannot_override_limits_even_when_confident():
    s = _state(amount=500.0, order_amt=500.0)
    s["recommendation"]["confidence"] = 1.0
    assert guardrails.evaluate(s)["decision"] == "escalate"


def test_missing_evidence_blocks_refund_but_not_deny():
    s = _state(); del s["evidence"]["fraud"]
    assert guardrails.evaluate(s)["decision"] == "escalate"
    s["recommendation"] = {"action": "deny", "confidence": 0.9}
    assert guardrails.evaluate(s)["decision"] == "auto_deny"


# ------------------------------------------------------------ end to end
def test_damaged_case_gets_refunded(conn):
    agent = ResolveAgent(conn, MockLLM(), runtime_kwargs={**FAST, "failure_rate": 0, "outage_rate": 0})
    conn.execute("UPDATE cases SET message='The item arrived broken.' WHERE case_id=?", (cid := first_case(conn, "damaged"),))
    final = agent.run(cid)
    assert final["outcome"]["decision"] == "auto_approve"
    assert conn.execute("SELECT COUNT(*) FROM refunds WHERE idempotency_key=?", (f"{cid}:refund",)).fetchone()[0] == 1


def test_high_value_escalates_then_operator_approves(conn):
    agent = ResolveAgent(conn, MockLLM(), runtime_kwargs={**FAST, "failure_rate": 0, "outage_rate": 0})
    cid = first_case(conn, "high_value")
    final = agent.run(cid)
    assert final["outcome"]["decision"] == "escalate"
    assert [p["case_id"] for p in approvals.pending(conn)] == [cid]
    approvals.approve(conn, cid, "tester", 300.0, "verified photos")
    row = conn.execute("SELECT final_outcome, resolved_by FROM cases WHERE case_id=?", (cid,)).fetchone()
    assert tuple(row) == ("refunded", "human:tester")
    with pytest.raises(ValueError):
        approvals.approve(conn, cid, "tester", 300.0)  # can't double-approve


def test_refund_is_idempotent(conn):
    from resolveai.tools import ToolRuntime
    rt = ToolRuntime(conn, "X", failure_rate=0, outage_rate=0, latency_scale=0)
    args = {"order_id": "O50000", "customer_id": "C1000", "amount": 10, "reason": "t", "idempotency_key": "k1"}
    a, b = rt.call("create_refund", args), rt.call("create_refund", args)
    assert a["data"]["refund_id"] == b["data"]["refund_id"] and b["data"]["idempotent_replay"]


def test_retries_recover_transient_failures(conn):
    from resolveai.tools import ToolRuntime
    rt = ToolRuntime(conn, "CASE-RETRY", failure_rate=0.5, outage_rate=0, latency_scale=0)
    results = [rt.call("search_refund_policy", {"query": f"damaged {i}"}) for i in range(20)]
    assert any(r["attempts"] > 1 and r["ok"] for r in results)


def test_outage_leads_to_safe_escalation(conn):
    agent = ResolveAgent(conn, MockLLM(), runtime_kwargs={**FAST, "failure_rate": 1.0})  # everything fails
    final = agent.run(first_case(conn, "damaged"))
    assert final["outcome"]["decision"] == "escalate"
    assert any("Missing evidence" in r for r in final["guardrail"]["reasons"])


# ---------------------------------------------------------------- loop guard
class StuckLLM(MockLLM):
    """A buggy planner that keeps asking for the same tool forever."""
    def next_tool(self, state):
        return {"name": "lookup_order", "args": {"order_id": state["order_id"], "customer_id": state["customer_id"]}}, 10, 5


class WanderingLLM(MockLLM):
    """Never repeats exactly, never says DONE: only the step cap can stop it."""
    def next_tool(self, state):
        return {"name": "search_refund_policy", "args": {"query": f"q{len(state.get('tool_calls', []))}"}}, 10, 5


class RogueLLM(MockLLM):
    """Tries to call the refund tool directly."""
    def next_tool(self, state):
        return {"name": "create_refund", "args": {"order_id": state["order_id"], "amount": 999}}, 10, 5


@pytest.mark.parametrize("llm,flag", [(StuckLLM(), "repeated_call"), (WanderingLLM(), "max_steps"),
                                      (RogueLLM(), "disallowed_tool")])
def test_loop_guard_forces_escalation(conn, llm, flag):
    agent = ResolveAgent(conn, llm, runtime_kwargs={**FAST, "failure_rate": 0, "outage_rate": 0})
    cid = first_case(conn, "damaged")
    final = agent.run(cid)
    assert final["loop_flag"].startswith(flag)
    assert final["outcome"]["decision"] == "escalate"
    assert final["steps"] <= config.MAX_STEPS + 1
    assert conn.execute("SELECT COUNT(*) FROM refunds WHERE idempotency_key=?", (f"{cid}:refund",)).fetchone()[0] == 0
