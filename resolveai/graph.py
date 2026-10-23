"""The agent: a LangGraph state machine.

    intake ──► planner ──► tools ──┐
                  ▲                │
                  └────────────────┘
                  │ (done)            │ (loop guard)
                  ▼                   │
               assess ──► guardrails ◄┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           execute         escalate          deny
              └───────────────┼───────────────┘
                              ▼
                          finalize ──► END

Short-term memory is the graph state itself (evidence, tool-call history, step count), checkpointed
per case via thread_id. Long-lived records (cases, audit log, approvals) live in SQLite.
"""
import json
import operator
import time
from typing import Annotated, Optional, TypedDict

from . import config, guardrails
from .db import get_conn, log_event, now_iso
from .llm import get_llm
from .tools import EVIDENCE_KEY, PLANNER_TOOLS, ToolRuntime

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    USING_LANGGRAPH = True
except ImportError:  # offline fallback, same API subset
    from ._langgraph_compat import END, START, MemorySaver, StateGraph
    USING_LANGGRAPH = False

ALLOWED_TOOLS = {t["name"] for t in PLANNER_TOOLS}


class AgentState(TypedDict, total=False):
    case_id: str
    customer_id: str
    order_id: str
    message: str
    category: str
    intent_confidence: float
    evidence: dict
    evidence_errors: dict
    pending_tool: Optional[dict]
    tool_calls: Annotated[list, operator.add]
    steps: Annotated[int, operator.add]
    loop_flag: Optional[str]
    recommendation: Optional[dict]
    guardrail: Optional[dict]
    outcome: Optional[dict]
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]


def _sig(name, args):
    return name + json.dumps(args, sort_keys=True)


CUSTOMER_REPLIES = {
    "auto_approve": "Good news: we've issued a refund of ${amount:.2f} for order {order_id}. It should appear in 3-5 business days.",
    "auto_deny": "We've reviewed order {order_id}. Unfortunately it isn't eligible for a refund under our policy ({why}).",
    "escalate": "Thanks for your patience. A specialist is reviewing order {order_id} and will get back to you within 24 hours.",
}


class ResolveAgent:
    def __init__(self, conn=None, llm=None, runtime_kwargs=None):
        self.conn = conn or get_conn()
        self.llm = llm or get_llm()
        self.runtime_kwargs = runtime_kwargs or {}
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _rt(self, state):
        return ToolRuntime(self.conn, state["case_id"], **self.runtime_kwargs)

    def intake(self, state):
        res, tin, tout = self.llm.classify(state["message"])
        log_event(self.conn, state["case_id"], "intake", "classify", res)
        return {"category": res["category"], "intent_confidence": res["confidence"],
                "evidence": {}, "evidence_errors": {}, "tokens_in": tin, "tokens_out": tout}

    def planner(self, state):
        if state.get("steps", 0) >= config.MAX_STEPS:
            log_event(self.conn, state["case_id"], "loop_guard", "max_steps", {"steps": state["steps"]}, ok=False)
            return {"pending_tool": None, "loop_flag": f"max_steps={config.MAX_STEPS}"}
        call, tin, tout = self.llm.next_tool(state)
        update = {"steps": 1, "tokens_in": tin, "tokens_out": tout, "pending_tool": call}
        if call:
            if call["name"] not in ALLOWED_TOOLS:
                log_event(self.conn, state["case_id"], "loop_guard", "disallowed_tool", call, ok=False)
                update.update(pending_tool=None, loop_flag=f"disallowed_tool:{call['name']}")
                return update
            sig = _sig(call["name"], call["args"])
            repeats = sum(1 for c in state.get("tool_calls", []) if _sig(c["name"], c["args"]) == sig)
            if repeats >= config.MAX_REPEAT_CALLS:
                log_event(self.conn, state["case_id"], "loop_guard", "repeated_call", {**call, "repeats": repeats}, ok=False)
                update.update(pending_tool=None, loop_flag=f"repeated_call:{call['name']}")
        log_event(self.conn, state["case_id"], "plan", call["name"] if call else "DONE", call or {})
        return update

    def tools(self, state):
        call = state["pending_tool"]
        res = self._rt(state).call(call["name"], call["args"])
        key = EVIDENCE_KEY[call["name"]]
        ev, errs = dict(state.get("evidence") or {}), dict(state.get("evidence_errors") or {})
        if res["ok"]:
            ev[key] = res["data"]
            errs.pop(key, None)
        else:
            errs[key] = res["error"]
        rec = {"name": call["name"], "args": call["args"], "ok": res["ok"], "attempts": res["attempts"],
               "latency_ms": res["latency_ms"], "error": res["error"]}
        return {"evidence": ev, "evidence_errors": errs, "tool_calls": [rec], "pending_tool": None}

    def assess(self, state):
        rec, tin, tout = self.llm.recommend(state)
        log_event(self.conn, state["case_id"], "recommendation", rec["action"], rec)
        return {"recommendation": rec, "tokens_in": tin, "tokens_out": tout}

    def check_guardrails(self, state):
        g = guardrails.evaluate(state)
        log_event(self.conn, state["case_id"], "guardrail", g["decision"], g)
        return {"guardrail": g}

    def execute(self, state):
        g, order = state["guardrail"], state["evidence"]["order"]
        res = self._rt(state).call("create_refund", {
            "order_id": order["order_id"], "customer_id": state["customer_id"], "amount": g["amount"],
            "reason": state.get("category"), "idempotency_key": f"{state['case_id']}:refund", "source": "agent",
        })
        if not res["ok"]:
            g2 = {"decision": "escalate", "amount": g["amount"], "reasons": [f"Refund execution failed: {res['error']}"]}
            return {"guardrail": g2, "outcome": self._escalate(state, g2)}
        return {"outcome": {"decision": "auto_approve", "final": "refunded", "amount": g["amount"],
                            "resolved_by": "agent", "refund_id": res["data"]["refund_id"]}}

    def escalate(self, state):
        return {"outcome": self._escalate(state, state["guardrail"])}

    def _escalate(self, state, g):
        self.conn.execute(
            "INSERT OR REPLACE INTO approvals(case_id, status, recommendation, reasons, evidence, created_at) VALUES (?,?,?,?,?,?)",
            (state["case_id"], "pending", json.dumps(state.get("recommendation")), json.dumps(g["reasons"]),
             json.dumps({"evidence": state.get("evidence"), "errors": state.get("evidence_errors")}, default=str), now_iso()),
        )
        self.conn.commit()
        log_event(self.conn, state["case_id"], "escalation", "human_review", {"reasons": g["reasons"]})
        return {"decision": "escalate", "final": "pending_review", "amount": None, "resolved_by": None}

    def deny(self, state):
        return {"outcome": {"decision": "auto_deny", "final": "denied", "amount": None, "resolved_by": "agent"}}

    def finalize(self, state):
        o, rec = state["outcome"], state.get("recommendation") or {}
        why = (rec.get("rationale") or "see policy").rstrip(".").lower()
        reply = CUSTOMER_REPLIES[o["decision"]].format(amount=o.get("amount") or 0, order_id=state["order_id"], why=why)
        summary = rec.get("rationale") or ""
        if o["decision"] == "escalate":
            summary = "; ".join(state["guardrail"]["reasons"])
        self.conn.execute(
            """UPDATE cases SET agent_decision=?, agent_amount=?, final_outcome=?, final_amount=?, resolved_by=?,
               category=?, summary=?, customer_reply=?, steps=?, tool_calls=?, tokens_in=?, tokens_out=?, updated_at=?
               WHERE case_id=?""",
            (o["decision"], (state["guardrail"] or {}).get("amount") if o["decision"] != "auto_deny" else None,
             o["final"], o.get("amount"), o.get("resolved_by"), state.get("category"), summary, reply,
             state.get("steps", 0), len(state.get("tool_calls", [])), state.get("tokens_in", 0),
             state.get("tokens_out", 0), now_iso(), state["case_id"]),
        )
        self.conn.commit()
        log_event(self.conn, state["case_id"], "outcome", o["decision"], {**o, "customer_reply": reply})
        return {"outcome": {**o, "customer_reply": reply}}

    # ----------------------------------------------------------------- routing
    @staticmethod
    def route_planner(state):
        if state.get("loop_flag"):
            return "guardrails"
        return "tools" if state.get("pending_tool") else "assess"

    @staticmethod
    def route_guardrails(state):
        return {"auto_approve": "execute", "auto_deny": "deny", "escalate": "escalate"}[state["guardrail"]["decision"]]

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("intake", self.intake)
        g.add_node("planner", self.planner)
        g.add_node("tools", self.tools)
        g.add_node("assess", self.assess)
        g.add_node("guardrails", self.check_guardrails)
        g.add_node("execute", self.execute)
        g.add_node("escalate", self.escalate)
        g.add_node("deny", self.deny)
        g.add_node("finalize", self.finalize)
        g.add_edge(START, "intake")
        g.add_edge("intake", "planner")
        g.add_conditional_edges("planner", self.route_planner,
                                {"tools": "tools", "assess": "assess", "guardrails": "guardrails"})
        g.add_edge("tools", "planner")
        g.add_edge("assess", "guardrails")
        g.add_conditional_edges("guardrails", self.route_guardrails,
                                {"execute": "execute", "deny": "deny", "escalate": "escalate"})
        for n in ("execute", "escalate", "deny"):
            g.add_edge(n, "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=MemorySaver())

    # --------------------------------------------------------------------- API
    def run(self, case_id):
        """Run the agent on an existing row in `cases`."""
        row = self.conn.execute("SELECT customer_id, order_id, message FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if not row:
            raise KeyError(case_id)
        init = {"case_id": case_id, "customer_id": row["customer_id"], "order_id": row["order_id"],
                "message": row["message"], "tool_calls": [], "steps": 0, "tokens_in": 0, "tokens_out": 0,
                "loop_flag": None, "recommendation": None, "guardrail": None, "pending_tool": None}
        log_event(self.conn, case_id, "case_start", self.llm.name, {"message": row["message"]})
        t0 = time.perf_counter()
        final = self.graph.invoke(init, config={"configurable": {"thread_id": case_id},
                                                "recursion_limit": config.RECURSION_LIMIT})
        latency = (time.perf_counter() - t0) * 1000
        self.conn.execute("UPDATE cases SET latency_ms=? WHERE case_id=?", (latency, case_id))
        self.conn.commit()
        final["latency_ms"] = latency
        return final

    def submit(self, customer_id, order_id, message, case_id=None):
        """Create a new (unlabeled) case from a ticket and run it."""
        if case_id is None:
            n = self.conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0] + 1
            case_id = f"LIVE-{n:04d}"
        self.conn.execute(
            "INSERT INTO cases(case_id, customer_id, order_id, message, scenario, created_at) VALUES (?,?,?,?,?,?)",
            (case_id, customer_id, order_id, message, "live", now_iso()),
        )
        self.conn.commit()
        return self.run(case_id)
