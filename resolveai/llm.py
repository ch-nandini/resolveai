"""LLM adapters. Three jobs, same interface for every provider:

  classify(message)    -> {"category", "confidence"}
  next_tool(state)     -> {"name", "args"} or None (done gathering evidence)
  recommend(state)     -> {"action": refund|deny|escalate, "amount", "confidence", "rationale", "policy_section"}

Each returns (result, tokens_in, tokens_out).

MockLLM is a deterministic stand-in so the whole pipeline (and the eval) runs offline and
reproducibly. Swap in a real model with LLM_PROVIDER=anthropic or openai.
"""
import json
import re
from datetime import datetime

from . import config
from .tools import PLANNER_TOOLS, EVIDENCE_KEY

CATEGORIES = ["not_delivered", "damaged", "wrong_item", "late_delivery", "changed_mind", "unknown"]

# ---------------------------------------------------------------------------
# Prompts (shared by real providers; MockLLM uses them only to estimate tokens)
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are ResolveAI, an agent that investigates customer refund requests for an online store. "
    "Be precise, follow the refund policy exactly, and never invent facts that are not in the evidence."
)


def classify_prompt(message):
    return (
        "Classify the customer's refund request into exactly one category: "
        + ", ".join(CATEGORIES)
        + ". Use 'unknown' if the problem is unclear.\n\n"
        f"Customer message:\n\"\"\"{message}\"\"\"\n\n"
        'Reply with JSON only: {"category": "...", "confidence": 0.0-1.0}'
    )


def _evidence_summary(state):
    ev = state.get("evidence") or {}
    errs = state.get("evidence_errors") or {}
    lines = []
    for k, v in ev.items():
        lines.append(f"- {k}: {json.dumps(v, default=str)}")
    for k, v in errs.items():
        lines.append(f"- {k}: FAILED after retries ({v})")
    return "\n".join(lines) or "- (none yet)"


def planner_prompt(state):
    calls = [f"{c['name']}({json.dumps(c['args'])}) -> {'ok' if c['ok'] else 'failed'}" for c in state.get("tool_calls", [])]
    return (
        f"Case {state['case_id']}. Customer {state['customer_id']} says about order {state['order_id']}:\n"
        f"\"\"\"{state['message']}\"\"\"\n"
        f"Classified as: {state.get('category')} (confidence {state.get('intent_confidence')}).\n\n"
        f"Evidence so far:\n{_evidence_summary(state)}\n\n"
        f"Tool calls so far:\n" + ("\n".join(calls) or "(none)") + "\n\n"
        "You must verify the order, the customer's history, the fraud risk and the relevant policy before "
        "a decision can be made. Call the next tool you need. Do not repeat a call that already succeeded. "
        "If a tool failed after retries, move on. When you have enough evidence, reply with the single word DONE."
    )


def recommend_prompt(state):
    policy = (state.get("evidence") or {}).get("policy", {}).get("sections", [])
    policy_text = "\n\n".join(f"{s['section']} {s['title']}\n{s['text']}" for s in policy) or "(policy unavailable)"
    return (
        f"Today is {config.SIM_NOW.date()}.\n"
        f"Customer message: \"\"\"{state['message']}\"\"\"\n"
        f"Request category: {state.get('category')}\n\n"
        f"Evidence:\n{_evidence_summary(state)}\n\n"
        f"Relevant policy:\n{policy_text}\n\n"
        "Apply the policy to the evidence and recommend ONE action:\n"
        "- refund: give the exact amount (full order total, or the partial amount the policy specifies)\n"
        "- deny: the policy says the customer is not eligible\n"
        "- escalate: the policy requires a human, identity can't be verified, or the facts conflict\n"
        "Do not apply the §8 automation limits yourself; a separate risk system enforces them.\n\n"
        'Reply with JSON only: {"action": "refund|deny|escalate", "amount": number|null, '
        '"confidence": 0.0-1.0, "rationale": "one sentence", "policy_section": "§N"}'
    )


def _est_tokens(text):
    return max(1, len(text) // 4)


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:200]}")
    return json.loads(m.group(0))


def _normalize_rec(rec):
    action = str(rec.get("action", "escalate")).lower().strip()
    if action not in ("refund", "deny", "escalate"):
        action = "escalate"
    amount = rec.get("amount")
    try:
        amount = round(float(amount), 2) if amount is not None else None
    except (TypeError, ValueError):
        amount = None
    try:
        conf = float(rec.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "action": action,
        "amount": amount if action == "refund" else None,
        "confidence": max(0.0, min(conf, 1.0)),
        "rationale": str(rec.get("rationale", ""))[:500],
        "policy_section": rec.get("policy_section"),
    }


class BaseLLM:
    name = "base"

    def classify(self, message):
        raise NotImplementedError

    def next_tool(self, state):
        raise NotImplementedError

    def recommend(self, state):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------
KEYWORDS = {
    "not_delivered": ["never arrived", "never showed", "never received", "hasn't arrived", "has not arrived",
                      "didn't arrive", "not delivered", "never got", "lost", "where is my", "missing package",
                      "no package", "nothing at my door", "still waiting", "never came", "not here yet"],
    "damaged": ["broken", "cracked", "damaged", "shattered", "defective", "stopped working", "doesn't work",
                "does not work", "dead on arrival", "torn", "dented", "faulty", "leaking", "ripped"],
    "wrong_item": ["wrong", "instead of", "not what i ordered", "different item", "isn't what i ordered",
                   "incorrect item", "mixed up", "sent me a"],
    "late_delivery": ["late", "delayed", "took forever", "after the promised", "past the delivery date",
                      "way past", "days after", "missed the delivery date"],
    "changed_mind": ["changed my mind", "don't need", "no longer need", "don't want", "ordered by mistake",
                     "by accident", "cheaper", "don't like", "not needed", "return it", "doesn't suit"],
}


class MockLLM(BaseLLM):
    """Keyword classifier + fixed evidence plan + policy rules. Deterministic, offline."""
    name = "mock"

    def classify(self, message):
        text = message.lower()
        scores = {c: sum(1 for kw in kws if kw in text) for c, kws in KEYWORDS.items()}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        (top, s1), (_, s2) = ranked[0], ranked[1]
        if s1 == 0:
            result = {"category": "unknown", "confidence": 0.3}
        else:
            result = {"category": top, "confidence": round(0.55 + 0.4 * (s1 - s2) / s1, 2)}
        p = classify_prompt(message)
        return result, _est_tokens(SYSTEM + p), _est_tokens(json.dumps(result))

    def next_tool(self, state):
        ev = state.get("evidence") or {}
        errs = state.get("evidence_errors") or {}
        plan = [
            ("lookup_order", {"order_id": state["order_id"], "customer_id": state["customer_id"]}),
            ("get_customer_history", {"customer_id": state["customer_id"]}),
            ("run_fraud_check", {"customer_id": state["customer_id"], "order_id": state["order_id"]}),
            ("search_refund_policy", {"query": state.get("category") or "refund"}),
        ]
        choice = None
        for name, args in plan:
            key = EVIDENCE_KEY[name]
            if key not in ev and key not in errs:
                choice = {"name": name, "args": args}
                break
        p = planner_prompt(state)
        out = json.dumps(choice) if choice else "DONE"
        return choice, _est_tokens(SYSTEM + p + json.dumps(PLANNER_TOOLS)), _est_tokens(out)

    def recommend(self, state):
        rec = self._rules(state)
        p = recommend_prompt(state)
        return rec, _est_tokens(SYSTEM + p), _est_tokens(json.dumps(rec))

    @staticmethod
    def _rules(state):
        cat = state.get("category")
        order = (state.get("evidence") or {}).get("order")
        conf = 0.9

        def R(action, amount=None, why="", sec=None):
            return {"action": action, "amount": amount, "confidence": conf, "rationale": why, "policy_section": sec}

        if order is None:
            return R("escalate", why="Order details unavailable", sec="§1")
        if not order.get("found"):
            return R("escalate", why="Order not found", sec="§1")
        if not order.get("owner_match"):
            return R("escalate", why="Order does not belong to this customer", sec="§1")
        if order.get("already_refunded"):
            return R("deny", why="Order was already refunded", sec="§7")
        if cat == "unknown":
            return R("escalate", why="Customer's problem is unclear", sec=None)

        amt = order["amount"]
        dsd = order.get("days_since_delivery")
        status = order.get("tracking_status")

        if cat == "not_delivered":
            if status in ("lost", "returned_to_sender"):
                return R("refund", amt, f"Tracking shows {status}", "§2")
            if status == "in_transit":
                if (order.get("days_past_promised") or 0) > 5:
                    return R("refund", amt, "In transit more than 5 days past promised date", "§2")
                return R("deny", why="Still within the delivery window", sec="§2")
            if status == "delivered":
                if order.get("signature_confirmed"):
                    return R("escalate", why="Carrier shows signed delivery; needs investigation", sec="§2")
                return R("refund", amt, "Delivered without signature and customer reports non-receipt", "§2")
        if cat in ("damaged", "wrong_item"):
            sec = "§3" if cat == "damaged" else "§4"
            if dsd is not None and dsd <= 60:
                return R("refund", amt, f"{cat.replace('_', ' ')} within 60 days", sec)
            return R("deny", why="Request is more than 60 days after delivery", sec=sec)
        if cat == "late_delivery":
            if (order.get("days_late") or 0) > 2:
                return R("refund", round(min(0.15 * amt, 25.0), 2), "Late delivery goodwill credit", "§5")
            return R("deny", why="Order arrived on time", sec="§5")
        if cat == "changed_mind":
            if order.get("final_sale"):
                return R("deny", why="Final-sale items aren't eligible for change-of-mind refunds", sec="§6")
            if dsd is not None and dsd <= 30:
                return R("refund", amt, "Change of mind within 30 days", "§6")
            return R("deny", why="More than 30 days since delivery", sec="§6")
        return R("escalate", why="No matching policy rule")


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------
class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, model=None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or config.ANTHROPIC_MODEL
        self.name = f"anthropic:{self.model}"

    def _call(self, user, tools=None, max_tokens=500):
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=SYSTEM,
                      messages=[{"role": "user", "content": user}])
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                               for t in tools]
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"name": b.name, "args": dict(b.input)} for b in resp.content if b.type == "tool_use"]
        return text, calls, resp.usage.input_tokens, resp.usage.output_tokens

    def classify(self, message):
        text, _, tin, tout = self._call(classify_prompt(message), max_tokens=100)
        try:
            r = _extract_json(text)
            cat = r.get("category") if r.get("category") in CATEGORIES else "unknown"
            return {"category": cat, "confidence": float(r.get("confidence", 0.5))}, tin, tout
        except Exception:
            return {"category": "unknown", "confidence": 0.0}, tin, tout

    def next_tool(self, state):
        _, calls, tin, tout = self._call(planner_prompt(state), tools=PLANNER_TOOLS, max_tokens=300)
        return (calls[0] if calls else None), tin, tout

    def recommend(self, state):
        text, _, tin, tout = self._call(recommend_prompt(state), max_tokens=400)
        try:
            return _normalize_rec(_extract_json(text)), tin, tout
        except Exception as e:
            return _normalize_rec({"action": "escalate", "confidence": 0, "rationale": f"unparseable output: {e}"}), tin, tout


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self, model=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or config.OPENAI_MODEL
        if not self.model:
            raise ValueError("Set OPENAI_MODEL to the model you want to use")
        self.name = f"openai:{self.model}"

    def _call(self, user, tools=None):
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [{"name": tc.function.name, "args": json.loads(tc.function.arguments or "{}")}
                 for tc in (msg.tool_calls or [])]
        return msg.content or "", calls, resp.usage.prompt_tokens, resp.usage.completion_tokens

    def classify(self, message):
        text, _, tin, tout = self._call(classify_prompt(message))
        try:
            r = _extract_json(text)
            cat = r.get("category") if r.get("category") in CATEGORIES else "unknown"
            return {"category": cat, "confidence": float(r.get("confidence", 0.5))}, tin, tout
        except Exception:
            return {"category": "unknown", "confidence": 0.0}, tin, tout

    def next_tool(self, state):
        _, calls, tin, tout = self._call(planner_prompt(state), tools=PLANNER_TOOLS)
        return (calls[0] if calls else None), tin, tout

    def recommend(self, state):
        text, _, tin, tout = self._call(recommend_prompt(state))
        try:
            return _normalize_rec(_extract_json(text)), tin, tout
        except Exception as e:
            return _normalize_rec({"action": "escalate", "confidence": 0, "rationale": f"unparseable output: {e}"}), tin, tout


def get_llm(provider=None):
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "mock":
        return MockLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAILLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")
