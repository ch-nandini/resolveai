"""Deterministic auto-resolution vs. human-escalation rules.

The LLM *recommends*; this module *decides*. Any refund that trips a risk rule goes to a human,
no matter how confident the model is. Denials are low-risk and can be automated, but still need
the order evidence and a confident read of the request.
"""
from . import config

REQUIRED_FOR_REFUND = ["order", "history", "fraud", "policy"]
REQUIRED_FOR_DENY = ["order"]

EVIDENCE_LABEL = {
    "order": "order lookup",
    "history": "customer history",
    "fraud": "fraud check",
    "policy": "policy search",
}


def evaluate(state):
    """Return {"decision": auto_approve|auto_deny|escalate, "amount": float|None, "reasons": [..]}"""
    reasons = []
    ev = state.get("evidence") or {}
    rec = state.get("recommendation")

    if state.get("loop_flag"):
        reasons.append(f"Loop guard tripped ({state['loop_flag']})")
    if not rec:
        return {"decision": "escalate", "amount": None, "reasons": reasons or ["No recommendation produced"]}

    action = rec.get("action")
    amount = rec.get("amount")
    confidence = min(state.get("intent_confidence", 1.0), rec.get("confidence", 1.0))

    if action == "escalate":
        reasons.append(f"Agent requested review: {rec.get('rationale', 'no rationale')}")

    required = REQUIRED_FOR_REFUND if action == "refund" else REQUIRED_FOR_DENY
    missing = [EVIDENCE_LABEL[k] for k in required if k not in ev]
    if missing:
        reasons.append("Missing evidence: " + ", ".join(missing))

    if confidence < config.MIN_CONFIDENCE:
        reasons.append(f"Low confidence ({confidence:.2f} < {config.MIN_CONFIDENCE})")

    if action == "refund":
        order = ev.get("order") or {}
        hist = ev.get("history") or {}
        fraud = ev.get("fraud") or {}
        if amount is None or amount <= 0:
            reasons.append("Refund amount missing or invalid")
        else:
            if order.get("amount") is not None and amount > order["amount"] + 0.01:
                reasons.append(f"Recommended ${amount:.2f} exceeds order total ${order['amount']:.2f}")
            if amount > config.AUTO_REFUND_LIMIT:
                reasons.append(f"Amount ${amount:.2f} over auto-refund limit ${config.AUTO_REFUND_LIMIT:.0f}")
        if fraud.get("score", 0) >= config.FRAUD_THRESHOLD:
            reasons.append(f"Fraud score {fraud['score']:.2f} ≥ {config.FRAUD_THRESHOLD}")
        if hist.get("refunds_last_30d", 0) >= config.MAX_REFUNDS_30D:
            reasons.append(f"{hist['refunds_last_30d']} refunds in last 30 days")
        if (hist.get("account_age_days") is not None and hist["account_age_days"] < config.NEW_ACCOUNT_DAYS
                and (amount or 0) > config.NEW_ACCOUNT_LIMIT):
            reasons.append(f"New account ({hist['account_age_days']}d) requesting ${amount:.2f}")
        if order.get("already_refunded"):
            reasons.append("Order already refunded")
    elif action not in ("deny", "escalate"):
        reasons.append(f"Unknown action '{action}'")

    if reasons:
        return {"decision": "escalate", "amount": amount, "reasons": reasons}
    if action == "refund":
        return {"decision": "auto_approve", "amount": round(amount, 2), "reasons": []}
    return {"decision": "auto_deny", "amount": None, "reasons": []}
