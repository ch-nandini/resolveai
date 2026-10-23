"""Run the agent on one case and print the step-by-step trace.

    python scripts/run_case.py CASE-0007          # re-run an existing eval case
    python scripts/run_case.py --customer C1000 --order O50000 --message "it arrived broken"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolveai import db  # noqa: E402
from resolveai.graph import ResolveAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", nargs="?")
    ap.add_argument("--customer"); ap.add_argument("--order"); ap.add_argument("--message")
    a = ap.parse_args()
    conn = db.get_conn()
    agent = ResolveAgent(conn)

    if a.case_id:
        # fresh run: clear previous outputs for this case
        conn.execute("DELETE FROM audit_log WHERE case_id=?", (a.case_id,))
        conn.execute("DELETE FROM approvals WHERE case_id=?", (a.case_id,))
        conn.execute("DELETE FROM refunds WHERE idempotency_key=?", (f"{a.case_id}:refund",))
        conn.commit()
        final = agent.run(a.case_id)
    else:
        final = agent.submit(a.customer, a.order, a.message)

    print(f"\n{final['case_id']}: \"{final['message']}\"")
    print(f"intent: {final['category']} ({final['intent_confidence']})")
    for i, c in enumerate(final["tool_calls"], 1):
        status = "ok" if c["ok"] else f"FAILED ({c['error']})"
        print(f"  step {i}: {c['name']}({json.dumps(c['args'])}) -> {status}, attempts={c['attempts']}, {c['latency_ms']}ms")
    print(f"recommendation: {final.get('recommendation')}")
    print(f"guardrails: {final['guardrail']}")
    print(f"outcome: {final['outcome']['decision']} -> {final['outcome']['final']}")
    print(f"reply to customer: {final['outcome']['customer_reply']}")
    print(f"tokens: {final['tokens_in']} in / {final['tokens_out']} out · {final['latency_ms']:.0f} ms")


if __name__ == "__main__":
    main()
