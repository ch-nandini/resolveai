"""Run the agent over the labeled dataset and compute metrics."""
import csv
import json
import os
import statistics
from collections import Counter, defaultdict

from . import config, datagen, db
from .graph import USING_LANGGRAPH, ResolveAgent

LABEL_FOR = {"auto_approve": "approve", "auto_deny": "deny", "escalate": "escalate"}


def pct(a, b):
    return round(100.0 * a / b, 1) if b else 0.0


def run(seed=42, limit=None, provider=None, progress=True):
    conn = db.reset()
    n = datagen.generate(conn, seed=seed)
    from .llm import get_llm
    agent = ResolveAgent(conn, llm=get_llm(provider))
    ids = [r[0] for r in conn.execute("SELECT case_id FROM cases ORDER BY case_id")]
    if limit:
        ids = ids[:limit]
    for i, cid in enumerate(ids, 1):
        agent.run(cid)
        if progress and (i % 25 == 0 or i == len(ids)):
            print(f"  {i}/{len(ids)} cases")
    return compute(conn, ids, agent.llm.name)


def compute(conn, ids, llm_name="?"):
    q = ",".join("?" * len(ids))
    cases = [dict(r) for r in conn.execute(f"SELECT * FROM cases WHERE case_id IN ({q})", ids)]
    calls = [dict(r) for r in conn.execute(f"SELECT * FROM audit_log WHERE case_id IN ({q}) AND event IN ('tool_call','tool_attempt')", ids)]
    guards = conn.execute(f"SELECT COUNT(*) FROM audit_log WHERE case_id IN ({q}) AND event='loop_guard'", ids).fetchone()[0]

    N = len(cases)
    pred = [LABEL_FOR[c["agent_decision"]] for c in cases]
    gold = [c["expected_decision"] for c in cases]
    correct = sum(p == g for p, g in zip(pred, gold))

    auto = sum(p != "escalate" for p in pred)
    tp = sum(p == "escalate" and g == "escalate" for p, g in zip(pred, gold))
    pe = sum(p == "escalate" for p in pred)
    ge = sum(g == "escalate" for g in gold)

    unsafe_approvals = [c["case_id"] for c, p, g in zip(cases, pred, gold) if p == "approve" and g != "approve"]
    wrong_denials = [c["case_id"] for c, p, g in zip(cases, pred, gold) if p == "deny" and g != "deny"]
    over_escalations = [c["case_id"] for c, p, g in zip(cases, pred, gold) if p == "escalate" and g != "escalate"]

    approvals_ok = [c for c, p, g in zip(cases, pred, gold) if p == g == "approve"]
    amount_ok = sum(abs((c["agent_amount"] or 0) - (c["expected_amount"] or 0)) < 0.01 for c in approvals_ok)

    final_calls = [c for c in calls if c["event"] == "tool_call" and c["name"] != "create_refund"]
    attempts = [c for c in calls if c["event"] == "tool_attempt" and c["name"] != "create_refund"]
    first_try = [c for c in final_calls if c["ok"] and c["attempt"] == 1]

    lat = sorted(c["latency_ms"] for c in cases)
    tin = [c["tokens_in"] for c in cases]
    tout = [c["tokens_out"] for c in cases]
    cost = [(i * config.PRICE_IN_PER_MTOK + o * config.PRICE_OUT_PER_MTOK) / 1e6 for i, o in zip(tin, tout)]

    resolved_auto_correct = sum(p == g and p != "escalate" for p, g in zip(pred, gold))

    by_scn = defaultdict(lambda: Counter())
    for c, p, g in zip(cases, pred, gold):
        by_scn[c["scenario"]]["n"] += 1
        by_scn[c["scenario"]]["correct"] += p == g
        by_scn[c["scenario"]][p] += 1
    scenario_rows = [{"scenario": s, "expected": next(c["expected_decision"] for c in cases if c["scenario"] == s),
                      "n": v["n"], "accuracy_pct": pct(v["correct"], v["n"]),
                      "approve": v["approve"], "deny": v["deny"], "escalate": v["escalate"]}
                     for s, v in sorted(by_scn.items())]

    confusion = {g: {p: sum(1 for pp, gg in zip(pred, gold) if gg == g and pp == p) for p in ("approve", "deny", "escalate")}
                 for g in ("approve", "deny", "escalate")}

    metrics = {
        "llm": llm_name,
        "orchestrator": "langgraph" if USING_LANGGRAPH else "langgraph-compat (offline fallback)",
        "cases": N,
        "autonomous_resolution_rate_pct": pct(auto, N),
        "correct_autonomous_resolution_rate_pct": pct(resolved_auto_correct, N),
        "decision_accuracy_pct": pct(correct, N),
        "escalation_precision_pct": pct(tp, pe),
        "escalation_recall_pct": pct(tp, ge),
        "unsafe_auto_approvals": len(unsafe_approvals),
        "wrong_auto_denials": len(wrong_denials),
        "over_escalations": len(over_escalations),
        "refund_amount_accuracy_pct": pct(amount_ok, len(approvals_ok)),
        "tool_calls": len(final_calls),
        "tool_first_attempt_success_pct": pct(len(first_try), len(final_calls)),
        "tool_success_after_retries_pct": pct(sum(c["ok"] for c in final_calls), len(final_calls)),
        "tool_retries": len(attempts) - len(final_calls),
        "loop_guard_trips": guards,
        "avg_steps_per_case": round(statistics.mean(c["steps"] for c in cases), 2),
        "avg_tool_calls_per_case": round(statistics.mean(c["tool_calls"] for c in cases), 2),
        "latency_p50_ms": round(lat[len(lat) // 2], 1),
        "latency_p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1),
        "avg_tokens_per_case": round(statistics.mean(i + o for i, o in zip(tin, tout))),
        "avg_cost_per_case_usd": round(statistics.mean(cost), 5),
        "cost_per_autonomous_resolution_usd": round(sum(cost) / max(auto, 1), 5),
        "confusion_matrix": confusion,
        "examples": {"unsafe_auto_approvals": unsafe_approvals[:10], "wrong_auto_denials": wrong_denials[:10],
                     "over_escalations": over_escalations[:10]},
    }
    return metrics, scenario_rows, cases


def write_reports(metrics, scenario_rows, cases, out_dir=None):
    out_dir = out_dir or config.REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "by_scenario": scenario_rows}, f, indent=2)
    cols = ["case_id", "scenario", "message", "category", "expected_decision", "agent_decision", "expected_amount",
            "agent_amount", "steps", "tool_calls", "tokens_in", "tokens_out", "latency_ms", "summary"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in sorted(cases, key=lambda c: c["case_id"]):
            w.writerow(c)
    m = metrics
    cm = m["confusion_matrix"]
    lines = [
        "# ResolveAI eval report", "",
        f"- LLM: `{m['llm']}` · Orchestrator: {m['orchestrator']} · Cases: {m['cases']}", "",
        "## Headline", "",
        "| Metric | Value |", "|---|---|",
        f"| Autonomous resolution rate | {m['autonomous_resolution_rate_pct']}% |",
        f"| Correct autonomous resolutions (of all cases) | {m['correct_autonomous_resolution_rate_pct']}% |",
        f"| Decision accuracy vs. labels | {m['decision_accuracy_pct']}% |",
        f"| Escalation precision / recall | {m['escalation_precision_pct']}% / {m['escalation_recall_pct']}% |",
        f"| Unsafe auto-approvals | {m['unsafe_auto_approvals']} |",
        f"| Wrong auto-denials | {m['wrong_auto_denials']} |",
        f"| Over-escalations (safe misses) | {m['over_escalations']} |",
        f"| Refund amount accuracy (correct approvals) | {m['refund_amount_accuracy_pct']}% |",
        f"| Tool calls: first-attempt / after-retry success | {m['tool_first_attempt_success_pct']}% / {m['tool_success_after_retries_pct']}% |",
        f"| Retries performed | {m['tool_retries']} |",
        f"| Loop-guard trips | {m['loop_guard_trips']} |",
        f"| Avg planner steps / tool calls per case | {m['avg_steps_per_case']} / {m['avg_tool_calls_per_case']} |",
        f"| Latency p50 / p95 | {m['latency_p50_ms']:.0f} ms / {m['latency_p95_ms']:.0f} ms |",
        f"| Avg tokens per case | {m['avg_tokens_per_case']} |",
        f"| Avg cost per case | ${m['avg_cost_per_case_usd']:.4f} |",
        "",
        "## Confusion matrix (rows = expected, cols = agent)", "",
        "| expected \\ agent | approve | deny | escalate |", "|---|---|---|---|",
    ]
    for g in ("approve", "deny", "escalate"):
        lines.append(f"| {g} | {cm[g]['approve']} | {cm[g]['deny']} | {cm[g]['escalate']} |")
    lines += ["", "## By scenario", "", "| Scenario | Expected | N | Accuracy | approve | deny | escalate |",
              "|---|---|---|---|---|---|---|"]
    for r in scenario_rows:
        lines.append(f"| {r['scenario']} | {r['expected']} | {r['n']} | {r['accuracy_pct']}% | {r['approve']} | {r['deny']} | {r['escalate']} |")
    if m["llm"] == "mock":
        lines += ["", "> Mock-LLM run: token counts are estimated from the real prompts (chars/4), latency is simulated "
                      "tool latency only, and prices are placeholders from config.py. Re-run with "
                      "`LLM_PROVIDER=anthropic` for real model numbers."]
    with open(os.path.join(out_dir, "eval_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
