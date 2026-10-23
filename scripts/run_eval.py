"""Generate the synthetic dataset, run the agent on every case, write reports/.

    python scripts/run_eval.py                      # mock LLM, all ~200 cases
    LLM_PROVIDER=anthropic python scripts/run_eval.py --limit 50
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolveai import config  # noqa: E402
from resolveai.evaluate import run, write_reports  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    ap.add_argument("--provider", default=None, help="mock | anthropic | openai (default: LLM_PROVIDER env)")
    args = ap.parse_args()

    print(f"Running eval (provider={args.provider or config.LLM_PROVIDER}, seed={args.seed})")
    metrics, scenarios, cases = run(seed=args.seed, limit=args.limit, provider=args.provider)
    write_reports(metrics, scenarios, cases)
    m = metrics
    print(f"""
Autonomous resolution rate : {m['autonomous_resolution_rate_pct']}%
Decision accuracy          : {m['decision_accuracy_pct']}%
Escalation precision/recall: {m['escalation_precision_pct']}% / {m['escalation_recall_pct']}%
Unsafe auto-approvals      : {m['unsafe_auto_approvals']}
Tool success (1st / retry) : {m['tool_first_attempt_success_pct']}% / {m['tool_success_after_retries_pct']}%
Avg steps per case         : {m['avg_steps_per_case']}
Latency p50 / p95          : {m['latency_p50_ms']:.0f} / {m['latency_p95_ms']:.0f} ms
Avg cost per case          : ${m['avg_cost_per_case_usd']:.4f}

Reports written to {config.REPORTS_DIR}/ (eval_report.md, results.csv, metrics.json)
Pending escalations are now in the approval queue: streamlit run dashboard/app.py""")


if __name__ == "__main__":
    main()
