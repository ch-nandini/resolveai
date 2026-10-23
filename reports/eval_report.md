# ResolveAI eval report

- LLM: `mock` · Orchestrator: langgraph · Cases: 200

## Headline

| Metric | Value |
|---|---|
| Autonomous resolution rate | 48.0% |
| Correct autonomous resolutions (of all cases) | 48.0% |
| Decision accuracy vs. labels | 83.5% |
| Escalation precision / recall | 68.3% / 100.0% |
| Unsafe auto-approvals | 0 |
| Wrong auto-denials | 0 |
| Over-escalations (safe misses) | 33 |
| Refund amount accuracy (correct approvals) | 100.0% |
| Tool calls: first-attempt / after-retry success | 92.2% / 99.5% |
| Retries performed | 70 |
| Loop-guard trips | 0 |
| Avg planner steps / tool calls per case | 5 / 4 |
| Latency p50 / p95 | 374 ms / 735 ms |
| Avg tokens per case | 4310 |
| Avg cost per case | $0.0048 |

## Confusion matrix (rows = expected, cols = agent)

| expected \ agent | approve | deny | escalate |
|---|---|---|---|
| approve | 63 | 0 | 27 |
| deny | 0 | 33 | 6 |
| escalate | 0 | 0 | 71 |

## By scenario

| Scenario | Expected | N | Accuracy | approve | deny | escalate |
|---|---|---|---|---|---|---|
| ambiguous_request | escalate | 10 | 100.0% | 0 | 0 | 10 |
| changed_mind | approve | 14 | 78.6% | 11 | 0 | 3 |
| changed_mind_final_sale | deny | 5 | 80.0% | 0 | 4 | 1 |
| changed_mind_outside_window | deny | 7 | 85.7% | 0 | 6 | 1 |
| damaged | approve | 20 | 70.0% | 14 | 0 | 6 |
| damaged_final_sale | approve | 6 | 83.3% | 5 | 0 | 1 |
| damaged_outside_window | deny | 5 | 100.0% | 0 | 5 | 0 |
| duplicate_request | deny | 6 | 100.0% | 0 | 6 | 0 |
| fraud_signals | escalate | 10 | 100.0% | 0 | 0 | 10 |
| high_value | escalate | 16 | 100.0% | 0 | 0 | 16 |
| high_value_denial | deny | 4 | 75.0% | 0 | 3 | 1 |
| late_claim_on_time | deny | 6 | 83.3% | 0 | 5 | 1 |
| late_delivery | approve | 12 | 75.0% | 9 | 0 | 3 |
| new_account | escalate | 10 | 100.0% | 0 | 0 | 10 |
| not_delivered_in_window | deny | 6 | 66.7% | 0 | 4 | 2 |
| not_delivered_lost | approve | 12 | 66.7% | 8 | 0 | 4 |
| not_delivered_no_signature | approve | 10 | 60.0% | 6 | 0 | 4 |
| not_delivered_signed | escalate | 8 | 100.0% | 0 | 0 | 8 |
| order_not_owned | escalate | 7 | 100.0% | 0 | 0 | 7 |
| serial_refunder | escalate | 10 | 100.0% | 0 | 0 | 10 |
| wrong_item | approve | 16 | 62.5% | 10 | 0 | 6 |

> Mock-LLM run: token counts are estimated from the real prompts (chars/4), latency is simulated tool latency only, and prices are placeholders from config.py. Re-run with `LLM_PROVIDER=anthropic` for real model numbers.
