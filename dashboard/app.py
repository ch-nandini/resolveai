"""ResolveAI operator dashboard.

    streamlit run dashboard/app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolveai import approvals, config, db  # noqa: E402
from resolveai.graph import ResolveAgent  # noqa: E402

st.set_page_config(page_title="ResolveAI Ops", page_icon="🛟", layout="wide")
conn = db.get_conn()


@st.cache_resource
def get_agent():
    return ResolveAgent(conn)


def trace_df(case_id):
    rows = conn.execute(
        "SELECT ts, event, name, ok, attempt, ROUND(latency_ms,1) AS latency_ms, payload FROM audit_log WHERE case_id=? ORDER BY id",
        (case_id,),
    ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if not df.empty:
        df["payload"] = df["payload"].str.slice(0, 300)
    return df


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛟 ResolveAI")
    operator = st.text_input("Operator", value="ops-1")
    st.caption(f"LLM provider: `{config.LLM_PROVIDER}`")
    st.divider()
    st.subheader("Submit a live ticket")
    with st.form("ticket"):
        cust = st.text_input("Customer ID", value="C1000")
        order = st.text_input("Order ID", value="")
        message = st.text_area("Customer message", value="My order arrived broken.")
        if st.form_submit_button("Run agent") and order:
            final = get_agent().submit(cust, order, message)
            o = final["outcome"]
            st.success(f"{final['case_id']}: {o['decision']} → {o['final']}")
            st.caption(o["customer_reply"])

counts = dict(conn.execute("SELECT status, COUNT(*) FROM approvals GROUP BY status").fetchall())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cases", conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
c2.metric("Pending review", counts.get("pending", 0))
c3.metric("Approved by ops", counts.get("approved", 0))
c4.metric("Rejected by ops", counts.get("rejected", 0))

tab_q, tab_cases, tab_metrics = st.tabs(["Approval queue", "Case explorer", "Eval metrics"])

# ------------------------------------------------------------ approval queue
with tab_q:
    queue = approvals.pending(conn)
    if not queue:
        st.info("Queue is empty. Run `python scripts/run_eval.py` or submit a ticket to populate it.")
    for item in queue:
        rec = item["recommendation"] or {}
        ev = (item["evidence"] or {}).get("evidence") or {}
        errs = (item["evidence"] or {}).get("errors") or {}
        order = ev.get("order") or {}
        hist = ev.get("history") or {}
        fraud = ev.get("fraud") or {}
        title = f"{item['case_id']} · {item['category'] or '?'} · ${order.get('amount', 0) or 0:.2f} · agent says: {rec.get('action', 'n/a')}"
        with st.expander(title):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**Customer message**  \n> {item['message']}")
                st.markdown("**Why it was escalated**")
                for r in item["reasons"] or []:
                    st.warning(r)
                if rec:
                    st.markdown(f"**Agent recommendation:** `{rec.get('action')}`"
                                + (f" ${rec['amount']:.2f}" if rec.get("amount") else "")
                                + f" · confidence {rec.get('confidence', 0):.2f} · {rec.get('policy_section') or ''}")
                    st.caption(rec.get("rationale", ""))
            with right:
                st.markdown("**Evidence**")
                st.write({
                    "item": order.get("item"),
                    "tracking": order.get("tracking_status"),
                    "days since delivery": order.get("days_since_delivery"),
                    "final sale": order.get("final_sale"),
                    "account age (days)": hist.get("account_age_days"),
                    "refunds last 30d": hist.get("refunds_last_30d"),
                    "fraud score": fraud.get("score"),
                    "fraud signals": fraud.get("signals"),
                })
                if errs:
                    st.error(f"Tool failures: {errs}")
            default_amt = float(rec.get("amount") or order.get("amount") or 0.0)
            a1, a2, b1, b2 = st.columns([1, 2, 1, 1], vertical_alignment="bottom")
            amt = a1.number_input("Refund $", min_value=0.0, value=default_amt, step=1.0, key=f"amt-{item['case_id']}")
            note = a2.text_input("Note", key=f"note-{item['case_id']}")
            if b1.button("✅ Approve", key=f"ok-{item['case_id']}", disabled=amt <= 0):
                approvals.approve(conn, item["case_id"], operator, amt, note)
                st.rerun()
            if b2.button("❌ Reject", key=f"no-{item['case_id']}"):
                approvals.reject(conn, item["case_id"], operator, note)
                st.rerun()
            with st.popover("Full trace"):
                st.dataframe(trace_df(item["case_id"]), use_container_width=True, hide_index=True)

# -------------------------------------------------------------- case explorer
with tab_cases:
    df = pd.read_sql_query(
        """SELECT case_id, scenario, category, expected_decision, agent_decision, final_outcome, resolved_by,
                  agent_amount, steps, tool_calls, ROUND(latency_ms) AS latency_ms, message, summary
           FROM cases ORDER BY case_id""", conn)
    f1, f2, f3 = st.columns(3)
    dec = f1.multiselect("Agent decision", sorted(df["agent_decision"].dropna().unique()))
    scn = f2.multiselect("Scenario", sorted(df["scenario"].dropna().unique()))
    only_wrong = f3.checkbox("Only mismatches vs. label")
    view = df
    if dec:
        view = view[view["agent_decision"].isin(dec)]
    if scn:
        view = view[view["scenario"].isin(scn)]
    if only_wrong:
        m = {"auto_approve": "approve", "auto_deny": "deny", "escalate": "escalate"}
        view = view[view["expected_decision"].notna() & (view["agent_decision"].map(m) != view["expected_decision"])]
    st.dataframe(view, use_container_width=True, hide_index=True)
    pick = st.selectbox("Inspect case", [""] + view["case_id"].tolist())
    if pick:
        row = df[df["case_id"] == pick].iloc[0]
        st.markdown(f"**{pick}** · {row['message']}")
        st.markdown(f"Decision: `{row['agent_decision']}` → `{row['final_outcome']}` · {row['summary']}")
        st.dataframe(trace_df(pick), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- metrics
with tab_metrics:
    path = os.path.join(config.REPORTS_DIR, "metrics.json")
    if not os.path.exists(path):
        st.info("No eval yet. Run `python scripts/run_eval.py`.")
    else:
        data = json.load(open(path))
        m = data["metrics"]
        st.caption(f"LLM: `{m['llm']}` · orchestrator: {m['orchestrator']} · {m['cases']} labeled cases")
        r1 = st.columns(4)
        r1[0].metric("Autonomous resolution", f"{m['autonomous_resolution_rate_pct']}%")
        r1[1].metric("Decision accuracy", f"{m['decision_accuracy_pct']}%")
        r1[2].metric("Escalation recall", f"{m['escalation_recall_pct']}%")
        r1[3].metric("Unsafe auto-approvals", m["unsafe_auto_approvals"])
        r2 = st.columns(4)
        r2[0].metric("Tool success after retries", f"{m['tool_success_after_retries_pct']}%")
        r2[1].metric("Avg steps / case", m["avg_steps_per_case"])
        r2[2].metric("p95 latency", f"{m['latency_p95_ms']:.0f} ms")
        r2[3].metric("Cost / case", f"${m['avg_cost_per_case_usd']:.4f}")
        st.subheader("Confusion matrix (rows = expected)")
        st.dataframe(pd.DataFrame(m["confusion_matrix"]).T, use_container_width=True)
        st.subheader("By scenario")
        sdf = pd.DataFrame(data["by_scenario"])
        st.bar_chart(sdf.set_index("scenario")[["approve", "deny", "escalate"]])
        st.dataframe(sdf, use_container_width=True, hide_index=True)
