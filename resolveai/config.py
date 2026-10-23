"""Central configuration. Everything tunable lives here (or in env vars)."""
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Simulation clock -------------------------------------------------------
# All dates in the synthetic world are relative to this fixed "now" so evals are reproducible.
SIM_NOW = datetime(2026, 10, 1, 12, 0, 0)

# --- Storage ----------------------------------------------------------------
DB_PATH = os.getenv("RESOLVEAI_DB", os.path.join(ROOT, "data", "resolveai.db"))
REPORTS_DIR = os.path.join(ROOT, "reports")

# --- LLM --------------------------------------------------------------------
# mock      -> deterministic offline stand-in (no API key, reproducible)
# anthropic -> Claude via the Anthropic SDK (needs ANTHROPIC_API_KEY)
# openai    -> OpenAI chat completions (needs OPENAI_API_KEY and OPENAI_MODEL)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

# Placeholder prices (USD per million tokens). Set these to your model's current pricing.
PRICE_IN_PER_MTOK = float(os.getenv("PRICE_IN_PER_MTOK", "1.0"))
PRICE_OUT_PER_MTOK = float(os.getenv("PRICE_OUT_PER_MTOK", "5.0"))

# --- Risk guardrails (enforced in code; the LLM cannot override these) -------
AUTO_REFUND_LIMIT = 200.00     # refunds above this always go to a human
FRAUD_THRESHOLD = 0.70         # fraud score at/above this always escalates
MAX_REFUNDS_30D = 3            # customers with this many refunds in 30 days escalate
NEW_ACCOUNT_DAYS = 14          # accounts younger than this...
NEW_ACCOUNT_LIMIT = 50.00      # ...asking for more than this escalate
MIN_CONFIDENCE = 0.60          # below this intent/recommendation confidence -> escalate

# --- Agent loop limits ------------------------------------------------------
MAX_STEPS = 10                 # max planner iterations per case
MAX_REPEAT_CALLS = 2           # identical (tool, args) calls allowed before loop guard trips
MAX_TOOL_RETRIES = 2           # retries per tool call on transient errors (3 attempts total)
RECURSION_LIMIT = 50           # hard cap on graph supersteps (backstop for the backstop)

# --- Fault / latency simulation ---------------------------------------------
TOOL_FAILURE_RATE = float(os.getenv("TOOL_FAILURE_RATE", "0.08"))   # transient failures per attempt
OUTAGE_RATE = float(os.getenv("OUTAGE_RATE", "0.015"))              # per-case hard outage of one tool
LATENCY_SCALE = float(os.getenv("LATENCY_SCALE", "1.0"))            # 0 = no simulated latency
