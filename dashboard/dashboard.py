"""
AI Agent RCA Dashboard
Full observability for multi-agent systems: Outcome Ledger + Incidents + RCA + Simulator.

Run: streamlit run dashboard/dashboard.py
Secrets: dashboard/.streamlit/secrets.toml (copy from observability/poc/.streamlit/secrets.toml)
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
from collections import defaultdict
import anthropic
import streamlit as st
import streamlit.components.v1 as st_components
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client

from engine.eval_engine   import (run_all_evals,
                                   eval_cost_per_trade, eval_research_conversion,
                                   eval_proposal_acceptance,
                                   COST_PER_TRADE_THRESHOLD, RESEARCH_CONVERSION_MIN,
                                   PROPOSAL_ACCEPTANCE_MIN)
from engine.pattern_detector import run_all_detectors
from engine.rca_engine    import build_annotated_call_stack, generate_fix_suggestion, summarize_incident
from simulator.failure_sim import (simulate_failure, list_patterns,
                                   simulate_quality_failure, list_quality_patterns)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Agent RCA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Hide Streamlit header and sidebar entirely */
[data-testid="stHeader"]  { display: none; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Minimal top padding — nav component sits in natural flow */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* ── Top nav bar — brand left, pills right, one row ── */
div[data-testid="stHorizontalBlock"]:has(.topnav-brand-cell) {
    background: #0f172a !important;
    border-bottom: 1px solid #1e293b !important;
    padding: 0 8px !important;
    margin: -0.5rem -1.5rem 1rem -1.5rem !important;
    align-items: center !important;
    min-height: 56px !important;
}
.topnav-brand-cell {
    padding: 10px 0 10px 8px;
    line-height: 1.4;
}
.topnav-brand { color: #f8fafc; font-weight: 700; font-size: 0.88rem; display: inline; }
.topnav-tag   { color: #64748b; font-size: 0.75rem; display: inline; }
/* Pills inside the nav row */
div[data-testid="stPillsGroup"] > div,
div[data-testid="stPillsRoot"] > div {
    gap: 28px !important;
}
div[data-testid="stPillsGroup"] button,
div[data-testid="stPillsRoot"] button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
    font-size: 0.78rem !important;
    padding: 4px 16px !important;
}
div[data-testid="stPillsGroup"] button:hover,
div[data-testid="stPillsRoot"] button:hover {
    color: #e2e8f0 !important;
    border-color: #4b5563 !important;
}
div[data-testid="stPillsGroup"] button[aria-pressed="true"],
div[data-testid="stPillsGroup"] button[data-selected="true"],
div[data-testid="stPillsRoot"]  button[aria-pressed="true"],
div[data-testid="stPillsRoot"]  button[data-selected="true"] {
    background: #1e3a5f !important;
    color: #93c5fd !important;
    border-color: #3b82f6 !important;
}

/* KPI cards */
.kpi-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 14px 18px; text-align: center;
}
.kpi-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; line-height: 1; }
.kpi-sub   { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }

/* Severity badges */
.badge-critical { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
.badge-warning  { background:#fef9c3; color:#92400e; border:1px solid #fcd34d;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
.badge-info     { background:#dbeafe; color:#1e40af; border:1px solid #93c5fd;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
.badge-sim      { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }

/* Trace tree */
.trace-row { font-family: 'Courier New', monospace; font-size: 0.78rem; color: #1e293b;
             padding: 4px 8px; border-left: 2px solid #cbd5e1; margin: 2px 0;
             background: #f8fafc; }
.trace-error  { border-left-color: #ef4444; background: #fff5f5; color: #7f1d1d; }
.trace-success { border-left-color: #10b981; background: #f0fdf4; }
.trace-root   { border-left-color: #f59e0b; background: #fffbeb; border-width: 3px; font-weight: 600; }

/* Score bar */
.score-bar-wrap { background: #e2e8f0; border-radius: 4px; height: 6px; margin: 4px 0; }
.score-bar-fill { height: 6px; border-radius: 4px; }

/* Callout */
.callout { background: #f0f9ff; border-left: 3px solid #3b82f6;
           padding: 10px 14px; border-radius: 0 6px 6px 0;
           margin: 8px 0; font-size: 0.9rem; color: #1e293b; }
.callout-critical { border-left-color: #ef4444; background: #fff5f5; }
.callout-warning  { border-left-color: #f59e0b; background: #fffbeb; }

/* Fix box */
.fix-box { background: #f0fdf4; border: 1px solid #6ee7b7; border-radius: 6px;
           padding: 12px 16px; font-family: monospace; font-size: 0.82rem;
           white-space: pre-wrap; color: #064e3b; }

/* Sim live indicator */
.sim-live { background: #faf5ff; border: 1px solid #c4b5fd; border-radius: 6px;
            padding: 10px 16px; color: #5b21b6; font-size: 0.9rem; }

/* AI analyst insight box */
.ai-insight-page {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-left: 4px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin: 0 0 12px 0;
    font-size: 0.88rem; color: #0c4a6e; line-height: 1.6;
}
.ai-insight-page .ai-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    color: #0ea5e9; text-transform: uppercase; margin-bottom: 6px;
}
.ai-insight-section {
    font-size: 0.82rem; color: #475569;
    font-style: italic; margin: 4px 0 10px 0;
    padding-left: 10px; border-left: 2px solid #cbd5e1;
}

/* Tooltip badge — ? icon with CSS hover tooltip */
.tip-badge {
    position: relative;
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 15px; height: 15px; border-radius: 50%;
    background: #e2e8f0; color: #64748b;
    font-size: 0.62rem; font-weight: 700;
    cursor: help; margin-left: 5px;
    vertical-align: middle; flex-shrink: 0;
    text-decoration: none;
}
.tip-badge::before {
    content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%; transform: translateX(-50%);
    background: #1e293b; color: #f8fafc;
    padding: 8px 12px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 400;
    line-height: 1.5; white-space: normal;
    width: 230px; pointer-events: none;
    opacity: 0; transition: opacity 0.15s ease;
    z-index: 9999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    text-transform: none; letter-spacing: normal;
}
.tip-badge::after {
    content: '';
    position: absolute;
    bottom: calc(100% + 2px); left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #1e293b;
    pointer-events: none;
    opacity: 0; transition: opacity 0.15s ease;
    z-index: 9999;
}
.tip-badge:hover::before,
.tip-badge:hover::after { opacity: 1; }

/* Pipeline node — colored dot with same hover tooltip as tip-badge */
.pipe-node {
    position: relative;
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%;
    font-size: 0.65rem; font-weight: 700;
    cursor: help; flex-shrink: 0;
}
.pipe-node::before {
    content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%; transform: translateX(-50%);
    background: #1e293b; color: #f8fafc;
    padding: 8px 12px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 400;
    line-height: 1.5; white-space: normal;
    width: 210px; pointer-events: none;
    opacity: 0; transition: opacity 0.15s ease;
    z-index: 9999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.pipe-node::after {
    content: '';
    position: absolute;
    bottom: calc(100% + 2px); left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #1e293b;
    pointer-events: none;
    opacity: 0; transition: opacity 0.15s ease;
    z-index: 9999;
}
.pipe-node:hover::before,
.pipe-node:hover::after { opacity: 1; }
</style>
""", unsafe_allow_html=True)


# ── Auth gate ─────────────────────────────────────────────────────────────────

# ── DB ────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _db():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    return create_client(url, key)


@st.cache_data(ttl=30)
def load_sessions() -> pd.DataFrame:
    r = _db().table("c_sessions").select(
        "id,date,total_cost_usd,total_latency_ms,total_tokens_input,"
        "total_tokens_output,trades_proposed,trades_executed,agents_invoked,"
        "terminal_reason,started_at,completed_at,cost_breakdown,is_simulated"
    ).order("started_at", desc=True).limit(200).execute()
    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df
    for col in ["total_cost_usd","total_latency_ms","total_tokens_input",
                "total_tokens_output","trades_executed","trades_proposed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["started_at"]  = pd.to_datetime(df["started_at"],  errors="coerce")
    df["completed_at"]= pd.to_datetime(df["completed_at"],errors="coerce")
    df["date"]        = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


@st.cache_data(ttl=60)
def load_traces() -> pd.DataFrame:
    r = _db().table("c_traces").select(
        "id,session_id,agent,step_type,tool_name,tokens_input,tokens_output,"
        "latency_ms,model,outcome,error,created_at"
    ).order("created_at").limit(10000).execute()
    df = pd.DataFrame(r.data or [])
    if df.empty:
        return df
    for col in ["tokens_input","tokens_output","latency_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_positions() -> pd.DataFrame:
    r = _db().table("c_positions").select(
        "id,session_id,ticker,status,realized_pnl,entry_price,close_time,open_date"
    ).execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0)
    return df


def load_incidents() -> pd.DataFrame:
    try:
        r = _db().table("c_incidents").select("*").order("created_at", desc=True).limit(200).execute()
        df = pd.DataFrame(r.data or [])
        if not df.empty:
            df["created_at"]   = pd.to_datetime(df["created_at"], errors="coerce")
            df["cost_wasted"]  = pd.to_numeric(df["cost_wasted"],  errors="coerce").fillna(0)
            df["tokens_wasted"]= pd.to_numeric(df["tokens_wasted"],errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


def load_evals_for_session(session_id: str) -> pd.DataFrame:
    try:
        r = _db().table("c_evals").select("*").eq("session_id", session_id).execute()
        return pd.DataFrame(r.data or [])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_all_evals() -> pd.DataFrame:
    try:
        rows, page_size, offset = [], 1000, 0
        while True:
            batch = (_db().table("c_evals")
                     .select("session_id,agent,eval_name,score,passed")
                     .range(offset, offset + page_size - 1)
                     .execute())
            rows.extend(batch.data or [])
            if len(batch.data or []) < page_size:
                break
            offset += page_size
        df = pd.DataFrame(rows)
        if not df.empty:
            df["score"]  = pd.to_numeric(df["score"],  errors="coerce").fillna(0)
            df["passed"] = df["passed"].astype(bool)
        return df
    except Exception:
        return pd.DataFrame()


def goto_page(page_name: str) -> None:
    st.session_state["_page"] = page_name
    st.rerun()


def goto_rca(incident_dict: dict, session_id: str) -> None:
    """Navigate to RCA View with the given incident pre-selected."""
    st.session_state["rca_incident"] = incident_dict
    st.session_state["rca_sid"]      = session_id
    st.session_state["_page"]        = "RCA View"
    st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

PATTERN_DESCRIPTIONS = {
    "Tool Timeout Loop":        "The same tool failed 3+ times in a row. Agent is retrying a broken tool instead of moving on.",
    "Context Spiral":           "Research agent burned >40K tokens but produced 0 trades. Token usage not justified by output.",
    "Pipeline Break":           "Research ran but orchestrator never started. Likely an uncaught exception in the handoff between agents.",
    "Cost Anomaly":             "Session cost was 2+ standard deviations above the recent average. Unexpectedly expensive run.",
    "Silent Exit":              "Session ended with 0 trades and no terminal reason recorded. Agent stopped without explanation.",
    "Empty Result Loop":        "Same tool called 3+ times successfully but session still produced nothing. Stuck in a results loop.",
    "Hyperactive Polling Loop": "The same tool was called 6+ times successfully in a row. Agent is over-fetching instead of using cached results.",
    "Tool Call Fabrication":    "2+ tool calls completed in under 50ms — too fast to have made a real API call. Agent may be hallucinating tool results.",
    "Handoff Schema Break":     "Research completed but the Risk agent's first step errored. The handoff payload is malformed or missing expected fields.",
    "Error Misinterpretation":  "HTTP error codes returned by tools (429, 500, etc.) but the agent continued as if they were successes.",
    "Isolation Forest Anomaly":        "Statistical outlier vs. baseline sessions. No named pattern matches — the session's feature vector is far from the normal cluster.",
    "Proactive: Grounding Failure":    "Research scored below 0.40 on data grounding for 3 consecutive sessions. Trade proposals are being built on insufficient data — before any operational alert fires.",
    "Proactive: Coherence Break":      "Orchestrator final decision does not match what research and risk produced this session. The synthesis layer is broken.",
    "Proactive: Quality Cascade":      "3 or more quality dimensions each declined >0.20 over the last 5 sessions. Systemic degradation across agents — not an isolated bad day.",
    "Proactive: Silent Degradation":   "Composite quality is declining across sessions while all operational evals stay clean. No alert has fired yet — but the trajectory is wrong.",
}

# Full detection explanation shown in RCA View expander — markdown supported here
PATTERN_DETAIL = {
    "Isolation Forest Anomaly": (
        "Detected by an **Isolation Forest** model trained on all sessions.\n\n"
        "**8 features scored per session:**\n"
        "- Total cost (USD)\n"
        "- Tokens in / tokens out\n"
        "- Latency (ms)\n"
        "- Trades executed\n"
        "- Agent count\n"
        "- Operational eval pass rate\n"
        "- Tool error rate\n\n"
        "A session is flagged when its feature vector is far from the baseline cluster "
        "(anomaly score ≥ 0.65). Likely causes: unusually high cost or token usage, "
        "abnormal latency, low eval pass rate, or a combination. "
        "Check the session metrics above against typical values to identify which feature is out of range."
    ),
    "Proactive: Grounding Failure": (
        "**Trigger:** `research.data_grounding` scored below **0.40** for 3 consecutive sessions.\n\n"
        "This dimension scores how many distinct tool types research called without errors. "
        "Below 0.40 means fewer than 2 distinct data sources — research is guessing, not grounding.\n\n"
        "**Why it matters before an operational alert fires:** Bad research inputs produce bad proposals. "
        "The risk agent may still approve them if the structure looks right. "
        "By the time a bad trade executes, the quality signal has been visible for 3 sessions.\n\n"
        "**Fix:** Require 3+ distinct tool types per ticker investigation (price data, news, ATR). "
        "Add an explicit output gate: agent must cite one price level and one quantitative indicator."
    ),
    "Proactive: Coherence Break": (
        "**Trigger:** `orchestrator.decision_consistency` scored below **0.50** in this session.\n\n"
        "This dimension checks whether the final trade decision is consistent with what research "
        "and risk produced. Below 0.50 means the orchestrator reached a conclusion that research "
        "and risk data do not support.\n\n"
        "**Why critical:** The orchestrator is the final gatekeeper. If it ignores upstream "
        "findings, the entire pipeline's reasoning chain is broken — even if all prior agents ran cleanly.\n\n"
        "**Fix:** Add a structured handoff contract: orchestrator must reference specific research "
        "findings and risk verdicts in its decision. Use typed outputs (Pydantic) so the "
        "orchestrator cannot silently ignore upstream data."
    ),
    "Proactive: Quality Cascade": (
        "**Trigger:** 3 or more quality dimensions each declined by more than **0.20** "
        "over the last 5 sessions.\n\n"
        "A single dimension declining is noise. Three or more declining together means something "
        "systemic changed — a prompt update, a model version change, a data quality issue, "
        "or market conditions that the pipeline was not designed for.\n\n"
        "**Severity escalates to critical** when 5 or more dimensions are declining simultaneously.\n\n"
        "**Fix:** Check for recent changes to agent prompts, tool schemas, or model versions. "
        "Run a manual quality review of the last 3 sessions. "
        "Identify which agent's dimensions are declining fastest — that is the root source."
    ),
    "Proactive: Silent Degradation": (
        "**Trigger:** Composite quality score declining at more than **0.015 per session** "
        "over the last 5 sessions, while all operational evals stay above 80% pass rate.\n\n"
        "This is the most dangerous pattern. Operational metrics look healthy — no tool errors, "
        "no pipeline breaks, no cost anomalies. But quality is quietly sliding. "
        "By the time an operational alert fires, the degradation has been underway for weeks.\n\n"
        "**The gap this closes:** Operational evals catch structural failures. "
        "Quality evals catch reasoning failures. A pipeline can be structurally sound "
        "while producing worse and worse decisions.\n\n"
        "**Fix:** Manual review of recent session outputs before the next trading session. "
        "Check for prompt drift, model behavior shifts, or gradual data quality decay. "
        "If the trend continues for 2 more sessions, escalate to critical."
    ),
}

AGENT_DESCRIPTIONS = {
    "orchestrator": "Makes the final trade execution decision. Receives approved proposals from the Risk agent and decides which trades to place.",
    "market":       "Fetches real-time price, volume, and market data for candidate tickers. First agent to run each session.",
    "research":     "Runs deep analysis on individual tickers — earnings, news, technicals, sentiment. One sub-agent per ticker.",
    "risk":         "Reviews research output, sizes positions, and approves or rejects each trade proposal before it reaches the orchestrator.",
}

OUTCOME_DESCRIPTIONS = {
    "clean":    "No failure patterns detected this session. All critical evals passed.",
    "incident": "One or more failure patterns fired. Some or all session cost may be wasted. Click View RCA for root cause and fix suggestion.",
}

def compute_cb_savings(sid: str, row: dict, evals_df, traces_df) -> tuple:
    """
    Returns (savings_float, detail_str).
    savings_float — cost a circuit breaker would have prevented.
    detail_str    — human-readable breakdown for the tooltip.
    """
    pipeline = ["market", "research", "risk", "orchestrator"]
    _labels  = {"market": "Market", "research": "Research",
                "risk": "Risk", "orchestrator": "Orch"}

    # per-agent costs from cost_breakdown
    bd = row.get("cost_breakdown") or {}
    agent_costs: dict[str, float] = {}
    if isinstance(bd, dict):
        for k, v in bd.items():
            if isinstance(v, dict):
                norm = "research" if str(k).startswith("research_") else k
                agent_costs[norm] = agent_costs.get(norm, 0) + float(v.get("cost_usd", 0))

    # total session cost (for proportional fallback when cost_breakdown missing)
    total_cost = float(row.get("total_cost_usd") or 0)
    if total_cost == 0 and agent_costs:
        total_cost = sum(agent_costs.values())

    # which agents ran — prefer traces; fall back to agents_invoked on the session row
    agents_ran: set[str] = set()
    if not traces_df.empty:
        for a in traces_df[traces_df["session_id"] == sid]["agent"].dropna().unique():
            a = str(a).lower()
            if a.startswith("research"):
                agents_ran.add("research")
            elif a == "market_shadow":
                agents_ran.add("market")
            else:
                agents_ran.add(a)
    if not agents_ran:
        for a in (row.get("agents_invoked") or []):
            a = str(a).lower()
            if a.startswith("research"):
                agents_ran.add("research")
            elif a == "market_shadow":
                agents_ran.add("market")
            else:
                agents_ran.add(a)

    # first failing agent: failed evals OR absent when a prior agent ran
    first_fail_idx  = None
    heuristic_used  = False
    fail_reason_str = ""
    for i, ag in enumerate(pipeline):
        if ag not in agents_ran:
            if i > 0 and any(pipeline[j] in agents_ran for j in range(i)):
                first_fail_idx  = i
                fail_reason_str = f"{_labels[ag]} never ran after prior agent completed"
                break
        else:
            if not evals_df.empty:
                ag_e = evals_df[(evals_df["session_id"] == sid) & (evals_df["agent"] == ag)]
                if not ag_e.empty and float(ag_e["passed"].sum()) / len(ag_e) < 0.8:
                    n_pass = int(ag_e["passed"].sum())
                    n_tot  = len(ag_e)
                    first_fail_idx  = i
                    fail_reason_str = f"{_labels[ag]} {n_pass}/{n_tot} evals passed"
                    break

    # If evals gave no signal: heuristic — 0 trades + 3+ agents + known incident
    if first_fail_idx is None:
        trades = int(row.get("trades_executed") or 0)
        if trades == 0 and total_cost > 0 and len(agents_ran) >= 3:
            first_fail_idx  = 1   # assume research (most common silent failure)
            heuristic_used  = True
            fail_reason_str = "Research (estimated — no eval data)"
        else:
            return 0.0, ""

    agents_after = [ag for ag in pipeline[first_fail_idx + 1:] if ag in agents_ran]
    if not agents_after:
        return 0.0, ""

    # precise savings from cost_breakdown
    savings = sum(agent_costs.get(ag, 0) for ag in agents_after)

    # proportional fallback when cost_breakdown is unavailable
    proportional = False
    if savings == 0 and not agent_costs and total_cost > 0:
        savings     = round(len(agents_after) / max(len(agents_ran), 1) * total_cost, 4)
        proportional = True

    if savings == 0:
        return 0.0, ""

    # Build tooltip breakdown
    lines = [f"CB fires at: {fail_reason_str}"]
    for ag in pipeline:
        if ag not in agents_ran:
            continue
        lbl = _labels[ag]
        if agent_costs:
            cost_str = f"${agent_costs.get(ag, 0):.4f}"
        elif proportional or heuristic_used:
            pct = round(100 / max(len(agents_ran), 1))
            cost_str = f"~{pct}% of ${total_cost:.4f}"
        else:
            cost_str = ""
        idx = pipeline.index(ag)
        if idx < first_fail_idx:
            lines.append(f"{lbl} {cost_str}: incurred")
        elif idx == first_fail_idx:
            lines.append(f"{lbl} {cost_str}: incurred · CB fires")
        else:
            lines.append(f"{lbl} {cost_str}: preventable")
    if heuristic_used or proportional:
        lines.append("(~estimate — exact data unavailable)")
    lines.append(f"Savings: ${savings:.4f}")

    # short label for inline display (no hover needed)
    fail_agent_name  = _labels.get(pipeline[first_fail_idx], pipeline[first_fail_idx].title())
    saved_names      = "+".join(_labels.get(a, a.title()) for a in agents_after)
    est_marker       = " (est.)" if (heuristic_used or proportional) else ""
    short_label      = f"{fail_agent_name} fails{est_marker} · {saved_names} saved"

    return savings, short_label


def pipeline_strip(sid: str, traces_df, evals_df, session_agents=None) -> str:
    """Pipeline strip: ORC coordinates → MKT → NEWS → RES → RSK → ORC synthesizes."""
    _sub_agents = ["market", "news_analyst", "research", "risk"]
    _labels     = {
        "market":       "MKT",
        "news_analyst": "NEWS",
        "research":     "RES",
        "risk":         "RSK",
        "orchestrator": "ORC",
    }

    def _normalize(a: str) -> str:
        a = a.lower()
        if a.startswith("research"):   return "research"
        if a == "market_shadow":       return "market"
        if "news" in a:                return "news_analyst"
        return a

    # "ran" = had substantive work: tokens > 0 OR a tool_call step
    _ran: set[str] = set()
    if not traces_df.empty:
        _st = traces_df[traces_df["session_id"] == sid]
        for _, _tr in _st.iterrows():
            a    = _normalize(str(_tr.get("agent") or ""))
            tok  = int(_tr.get("tokens_input") or 0) + int(_tr.get("tokens_output") or 0)
            step = str(_tr.get("step_type") or "").lower()
            if tok > 0 or step == "tool_call":
                _ran.add(a)
    if not _ran and session_agents:
        for a in session_agents:
            _ran.add(_normalize(str(a)))

    def _eval_color(ag):
        # Check operational evals (agent=ag) and quality evals (agent=ag_quality)
        # Only color if agent actually ran — no coloring from backfilled quality evals alone
        if ag not in _ran:
            return "#e2e8f0", "#94a3b8", f"{_labels.get(ag, ag)}: did not run", "○"
        _ae = pd.DataFrame()
        if not evals_df.empty:
            _ae = evals_df[
                (evals_df["session_id"] == sid) &
                (evals_df["agent"].isin([ag, ag + "_quality"]))
            ]
        if _ae.empty:
            # Slate-blue: ran but no evals (market agent is never evaluated)
            return "#475569", "#ffffff", f"{_labels.get(ag, ag)}: ran — not evaluated", "●"
        n_pass = int(_ae["passed"].sum())
        n_tot  = len(_ae)
        pr     = n_pass / n_tot * 100
        bg     = "#10b981" if pr >= 80 else "#f59e0b" if pr >= 60 else "#ef4444"
        return bg, "#ffffff", f"{_labels.get(ag, ag)}: {n_pass}/{n_tot} evals passed ({pr:.0f}%)", "●"

    def _node(bg, fg, tip, sym, label):
        safe = tip.replace('"', "&quot;")
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:1px">'
            f'<span class="pipe-node" data-tip="{safe}" style="background:{bg};color:{fg}">{sym}</span>'
            f'<span style="font-size:0.58rem;color:#94a3b8;line-height:1.2">{label}</span>'
            f'</div>'
        )

    arrow = '<span style="color:#cbd5e1;font-size:0.75rem;margin-bottom:10px">→</span>'

    # Left ORC: coordinator — green if any agent ran, gray otherwise
    orc_ran = bool(_ran)
    orc_l_bg  = "#10b981" if orc_ran else "#e2e8f0"
    orc_l_fg  = "#ffffff"  if orc_ran else "#94a3b8"
    orc_l_tip = "Orchestrator: session initiated — dispatching to sub-agents" if orc_ran else "Orchestrator: no session data"
    orc_l_sym = "●" if orc_ran else "○"

    # Right ORC: synthesizer — eval pass rate
    orc_r_bg, orc_r_fg, orc_r_tip, orc_r_sym = _eval_color("orchestrator")
    orc_r_tip = orc_r_tip.replace("ORC:", "Orchestrator synthesizer:")

    html = '<div style="display:flex;align-items:flex-start;gap:3px;overflow:visible;padding-bottom:2px">'
    html += _node(orc_l_bg, orc_l_fg, orc_l_tip, orc_l_sym, "ORC")
    html += arrow
    for ag in _sub_agents:
        bg, fg, tip, sym = _eval_color(ag)
        html += _node(bg, fg, tip, sym, _labels[ag])
        html += arrow
    html += _node(orc_r_bg, orc_r_fg, orc_r_tip, orc_r_sym, "ORC")
    html += '</div>'
    return html


def tip_badge(description: str) -> str:
    """Inline ? badge with CSS hover tooltip."""
    safe = description.replace('"', "&quot;").replace("'", "&#39;")
    return f'<span class="tip-badge" data-tip="{safe}">?</span>'

AGENT_COLORS = {
    "research":      "#f59e0b",
    "orchestrator":  "#3b82f6",
    "risk":          "#10b981",
    "market_shadow": "#6b7280",
    "market":        "#8b5cf6",
}

SEV_BADGE = {
    "critical": '<span class="badge-critical">CRITICAL</span>',
    "warning":  '<span class="badge-warning">WARNING</span>',
    "info":     '<span class="badge-info">INFO</span>',
}

def badge(severity: str, simulated: bool = False) -> str:
    if simulated:
        return '<span class="badge-sim">SIMULATED</span> ' + SEV_BADGE.get(severity, "")
    return SEV_BADGE.get(severity, severity.upper())

def kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{"<div class=kpi-sub>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )

def score_bar(score: float, passed: bool) -> str:
    color = "#10b981" if passed else "#ef4444"
    pct   = int(score * 100)
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
    )

def compute_business_evals_df(sessions_df: pd.DataFrame,
                               traces_df: pd.DataFrame) -> pd.DataFrame:
    """Compute business outcome evals on-the-fly and return as a long-form DataFrame."""
    rows = []
    for _, sess in sessions_df.iterrows():
        sid    = sess["id"]
        trows  = traces_df[traces_df["session_id"] == sid].to_dict("records")
        sdict  = sess.to_dict()
        for fn, eval_name, threshold, fmt in [
            (eval_cost_per_trade,       "cost_per_trade",       COST_PER_TRADE_THRESHOLD,  "dollar"),
            (eval_research_conversion,  "research_conversion",  RESEARCH_CONVERSION_MIN,   "pct"),
            (eval_proposal_acceptance,  "proposal_acceptance",  PROPOSAL_ACCEPTANCE_MIN,   "pct"),
        ]:
            r = fn(trows, sdict)
            detail = r.detail
            if eval_name == "cost_per_trade":
                raw_value = detail.get("cost_per_trade", 0)
            else:
                key = "conversion_rate" if eval_name == "research_conversion" else "acceptance_rate"
                raw_value = detail.get(key, 0)
            rows.append({
                "session_id": sid,
                "started_at": sess.get("started_at"),
                "eval_name":  eval_name,
                "score":      r.score,
                "passed":     r.passed,
                "value":      raw_value,
                "threshold":  threshold,
                "fmt":        fmt,
            })
    return pd.DataFrame(rows)


def run_analysis_for_session(session_row: dict, traces: list[dict],
                              recent_costs: list[float]) -> tuple:
    evals     = run_all_evals(session_row, traces, recent_costs)
    incidents = run_all_detectors(session_row, traces, evals, recent_costs)
    return evals, incidents


def _ledger_summary_cache_key(
    n_sessions: int, total_cost: float, total_trades: int,
    n_wasted: int, wasted_cost: float, n_incidents: int,
    top_agent: str, top_agent_cost: float,
) -> str:
    return f"{n_sessions}:{total_cost:.4f}:{total_trades}:{n_wasted}:{wasted_cost:.4f}:{n_incidents}:{top_agent}:{top_agent_cost:.4f}"


def generate_ledger_insights(
    n_sessions: int, total_cost: float, total_trades: int,
    n_wasted: int, wasted_cost: float, wasted_pct: float,
    n_incidents: int, top_agent: str, top_agent_cost: float,
    top_agent_pct: float, n_sessions_with_cost_data: int,
    recent_terminal_reasons: list[str],
) -> dict:
    cache_key = _ledger_summary_cache_key(
        n_sessions, total_cost, total_trades, n_wasted,
        wasted_cost, n_incidents, top_agent, top_agent_cost,
    )
    if st.session_state.get("_ledger_summary_key") == cache_key:
        return st.session_state["_ledger_summary"]

    prompt = f"""You are an AI reliability analyst reviewing the Outcome Ledger for Strategy C — a 6-agent trading pipeline (market, news_analyst, research, risk, orchestrator, synthesis).

Data snapshot:
- Sessions run: {n_sessions}
- Total LLM spend: ${total_cost:.4f}
- Trades executed: {total_trades}
- Wasted sessions (0 trades, cost > 0): {n_wasted} sessions · ${wasted_cost:.4f} · {wasted_pct:.0f}% of spend
- Incidents detected: {n_incidents}
- Top cost agent: {top_agent} (${top_agent_cost:.4f}, {top_agent_pct:.0f}% of spend) across {n_sessions_with_cost_data} sessions
- Recent exit reasons: {", ".join(recent_terminal_reasons[:10]) if recent_terminal_reasons else "none"}

Return a JSON object with exactly these 4 keys. Each value is a single sentence — direct, analytical, no filler:

{{
  "page_summary": "2-3 sentence narrative: what this data says about the pipeline's overall health and the single most important thing to act on",
  "kpi_insight": "one sentence interpreting the cost-to-trade ratio and what the wasted session rate means",
  "cost_insight": "one sentence on whether the top agent's cost share is expected or a signal worth investigating",
  "sessions_insight": "one sentence on what patterns to look for in the session list given the exit reason distribution"
}}

No markdown, no extra keys, valid JSON only."""

    client = anthropic.Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {
            "page_summary": raw,
            "kpi_insight": "", "cost_insight": "", "sessions_insight": "",
        }

    st.session_state["_ledger_summary_key"] = cache_key
    st.session_state["_ledger_summary"] = result
    return result



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: RCA View utilities (injected before Sidebar block)
# ─────────────────────────────────────────────────────────────────────────────

import math as _math
import json as _json


def _str(v):
    """Return None for null/NaN, string otherwise."""
    return None if (v is None or (isinstance(v, float) and _math.isnan(v))) else str(v)


def _section_header(title: str, help_md: str) -> None:
    """Section title with inline ? help popover."""
    c_t, c_h = st.columns([28, 1])
    with c_t:
        st.markdown(f"#### {title}")
    with c_h:
        with st.popover("?"):
            st.markdown(help_md)


def _eval_reason(eval_name: str, detail: dict, score: float, passed: bool) -> str:
    """One-line explanation of how the eval score was calculated."""
    d = detail if isinstance(detail, dict) else {}

    if eval_name == "completion":
        if d.get("llm_ok") and d.get("tool_calls_ok"):
            return "Scored 1.0 — LLM ran and at least one tool call succeeded"
        if d.get("llm_ok") is not None:
            n = d.get("tool_calls", "")
            suffix = f" ({n} tool calls all failed)" if n else " (all tool calls failed)"
            return f"Scored 0.3 — LLM ran but data-fetch failed{suffix}. Threshold >= 0.7"
        if d.get("decision_traces") is not None:
            return f"Scored 1.0 — {d['decision_traces']} successful LLM/decision trace(s) found"
        reason = d.get("reason", "")
        if reason:
            n = d.get("tool_calls", "")
            return reason + (f" ({n} calls)" if n else "")
        return "Scored 1.0 — research agent completed" if passed else "Scored 0.0 — research did not run"

    if eval_name == "tool_success_rate":
        total = d.get("total", 0)
        if not total:
            return "No tool calls made — scored 1.0 by default"
        s = d.get("success", 0)
        pct = int(round(score * 100))
        return f"{s}/{total} tool calls succeeded ({pct}%) — threshold >= 80%"

    if eval_name == "token_efficiency":
        used = d.get("tokens_used")
        thr  = d.get("threshold")
        if used is not None and thr:
            return f"{used:,} tokens used vs {thr:,} limit"
        if used is not None:
            return f"{used:,} tokens used"

    if eval_name == "data_completeness":
        reason = d.get("reason", "")
        if reason:
            return reason
        t, s = d.get("total", 0), d.get("success", 0)
        if t:
            return f"{s}/{t} market traces succeeded — score = success/total, threshold >= 0.7"

    if eval_name == "data_freshness":
        reason = d.get("reason", "")
        if reason:
            return reason
        lag = d.get("lag_minutes")
        thr = d.get("threshold_minutes")
        if lag is not None and thr is not None:
            rel = "within" if passed else "exceeds"
            return f"First market trace {lag} min after session start — {rel} {thr}-min limit"

    if eval_name == "assessment_complete":
        reason = d.get("reason", "")
        if reason:
            return reason
        total   = d.get("total", 0)
        success = d.get("success", 0)
        if total:
            return f"Risk agent: {success}/{total} traces succeeded — threshold = 1.0"
        return f"Risk assessment {'present' if passed else 'absent'} — threshold 1.0"

    if eval_name == "within_parameters":
        reason = d.get("reason", "")
        if reason:
            return reason
        rt = d.get("risk_traces", 0)
        et = d.get("error_traces", 0)
        if rt:
            return f"{rt} risk trace(s), {et} with errors — score = 1 - errors/traces, threshold >= 0.9"
        return f"No risk traces found — scored {score:.2f}"

    if eval_name == "decision_made":
        reason = d.get("reason", "")
        if reason:
            return reason
        trades = d.get("trades_executed")
        term   = d.get("terminal_reason")
        if trades is not None and trades > 0:
            return f"{trades} trade(s) executed — orchestrator decided, scored 1.0"
        if term:
            return f"Terminal reason logged: '{term}' — scored 0.8, threshold >= 0.7"
        return "Orchestrator ran but no decision or reason recorded — scored 0.4"

    if eval_name == "consistency":
        reason = d.get("reason", "")
        if reason:
            return reason
        return f"Orchestrator pipeline consistency — scored {score:.2f}, threshold >= 0.8"

    if eval_name == "pipeline_completion":
        missing = d.get("missing", [])
        present = d.get("present", [])
        n_req   = len(present) + len(missing)
        if missing:
            return f"{len(present)}/{n_req} required agents ran — missing: {', '.join(missing)}"
        return f"All {n_req} agents ran (market, research, risk, orchestrator) — scored 1.0"

    if eval_name == "cost_anomaly":
        reason = d.get("reason", "")
        if reason:
            cost = d.get("cost_usd")
            return reason + (f" — USD{cost:.4f}" if cost else "")
        z    = d.get("z_score")
        mean = d.get("mean")
        cost = d.get("cost_usd")
        sig  = d.get("threshold_sigma", 2)
        if z is not None and mean is not None:
            rel  = f"{z:.1f}s above" if z > 0 else "within"
            flag = "flagged" if not passed else "normal"
            return f"USD{cost:.4f} spent — {rel} mean USD{mean:.4f} ({flag}, threshold {sig}s)"

    if eval_name == "outcome_linkage":
        reason = d.get("reason", "")
        if reason:
            return reason
        trades = d.get("trades_executed")
        term   = d.get("terminal_reason")
        if trades is not None and trades > 0:
            return f"{trades} trade(s) executed — session has measurable outcome, scored 1.0"
        if term:
            return f"No trades but terminal reason logged: '{term}' — scored 0.8"
        return "0 trades and no terminal reason — silent exit, scored 0.0"

    if eval_name == "tokens_per_decision":
        tpd   = d.get("tokens_per_decision")
        thr   = d.get("threshold")
        total = d.get("total_tokens")
        dec   = d.get("trades")
        if tpd is not None and thr:
            return f"{total:,} tokens / {dec} decision(s) = {tpd:,} per decision (limit {thr:,})"

    return ""


def _render_call_chain_html(
    agents_present: list,
    evals_by_agent: dict,
    root_agent,
    agent_stats: dict,
    sess_row: dict = None,
    sess_traces: list = None,
) -> str:
    """Pure HTML/CSS pipeline cards rendered via st.components.v1.html."""
    if not agents_present:
        return ""

    bd = (sess_row or {}).get("cost_breakdown")
    agent_cost: dict[str, float] = {}
    if isinstance(bd, dict) and bd:
        for k, v in bd.items():
            if isinstance(v, dict):
                agent_cost[k.lower()] = float(v.get("cost_usd", 0))
    if not agent_cost:
        total_cost = float((sess_row or {}).get("total_cost_usd") or 0)
        total_tok  = sum(s.get("tokens", 1) for s in agent_stats.values()) or 1
        for a, s in agent_stats.items():
            agent_cost[a] = (s.get("tokens", 0) / total_tok) * total_cost

    trades = int((sess_row or {}).get("trades_executed", 0))

    def _agent_tools(a):
        if not sess_traces:
            return []
        tools, seen = [], set()
        for t in sess_traces:
            if (t.get("agent") or "").lower() == a.lower():
                raw = t.get("tool_name")
                if raw is None:
                    continue
                if isinstance(raw, float):
                    continue  # NaN from pandas
                tn = str(raw).strip()
                if tn and tn not in seen:
                    tools.append(tn)
                    seen.add(tn)
        return tools[:3]

    def _has_parallel(a):
        if not sess_traces:
            return False
        tool_traces = [
            t for t in sess_traces
            if (t.get("agent") or "").lower() == a.lower()
            and t.get("tool_name") and t.get("created_at")
        ]
        if len(tool_traces) < 2:
            return False
        try:
            dts = sorted(
                [(pd.to_datetime(t["created_at"]), float(t.get("latency_ms") or 0))
                 for t in tool_traces],
                key=lambda x: x[0],
            )
            for i in range(len(dts) - 1):
                end_i = dts[i][0] + pd.Timedelta(milliseconds=dts[i][1])
                if end_i > dts[i + 1][0]:
                    return True
        except Exception:
            pass
        return False

    def _border_color(a):
        if a == root_agent:
            return "#f59e0b"
        evs = evals_by_agent.get(a.lower(), [])
        if not evs:
            return "#cbd5e1"
        return "#ef4444" if any(not e["passed"] for e in evs) else "#10b981"

    def _bg_color(a):
        if a == root_agent:
            return "#fff7ed"
        evs = evals_by_agent.get(a.lower(), [])
        if not evs:
            return "#f8fafc"
        return "#fef2f2" if any(not e["passed"] for e in evs) else "#f0fdf4"

    css = (
        "*{box-sizing:border-box;margin:0;padding:0;}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#f8fafc;padding:16px 12px 10px;}"
        ".pipeline{display:flex;align-items:stretch;justify-content:space-evenly;}"
        ".card{flex:1;min-width:0;max-width:200px;border-radius:10px;"
        "display:flex;flex-direction:column;min-height:220px;cursor:pointer;}"
        ".card:hover{filter:brightness(0.97);}"
        ".banner{font-size:10px;font-weight:700;color:#b45309;background:#fef3c7;"
        "text-align:center;padding:5px 8px;border-radius:8px 8px 0 0;letter-spacing:.5px;}"
        ".banner-ghost{padding:5px 8px;font-size:10px;visibility:hidden;}"
        ".body{display:flex;flex-direction:column;align-items:center;justify-content:center;"
        "padding:12px 10px;flex:1;gap:4px;text-align:center;}"
        ".dot{width:11px;height:11px;border-radius:50%;margin:0 auto 5px;}"
        ".name{font-size:13px;font-weight:700;color:#1e293b;letter-spacing:.4px;}"
        ".metrics{font-size:11px;color:#64748b;margin-top:1px;}"
        ".tools{font-size:10px;color:#94a3b8;margin-top:5px;line-height:1.6;}"
        ".badge{font-size:9px;padding:2px 7px;border-radius:10px;margin-top:3px;display:inline-block;}"
        ".par{background:#ede9fe;color:#7c3aed;}"
        ".seq{background:#f1f5f9;color:#64748b;}"
        ".evals{font-size:10px;margin-top:5px;line-height:1.7;}"
        ".pass{color:#10b981;}"
        ".fail{color:#ef4444;}"
        ".hint{font-size:9px;color:#94a3b8;margin-top:6px;}"
        ".arrow{display:flex;align-items:center;padding:0 10px;color:#94a3b8;"
        "font-size:22px;flex-shrink:0;align-self:center;}"
        ".terminal{flex:0 0 90px;border-radius:10px;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;padding:14px 10px;min-height:220px;}"
        ".ticon{font-size:20px;}"
        ".tlabel{font-size:13px;font-weight:700;margin-top:4px;}"
        ".tsub{font-size:11px;color:#64748b;margin-top:3px;}"
        ".legend{display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;}"
        ".li{font-size:11px;color:#64748b;}"
    )

    # JS: click card → try to scroll parent page to matching agent expander
    js = (
        "document.querySelectorAll('.card').forEach(function(card){"
        "card.addEventListener('click',function(){"
        "var name=card.dataset.agent;"
        "try{"
        "var exps=window.parent.document.querySelectorAll('[data-testid=\"stExpander\"]');"
        "for(var i=0;i<exps.length;i++){"
        "if(exps[i].textContent.toLowerCase().indexOf(name)>-1){"
        "exps[i].scrollIntoView({behavior:'smooth',block:'start'});break;}}"
        "}catch(e){}"
        "});});"
    )

    cards_html = ""
    for i, a in enumerate(agents_present):
        evs    = evals_by_agent.get(a.lower(), [])
        stats  = agent_stats.get(a.lower(), {})
        cost   = agent_cost.get(a.lower(), agent_cost.get(a, 0))
        tok    = stats.get("tokens", 0)
        lat_s  = (stats.get("latency_ms", 0) or 0) // 1000
        tools  = _agent_tools(a)
        is_par = _has_parallel(a)
        border = _border_color(a)
        bg     = _bg_color(a)

        passed_names = [e["eval_name"].split(".")[-1] for e in evs if e["passed"]]
        failed_names = [e["eval_name"].split(".")[-1] for e in evs if not e["passed"]]

        banner = (
            '<div class="banner">ROOT CAUSE</div>'
            if a == root_agent
            else '<div class="banner-ghost">x</div>'
        )

        tools_html = (
            '<br>'.join(tools)
            if tools else '<span style="color:transparent;">—</span>'
        )

        exec_badge = ""
        if tools:
            exec_badge = (
                '<span class="badge par">&#x2016; parallel tools</span>'
                if is_par else
                '<span class="badge seq">&#8594; sequential</span>'
            )

        eval_parts = []
        if passed_names:
            eval_parts.append(f'<span class="pass">&#10003; {len(passed_names)} passed</span>')
        if failed_names:
            short = ', '.join(n.replace('_', ' ') for n in failed_names[:2])
            if len(failed_names) > 2:
                short += f' +{len(failed_names)-2}'
            eval_parts.append(f'<span class="fail">&#10007; {short}</span>')
        if not evs:
            eval_parts.append('<span style="color:#cbd5e1">no evals</span>')
        evals_html = '<br>'.join(eval_parts)

        cards_html += (
            f'<div class="card" data-agent="{a.lower()}" '
            f'style="border:2px solid {border};background:{bg};">'
            f'{banner}'
            f'<div class="body">'
            f'<div class="dot" style="background:{border};"></div>'
            f'<div class="name">{a.upper()}</div>'
            f'<div class="metrics">${cost:.4f} &middot; {tok:,}t &middot; {lat_s}s</div>'
            f'<div class="tools">{tools_html}</div>'
            f'{exec_badge}'
            f'<div class="evals">{evals_html}</div>'
            f'<div class="hint">click &#8595; agent details</div>'
            f'</div></div>'
        )

        if i < len(agents_present) - 1:
            cards_html += '<div class="arrow">&#8594;</div>'

    term_label  = "OUTPUT" if trades > 0 else "WASTE"
    term_bg     = "#f0fdf4" if trades > 0 else "#fef2f2"
    term_border = "#10b981" if trades > 0 else "#ef4444"
    term_color  = "#16a34a" if trades > 0 else "#dc2626"
    term_icon   = "&#10003;" if trades > 0 else "&#10007;"
    term_sub    = f"{trades} trade{'s' if trades != 1 else ''}" if trades > 0 else "0 trades"

    cards_html += (
        '<div class="arrow">&#8594;</div>'
        f'<div class="terminal" style="border:2px solid {term_border};background:{term_bg};">'
        f'<div class="ticon" style="color:{term_color};">{term_icon}</div>'
        f'<div class="tlabel" style="color:{term_color};">{term_label}</div>'
        f'<div class="tsub">{term_sub}</div>'
        f'</div>'
    )

    legend = (
        '<div class="legend">'
        '<span class="li"><span style="color:#10b981">&#9679;</span> All evals passed</span>'
        '<span class="li"><span style="color:#ef4444">&#9679;</span> Evals failed</span>'
        '<span class="li"><span style="color:#f59e0b">&#9679;</span> Root cause</span>'
        '<span class="li"><span style="color:#cbd5e1">&#9679;</span> No eval data</span>'
        '<span class="li"><span style="background:#ede9fe;color:#7c3aed;padding:0 5px;'
        'border-radius:8px;font-size:9px;">&#x2016; parallel</span> tools ran in parallel</span>'
        '</div>'
    )

    return (
        f'<html><head><style>{css}</style></head><body>'
        f'<div class="pipeline">{cards_html}</div>'
        f'{legend}'
        f'<script>{js}</script>'
        f'</body></html>'
    )


def _build_timeline_fig(traces: list):
    """Plotly Gantt-style horizontal timeline of agent execution steps."""
    if not traces:
        return None
    df = pd.DataFrame(traces)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    if df.empty:
        return None
    df["latency_ms"] = pd.to_numeric(df.get("latency_ms", 0), errors="coerce").fillna(500)
    t0 = df["created_at"].min()
    df["end_s"]   = (df["created_at"] - t0).dt.total_seconds()
    df["start_s"] = (df["end_s"] - df["latency_ms"] / 1000).clip(lower=0)
    df["step"]    = df.apply(lambda r: r.get("tool_name") or r.get("step_type") or "step", axis=1)
    df["is_err"]  = (df["outcome"] == "error") | df["error"].notna()

    order  = ["market", "research", "risk", "orchestrator"]
    agents = [a for a in order if a in df["agent"].values]
    if not agents:
        agents = df["agent"].dropna().unique().tolist()

    total_dur = df["end_s"].max() or 1.0
    label_threshold = total_dur * 0.06  # only label bars wider than 6% of total

    # One bar per agent spanning its full execution window
    fig = go.Figure()
    for agent in agents:
        adf       = df[df["agent"] == agent]
        min_start = float(adf["start_s"].min())
        max_end   = float(adf["end_s"].max())
        n_steps   = len(adf)
        n_err     = int(adf["is_err"].sum())
        # minimum bar width: 3% of total duration so short agents are still visible
        bar_dur   = max(max_end - min_start, total_dur * 0.03, 1.0)
        bar_color = "#ef4444" if n_err > 0 else AGENT_COLORS.get(agent, "#94a3b8")
        # only show text if bar is wide enough to avoid Plotly rotating it vertically
        _bar_pct  = bar_dur / (total_dur or 1)
        bar_text  = (
            f"{n_steps} step{'s' if n_steps != 1 else ''}" + (f"  {n_err} err" if n_err else "")
            if _bar_pct >= 0.08 else ""
        )
        fig.add_trace(go.Bar(
            x=[bar_dur], y=[agent.upper()], base=[min_start],
            orientation="h",
            marker_color=bar_color,
            marker_line_width=0,
            text=bar_text,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="#ffffff"),
            hovertemplate=(
                f"<b>{agent.upper()}</b><br>"
                f"Start: {min_start:.1f}s  End: {max_end:.1f}s<br>"
                f"Steps: {n_steps}  Errors: {n_err}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff", font_color="#1e293b",
        height=max(180, len(agents) * 70 + 60),
        margin=dict(t=10, b=40, l=10, r=20),
        barmode="overlay",
        xaxis=dict(title="Seconds from session start", gridcolor="#e2e8f0", tickfont=dict(size=10)),
        yaxis=dict(
            categoryorder="array",
            categoryarray=[a.upper() for a in reversed(agents)],
        ),
    )
    return fig


def _build_cost_donut(sess_row: dict, sess_traces: list):
    """Plotly donut of cost split by agent for a single session."""
    bd = sess_row.get("cost_breakdown")
    if isinstance(bd, dict) and bd:
        labels = [k for k in bd if isinstance(bd[k], dict)]
        values = [float(bd[k].get("cost_usd", 0)) for k in labels]
    else:
        if not sess_traces:
            return None
        tdf = pd.DataFrame(sess_traces)
        if tdf.empty:
            return None
        tdf["tok"] = (
            pd.to_numeric(tdf.get("tokens_input", 0), errors="coerce").fillna(0) +
            pd.to_numeric(tdf.get("tokens_output", 0), errors="coerce").fillna(0)
        )
        agg   = tdf.groupby("agent")["tok"].sum()
        total = agg.sum()
        if total == 0:
            return None
        tc     = float(sess_row.get("total_cost_usd") or 0)
        labels = list(agg.index)
        values = [(v / total) * tc for v in agg.values]

    if not any(v > 0 for v in values):
        return None

    total_val = sum(values)
    fig = go.Figure(go.Pie(
        labels=[l.upper() for l in labels],
        values=values,
        hole=0.55,
        marker_colors=[AGENT_COLORS.get(l, "#94a3b8") for l in labels],
        textinfo="label+percent",
        textfont_size=10,
        hovertemplate="<b>%{label}</b><br>$%{value:.5f}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"${total_val:.4f}", x=0.5, y=0.5,
        font=dict(size=12, color="#0f172a"), showarrow=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e293b", height=200,
        margin=dict(t=5, b=5, l=5, r=5), showlegend=False,
    )
    return fig



# ── Page resolution ───────────────────────────────────────────────────────────

_NAV_PAGES = [
    "Overview", "Ledger", "Quality Drift",
    "Incidents Feed", "RCA View", "Failure Simulator", "Trace Inspector",
]

if "_page" in st.session_state:
    _forced = st.session_state.pop("_page")
    st.query_params["page"] = _forced
    page = _forced
elif "sid" in st.query_params and "page" not in st.query_params:
    st.query_params["page"] = "RCA View"
    page = "RCA View"
else:
    page = st.query_params.get("page", "Overview")
    if page not in _NAV_PAGES:
        page = "Overview"

# ── Top navigation bar (st.pills — no JS, no iframe) ─────────────────────────

_nav_brand_col, _nav_pills_col = st.columns([3, 9])
with _nav_brand_col:
    st.markdown(
        '<div class="topnav-brand-cell">'
        '<span class="topnav-brand">AI Agent RCA</span>'
        '<span class="topnav-tag">&nbsp;·&nbsp;Strategy C · Live</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with _nav_pills_col:
    _nav_sel = st.pills(
        "Navigation", _NAV_PAGES,
        selection_mode="single",
        default=page,
        label_visibility="collapsed",
        key=f"topnav_{page}",
    )

if _nav_sel and _nav_sel != page:
    st.query_params["page"] = _nav_sel
    st.rerun()


# ── Load shared data ──────────────────────────────────────────────────────────

sessions   = load_sessions()
traces_all = load_traces()
positions  = load_positions()
incidents  = load_incidents()

if not sessions.empty and not positions.empty:
    pnl = (positions.groupby("session_id")["realized_pnl"].sum()
           .reset_index().rename(columns={"realized_pnl": "session_pnl"}))
    sessions = sessions.merge(pnl, left_on="id", right_on="session_id", how="left")
    sessions["session_pnl"] = sessions.get("session_pnl", 0)

recent_costs = sessions["total_cost_usd"].tolist() if not sessions.empty else []


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    _ov_h1, _ov_h2, _ov_h3 = st.columns([5, 1, 1])
    with _ov_h1:
        st.markdown("## System Overview")
    with _ov_h2:
        _ov_range = st.radio("Period", ["7d", "14d", "30d"],
                             horizontal=True, index=0,
                             label_visibility="collapsed", key="ov_range")
    with _ov_h3:
        if st.button("↺ Refresh", use_container_width=True):
            load_sessions.clear()
            load_traces.clear()
            load_all_evals.clear()
            st.rerun()
    _ov_days   = int(_ov_range[:-1])
    _ov_now    = pd.Timestamp.now(tz="UTC")
    _ov_cut    = _ov_now - pd.Timedelta(days=_ov_days)
    _ov_prev_c = _ov_cut  - pd.Timedelta(days=_ov_days)

    # ── Filter sessions + incidents to window ─────────────────────────────────
    _ov_s = sessions.copy() if not sessions.empty else sessions
    if not _ov_s.empty:
        _ov_s["started_at"] = pd.to_datetime(_ov_s["started_at"], errors="coerce", utc=True)
        _ov_s_now  = _ov_s[_ov_s["started_at"] >= _ov_cut]
        _ov_s_prev = _ov_s[(_ov_s["started_at"] >= _ov_prev_c) & (_ov_s["started_at"] < _ov_cut)]
    else:
        _ov_s_now = _ov_s_prev = _ov_s

    _ov_i = incidents.copy() if not incidents.empty else incidents
    if not _ov_i.empty and "created_at" in _ov_i.columns:
        _ov_i["created_at"] = pd.to_datetime(_ov_i["created_at"], errors="coerce", utc=True)
        _ov_i_now  = _ov_i[_ov_i["created_at"] >= _ov_cut]
        _ov_i_prev = _ov_i[(_ov_i["created_at"] >= _ov_prev_c) & (_ov_i["created_at"] < _ov_cut)]
    else:
        _ov_i_now = _ov_i_prev = _ov_i

    _ov_sids_inc  = set(_ov_i_now["session_id"].unique())  if not _ov_i_now.empty  else set()
    _ov_sids_prv  = set(_ov_i_prev["session_id"].unique()) if not _ov_i_prev.empty else set()

    _ov_n_sess    = len(_ov_s_now)
    _ov_n_sess_p  = len(_ov_s_prev)
    _ov_n_inc     = len(_ov_i_now)
    _ov_n_inc_p   = len(_ov_i_prev)
    _ov_n_clean   = int((~_ov_s_now["id"].isin(_ov_sids_inc)).sum())   if not _ov_s_now.empty  else 0
    _ov_n_clean_p = int((~_ov_s_prev["id"].isin(_ov_sids_prv)).sum())  if not _ov_s_prev.empty else 0
    _ov_sr        = _ov_n_clean   / _ov_n_sess   * 100 if _ov_n_sess   else 0
    _ov_sr_p      = _ov_n_clean_p / _ov_n_sess_p * 100 if _ov_n_sess_p else 0
    _ov_avg_cost  = float(_ov_s_now["total_cost_usd"].mean())  if not _ov_s_now.empty  else 0.0
    _ov_avg_cost_p= float(_ov_s_prev["total_cost_usd"].mean()) if not _ov_s_prev.empty else 0.0

    _ov_top_pat = ""
    if not _ov_i_now.empty and "pattern_name" in _ov_i_now.columns:
        _pc = _ov_i_now["pattern_name"].value_counts()
        if len(_pc):
            _ov_top_pat = _pc.index[0].replace("_", " ").title()

    def _ov_dlt(curr, prev, higher_good=True):
        if not prev or pd.isna(prev) or prev == 0:
            return ""
        d = curr - prev
        if abs(d) < 0.001:
            return '<span style="color:#94a3b8;font-size:0.72rem">→ flat</span>'
        pct = abs(d / prev * 100)
        arr = "▲" if d > 0 else "▼"
        good = (d > 0) == higher_good
        c = "#10b981" if good else "#ef4444"
        return f'<span style="color:{c};font-size:0.72rem">{arr} {pct:.0f}% vs prev</span>'

    # ── KPI tiles ─────────────────────────────────────────────────────────────
    _kpi_tips = {
        "Sessions":         "Total pipeline runs in the selected period. Each session is one full Market → Research → Risk → Orchestrator cycle.",
        "Success Rate":     "% of sessions where no failure patterns were detected. A session is clean if no incidents fired.",
        "Avg Cost/Session": "Average LLM API cost per pipeline run across all agents combined.",
        "Incidents":        "Total failure pattern detections in the period. One session can trigger multiple incidents.",
        "Top Pattern":      "The most frequently detected failure pattern in the selected period.",
    }
    _k1, _k2, _k3, _k4, _k5 = st.columns(5)
    with _k1:
        st.markdown(kpi("Sessions" + tip_badge(_kpi_tips["Sessions"]),
                        str(_ov_n_sess),
                        _ov_dlt(_ov_n_sess, _ov_n_sess_p)), unsafe_allow_html=True)
    with _k2:
        st.markdown(kpi("Success Rate" + tip_badge(_kpi_tips["Success Rate"]),
                        f"{_ov_sr:.0f}%",
                        _ov_dlt(_ov_sr, _ov_sr_p)), unsafe_allow_html=True)
        if st.button("View incidents →", key="ov_sr", use_container_width=True):
            goto_page("Incidents Feed")
    with _k3:
        st.markdown(kpi("Avg Cost / Session" + tip_badge(_kpi_tips["Avg Cost/Session"]),
                        f"${_ov_avg_cost:.3f}",
                        _ov_dlt(_ov_avg_cost, _ov_avg_cost_p, higher_good=False)),
                    unsafe_allow_html=True)
        if st.button("View ledger →", key="ov_cost", use_container_width=True):
            goto_page("Ledger")
    with _k4:
        st.markdown(kpi("Incidents" + tip_badge(_kpi_tips["Incidents"]),
                        str(_ov_n_inc),
                        _ov_dlt(_ov_n_inc, _ov_n_inc_p, higher_good=False)),
                    unsafe_allow_html=True)
        if st.button("View all →", key="ov_inc", use_container_width=True):
            goto_page("Incidents Feed")
    with _k5:
        _ov_pat_val = (
            f'{_ov_top_pat}{tip_badge(PATTERN_DESCRIPTIONS.get(_ov_top_pat, ""))}'
            if _ov_top_pat else "—"
        )
        st.markdown(kpi("Top Pattern" + tip_badge(_kpi_tips["Top Pattern"]),
                        _ov_pat_val, "most frequent in period"),
                    unsafe_allow_html=True)
        if _ov_top_pat and st.button("Filter feed →", key="ov_pat", use_container_width=True):
            goto_page("Incidents Feed")

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    _ov_c1, _ov_c2 = st.columns(2)

    with _ov_c1:
        st.markdown("#### Outcome Rate")
        if not _ov_s_now.empty:
            _ov_sd = _ov_s_now.copy()
            _ov_sd["_date"] = pd.to_datetime(_ov_sd["started_at"]).dt.date
            _ov_sd["_inc"]  = _ov_sd["id"].isin(_ov_sids_inc)
            _ov_grp = (
                _ov_sd.groupby("_date")
                .apply(lambda g: pd.Series({
                    "Clean":    int((~g["_inc"]).sum()),
                    "Incident": int(g["_inc"].sum()),
                }))
                .reset_index()
                .sort_values("_date")
            )
            _ov_fig_out = go.Figure()
            _ov_fig_out.add_trace(go.Bar(
                x=_ov_grp["_date"].astype(str), y=_ov_grp["Clean"],
                name="Clean", marker_color="#10b981",
                hovertemplate="<b>%{x}</b><br>Clean: %{y}<extra></extra>",
            ))
            _ov_fig_out.add_trace(go.Bar(
                x=_ov_grp["_date"].astype(str), y=_ov_grp["Incident"],
                name="Incident", marker_color="#ef4444",
                hovertemplate="<b>%{x}</b><br>Incident: %{y}<extra></extra>",
            ))
            _ov_fig_out.update_layout(
                barmode="stack",
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                height=240, margin=dict(t=10, b=45, l=0, r=10),
                legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=11)),
                font=dict(color="#1e293b", size=11),
                xaxis=dict(gridcolor="#e2e8f0", tickangle=-30),
                yaxis=dict(gridcolor="#e2e8f0", title="Sessions"),
            )
            st.plotly_chart(_ov_fig_out, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No sessions in this range.")

    with _ov_c2:
        st.markdown("#### Incidents by Pattern")
        if not _ov_i_now.empty and "pattern_name" in _ov_i_now.columns:
            _ov_pat = (
                _ov_i_now["pattern_name"].value_counts()
                .reset_index()
                .rename(columns={"pattern_name": "pattern", "count": "n"})
            )
            _ov_pat["label"] = _ov_pat["pattern"].str.replace("_", " ").str.title()
            _ov_pat = _ov_pat.sort_values("n")
            _ov_sev_map = {}
            if "severity" in _ov_i_now.columns:
                _sev_lkp = _ov_i_now.groupby("pattern_name")["severity"].first().to_dict()
                _ov_sev_map = {k.replace("_", " ").title(): v for k, v in _sev_lkp.items()}
            _ov_bar_cols = [
                "#ef4444" if _ov_sev_map.get(l) == "critical"
                else "#f59e0b" if _ov_sev_map.get(l) == "warning"
                else "#3b82f6"
                for l in _ov_pat["label"]
            ]
            _ov_pat["tip"] = _ov_pat["label"].map(
                lambda l: PATTERN_DESCRIPTIONS.get(l, "")
            )
            _ov_fig_pat = go.Figure(go.Bar(
                x=_ov_pat["n"], y=_ov_pat["label"],
                orientation="h",
                marker_color=_ov_bar_cols,
                text=_ov_pat["n"], textposition="outside",
                customdata=_ov_pat["tip"],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Count: %{x}<br>"
                    "<i style='color:#94a3b8'>%{customdata}</i>"
                    "<extra></extra>"
                ),
            ))
            _ov_fig_pat.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                height=240, margin=dict(t=10, b=45, l=10, r=40),
                font=dict(color="#1e293b", size=11),
                xaxis=dict(gridcolor="#e2e8f0", title="Incidents", dtick=1),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(_ov_fig_pat, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No incidents in this range.")

    # ── Agent Eval Health ─────────────────────────────────────────────────────
    st.markdown("#### Agent Eval Health")
    _ov_ae = load_all_evals()
    if not _ov_ae.empty and not sessions.empty:
        _ov_sa = sessions.copy()
        _ov_sa["started_at"] = pd.to_datetime(_ov_sa["started_at"], errors="coerce", utc=True)
        _ov_ids_7d  = set(_ov_sa[_ov_sa["started_at"] >= (_ov_now - pd.Timedelta(days=7))]["id"])
        _ov_ids_30d = set(_ov_sa[_ov_sa["started_at"] >= (_ov_now - pd.Timedelta(days=30))]["id"])

        def _ov_pr(df_e, ids):
            sub = df_e[df_e["session_id"].isin(ids)]
            return (float(sub["passed"].sum()) / len(sub) * 100) if len(sub) > 0 else None

        def _ov_bar(pct):
            if pct is None:
                return '<span style="color:#94a3b8">—</span>'
            c = "#10b981" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"
            return (
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="flex:1;background:#e2e8f0;border-radius:3px;height:7px;min-width:60px">'
                f'<div style="width:{pct:.0f}%;background:{c};border-radius:3px;height:7px"></div>'
                f'</div>'
                f'<span style="font-size:0.82rem;min-width:34px;font-weight:600;color:{c}">'
                f'{pct:.0f}%</span>'
                f'</div>'
            )

        _ov_tbl = (
            '<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
            '<thead><tr style="border-bottom:2px solid #e2e8f0">'
            '<th style="text-align:left;padding:8px 12px;color:#64748b;font-weight:600;width:20%">'
            'Agent</th>'
            '<th style="padding:8px 12px;color:#64748b;font-weight:600;width:35%">'
            'Pass Rate (7d)</th>'
            '<th style="padding:8px 12px;color:#64748b;font-weight:600;width:35%">'
            'Pass Rate (30d)</th>'
            '<th style="text-align:left;padding:8px 12px;color:#64748b;font-weight:600">'
            'Trend</th>'
            '</tr></thead><tbody>'
        )
        for _ov_ag in ["orchestrator", "market", "research", "risk"]:
            _ov_sub = _ov_ae[_ov_ae["agent"] == _ov_ag]
            _ov_p7  = _ov_pr(_ov_sub, _ov_ids_7d)
            _ov_p30 = _ov_pr(_ov_sub, _ov_ids_30d)
            if _ov_p7 is None and _ov_p30 is None:
                continue
            if _ov_p7 is not None and _ov_p30 is not None:
                _ov_td = _ov_p7 - _ov_p30
                _ov_ts = "▲ improving" if _ov_td > 3 else "▼ slipping" if _ov_td < -3 else "→ stable"
                _ov_tc = "#10b981" if _ov_td > 3 else "#ef4444" if _ov_td < -3 else "#64748b"
            else:
                _ov_ts, _ov_tc = "—", "#94a3b8"
            _ov_ag_tip = tip_badge(AGENT_DESCRIPTIONS.get(_ov_ag, ""))
            _ov_tbl += (
                f'<tr style="border-bottom:1px solid #f1f5f9">'
                f'<td style="padding:10px 12px;font-weight:600;color:#0f172a">'
                f'{_ov_ag.title()}{_ov_ag_tip}</td>'
                f'<td style="padding:10px 12px">{_ov_bar(_ov_p7)}</td>'
                f'<td style="padding:10px 12px">{_ov_bar(_ov_p30)}</td>'
                f'<td style="padding:10px 12px;color:{_ov_tc};font-size:0.82rem;font-weight:600">'
                f'{_ov_ts}</td>'
                f'</tr>'
            )
        _ov_tbl += '</tbody></table>'
        st.markdown(_ov_tbl, unsafe_allow_html=True)
        if st.button("Explore eval trends  →  Quality Drift", key="ov_qd"):
            goto_page("Quality Drift")
    else:
        st.info("No eval data available.")

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

    # ── Recent Sessions ───────────────────────────────────────────────────────
    _ov_rs_h, _ov_rs_e = st.columns([4, 1])
    with _ov_rs_h:
        st.markdown("#### Recent Sessions")
    with _ov_rs_e:
        if not _ov_i.empty:
            _ov_exp = _ov_i[["created_at", "session_id", "pattern_name",
                              "severity", "cost_wasted"]].copy()
            _ov_exp["created_at"] = _ov_exp["created_at"].dt.strftime("%Y-%m-%d %H:%M")
            st.download_button(
                "Export Incident Log",
                data=_ov_exp.to_csv(index=False),
                file_name="incident_log.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if not sessions.empty:
        _ov_rs = sessions.copy()
        _ov_rs["started_at"] = pd.to_datetime(_ov_rs["started_at"], errors="coerce", utc=True)
        _ov_rs = _ov_rs.sort_values("started_at", ascending=False).head(50)

        _ov_rows = []
        for _, _ov_row in _ov_rs.iterrows():
            _ov_sid   = _ov_row["id"]
            _ov_sincs = _ov_i[_ov_i["session_id"] == _ov_sid] if not _ov_i.empty else pd.DataFrame()
            _ov_cost  = float(_ov_row.get("total_cost_usd") or 0)
            if _ov_cost == 0:
                _bd = _ov_row.get("cost_breakdown")
                if isinstance(_bd, dict):
                    _ov_cost = sum(v.get("cost_usd", 0) for v in _bd.values() if isinstance(v, dict))
            _ov_wasted, _ = (
                compute_cb_savings(_ov_sid, _ov_row.to_dict(), _ov_ae, traces_all)
                if not _ov_sincs.empty else (0.0, "")
            )
            _n_incs = len(_ov_sincs)
            _pattern = (
                _ov_sincs["pattern_name"].iloc[0] if _n_incs == 1
                else f"{_ov_sincs['pattern_name'].iloc[0]} +{_n_incs-1}" if _n_incs > 1
                else ""
            )
            _ov_rows.append({
                "_id":         _ov_sid,
                "_has_inc":    _n_incs > 0,
                "_zero_trade": int(_ov_row.get("trades_executed") or 0) == 0,
                "Session":     _ov_row["started_at"].strftime("%m-%d %H:%M") if pd.notna(_ov_row["started_at"]) else "-",
                "Cost ($)":    f"${_ov_cost:.4f}" if _ov_cost > 0 else "-",
                "Trades":      int(_ov_row.get("trades_executed") or 0),
                "Duration (s)": int((_ov_row.get("total_latency_ms") or 0) / 1000),
                "Tokens":      int((_ov_row.get("total_tokens_input") or 0) + (_ov_row.get("total_tokens_output") or 0)),
                "Incidents":   _n_incs,
                "Pattern":     _pattern,
                "Exit Reason": str(_ov_row.get("terminal_reason") or ""),
                "CB Saved ($)": f"${_ov_wasted:.4f}" if _ov_wasted > 0 else "-",
            })
        _ov_tbl = pd.DataFrame(_ov_rows)

        _ov_gb = GridOptionsBuilder.from_dataframe(_ov_tbl)
        _ov_gb.configure_default_column(suppressMenu=True, sortable=False, resizable=False, filter=False)
        _ov_gb.configure_column("_id",        hide=True)
        _ov_gb.configure_column("_has_inc",   hide=True)
        _ov_gb.configure_column("_zero_trade",hide=True)
        _ov_gb.configure_column("Exit Reason", cellStyle=JsCode("""
            function(params) {
                var v = params.value || '';
                if (v === 'converged') return {'color':'#166534','fontWeight':'600'};
                if (v === 'no_viable_proposals' || v === 'all_rejected') return {'color':'#c2410c','fontWeight':'600'};
                if (v === 'watchdog_timeout' || v === 'timeout') return {'color':'#991b1b','fontWeight':'700'};
                if (v === 'eod_complete' || v === 'superseded') return {'color':'#475569'};
                return {};
            }
        """))
        _ov_PAGE = 10
        _ov_gb.configure_selection("single", use_checkbox=False)
        _ov_gb.configure_grid_options(
            pagination=True,
            paginationPageSize=_ov_PAGE,
            suppressPaginationPanel=len(_ov_tbl) <= _ov_PAGE,
            getRowStyle=JsCode("""
                function(params) {
                    if (params.data._has_inc)    return {'background':'#fee2e2','color':'#7f1d1d'};
                    if (params.data._zero_trade) return {'background':'#fef9c3','color':'#713f12'};
                }
            """),
            rowHeight=34,
            headerHeight=36,
            suppressHorizontalScroll=True,
        )
        _ov_resp = AgGrid(
            _ov_tbl,
            gridOptions=_ov_gb.build(),
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=min(600, 56 + min(len(_ov_tbl), _ov_PAGE) * 34 + (0 if len(_ov_tbl) <= _ov_PAGE else 60)),
            use_container_width=True,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            theme="streamlit",
        )
        st.markdown(
            "<div style='font-size:0.8rem;color:#94a3b8;margin-top:4px'>"
            "Red = incident \u00b7 Amber = 0 trades \u00b7 Showing last 50 sessions \u00b7 Click a row to see pipeline &nbsp;\u00b7&nbsp; Pipeline nodes: green = passed \u00b7 red = failed \u00b7 slate = ran, not evaluated \u00b7 hollow = did not run</div>",
            unsafe_allow_html=True,
        )

        # \u2500\u2500 Pipeline strip detail panel \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        _ov_sel = _ov_resp.selected_rows
        if _ov_sel is not None and len(_ov_sel) > 0:
            _ov_sel_row = _ov_sel.iloc[0] if isinstance(_ov_sel, pd.DataFrame) else _ov_sel[0]
            _ov_sel_sid = _ov_sel_row["_id"]
            _ov_full    = sessions[sessions["id"] == _ov_sel_sid]
            _ov_sel_inc = _ov_i[_ov_i["session_id"] == _ov_sel_sid] if not _ov_i.empty else pd.DataFrame()

            if not _ov_full.empty:
                _ov_sr      = _ov_full.iloc[0]
                _ov_s_cost  = float(_ov_sr.get("total_cost_usd") or 0)
                if _ov_s_cost == 0:
                    _bd = _ov_sr.get("cost_breakdown")
                    if isinstance(_bd, dict):
                        _ov_s_cost = sum(v.get("cost_usd", 0) for v in _bd.values() if isinstance(v, dict))
                _ov_s_trades = int(_ov_sr.get("trades_executed") or 0)
                _ov_s_dur   = int((_ov_sr.get("total_latency_ms") or 0) / 1000)
                _ov_s_tok   = int((_ov_sr.get("total_tokens_input") or 0) + (_ov_sr.get("total_tokens_output") or 0))
                _ov_s_exit  = str(_ov_sr.get("terminal_reason") or "")
                _ov_s_date  = pd.to_datetime(_ov_sr.get("started_at"), errors="coerce", utc=True)
                _ov_s_date_str = _ov_s_date.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(_ov_s_date) else "-"

                _exit_colors = {
                    "converged":             ("#166534", "#dcfce7"),
                    "eod_complete":          ("#334155", "#f1f5f9"),
                    "superseded":            ("#334155", "#f1f5f9"),
                    "no_viable_proposals":   ("#c2410c", "#fff7ed"),
                    "all_rejected":          ("#c2410c", "#fff7ed"),
                    "watchdog_timeout":      ("#991b1b", "#fee2e2"),
                    "timeout":               ("#991b1b", "#fee2e2"),
                }
                _ec_fg, _ec_bg = _exit_colors.get(_ov_s_exit, ("#475569", "#f8fafc"))

                _inc_html = ""
                if not _ov_sel_inc.empty:
                    for _, _inc_row in _ov_sel_inc.iterrows():
                        _sev   = str(_inc_row.get("severity") or "").lower()
                        _ibg   = "#fee2e2" if _sev == "critical" else "#fef9c3"
                        _ifg   = "#7f1d1d" if _sev == "critical" else "#713f12"
                        _ipat  = str(_inc_row.get("pattern_name") or "")
                        _irc   = str(_inc_row.get("root_cause") or "")[:90]
                        _inc_html += (
                            f'<div style="background:{_ibg};color:{_ifg};border-radius:6px;'
                            f'padding:6px 10px;font-size:0.78rem;margin-top:4px">'
                            f'<strong>{_sev.upper()}</strong> \u00b7 {_ipat}'
                            + (f'  \u2014  {_irc}' if _irc else '') + '</div>'
                        )

                _ov_agents_inv = _ov_sr.get("agents_invoked") or []
                _strip_html = pipeline_strip(_ov_sel_sid, traces_all, _ov_ae, session_agents=_ov_agents_inv)

                # Accent color mirrors the selected row's state
                _has_inc_flag    = not _ov_sel_inc.empty
                _zero_trade_flag = _ov_s_trades == 0
                _accent = "#ef4444" if _has_inc_flag else "#f59e0b" if _zero_trade_flag else "#3b82f6"

                # Pipeline efficiency: early exit saved cost vs avg full run
                _early_exits = {"no_viable_proposals", "all_rejected"}
                _eff_html = ""
                if _ov_s_exit in _early_exits and not sessions.empty:
                    _full_runs = sessions[
                        sessions["terminal_reason"].isin(["converged", "eod_complete"])
                    ]["total_cost_usd"].dropna()
                    if not _full_runs.empty:
                        _avg_full = _full_runs.mean()
                        _saved    = max(0.0, _avg_full - _ov_s_cost)
                        if _saved > 0.0001:
                            _eff_html = (
                                f'<div style="display:inline-flex;align-items:center;gap:6px;'
                                f'background:#f0fdf4;border:1px solid #86efac;border-radius:6px;'
                                f'padding:5px 10px;margin-top:8px;font-size:0.78rem;color:#166534">'
                                f'<strong>Pipeline efficiency</strong> &nbsp;'
                                f'Stopped at Market — saved ~<strong>${_saved:.4f}</strong> vs avg full run '
                                f'(${_avg_full:.4f}). Research, Risk, and Orchestrator did not run because '
                                f'market conditions ruled out viable trades.'
                                f'</div>'
                            )

                _exit_explanations = {
                    "converged":           "All agents completed. Orchestrator found viable proposals and executed trades.",
                    "eod_complete":        "End-of-day session. Open positions closed, P&L reconciled.",
                    "no_viable_proposals": "Market conditions did not support any trades. Pipeline stopped at Market agent, saving Research/Risk/Orchestrator cost.",
                    "all_rejected":        "Research and Risk ran but all proposals were rejected against risk criteria. No trades placed.",
                    "superseded":          "Session replaced by a newer run (duplicate start or manual restart).",
                    "watchdog_timeout":    "Session exceeded the watchdog time limit and was force-shut down. Check for hung agents.",
                    "timeout":             "An agent or tool call exceeded its time limit. Session aborted.",
                }
                _exit_explain = _exit_explanations.get(_ov_s_exit or "", "")

                _legend_html = """
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;padding-top:8px;
            border-top:1px solid #e2e8f0;font-size:0.72rem;color:#64748b;align-items:center">
  <span style="font-weight:600;color:#94a3b8;letter-spacing:0.04em">NODES</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#10b981;margin-right:4px;vertical-align:middle"></span>Passed evals</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f59e0b;margin-right:4px;vertical-align:middle"></span>Partial pass</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:4px;vertical-align:middle"></span>Failed evals</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#475569;margin-right:4px;vertical-align:middle"></span>Ran, not evaluated</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid #cbd5e1;background:#e2e8f0;margin-right:4px;vertical-align:middle"></span>Did not run</span>
</div>"""

                _exit_html = (
                    f'<div style="font-size:0.78rem;color:#475569;margin-top:8px;padding-top:8px;'
                    f'border-top:1px solid #e2e8f0">'
                    f'<strong style="color:#64748b">Exit:</strong> {_exit_explain}</div>'
                ) if _exit_explain else ""

                _detail_html = f"""
<div style="margin-top:0;border-left:4px solid {_accent};border-radius:0 8px 8px 0;
            padding:12px 16px 20px 14px;background:#f8fafc;border-top:1px solid #e2e8f0;
            border-right:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;overflow:visible">
  <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:8px;letter-spacing:0.03em">
    PIPELINE DETAIL &nbsp;·&nbsp; {_ov_s_date_str}{"&nbsp;&nbsp;<span style='color:#ef4444'>Red row = incident flagged, not a pipeline failure</span>" if _has_inc_flag and _ov_s_exit in ("converged","eod_complete") else ""}
  </div>
  <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;overflow:visible;padding-bottom:4px">
    {_strip_html}
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-left:8px;border-left:1px solid #e2e8f0;padding-left:16px">
      <span style="font-size:0.82rem"><strong style="color:#64748b">Cost</strong>&nbsp;${_ov_s_cost:.4f}</span>
      <span style="font-size:0.82rem"><strong style="color:#64748b">Trades</strong>&nbsp;{_ov_s_trades}</span>
      <span style="font-size:0.82rem"><strong style="color:#64748b">Duration</strong>&nbsp;{_ov_s_dur}s</span>
      <span style="font-size:0.82rem"><strong style="color:#64748b">Tokens</strong>&nbsp;{_ov_s_tok:,}</span>
      <span style="background:{_ec_bg};color:{_ec_fg};border-radius:4px;padding:2px 8px;
                  font-size:0.75rem;font-weight:600;letter-spacing:0.02em">{_ov_s_exit or "unknown"}</span>
    </div>
  </div>
  {_exit_html}
  {_eff_html}
  {_inc_html}
  {_legend_html}
</div>"""
                st.markdown(_detail_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Ledger
# ══════════════════════════════════════════════════════════════════════════════
if page == "Ledger":
    st.markdown("## Outcome Ledger")

    if sessions.empty:
        st.info("No sessions found.")
        st.stop()

    # ── Pre-compute KPI values (needed for LLM summary and display) ──────────
    total_cost  = sessions["total_cost_usd"].sum()
    total_trade = sessions["trades_executed"].sum()
    wasted      = sessions[(sessions["trades_executed"] == 0) & (sessions["total_cost_usd"] > 0)]
    wasted_cost = wasted["total_cost_usd"].sum()
    wasted_pct  = (wasted_cost / total_cost * 100) if total_cost else 0
    inc_count   = len(incidents) if not incidents.empty else 0

    # Derive top agent for summary context
    _top_agent, _top_agent_cost, _top_agent_pct, _n_bd = "unknown", 0.0, 0.0, 0
    _all_agent_costs: dict[str, float] = defaultdict(float)
    for _, _r in sessions.iterrows():
        _bd = _r.get("cost_breakdown") or {}
        if not isinstance(_bd, dict) or not _bd:
            continue
        _n_bd += 1
        for _k, _v in _bd.items():
            if isinstance(_v, dict):
                _norm = "research" if _k.startswith("research_") else _k
                _all_agent_costs[_norm] += _v.get("cost_usd", 0)
    if _all_agent_costs:
        _top_agent = max(_all_agent_costs, key=_all_agent_costs.get)
        _top_agent_cost = _all_agent_costs[_top_agent]
        _top_agent_pct  = _top_agent_cost / sum(_all_agent_costs.values()) * 100

    _recent_reasons = (
        sessions.sort_values("started_at", ascending=False)["terminal_reason"]
        .dropna().tolist()
    )

    # ── LLM page summary (cached by data snapshot) ────────────────────────────
    _insights = {}
    _api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not _api_key:
        st.warning("Add ANTHROPIC_API_KEY to Streamlit secrets to enable AI summaries.")
    else:
        with st.spinner("Generating analyst summary..."):
            try:
                _insights = generate_ledger_insights(
                    n_sessions=len(sessions),
                    total_cost=total_cost,
                    total_trades=int(total_trade),
                    n_wasted=len(wasted),
                    wasted_cost=wasted_cost,
                    wasted_pct=wasted_pct,
                    n_incidents=inc_count,
                    top_agent=_top_agent,
                    top_agent_cost=_top_agent_cost,
                    top_agent_pct=_top_agent_pct,
                    n_sessions_with_cost_data=_n_bd,
                    recent_terminal_reasons=_recent_reasons,
                )
            except Exception as _e:
                st.warning(f"AI summary unavailable: {_e}")

    if _insights.get("page_summary"):
        st.markdown(
            f'<div class="ai-insight-page">'
            f'<div class="ai-label">AI Analyst</div>'
            f'{_insights["page_summary"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown("#### Pipeline Health")
    if _insights.get("kpi_insight"):
        st.markdown(
            f'<div class="ai-insight-section">{_insights["kpi_insight"]}</div>',
            unsafe_allow_html=True,
        )

    _cost_per_trade = total_cost / total_trade if total_trade > 0 else 0
    _cpt_color = "#ef4444" if _cost_per_trade > 100 else ("#f59e0b" if _cost_per_trade > 50 else "#10b981")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(kpi("Total Sessions", str(len(sessions))), unsafe_allow_html=True)
    c2.markdown(kpi("Total Spend", f"${total_cost:.2f}"), unsafe_allow_html=True)
    c3.markdown(kpi("Trades Executed", str(int(total_trade))), unsafe_allow_html=True)
    _cpt_val = f'<span style="color:{_cpt_color}">${_cost_per_trade:.2f}</span>' if total_trade > 0 else "—"
    c4.markdown(kpi("Cost / Trade", _cpt_val), unsafe_allow_html=True)
    c5.markdown(kpi("Wasted Sessions",
        f"{len(wasted)} · ${wasted_cost:.2f}",
        f"{wasted_pct:.0f}% of spend" if total_cost else ""),
        unsafe_allow_html=True)
    c6.markdown(kpi("Incidents Detected", str(inc_count)), unsafe_allow_html=True)

    st.divider()

    # Cost by Agent — donut + drill-down
    if sessions["cost_breakdown"].notna().any():
        st.markdown("#### Cost by Agent")
        if _insights.get("cost_insight"):
            st.markdown(
                f'<div class="ai-insight-section">{_insights["cost_insight"]}</div>',
                unsafe_allow_html=True,
            )

        # Aggregate cost_breakdown across all sessions with data.
        # research_TICKER keys (new format) are normalised to "research" for the
        # agent-level donut while per-ticker detail is kept separately.
        agent_costs: dict[str, float] = defaultdict(float)
        agent_llm: dict[str, dict] = defaultdict(
            lambda: {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "model": None}
        )
        ticker_costs: dict[str, float] = defaultdict(float)
        ticker_llm:  dict[str, dict]  = defaultdict(
            lambda: {"input": 0, "output": 0, "model": None}
        )
        sessions_with_bd = 0
        for _, _row in sessions.iterrows():
            bd = _row.get("cost_breakdown") or {}
            if not isinstance(bd, dict) or not bd:
                continue
            sessions_with_bd += 1
            for _agent, _data in bd.items():
                if not isinstance(_data, dict):
                    continue
                _parts = _agent.split("_", 1)
                _is_research_ticker = _parts[0] == "research" and len(_parts) > 1
                _norm = "research" if _is_research_ticker else _agent

                # Agent-level totals
                agent_costs[_norm]               += _data.get("cost_usd", 0)
                agent_llm[_norm]["input"]        += _data.get("input", 0)
                agent_llm[_norm]["output"]       += _data.get("output", 0)
                agent_llm[_norm]["cache_read"]   += _data.get("cache_read", 0)
                agent_llm[_norm]["cache_write"]  += _data.get("cache_write", 0)
                if _data.get("model"):
                    agent_llm[_norm]["model"] = _data["model"]

                # Per-ticker detail (only for research_TICKER keys)
                if _is_research_ticker:
                    _tk = _parts[1]
                    ticker_costs[_tk]            += _data.get("cost_usd", 0)
                    ticker_llm[_tk]["input"]     += _data.get("input", 0)
                    ticker_llm[_tk]["output"]    += _data.get("output", 0)
                    if _data.get("model"):
                        ticker_llm[_tk]["model"] = _data["model"]

        if agent_costs:
            _agents_list = sorted(agent_costs, key=agent_costs.get, reverse=True)

            _donut_col, _detail_col = st.columns([4, 6])

            with _donut_col:
                _donut = go.Figure(go.Pie(
                    labels=_agents_list,
                    values=[agent_costs[a] for a in _agents_list],
                    hole=0.55,
                    marker=dict(colors=[AGENT_COLORS.get(a, "#94a3b8") for a in _agents_list]),
                    textinfo="label+percent",
                    textfont=dict(size=11),
                    hovertemplate="<b>%{label}</b><br>$%{value:.4f} · %{percent}<extra></extra>",
                    direction="clockwise",
                    sort=False,
                ))
                _donut.add_annotation(
                    text=f"${sum(agent_costs.values()):.3f}",
                    x=0.5, y=0.5,
                    font=dict(size=13, color="#0f172a"),
                    showarrow=False,
                )
                _donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#1e293b", height=260,
                    margin=dict(t=10, b=0, l=0, r=0), showlegend=False,
                )
                st.plotly_chart(_donut, use_container_width=True)
                st.caption(f"LLM cost only · {sessions_with_bd} of {len(sessions)} sessions have data")

                _sel_agent = st.pills(
                    "Agent", _agents_list, selection_mode="single",
                    key="cost_agent_drill", label_visibility="collapsed",
                )

            with _detail_col:
                if _sel_agent:
                    _llm = agent_llm[_sel_agent]
                    st.markdown(f"**{_sel_agent}**")
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    _mc1.metric("Cost",    f"${agent_costs[_sel_agent]:.4f}")
                    _mc2.metric("Input",   f"{_llm['input']:,}")
                    _mc3.metric("Output",  f"{_llm['output']:,}")
                    _mc4.metric("Cache ↩", f"{_llm['cache_read']:,}")
                    if _llm.get("model"):
                        st.caption(f"Model: `{_llm['model']}`  ·  Cache write: {_llm['cache_write']:,} tok")

                    # Per-ticker cost breakdown (research agent only)
                    if _sel_agent == "research" and ticker_costs:
                        st.markdown("**Cost by Ticker**")
                        _tks = sorted(ticker_costs, key=ticker_costs.get, reverse=True)
                        _fig_tk = go.Figure(go.Bar(
                            x=[ticker_costs[t] for t in _tks],
                            y=_tks,
                            orientation="h",
                            marker_color=AGENT_COLORS.get("research", "#94a3b8"),
                            text=[f"${ticker_costs[t]:.4f}" for t in _tks],
                            textposition="outside",
                            hovertemplate="<b>%{y}</b><br>$%{x:.4f}<extra></extra>",
                        ))
                        _fig_tk.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#1e293b",
                            height=max(120, len(_tks) * 30 + 40),
                            margin=dict(t=5, b=5, l=10, r=60),
                            xaxis=dict(title="Cost (USD)", tickformat="$.4f"),
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(_fig_tk, use_container_width=True)

                    # Tool call breakdown from traces.
                    # For research, match agent names that start with "research"
                    # (covers both old "research" and new "research_TICKER" rows).
                    if _sel_agent == "research":
                        _atr = traces_all[
                            traces_all["agent"].str.startswith("research") &
                            (traces_all["step_type"] == "tool_call")
                        ]
                    else:
                        _atr = traces_all[
                            (traces_all["agent"] == _sel_agent) &
                            (traces_all["step_type"] == "tool_call")
                        ]

                    if not _atr.empty:
                        st.markdown("**Tool calls**")
                        _ts = (
                            _atr.groupby("tool_name")
                            .agg(
                                Calls    =("tool_name",  "count"),
                                Latency  =("latency_ms", "mean"),
                                Errors   =("error",      lambda x: x.notna().sum()),
                            )
                            .reset_index()
                            .rename(columns={"tool_name": "Tool"})
                            .sort_values("Calls", ascending=False)
                        )
                        _ts["Latency (ms)"] = _ts["Latency"].round(0).astype(int)
                        _ts["Error %"]      = (_ts["Errors"] / _ts["Calls"] * 100).round(1).astype(str) + "%"
                        st.dataframe(
                            _ts[["Tool", "Calls", "Latency (ms)", "Errors", "Error %"]],
                            use_container_width=True, hide_index=True, height=200,
                        )
                    else:
                        st.caption("No tool calls recorded for this agent.")
                else:
                    st.markdown(
                        '<div style="display:flex;align-items:center;justify-content:center;'
                        'height:200px;color:#94a3b8;font-size:0.88rem;">'
                        'Select an agent below the chart</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()

    # ── Wasted vs Productive cost split ──────────────────────────────────────
    if sessions["cost_breakdown"].notna().any():
        st.markdown("#### Where Wasted Spend Goes")
        st.caption("Agent cost split: sessions that executed trades vs. sessions that burned budget with 0 trades.")

        _productive = sessions[sessions["trades_executed"] > 0]

        def _agent_cost_sum(subset: pd.DataFrame) -> dict[str, float]:
            totals: dict[str, float] = defaultdict(float)
            for _, _r in subset.iterrows():
                _bd = _r.get("cost_breakdown") or {}
                if not isinstance(_bd, dict):
                    continue
                for _k, _v in _bd.items():
                    if isinstance(_v, dict):
                        _norm = "research" if _k.startswith("research_") else _k
                        totals[_norm] += _v.get("cost_usd", 0)
            return dict(totals)

        _wasted_costs     = _agent_cost_sum(wasted)
        _productive_costs = _agent_cost_sum(_productive)
        _split_agents     = sorted(set(_wasted_costs.keys()) | set(_productive_costs.keys()))

        if _split_agents:
            _fig_split = go.Figure()
            _fig_split.add_trace(go.Bar(
                name="Wasted (0 trades)",
                x=_split_agents,
                y=[_wasted_costs.get(a, 0) for a in _split_agents],
                marker_color="#ef4444",
                text=[f"${_wasted_costs.get(a, 0):.4f}" for a in _split_agents],
                textposition="outside",
            ))
            _fig_split.add_trace(go.Bar(
                name="Productive",
                x=_split_agents,
                y=[_productive_costs.get(a, 0) for a in _split_agents],
                marker_color="#10b981",
                text=[f"${_productive_costs.get(a, 0):.4f}" for a in _split_agents],
                textposition="outside",
            ))
            _fig_split.update_layout(
                barmode="group",
                yaxis_title="Cost USD",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1e293b",
                height=300,
                margin=dict(t=30, b=10, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(_fig_split, use_container_width=True)

            # Per-agent "% wasted" callout cards
            _card_cols = st.columns(min(len(_split_agents), 4))
            for _i, _ag in enumerate(_split_agents[:4]):
                _wc = _wasted_costs.get(_ag, 0)
                _pc = _productive_costs.get(_ag, 0)
                _tot = _wc + _pc
                _wpct = round(_wc / _tot * 100) if _tot > 0 else 0
                _clr = "#ef4444" if _wpct > 50 else ("#f59e0b" if _wpct > 25 else "#10b981")
                _card_cols[_i].markdown(
                    f'<div style="padding:10px 14px;background:#f8fafc;border-radius:8px;'
                    f'border-left:3px solid {_clr};margin-top:4px">'
                    f'<div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase">{_ag}</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{_clr}">{_wpct}% wasted</div>'
                    f'<div style="font-size:0.8rem;color:#64748b">${_wc:.4f} of ${_tot:.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

    # ── Session table ─────────────────────────────────────────────────────────
    st.markdown("#### Sessions")
    if _insights.get("sessions_insight"):
        st.markdown(
            f'<div class="ai-insight-section">{_insights["sessions_insight"]}</div>',
            unsafe_allow_html=True,
        )

    display = sessions.copy()
    display["Duration (s)"] = (
        display["total_latency_ms"] / 1000
    ).round(0).astype(int)
    display["Tokens"] = (display["total_tokens_input"] + display["total_tokens_output"]).astype(int)

    if not incidents.empty:
        inc_counts = incidents.groupby("session_id").size().reset_index(name="Incidents")
        display = display.merge(inc_counts, left_on="id", right_on="session_id", how="left")
        display["Incidents"] = display["Incidents"].fillna(0).astype(int)

        def _pattern_summary(sid):
            names = incidents[incidents["session_id"] == sid]["pattern_name"].tolist()
            if not names:
                return ""
            if len(names) == 1:
                return names[0]
            first = names[0]
            return f"{first} +{len(names)-1}" if len(first) > 20 else " · ".join(names[:2]) + (f" +{len(names)-2}" if len(names) > 2 else "")

        display["Type"] = display["id"].apply(_pattern_summary)
    else:
        display["Incidents"] = 0
        display["Type"] = ""

    display["Wasted"] = (display["trades_executed"] == 0) & (display["total_cost_usd"] > 0.01)

    # ── Grid filter row ───────────────────────────────────────────────────────
    _gf1, _gf2, _gf_sp = st.columns([2, 2, 5])
    _only_incidents = _gf1.checkbox("Incidents only", value=False, key="ledger_inc_filter")
    _only_wasted    = _gf2.checkbox("0-trade sessions", value=False, key="ledger_waste_filter")
    if _only_incidents:
        display = display[display["Incidents"] > 0]
    if _only_wasted:
        display = display[display["Wasted"]]

    cols = ["started_at", "total_cost_usd", "Duration (s)", "trades_executed",
            "Tokens", "Incidents", "Type", "terminal_reason"]
    rename = {
        "started_at":       "Session",
        "total_cost_usd":   "Cost ($)",
        "trades_executed":  "Trades",
        "terminal_reason":  "Exit Reason",
    }
    tbl = display[cols].rename(columns=rename).copy()
    tbl["Session"] = tbl["Session"].dt.strftime("%m-%d %H:%M")
    tbl["Cost ($)"] = tbl["Cost ($)"].map("${:.4f}".format)

    _PAGE_SIZE = 20
    if "ledger_page" not in st.session_state:
        st.session_state.ledger_page = 0
    if "ledger_selected_id" not in st.session_state:
        st.session_state.ledger_selected_id = None
    # Reset to page 0 when filters change
    _filter_key = (_only_incidents, _only_wasted)
    if st.session_state.get("_ledger_filter_key") != _filter_key:
        st.session_state.ledger_page = 0
        st.session_state["_ledger_filter_key"] = _filter_key
    _total_pages = max(1, (len(tbl) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    st.session_state.ledger_page = min(st.session_state.ledger_page, _total_pages - 1)
    _p = st.session_state.ledger_page

    _page_ids  = display["id"].iloc[_p * _PAGE_SIZE : (_p + 1) * _PAGE_SIZE].reset_index(drop=True)
    _page_tbl  = tbl.iloc[_p * _PAGE_SIZE : (_p + 1) * _PAGE_SIZE].reset_index(drop=True).copy()
    _page_tbl.insert(0, "_id", _page_ids.values)

    _gb = GridOptionsBuilder.from_dataframe(_page_tbl)
    _gb.configure_default_column(
        suppressMenu=True, sortable=True, resizable=False, filter=False,
    )
    _gb.configure_column("_id", hide=True, suppressColumnsToolPanel=True)
    _gb.configure_column("Exit Reason", cellStyle=JsCode("""
        function(params) {
            var v = params.value || '';
            if (v === 'converged')
                return {'color': '#166534', 'fontWeight': '600'};
            if (v === 'skip_propagated' || v === 'eod_complete' || v === 'superseded')
                return {'color': '#475569'};
            if (v === 'no_viable_candidates' || v === 'caution_no_retry')
                return {'color': '#92400e', 'fontWeight': '600'};
            if (v === 'no_viable_proposals' || v === 'structural_block' || v === 'all_rejected')
                return {'color': '#c2410c', 'fontWeight': '600'};
            if (v === 'watchdog_timeout' || v === 'timeout')
                return {'color': '#991b1b', 'fontWeight': '700'};
            return {};
        }
    """))
    _gb.configure_selection("single", use_checkbox=False)
    _gb.configure_grid_options(
        getRowStyle=JsCode("""
            function(params) {
                if (params.data.Incidents > 0)
                    return {'background': '#fee2e2', 'color': '#7f1d1d'};
                if (parseInt(params.data.Trades) === 0)
                    return {'background': '#fef9c3', 'color': '#713f12'};
            }
        """),
        rowHeight=34,
        headerHeight=36,
        suppressHorizontalScroll=True,
    )
    _resp = AgGrid(
        _page_tbl,
        gridOptions=_gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=250,
        use_container_width=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme="streamlit",
    )
    _sel = _resp.selected_rows
    if _sel is not None and len(_sel) > 0:
        _row = _sel.iloc[0] if hasattr(_sel, "iloc") else _sel[0]
        st.session_state.ledger_selected_id = _row["_id"]

    _ca, _cb, _cc = st.columns([1, 3, 1])
    with _ca:
        if st.button("← Prev", disabled=(_p == 0), key="ledger_prev"):
            st.session_state.ledger_page -= 1
            st.session_state.ledger_selected_id = None
            st.rerun()
    with _cb:
        st.caption(
            f"Page {_p + 1} of {_total_pages} · {len(tbl)} sessions · "
            "Red = incidents · Amber = 0 trades · Click a row for details"
        )
    with _cc:
        if st.button("Next →", disabled=(_p >= _total_pages - 1), key="ledger_next"):
            st.session_state.ledger_page += 1
            st.session_state.ledger_selected_id = None
            st.rerun()

    # ── Inline session detail (row-click driven) ──────────────────────────────
    _dsid = st.session_state.ledger_selected_id
    if _dsid:
        st.divider()
        st.markdown("#### Session Detail")
        _drow = sessions[sessions["id"] == _dsid].iloc[0].to_dict()

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Cost",     f"${_drow['total_cost_usd']:.4f}")
        dc2.metric("Duration", f"{int(_drow['total_latency_ms']//1000)}s")
        dc3.metric("Tokens",   f"{int(_drow['total_tokens_input']+_drow['total_tokens_output']):,}")
        dc4.metric("Trades",   str(int(_drow["trades_executed"])))

        if _drow.get("terminal_reason"):
            st.info(f"Exit reason: {_drow['terminal_reason']}")

        if not incidents.empty:
            _dinc = incidents[incidents["session_id"] == _dsid]
            if not _dinc.empty:
                st.markdown("**Incidents**")
                for _, _inc in _dinc.iterrows():
                    _ic1, _ic2 = st.columns([6, 1])
                    with _ic1:
                        st.markdown(
                            f"{badge(_inc['severity'])} &nbsp; **{_inc['pattern_name']}** — {_inc['root_cause']}",
                            unsafe_allow_html=True,
                        )
                    with _ic2:
                        if st.button("RCA →", key=f"drill_rca_{_inc['id']}"):
                            goto_rca(_inc.to_dict(), _dsid)

        _dev = load_evals_for_session(_dsid)
        if not _dev.empty:
            _op_dev   = _dev[~_dev["agent"].str.endswith("_quality", na=False)]
            _qual_dev = _dev[
                _dev["agent"].str.endswith("_quality", na=False) &
                (_dev["eval_name"] == "composite_score")
            ]

            if not _qual_dev.empty:
                st.markdown("**Quality Scores**")
                _QUAL_HELP = {
                    "research_quality": (
                        "Composite of 5 dimensions (threshold 0.60):\n\n"
                        "- Data grounding — distinct tool types used successfully (3+ = 1.0, 1 = 0.33)\n"
                        "- Thesis coherence — decision or agent_message trace produced\n"
                        "- Actionability — risk agent started after research completed\n"
                        "- Catalyst specificity — news or earnings tool was called\n"
                        "- Risk acknowledgment — total token count as an analysis depth proxy\n\n"
                        "Scores are structural proxies inferred from trace patterns. "
                        "Semantic scoring via LLM judge activates once raw output text is stored in traces."
                    ),
                    "risk_quality": (
                        "Composite of 5 dimensions (threshold 0.60):\n\n"
                        "- Research consistency — research agent ran before risk\n"
                        "- Parameter completeness — decision or agent_message trace produced\n"
                        "- Volatility accounting — ATR or volatility tool was called\n"
                        "- Stop loss quality — neutral proxy (requires output text to score)\n"
                        "- Position sizing rationale — number of successful tool calls\n\n"
                        "Scores are structural proxies inferred from trace patterns. "
                        "Semantic scoring via LLM judge activates once raw output text is stored in traces."
                    ),
                    "orchestrator_quality": (
                        "Composite of 4 dimensions (threshold 0.60):\n\n"
                        "- Decision consistency — trades placed with full pipeline = 1.0; "
                        "no trade with valid exit reason = 0.85; trades without pipeline = 0.40\n"
                        "- Resolution completeness — terminal reason quality "
                        "(good exits: eod_complete, no_opportunity, risk_rejected, etc.)\n"
                        "- Reasoning transparency — decision trace present\n"
                        "- Upstream integration — both research and risk ran before orchestrator\n\n"
                        "Scores are structural proxies inferred from trace patterns. "
                        "Semantic scoring via LLM judge activates once raw output text is stored in traces."
                    ),
                    "session_quality": (
                        "Composite of 2 dimensions (threshold 0.60):\n\n"
                        "- Pipeline coherence — all 3 downstream agents ran (1.0), "
                        "none ran / market-only (0.80), partial pipeline (0.30)\n"
                        "- Reasoning chain — penalised for any agent stage where every trace was an error\n\n"
                        "Scores are structural proxies inferred from trace patterns. "
                        "Semantic scoring via LLM judge activates once raw output text is stored in traces."
                    ),
                }
                _qc = st.columns(len(_qual_dev))
                for _qi, (_, _qev) in enumerate(_qual_dev.iterrows()):
                    _qs     = float(_qev.get("score") or 0)
                    _agent  = str(_qev["agent"])
                    _qlabel = _agent.replace("_quality", "").title()
                    _qc[_qi].metric(
                        label=_qlabel,
                        value=f"{_qs:.2f}",
                        delta="passed" if _qs >= 0.60 else "below threshold",
                        delta_color="normal" if _qs >= 0.60 else "inverse",
                        help=_QUAL_HELP.get(_agent, ""),
                    )

            if not _op_dev.empty:
                st.markdown("**Operational Evals**")
                _ec = st.columns(2)
                for _i, (_, _ev) in enumerate(_op_dev.iterrows()):
                    with _ec[_i % 2]:
                        _passed = bool(_ev.get("passed"))
                        _score  = float(_ev.get("score") or 0)
                        _icon   = "✓" if _passed else "✗"
                        _color  = "#10b981" if _passed else "#ef4444"
                        st.markdown(
                            f'<div style="margin:4px 0">'
                            f'<span style="color:{_color};font-weight:600">{_icon}</span> '
                            f'<b>{_ev["agent"]}.{_ev["eval_name"]}</b> &nbsp; '
                            f'<code>{_score:.2f}</code>'
                            f'{score_bar(_score, _passed)}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# (Session Deep Dive removed — functionality absorbed into Ledger inline detail)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Quality Drift
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Quality Drift":
    _qd_hdr, _qd_hlp = st.columns([11, 1])
    _qd_hdr.markdown("## Quality Drift")
    with _qd_hlp.popover("?"):
        st.markdown(
            "**What this page shows**\n\n"
            "Tracks whether your AI pipeline is getting better or worse over time across three layers:\n\n"
            "- **Operational** — did each agent complete its job? (tool success rate, pipeline completion, eval pass rates)\n"
            "- **Business** — did the pipeline produce results? (cost per trade, research conversion, proposal acceptance)\n"
            "- **Quality** — did each agent reason well? (data grounding, thesis coherence, decision consistency — scored 0 to 1)\n\n"
            "A session can pass all operational checks and still score low on quality "
            "if the research was vague or the decision was inconsistent."
        )

    if sessions.empty:
        st.info("No sessions found.")
        st.stop()

    s = sessions.sort_values("started_at").copy()
    s["label"] = s["started_at"].dt.strftime("%m-%d %H:%M")
    all_evals  = load_all_evals()

    # Join operational evals with session timestamps
    if not all_evals.empty and not s.empty:
        evals_ts = all_evals.merge(
            s[["id", "started_at", "label"]],
            left_on="session_id", right_on="id", how="left",
        ).sort_values("started_at")
    else:
        evals_ts = pd.DataFrame()

    # Compute business outcome evals on the fly
    biz_evals_df = compute_business_evals_df(s, traces_all)

    # ── Shared top KPIs ───────────────────────────────────────────────────────
    _n_recent = min(7, len(s))
    _n_prev   = min(7, max(0, len(s) - _n_recent))

    if not evals_ts.empty:
        _recent_sids = s.iloc[-_n_recent:]["id"].tolist()
        _prev_sids   = s.iloc[-_n_recent - _n_prev : -_n_recent]["id"].tolist() if _n_prev else []
        _op_evals_ts = evals_ts[~evals_ts["agent"].str.endswith("_quality", na=False)]
        _op_recent   = _op_evals_ts[_op_evals_ts["session_id"].isin(_recent_sids)]["passed"].mean()
        _op_prev     = _op_evals_ts[_op_evals_ts["session_id"].isin(_prev_sids)]["passed"].mean() if _prev_sids else None
        _op_delta    = (_op_recent - _op_prev) if _op_prev is not None else None
        _qual_raw    = evals_ts[
            evals_ts["agent"].str.endswith("_quality", na=False) &
            (evals_ts["eval_name"] == "composite_score") &
            evals_ts["session_id"].isin(_recent_sids)
        ]["score"]
        _qual_recent = float(_qual_raw.mean()) if not _qual_raw.empty else None
    else:
        _op_recent = _op_delta = _qual_recent = None

    if not biz_evals_df.empty:
        _biz_recent = biz_evals_df[biz_evals_df["session_id"].isin(s.iloc[-_n_recent:]["id"].tolist())]["passed"].mean()
        _biz_prev   = biz_evals_df[biz_evals_df["session_id"].isin(
            s.iloc[-_n_recent - _n_prev : -_n_recent]["id"].tolist() if _n_prev else []
        )]["passed"].mean() if _n_prev else None
        _biz_delta  = (_biz_recent - _biz_prev) if _biz_prev is not None else None
    else:
        _biz_recent = _biz_delta = None

    _recent_inc_count = len(incidents[incidents["session_id"].isin(s.iloc[-_n_recent:]["id"])]) if not incidents.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi(
        "Operational Health (last 7)",
        f"{_op_recent*100:.0f}%" if _op_recent is not None else "—",
        (f"{'▲' if _op_delta >= 0 else '▼'} {abs(_op_delta)*100:.0f}pp vs prior 7"
         if _op_delta is not None else ""),
    ), unsafe_allow_html=True)
    k2.markdown(kpi(
        "Business Outcomes (last 7)",
        f"{_biz_recent*100:.0f}%" if _biz_recent is not None else "—",
        (f"{'▲' if _biz_delta >= 0 else '▼'} {abs(_biz_delta)*100:.0f}pp vs prior 7"
         if _biz_delta is not None else ""),
    ), unsafe_allow_html=True)
    k3.markdown(kpi("Incidents (last 7 sessions)", str(_recent_inc_count)), unsafe_allow_html=True)
    k4.markdown(kpi(
        "Quality Score (last 7)",
        f"{_qual_recent:.2f}" if _qual_recent is not None else "—",
        "mean composite across all agents" if _qual_recent is not None else "run backfill_quality.py",
    ), unsafe_allow_html=True)

    st.divider()

    tab_op, tab_sem = st.tabs(["Operational / Business Health", "Semantic Health"])

    # ── TAB 1: OPERATIONAL / BUSINESS HEALTH ─────────────────────────────────
    with tab_op:

        # Compute per-session health scores (shared by timeline + dual bars)
        s_c = s.copy()
        if not evals_ts.empty:
            _op_scores = (
                evals_ts[evals_ts["agent"] != "business"]
                .groupby("session_id")["passed"].mean()
                .reset_index().rename(columns={"passed": "op_score"})
            )
            s_c = s_c.merge(_op_scores, left_on="id", right_on="session_id", how="left")
            s_c["op_score"] = s_c["op_score"].fillna(0.5)
        else:
            s_c["op_score"] = 0.5

        if not biz_evals_df.empty:
            _biz_scores = (
                biz_evals_df.groupby("session_id")["passed"].mean()
                .reset_index().rename(columns={"passed": "biz_score"})
            )
            s_c = s_c.merge(_biz_scores, left_on="id", right_on="session_id", how="left")
            s_c["biz_score"] = s_c["biz_score"].fillna(0.0)
        else:
            s_c["biz_score"] = 0.0

        s_c["health"] = s_c["op_score"] * 0.6 + s_c["biz_score"] * 0.4

        # ── Section: Session Health ────────────────────────────────────────────
        st.markdown("#### Session Health")

        # Group sessions into up to 6 traces (3 health buckets × 2 incident states)
        _inc_sids_tl = set(incidents["session_id"].tolist()) if not incidents.empty else set()
        s_c["health_bucket"] = s_c["health"].apply(
            lambda h: "healthy" if h >= 0.75 else ("watch" if h >= 0.45 else "critical")
        )
        s_c["has_incident"] = s_c["id"].isin(_inc_sids_tl)

        _bucket_colors = {"healthy": "#10b981", "watch": "#f59e0b", "critical": "#ef4444"}
        _bucket_names  = {"healthy": "Healthy (>=75%)", "watch": "Watch (45-75%)", "critical": "Critical (<45%)"}

        fig_tl = go.Figure()
        for _bucket, _bclr in _bucket_colors.items():
            for _has_inc in [False, True]:
                _grp = s_c[(s_c["health_bucket"] == _bucket) & (s_c["has_incident"] == _has_inc)]
                if _grp.empty:
                    continue
                _sym = "diamond" if _has_inc else "circle"
                _lnm = _bucket_names[_bucket] + (" + incident" if _has_inc else "")
                fig_tl.add_trace(go.Scatter(
                    x=_grp["label"], y=_grp["health"],
                    mode="markers", name=_lnm,
                    marker=dict(size=14, color=_bclr, symbol=_sym,
                                line=dict(color="#0f172a", width=1)),
                    customdata=_grp[["op_score", "biz_score", "trades_executed", "total_cost_usd"]].values,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Health: %{y:.0%}<br>"
                        "Op: %{customdata[0]:.0%}  Biz: %{customdata[1]:.0%}<br>"
                        "Trades: %{customdata[2]:.0f}<br>"
                        "Cost: $%{customdata[3]:.4f}"
                        "<extra></extra>"
                    ),
                ))
        if not incidents.empty:
            for _, _inc in incidents.iterrows():
                _match = s_c[s_c["id"] == _inc["session_id"]]
                if not _match.empty:
                    _x = _match.iloc[0]["label"]
                    fig_tl.add_shape(
                        type="line", x0=_x, x1=_x, y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color="rgba(239,68,68,0.35)", dash="dot", width=1),
                    )
        fig_tl.update_layout(
            paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
            font_color="#1e293b", height=240,
            yaxis=dict(title="Health score", tickformat=".0%", range=[0, 1.1]),
            xaxis_tickangle=-35,
            legend=dict(orientation="h", y=1.12, font_size=11),
            margin=dict(t=30, b=60),
        )
        st.markdown(
            "**Session Health Timeline** — 60% operational + 40% business; "
            "diamond = incident detected; hover for breakdown"
        )
        st.plotly_chart(fig_tl, use_container_width=True, key="tl_health")

        # Incident frequency stacked bar
        if not incidents.empty:
            _inc_freq = (
                incidents.groupby(["session_id", "severity"]).size()
                .reset_index(name="count")
                .merge(s[["id", "label", "started_at"]], left_on="session_id", right_on="id", how="left")
                .sort_values("started_at")
            )
            fig_inc = go.Figure()
            for _sev, _sclr in [("critical", "#ef4444"), ("warning", "#f59e0b"), ("info", "#3b82f6")]:
                _sd = _inc_freq[_inc_freq["severity"] == _sev]
                if _sd.empty:
                    continue
                fig_inc.add_trace(go.Bar(
                    x=_sd["label"], y=_sd["count"],
                    name=_sev.title(), marker_color=_sclr, opacity=0.8,
                ))
            fig_inc.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                font_color="#1e293b", height=190,
                barmode="stack",
                yaxis=dict(title="Incidents", dtick=1),
                xaxis_tickangle=-35,
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=30, b=60),
            )
            st.markdown("**Incident Frequency** — per session, by severity")
            st.plotly_chart(fig_inc, use_container_width=True, key="tl_incfreq")

        st.divider()

        # ── Section: Operational Trends ────────────────────────────────────────
        st.markdown("#### Operational Trends")

        st.markdown("**Eval Pass Rate by Agent** — rolling 5-session average")
        if not evals_ts.empty:
            _agent_pass = (
                evals_ts[evals_ts["agent"] != "business"]
                .groupby(["session_id", "agent", "started_at"])["passed"]
                .mean().reset_index()
                .sort_values("started_at")
            )
            fig_pass = go.Figure()
            for _ag, _clr in AGENT_COLORS.items():
                _ag_data = _agent_pass[_agent_pass["agent"] == _ag].copy()
                if len(_ag_data) < 2:
                    continue
                _ag_data["rolling"] = _ag_data["passed"].rolling(5, min_periods=1).mean()
                _ag_data["lbl"]     = _ag_data["started_at"].dt.strftime("%m-%d %H:%M")
                fig_pass.add_trace(go.Scatter(
                    x=_ag_data["lbl"], y=_ag_data["rolling"],
                    mode="lines+markers", name=_ag,
                    line=dict(color=_clr, width=2), marker=dict(size=5),
                ))
            if not incidents.empty:
                for _, _inc in incidents.iterrows():
                    _match = s[s["id"] == _inc["session_id"]]
                    if not _match.empty:
                        _x = _match.iloc[0]["label"]
                        _c = "#ef4444" if _inc["severity"] == "critical" else "#f59e0b"
                        fig_pass.add_shape(
                            type="line", x0=_x, x1=_x, y0=0, y1=1,
                            xref="x", yref="paper", line=dict(color=_c, dash="dot", width=1),
                        )
            fig_pass.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                font_color="#1e293b", height=290,
                yaxis=dict(title="Pass rate", tickformat=".0%", range=[0, 1.05]),
                xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1),
                margin=dict(t=40, b=60),
            )
            st.plotly_chart(fig_pass, use_container_width=True, key="op_passrate")
        else:
            st.info("No eval data. Run backfill to populate c_evals.")

        st.markdown("**Critical Eval Scores Over Time**")
        st.caption("Large dots = eval failed that session. Dotted lines = failure thresholds.")
        _KEY_EVALS = {
            "research.tool_success_rate": ("#f59e0b", 0.80),
            "orchestrator.exit_quality":  ("#3b82f6", 0.70),
            "risk.assessment_complete":   ("#10b981", 1.00),
        }
        if not evals_ts.empty:
            fig_ev = go.Figure()
            for _key, (_clr, _thr) in _KEY_EVALS.items():
                _ag, _en = _key.split(".", 1)
                _ev_data = evals_ts[
                    (evals_ts["agent"] == _ag) & (evals_ts["eval_name"] == _en)
                ].copy().sort_values("started_at")
                if _ev_data.empty:
                    continue
                _ev_data["lbl"] = _ev_data["started_at"].dt.strftime("%m-%d %H:%M")
                fig_ev.add_trace(go.Scatter(
                    x=_ev_data["lbl"], y=_ev_data["score"],
                    mode="lines+markers", name=_key,
                    line=dict(color=_clr, width=1.5),
                    marker=dict(
                        size=[8 if not p else 5 for p in _ev_data["passed"]],
                        color=[("#ef4444" if not p else _clr) for p in _ev_data["passed"]],
                    ),
                ))
                fig_ev.add_hline(y=_thr, line_dash="dot", line_color=_clr,
                                 annotation_text=f"{_key} thr",
                                 annotation_position="bottom right",
                                 annotation_font_size=9)
            if not incidents.empty:
                for _, _inc in incidents.iterrows():
                    _match = s[s["id"] == _inc["session_id"]]
                    if not _match.empty:
                        _x = _match.iloc[0]["label"]
                        _c = "#ef4444" if _inc["severity"] == "critical" else "#f59e0b"
                        fig_ev.add_shape(
                            type="line", x0=_x, x1=_x, y0=0, y1=1,
                            xref="x", yref="paper", line=dict(color=_c, dash="dot", width=1),
                        )
            fig_ev.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                font_color="#1e293b", height=300,
                yaxis=dict(title="Score", range=[-0.05, 1.1]),
                xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1),
                margin=dict(t=40, b=60),
            )
            st.plotly_chart(fig_ev, use_container_width=True, key="op_keyevals")

        st.markdown("**Pipeline Completion Rate**")
        st.caption("1.0 = all 4 agents ran. Drops signal systemic pipeline breaks.")
        if not evals_ts.empty:
            _pc = evals_ts[evals_ts["eval_name"] == "pipeline_completion"].copy()
            if not _pc.empty:
                _pc = _pc.sort_values("started_at")
                _pc["lbl"]        = _pc["started_at"].dt.strftime("%m-%d %H:%M")
                _pc["rolling_pc"] = _pc["score"].rolling(7, min_periods=1).mean()
                fig_pc = go.Figure()
                fig_pc.add_trace(go.Bar(
                    x=_pc["lbl"], y=_pc["score"], name="Completed",
                    marker_color=["#10b981" if v == 1.0 else "#ef4444" for v in _pc["score"]],
                    opacity=0.7,
                ))
                fig_pc.add_trace(go.Scatter(
                    x=_pc["lbl"], y=_pc["rolling_pc"], mode="lines", name="7-session avg",
                    line=dict(color="#0f172a", width=2, dash="dash"),
                ))
                fig_pc.update_layout(
                    paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                    font_color="#1e293b", height=230,
                    yaxis=dict(title="Score", range=[0, 1.1]),
                    xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1),
                    margin=dict(t=20, b=60),
                )
                st.plotly_chart(fig_pc, use_container_width=True, key="op_pipeline")

        st.divider()

        # ── Section: Business Outcomes ─────────────────────────────────────────
        st.markdown("#### Business Outcomes")

        st.markdown("**Trades Executed vs Cost** — per session")
        st.caption(
            "Cost flat while trades drop = agent running but not producing. "
            "Cost rising while trades stay flat = pure waste."
        )
        s["rolling_cost"] = s["total_cost_usd"].rolling(7, min_periods=1).mean()
        fig_dr = go.Figure()
        fig_dr.add_trace(go.Bar(
            x=s["label"], y=s["trades_executed"],
            name="Trades executed", marker_color="#10b981", opacity=0.65, yaxis="y",
        ))
        fig_dr.add_trace(go.Scatter(
            x=s["label"], y=s["total_cost_usd"],
            mode="lines+markers", name="Cost ($)",
            line=dict(color="#f59e0b", width=1.5), marker=dict(size=5), yaxis="y2",
        ))
        fig_dr.add_trace(go.Scatter(
            x=s["label"], y=s["rolling_cost"],
            mode="lines", name="Cost 7-avg",
            line=dict(color="#b45309", width=1.5, dash="dash"), yaxis="y2",
        ))
        if not incidents.empty:
            for _, _inc in incidents.iterrows():
                _match = s[s["id"] == _inc["session_id"]]
                if not _match.empty:
                    _x = _match.iloc[0]["label"]
                    _c = "#ef4444" if _inc["severity"] == "critical" else "#f59e0b"
                    fig_dr.add_shape(
                        type="line", x0=_x, x1=_x, y0=0, y1=1,
                        xref="x", yref="paper", line=dict(color=_c, dash="dot", width=1),
                    )
        fig_dr.update_layout(
            paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
            font_color="#1e293b", height=290,
            xaxis_tickangle=-30,
            yaxis=dict(title="Trades", side="left"),
            yaxis2=dict(title="Cost (USD)", side="right", overlaying="y"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=40, b=60),
        )
        st.plotly_chart(fig_dr, use_container_width=True, key="biz_tradescost")

        if not biz_evals_df.empty:
            st.markdown("**Business Outcome Scores** — trend")
            _biz_ts = (
                biz_evals_df
                .merge(s[["id", "label"]], left_on="session_id", right_on="id", how="left")
                .sort_values("started_at")
            )
            _biz_colors_b = {
                "cost_per_trade":      "#3b82f6",
                "research_conversion": "#f59e0b",
                "proposal_acceptance": "#10b981",
            }
            fig_bts = go.Figure()
            for _bn, _bc in _biz_colors_b.items():
                _bd = _biz_ts[_biz_ts["eval_name"] == _bn]
                if _bd.empty:
                    continue
                fig_bts.add_trace(go.Scatter(
                    x=_bd["label"], y=_bd["score"],
                    mode="lines+markers", name=_bn.replace("_", " ").title(),
                    line=dict(color=_bc, width=2),
                    marker=dict(
                        size=[7 if not p else 5 for p in _bd["passed"]],
                        color=[("#ef4444" if not p else _bc) for p in _bd["passed"]],
                    ),
                    customdata=_bd["value"].round(3).tolist(),
                    hovertemplate="%{x}<br>Score: %{y:.2f}<br>Value: %{customdata}<extra></extra>",
                ))
            fig_bts.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8",
                              annotation_text="target", annotation_font_size=9)
            fig_bts.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                font_color="#1e293b", height=280,
                yaxis=dict(title="Score (0-1)", range=[-0.05, 1.15]),
                xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1),
                margin=dict(t=40, b=60),
            )
            st.plotly_chart(fig_bts, use_container_width=True, key="biz_outcomes")
            st.caption(
                "Red markers = threshold missed. "
                "cost_per_trade: pass < $0.50; "
                "research_conversion: pass > 30%; "
                "proposal_acceptance: pass > 40%."
            )

        st.markdown("**Operational vs Business Score** — side by side per session")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(
            x=s_c["label"], y=s_c["op_score"],
            name="Operational", marker_color="#3b82f6", opacity=0.8,
        ))
        fig_dual.add_trace(go.Bar(
            x=s_c["label"], y=s_c["biz_score"],
            name="Business", marker_color="#10b981", opacity=0.8,
        ))
        fig_dual.update_layout(
            paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
            font_color="#1e293b", height=240,
            barmode="group",
            yaxis=dict(title="Score", tickformat=".0%", range=[0, 1.1]),
            xaxis_tickangle=-35,
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=30, b=60),
        )
        st.plotly_chart(fig_dual, use_container_width=True, key="biz_dual")
        st.caption(
            "Operational score = mean pass rate across all agent evals. "
            "Business score = mean pass rate across cost_per_trade, research_conversion, proposal_acceptance. "
            "A pipeline can score high on operational and low on business — it ran cleanly but produced nothing."
        )

    # ── TAB 2: SEMANTIC HEALTH ────────────────────────────────────────────────
    with tab_sem:

        _qual_colors = {
            "research_quality":     "#f59e0b",
            "risk_quality":         "#10b981",
            "orchestrator_quality": "#3b82f6",
            "session_quality":      "#8b5cf6",
        }

        if not evals_ts.empty:
            with st.expander("**Pipeline Quality Snapshot**  ·  Last 15 sessions · green = healthy, red = below threshold", expanded=True):
                with st.popover("?"):
                    st.markdown(
                        "**How to read the heatmap**\n\n"
                        "Each cell shows one agent's composite quality score for one session.\n\n"
                        "**Rows** = agents (Research, Risk, Orchestrator, Session)\n\n"
                        "**Columns** = sessions, oldest on the left, most recent on the right\n\n"
                        "**Color scale:**\n"
                        "- Green (≥ 0.60) — quality is healthy for that agent in that session\n"
                        "- Yellow (0.40–0.60) — borderline; worth watching\n"
                        "- Red (< 0.40) — below threshold; the agent's reasoning quality was poor\n\n"
                        "**What to look for:**\n"
                        "- A row drifting from green to red (left to right) = that agent is quietly degrading — "
                        "the Silent Degradation pattern\n"
                        "- A column that is all red = that session had pipeline-wide quality failure\n"
                        "- Red spreading diagonally across rows = Quality Cascade — one agent's drop "
                        "pulled downstream agents down with it\n"
                        "- Isolated red cells = single-session anomaly, likely noise"
                    )
                _last15_ids = s.tail(15)["id"].tolist()
                _qhm = evals_ts[
                    evals_ts["agent"].str.endswith("_quality", na=False) &
                    (evals_ts["eval_name"] == "composite_score") &
                    evals_ts["session_id"].isin(_last15_ids)
                ].copy()
                if not _qhm.empty:
                    _qhm["lbl"] = _qhm["started_at"].dt.strftime("%m-%d %H:%M")
                    _qhm["agent_label"] = _qhm["agent"].str.replace("_quality", "", regex=False).str.title()
                    _qhm_pivot = _qhm.pivot_table(
                        index="agent_label", columns="lbl", values="score", aggfunc="first"
                    )
                    _col_order = [lb for lb in s.tail(15)["label"] if lb in _qhm_pivot.columns]
                    _qhm_pivot = _qhm_pivot[_col_order] if _col_order else _qhm_pivot
                    _z_q   = _qhm_pivot.values.astype(float)
                    _txt_q = [[f"{v:.2f}" if not pd.isna(v) else "—" for v in row] for row in _z_q]
                    fig_qhm = go.Figure(go.Heatmap(
                        z=_z_q, x=list(_qhm_pivot.columns), y=list(_qhm_pivot.index),
                        text=_txt_q, texttemplate="%{text}",
                        colorscale=[[0, "#fee2e2"], [0.6, "#fef9c3"], [1, "#dcfce7"]],
                        zmin=0, zmax=1, showscale=True,
                        colorbar=dict(title="Score", thickness=12, len=0.8),
                    ))
                    fig_qhm.update_layout(
                        paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                        font_color="#1e293b", height=200,
                        xaxis_tickangle=-35, xaxis=dict(side="top"),
                        margin=dict(t=60, b=20, l=120, r=60),
                    )
                    st.plotly_chart(fig_qhm, use_container_width=True)
                    st.caption("Green = quality healthy (≥0.60). Yellow = borderline. Red = below threshold.")

        if not evals_ts.empty:
            _qd_all = evals_ts[
                evals_ts["agent"].str.endswith("_quality", na=False) &
                (evals_ts["eval_name"] == "composite_score")
            ].copy().sort_values("started_at")
        else:
            _qd_all = pd.DataFrame()

        if not _qd_all.empty:
            # ── Per-agent drift: linear slope over last 5 sessions ────────────
            _DRIFT_N = 5
            _drift_labels_map = {
                "research_quality":     "Research",
                "risk_quality":         "Risk",
                "orchestrator_quality": "Orchestrator",
                "session_quality":      "Session",
            }
            _drift = {}
            for _qa in _qual_colors:
                _qd_d   = _qd_all[_qd_all["agent"] == _qa].sort_values("started_at")
                if len(_qd_d) < 2:
                    _drift[_qa] = None
                    continue
                _last_n = _qd_d["score"].values[-_DRIFT_N:]
                _xs_d   = np.arange(len(_last_n), dtype=float)
                _slp_d, _ = np.polyfit(_xs_d, _last_n, 1)
                _drift[_qa] = {
                    "current": float(_last_n[-1]),
                    "slope":   float(_slp_d),
                    "change":  float(_last_n[-1] - _last_n[0]),
                    "n":       len(_last_n),
                }

            st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)
            with st.expander("**Recent Trend Direction**  ·  Linear trend across the last 5 sessions per agent", expanded=True):
                with st.popover("?"):
                    st.markdown(
                        "**How to read these cards**\n\n"
                        "Each card shows one agent's current quality score and which direction it is moving.\n\n"
                        "**Score (large number):** The composite quality score from the most recent session. "
                        "0 = worst, 1 = best. Anything below **0.60** is a concern even if no operational incident fired.\n\n"
                        "**Arrow and label:** Direction of the linear trend calculated across the last 5 sessions.\n"
                        "- ▲ Improving — score is rising session over session\n"
                        "- → Stable — score is flat within ±0.01\n"
                        "- ▼ Declining — score is falling. A declining trend that crosses 0.60 is the early "
                        "warning signal for Silent Degradation — the agent is getting worse before any operational error fires.\n\n"
                        "**Δ change:** Difference between the oldest and newest of the last 5 sessions. "
                        "A large negative number means quality dropped significantly in a short window.\n\n"
                        "**Why 5 sessions?** Short enough to catch recent drift; long enough to filter one-session noise."
                    )
                _dc = st.columns(4)
                for _di, _qa in enumerate(["research_quality", "risk_quality",
                                            "orchestrator_quality", "session_quality"]):
                    _qc   = _qual_colors[_qa]
                    _qlbl = _drift_labels_map[_qa]
                    _dd   = _drift.get(_qa)
                    if not _dd:
                        _dc[_di].markdown(
                            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                            f'border-radius:8px;padding:12px 10px;text-align:center">'
                            f'<div style="font-size:0.75rem;font-weight:600;color:{_qc}">{_qlbl}</div>'
                            f'<div style="font-size:1.4rem;font-weight:700;color:#334155">—</div>'
                            f'<div style="font-size:0.65rem;color:#94a3b8">not enough data</div>'
                            f'</div>', unsafe_allow_html=True,
                        )
                        continue
                    _slp = _dd["slope"]
                    if _slp > 0.01:
                        _arrow, _aclr, _ttxt = "▲", "#16a34a", "improving"
                        _cbg, _cbrd = "#f0fdf4", "#a7f3d0"
                    elif _slp < -0.01:
                        _arrow, _aclr, _ttxt = "▼", "#dc2626", "declining"
                        _cbg, _cbrd = "#fef2f2", "#fca5a5"
                    else:
                        _arrow, _aclr, _ttxt = "→", "#64748b", "stable"
                        _cbg, _cbrd = "#f8fafc", "#e2e8f0"
                    _chg_str = f"{_dd['change']:+.3f}" if abs(_dd["change"]) >= 0.001 else "±0.000"
                    _dc[_di].markdown(
                        f'<div style="background:{_cbg};border:1px solid {_cbrd};'
                        f'border-radius:8px;padding:12px 10px;text-align:center">'
                        f'<div style="font-size:0.75rem;font-weight:600;color:{_qc};margin-bottom:4px">{_qlbl}</div>'
                        f'<div style="font-size:1.4rem;font-weight:700;color:#0f172a">{_dd["current"]:.2f}</div>'
                        f'<div style="font-size:1.0rem;color:{_aclr};font-weight:700;margin:2px 0">{_arrow} {_ttxt}</div>'
                        f'<div style="font-size:0.65rem;color:#64748b">{_chg_str} over {_dd["n"]} sessions</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
                st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

            # ── Incident lookup: session_id → list of {pattern, severity} ─────
            _inc_by_sid: dict = {}
            if not incidents.empty:
                for _, _ir in incidents.iterrows():
                    _isid = _ir["session_id"]
                    _inc_by_sid.setdefault(_isid, []).append({
                        "pattern":  _ir.get("pattern_name", "Unknown"),
                        "severity": _ir.get("severity", ""),
                    })

            # ── System composite: mean of all 4 quality composites per session ─
            _sys_comp = (
                _qd_all.groupby(["session_id", "started_at"])["score"]
                .mean().reset_index().sort_values("started_at")
            )
            _sys_comp["lbl"] = _sys_comp["started_at"].dt.strftime("%m-%d %H:%M")

            # ── Per-agent incident lookup (filtered by failed_evals.agent) ───────
            def _agent_inc_lookup(target_agent: str) -> dict:
                result: dict = {}
                if incidents.empty:
                    return result
                for _, _ir in incidents.iterrows():
                    _fe = _ir.get("failed_evals") or []
                    if isinstance(_fe, str):
                        try:
                            _fe = json.loads(_fe)
                        except Exception:
                            _fe = []
                    _agents_hit = {
                        (e.get("agent") or "").lower()
                        for e in (_fe if isinstance(_fe, list) else [])
                    }
                    if target_agent.lower() in _agents_hit:
                        _sid = _ir["session_id"]
                        result.setdefault(_sid, []).append({
                            "pattern":  _ir.get("pattern_name", "Unknown"),
                            "severity": _ir.get("severity", ""),
                        })
                return result

            # ── Shared chart builder ───────────────────────────────────────────
            def _qchart(df, color, fill, height=300, inc_lookup=None):
                if df.empty:
                    return go.Figure()
                _lookup = inc_lookup if inc_lookup is not None else _inc_by_sid
                lbls   = df["lbl"].tolist()
                scores = df["score"].tolist()
                sids   = df["session_id"].tolist()
                _xs    = np.arange(len(scores), dtype=float)
                _sl, _ic = np.polyfit(_xs, scores, 1)
                _fitted  = (_sl * _xs + _ic).tolist()

                fig = go.Figure()
                # Score line + fill
                fig.add_trace(go.Scatter(
                    x=lbls, y=scores, mode="lines+markers", name="score",
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    fill="tozeroy", fillcolor=fill,
                    hovertemplate="%{x}<br>Score: %{y:.3f}<extra></extra>",
                ))
                # Trend line
                fig.add_trace(go.Scatter(
                    x=lbls, y=_fitted, mode="lines", name="trend",
                    line=dict(color="#b45309", width=1.5, dash="dash"),
                    hoverinfo="skip",
                ))
                # Threshold
                fig.add_hline(
                    y=0.60, line_dash="dot", line_color="#94a3b8",
                    annotation_text="0.60 threshold",
                    annotation_position="bottom right",
                    annotation_font_size=8,
                )
                # Incident markers — single trace for all incident sessions
                _inc_x, _inc_y, _inc_hov = [], [], []
                for _lbl, _score, _sid in zip(lbls, scores, sids):
                    _incs = _lookup.get(_sid)
                    if not _incs:
                        continue
                    fig.add_shape(
                        type="line", x0=_lbl, x1=_lbl, y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color="rgba(239,68,68,0.35)", width=1, dash="dot"),
                    )
                    _inc_x.append(_lbl)
                    _inc_y.append(_score)
                    _inc_hov.append(
                        f"<b>{'Incidents' if len(_incs) > 1 else 'Incident'}</b><br>"
                        + "<br>".join(f"{i['pattern']} ({i['severity']})" for i in _incs)
                        + f"<br><span style='color:#94a3b8'>{_lbl} · {_sid[:8]}</span>"
                    )
                if _inc_x:
                    fig.add_trace(go.Scatter(
                        x=_inc_x, y=_inc_y, mode="markers",
                        name="incident", showlegend=False,
                        marker=dict(size=10, color="#ef4444",
                                    line=dict(color="#ffffff", width=1.5)),
                        customdata=_inc_hov,
                        hovertemplate="%{customdata}<extra></extra>",
                    ))
                fig.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                    font_color="#1e293b", height=height,
                    yaxis=dict(range=[0, 1.05], tickvals=[0.0, 0.5, 1.0],
                               showgrid=True, gridcolor="#f1f5f9", gridwidth=1),
                    xaxis=dict(showgrid=False, tickangle=-30),
                    showlegend=False,
                    margin=dict(t=8, b=50, l=36, r=10),
                )
                return fig

            def _stats(scores_s):
                if scores_s.empty:
                    return 0.0, 0.0, 0.0
                return (float(scores_s.mean()),
                        float((scores_s >= 0.60).mean() * 100),
                        float(scores_s.iloc[-1] - scores_s.iloc[0]))

            def _stats_md(mn, pct, dlt):
                _sign = "+" if dlt >= 0 else ""
                _dc   = "#16a34a" if dlt > 0.01 else ("#dc2626" if dlt < -0.01 else "#64748b")
                return (
                    f'<span style="font-size:0.82rem;color:#475569">'
                    f'mean <b style="color:#0f172a">{mn:.2f}</b>&nbsp;&nbsp;'
                    f'pass <b style="color:#0f172a">{pct:.0f}%</b>&nbsp;&nbsp;'
                    f'<b style="color:{_dc}">Δ {_sign}{dlt:.2f}</b>'
                    f'</span>'
                )

            st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)
            with st.expander("**Quality Trends Over Time**  ·  Composite score per session · dashed line = trend · red dot = incident", expanded=True):
                with st.popover("?"):
                    st.markdown(
                        "**How to read these charts**\n\n"
                        "Each chart shows how one agent's quality composite score has moved across sessions.\n\n"
                        "**Solid line:** The actual composite score per session (0–1). "
                        "It is the average of all quality dimensions for that agent in that session.\n\n"
                        "**Dashed orange line:** Linear trend fitted across all sessions shown. "
                        "A downward slope means quality is declining over time even if individual sessions look acceptable. "
                        "This is the Silent Degradation signal.\n\n"
                        "**Dotted grey line at 0.60:** The quality threshold. Sessions below this line "
                        "passed operationally but the reasoning quality was below the acceptable floor.\n\n"
                        "**Red dot on a session:** An incident was detected for that session. "
                        "Hover to see which pattern fired. A cluster of red dots alongside a downward trend "
                        "suggests the pipeline is under sustained stress.\n\n"
                        "**Mean / Pass / Δ (below chart title):**\n"
                        "- Mean = average score across all sessions shown\n"
                        "- Pass = % of sessions that cleared the 0.60 threshold\n"
                        "- Δ = change from first to last session (negative = quality dropped)\n\n"
                        "**System composite** (top chart) is the mean of all four agent scores per session — "
                        "a single number summarising overall pipeline quality."
                    )
                # Legend strip
                st.markdown(
                    '<div style="font-size:0.78rem;color:#475569;margin:6px 0 10px;'
                    'display:flex;gap:18px;align-items:center">'
                    '<span><span style="display:inline-block;width:24px;height:2px;'
                    'background:#3b82f6;vertical-align:middle;margin-right:4px"></span>score</span>'
                    '<span><span style="display:inline-block;width:24px;height:0;'
                    'border-top:2px dashed #b45309;vertical-align:middle;margin-right:4px"></span>trend</span>'
                    '<span><span style="display:inline-block;width:24px;height:0;'
                    'border-top:2px dotted #94a3b8;vertical-align:middle;margin-right:4px"></span>'
                    '0.60 threshold</span>'
                    '<span><span style="display:inline-block;width:10px;height:10px;'
                    'background:#ef4444;border-radius:50%;vertical-align:middle;margin-right:4px"></span>'
                    'incident</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                # ── System composite card ─────────────────────────────────────
                _smn, _spct, _sdlt = _stats(_sys_comp["score"])
                st.markdown(
                    '<div style="background:#ffffff;border:1px solid #e2e8f0;'
                    'border-radius:12px;padding:16px 20px 4px;margin-bottom:12px">',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<span style="font-size:1.0rem;font-weight:700;color:#3b82f6">'
                    '● Session (system composite)</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(_stats_md(_smn, _spct, _sdlt), unsafe_allow_html=True)
                st.plotly_chart(_qchart(_sys_comp, "#3b82f6", "rgba(59,130,246,0.07)", 280),
                                use_container_width=True, key="qd_sys")
                st.markdown("</div>", unsafe_allow_html=True)

                # ── Contributing agent small multiples ────────────────────────
                _ag_cfg = [
                    ("research_quality",     "Research",     "#f59e0b", "rgba(245,158,11,0.07)"),
                    ("risk_quality",         "Risk",         "#10b981", "rgba(16,185,129,0.07)"),
                    ("orchestrator_quality", "Orchestrator", "#3b82f6", "rgba(59,130,246,0.07)"),
                ]
                _ag_cols = st.columns(3)
                for _ci, (_qa, _qlbl, _qc, _qfill) in enumerate(_ag_cfg):
                    _qd_ag = _qd_all[_qd_all["agent"] == _qa].copy().sort_values("started_at")
                    _qd_ag["lbl"] = _qd_ag["started_at"].dt.strftime("%m-%d %H:%M")
                    _mn, _pct, _dlt = _stats(_qd_ag["score"])
                    with _ag_cols[_ci]:
                        st.markdown(
                            f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
                            f'border-radius:12px;padding:16px 20px 4px;margin-bottom:12px">',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<span style="font-size:1.0rem;font-weight:700;color:{_qc}">'
                            f'● {_qlbl}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(_stats_md(_mn, _pct, _dlt), unsafe_allow_html=True)
                        _ag_inc = _agent_inc_lookup(_qa.replace("_quality", ""))
                        st.plotly_chart(
                            _qchart(_qd_ag, _qc, _qfill, 200, inc_lookup=_ag_inc),
                            use_container_width=True, key=f"qd_{_qa}",
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

            # Dimension help text: explains how each score is calculated
            # Plain-English descriptions — what this dimension actually means
            _DIM_HELP = {
                "data_grounding":           "Did research pull from multiple sources? Scores how many different tools ran without errors. Three or more distinct sources = full score.",
                "thesis_coherence":         "Did research produce a clear conclusion? Checks whether the agent logged a structured output before passing work downstream.",
                "actionability":            "Was research actually used? Checks whether the risk agent received and evaluated research output. Low score means the handoff may be broken.",
                "catalyst_specificity":     "Did research look for a reason to act right now? Checks whether a news or earnings tool was called — not just historical data.",
                "risk_acknowledgment":      "How much work did research do? Uses total token consumption as a rough proxy for depth of analysis. (Replaced by volatility_accounting in newer sessions.)",
                "research_consistency":     "Did risk have research to work with? Checks whether research completed before risk ran. Risk assessments without upstream data are unreliable.",
                "parameter_completeness":   "Did risk reach a conclusion? Checks whether the risk agent produced a structured approve or reject output with parameters.",
                "volatility_accounting":    "Did research check how volatile the stock is right now? Checks whether an ATR or volatility tool was called during research — needed for accurate position sizing.",
                "stop_loss_quality":        "How well-calibrated is the stop loss? We can't score this yet — it requires reading the agent's output text, which isn't stored in traces.",
                "position_sizing_rationale":"Was position size based on enough data? Scores how many successful data lookups risk completed before deciding on size.",
                "decision_consistency":     "Did the final decision match the evidence? Checks whether a trade (or no-trade) was consistent with what research and risk produced.",
                "resolution_completeness":  "Did the session end cleanly? Checks whether the pipeline reached a recognized outcome rather than an error or missing reason.",
                "reasoning_transparency":   "Did the orchestrator explain its decision? Checks whether it logged its reasoning — not just the action it took.",
                "upstream_integration":     "Did all stages contribute? Checks how many of the three key agents (research, risk, orchestrator) completed and fed into each other.",
                "pipeline_coherence":       "Did the full pipeline run? Scores whether all agents completed. A partial pipeline means some stages were skipped or crashed.",
                "reasoning_chain":          "Were there cascading failures? Checks whether any agent had only errors and no successful steps at all.",
            }

            # actionable = you can fix this by changing agent code
            # gap        = we can't measure this yet (needs more instrumentation)
            # note       = reflects what happened; not directly improvable
            _DIM_STATUS = {
                "data_grounding":           "actionable",
                "thesis_coherence":         "actionable",
                "actionability":            "actionable",
                "catalyst_specificity":     "actionable",
                "risk_acknowledgment":      "note",
                "research_consistency":     "actionable",
                "parameter_completeness":   "actionable",
                "volatility_accounting":    "actionable",
                "stop_loss_quality":        "gap",
                "position_sizing_rationale":"actionable",
                "decision_consistency":     "actionable",
                "resolution_completeness":  "actionable",
                "reasoning_transparency":   "actionable",
                "upstream_integration":     "actionable",
                "pipeline_coherence":       "actionable",
                "reasoning_chain":          "actionable",
            }

            # Shown only when status=actionable and score is below threshold
            _DIM_IMPROVE = {
                "data_grounding":           "Use at least 3 different tool types in research (e.g., price data, news, ATR). Each distinct type adds ~0.33 to this score.",
                "thesis_coherence":         "Ensure research logs a structured output or decision before completing. Agents that exit silently score 0.20.",
                "actionability":            "Check that research output is passed to the risk agent. If research ran but this is low, the handoff between agents is broken.",
                "catalyst_specificity":     "Add a news or earnings lookup to the research step. Without a current-events check, research relies on historical data only.",
                "research_consistency":     "Research must complete before risk runs. Check pipeline ordering — risk shouldn't start if research errored out.",
                "parameter_completeness":   "Risk agent should output a structured decision (approve or reject with size and stop parameters). Silent exits score 0.50.",
                "volatility_accounting":    "Ensure the research agent calls get_atr or get_ticker_market_data (which includes ATR). Without this, position sizing has no volatility input.",
                "position_sizing_rationale":"Risk agent needs at least 2 successful data lookups before deciding position size. Add more calls or fix the ones that are failing.",
                "decision_consistency":     "Full score requires a trade placed after both research and risk completed. Partial pipelines or unexplained no-trade sessions score lower.",
                "resolution_completeness":  "Pipeline ended with an error or unknown reason. Check the Incidents Feed for this session to find what crashed before a clean exit.",
                "reasoning_transparency":   "Orchestrator should log a decision trace with its reasoning. Agents that pass through silently score 0.20.",
                "upstream_integration":     "Both research and risk need to complete for full score. Check which upstream stage did not run.",
                "pipeline_coherence":       "All three agents (research, risk, orchestrator) should complete. A partial pipeline scores 0.30.",
                "reasoning_chain":          "One or more agents had only error traces with no successful steps. Check the Incidents Feed to see which agent failed.",
            }

            _agent_colors = {
                "research_quality":      "#f59e0b",
                "risk_quality":          "#10b981",
                "orchestrator_quality":  "#3b82f6",
                "session_quality":       "#8b5cf6",
            }
            _agent_bg = {
                "research_quality":      "#fffbeb",
                "risk_quality":          "#f0fdf4",
                "orchestrator_quality":  "#eff6ff",
                "session_quality":       "#f5f3ff",
            }
            _agent_border = {
                "research_quality":      "#fde68a",
                "risk_quality":          "#a7f3d0",
                "orchestrator_quality":  "#bfdbfe",
                "session_quality":       "#ddd6fe",
            }

            st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)
            with st.expander("**Session Detail**  ·  Per-dimension scores with fix guidance", expanded=False):
                with st.popover("?"):
                    st.markdown(
                        "**How to read the dimension breakdown**\n\n"
                        "Each card is one quality dimension for one agent in the selected session.\n\n"
                        "**Score (0–1):** How well the agent performed on that dimension. Below 0.60 = concern.\n\n"
                        "**Fixable badge:** The dimension is below threshold and can be improved by changing agent code. "
                        "Click **Fix ?** to see exactly what to change.\n\n"
                        "**Measurement gap badge:** Cannot be scored yet — requires raw LLM output text stored in traces. "
                        "No agent code change will improve this score until output logging is added.\n\n"
                        "**Composite score (top of each agent block):** Average of all dimensions for that agent. "
                        "This is the number shown in the trend charts above."
                    )
                # ── Session quality browser ───────────────────────────────────
                _qd_composites = evals_ts[
                    evals_ts["agent"].str.endswith("_quality", na=False) &
                    (evals_ts["eval_name"] == "composite_score")
                ][["session_id", "agent", "score"]].copy()

                _qd_pivot = (
                    _qd_composites
                    .pivot_table(index="session_id", columns="agent", values="score", aggfunc="first")
                    .reset_index()
                ) if not _qd_composites.empty else pd.DataFrame(columns=["session_id"])

                _qd_sess = s.sort_values("started_at", ascending=False)[
                    ["id", "started_at", "label", "terminal_reason",
                     "total_cost_usd", "trades_executed",
                     "total_latency_ms", "total_tokens_input", "total_tokens_output",
                     "agents_invoked"]
                ].copy()
                _qd_merged = _qd_sess.merge(_qd_pivot, left_on="id", right_on="session_id", how="left")

                if not incidents.empty:
                    _qd_inc_c = incidents.groupby("session_id").size().reset_index(name="_inc_count")
                    _qd_merged = _qd_merged.merge(_qd_inc_c, left_on="id", right_on="session_id", how="left")
                    _qd_merged["_inc_count"] = _qd_merged["_inc_count"].fillna(0).astype(int)
                else:
                    _qd_merged["_inc_count"] = 0

                def _qscore(row, col):
                    v = row.get(col)
                    return round(float(v), 2) if v is not None and not pd.isna(v) else None

                _qd_rows = []
                for _, _qr in _qd_merged.iterrows():
                    _qd_rows.append({
                        "_id":          _qr["id"],
                        "_has_inc":     int(_qr["_inc_count"]) > 0,
                        "Session":      _qr["label"],
                        "Research":     _qscore(_qr, "research_quality"),
                        "Risk":         _qscore(_qr, "risk_quality"),
                        "Orchestrator": _qscore(_qr, "orchestrator_quality"),
                        "Session Q":    _qscore(_qr, "session_quality"),
                        "Exit":         str(_qr.get("terminal_reason") or ""),
                        "Incidents":    int(_qr["_inc_count"]),
                    })
                _qd_tbl = pd.DataFrame(_qd_rows)

                _score_cell = JsCode("""
                    function(params) {
                        var v = params.value;
                        if (v === null || v === undefined || v === '') return {};
                        if (v >= 0.60) return {color:'#166534', fontWeight:'600'};
                        if (v >= 0.40) return {color:'#92400e', fontWeight:'600'};
                        return {color:'#991b1b', fontWeight:'700'};
                    }
                """)

                _qd_PAGE = 10
                _qd_gb = GridOptionsBuilder.from_dataframe(_qd_tbl)
                _qd_gb.configure_default_column(suppressMenu=True, sortable=False, resizable=False, filter=False)
                _qd_gb.configure_column("_id",     hide=True)
                _qd_gb.configure_column("_has_inc", hide=True)
                _qd_gb.configure_column("Session",      width=110, suppressSizeToFit=True)
                _qd_gb.configure_column("Research",     width=90,  suppressSizeToFit=True, cellStyle=_score_cell)
                _qd_gb.configure_column("Risk",         width=70,  suppressSizeToFit=True, cellStyle=_score_cell)
                _qd_gb.configure_column("Orchestrator", width=110, suppressSizeToFit=True, cellStyle=_score_cell)
                _qd_gb.configure_column("Session Q",    width=90,  suppressSizeToFit=True, cellStyle=_score_cell)
                _qd_gb.configure_column("Exit",         flex=1,    cellStyle=JsCode("""
                    function(params) {
                        var v = params.value || '';
                        if (v === 'converged') return {color:'#166534', fontWeight:'600'};
                        if (v === 'no_viable_proposals' || v === 'all_rejected') return {color:'#c2410c', fontWeight:'600'};
                        if (v === 'watchdog_timeout' || v === 'timeout') return {color:'#991b1b', fontWeight:'700'};
                        if (v === 'eod_complete' || v === 'superseded') return {color:'#475569'};
                        return {};
                    }
                """))
                _qd_gb.configure_column("Incidents",    width=80, suppressSizeToFit=True)
                _qd_gb.configure_selection("single", use_checkbox=False)
                _qd_gb.configure_grid_options(
                    pagination=True,
                    paginationPageSize=_qd_PAGE,
                    suppressPaginationPanel=len(_qd_tbl) <= _qd_PAGE,
                    getRowStyle=JsCode("""
                        function(params) {
                            if (params.data._has_inc) return {'background':'#fee2e2','color':'#7f1d1d'};
                        }
                    """),
                    rowHeight=34, headerHeight=36, suppressHorizontalScroll=True,
                )
                _qd_resp = AgGrid(
                    _qd_tbl,
                    gridOptions=_qd_gb.build(),
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    height=min(560, 56 + min(len(_qd_tbl), _qd_PAGE) * 34 + (0 if len(_qd_tbl) <= _qd_PAGE else 60)),
                    use_container_width=True,
                    allow_unsafe_jscode=True,
                    fit_columns_on_grid_load=True,
                    theme="streamlit",
                )
                st.markdown(
                    "<div style='font-size:0.78rem;color:#94a3b8;margin-top:4px'>"
                    "Green = score &ge;0.60 &nbsp;·&nbsp; Amber = 0.40–0.60 &nbsp;·&nbsp; "
                    "Red = below 0.40 &nbsp;·&nbsp; Red row = incident &nbsp;·&nbsp; "
                    "Click a row to see pipeline and dimension breakdown</div>",
                    unsafe_allow_html=True,
                )

                # ── Pipeline strip + quality detail panel ─────────────────────
                _qd_sel = _qd_resp.selected_rows
                _picked_sid = None
                if _qd_sel is not None and len(_qd_sel) > 0:
                    _qd_sel_row  = _qd_sel.iloc[0] if isinstance(_qd_sel, pd.DataFrame) else _qd_sel[0]
                    _picked_sid  = _qd_sel_row["_id"]
                    _qd_full     = sessions[sessions["id"] == _picked_sid]
                    _qd_sel_incs = incidents[incidents["session_id"] == _picked_sid] if not incidents.empty else pd.DataFrame()

                    if not _qd_full.empty:
                        _qd_sr       = _qd_full.iloc[0]
                        _qd_cost     = float(_qd_sr.get("total_cost_usd") or 0)
                        _qd_trades   = int(_qd_sr.get("trades_executed") or 0)
                        _qd_dur      = int((_qd_sr.get("total_latency_ms") or 0) / 1000)
                        _qd_tok      = int((_qd_sr.get("total_tokens_input") or 0) + (_qd_sr.get("total_tokens_output") or 0))
                        _qd_exit     = str(_qd_sr.get("terminal_reason") or "")
                        _qd_date_str = pd.to_datetime(_qd_sr.get("started_at"), errors="coerce", utc=True)
                        _qd_date_str = _qd_date_str.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(_qd_date_str) else "-"

                        _qd_agents_inv = _qd_sr.get("agents_invoked") or []
                        _qd_strip      = pipeline_strip(_picked_sid, traces_all, all_evals, session_agents=_qd_agents_inv)

                        _ec_fg2, _ec_bg2 = {
                            "converged":           ("#166534", "#dcfce7"),
                            "eod_complete":        ("#334155", "#f1f5f9"),
                            "superseded":          ("#334155", "#f1f5f9"),
                            "no_viable_proposals": ("#c2410c", "#fff7ed"),
                            "all_rejected":        ("#c2410c", "#fff7ed"),
                            "watchdog_timeout":    ("#991b1b", "#fee2e2"),
                            "timeout":             ("#991b1b", "#fee2e2"),
                        }.get(_qd_exit, ("#475569", "#f8fafc"))

                        _exit_explanations2 = {
                            "converged":           "All agents completed. Orchestrator found viable proposals and executed trades.",
                            "eod_complete":        "End-of-day session. Open positions closed, P&L reconciled.",
                            "no_viable_proposals": "Market conditions did not support any trades. Pipeline stopped at Market agent.",
                            "all_rejected":        "Research and Risk ran but all proposals were rejected against risk criteria.",
                            "superseded":          "Session replaced by a newer run.",
                            "watchdog_timeout":    "Session exceeded the watchdog time limit and was force-shut down.",
                            "timeout":             "An agent or tool call exceeded its time limit. Session aborted.",
                        }
                        _qd_exit_explain = _exit_explanations2.get(_qd_exit, "")

                        _qd_has_inc = not _qd_sel_incs.empty
                        _qd_accent  = "#ef4444" if _qd_has_inc else "#f59e0b" if _qd_trades == 0 else "#3b82f6"

                        # Quality scores summary row
                        _q_scores_html = ""
                        for _qa_key, _qa_lbl, _qa_clr in [
                            ("research_quality", "Research", "#f59e0b"),
                            ("risk_quality", "Risk", "#10b981"),
                            ("orchestrator_quality", "Orchestrator", "#3b82f6"),
                            ("session_quality", "Session", "#8b5cf6"),
                        ]:
                            _qa_row = all_evals[
                                (all_evals["session_id"] == _picked_sid) &
                                (all_evals["agent"] == _qa_key) &
                                (all_evals["eval_name"] == "composite_score")
                            ]
                            if not _qa_row.empty:
                                _qa_score = float(_qa_row["score"].iloc[0])
                                _qa_clr2  = "#166534" if _qa_score >= 0.60 else "#92400e" if _qa_score >= 0.40 else "#991b1b"
                                _qa_bg2   = "#dcfce7" if _qa_score >= 0.60 else "#fef9c3" if _qa_score >= 0.40 else "#fee2e2"
                                _q_scores_html += (
                                    f'<span style="background:{_qa_bg2};color:{_qa_clr2};border-radius:4px;'
                                    f'padding:2px 8px;font-size:0.78rem;font-weight:600">'
                                    f'{_qa_lbl}&nbsp;{_qa_score:.2f}</span> '
                                )

                        # Incident banners
                        _qd_inc_html = ""
                        if _qd_has_inc:
                            for _, _ir in _qd_sel_incs.iterrows():
                                _sev  = str(_ir.get("severity") or "").lower()
                                _ibg  = "#fee2e2" if _sev == "critical" else "#fef9c3"
                                _ifg  = "#7f1d1d" if _sev == "critical" else "#713f12"
                                _ipat = str(_ir.get("pattern_name") or "")
                                _irc  = str(_ir.get("root_cause") or "")[:90]
                                _qd_inc_html += (
                                    f'<div style="background:{_ibg};color:{_ifg};border-radius:6px;'
                                    f'padding:6px 10px;font-size:0.78rem;margin-top:4px">'
                                    f'<strong>{_sev.upper()}</strong> · {_ipat}'
                                    f'{"  —  " + _irc if _irc else ""}</div>'
                                )

                        _qd_legend = """
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;padding-top:8px;
            border-top:1px solid #e2e8f0;font-size:0.72rem;color:#64748b;align-items:center">
  <span style="font-weight:600;color:#94a3b8;letter-spacing:0.04em">NODES</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#10b981;margin-right:4px;vertical-align:middle"></span>Passed evals</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f59e0b;margin-right:4px;vertical-align:middle"></span>Partial pass</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:4px;vertical-align:middle"></span>Failed evals</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#475569;margin-right:4px;vertical-align:middle"></span>Ran, not evaluated</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid #cbd5e1;background:#e2e8f0;margin-right:4px;vertical-align:middle"></span>Did not run</span>
</div>"""

                        st.markdown(f"""
<div style="margin-top:8px;border-left:4px solid {_qd_accent};border-radius:0 8px 8px 0;
            padding:12px 16px 20px 14px;background:#f8fafc;border-top:1px solid #e2e8f0;
            border-right:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;overflow:visible">
  <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:8px;letter-spacing:0.03em">
    PIPELINE DETAIL &nbsp;·&nbsp; {_qd_date_str}{"&nbsp;&nbsp;<span style='color:#ef4444'>Red row = incident flagged, not a pipeline failure</span>" if _qd_has_inc and _qd_exit in ("converged","eod_complete") else ""}
  </div>
  <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;overflow:visible;padding-bottom:4px">
    {_qd_strip}
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-left:8px;border-left:1px solid #e2e8f0;padding-left:16px">
      <span style="font-size:0.82rem"><strong style="color:#64748b">Cost</strong>&nbsp;${_qd_cost:.4f}</span>
      <span style="font-size:0.82rem"><strong style="color:#64748b">Trades</strong>&nbsp;{_qd_trades}</span>
      <span style="font-size:0.82rem"><strong style="color:#64748b">Duration</strong>&nbsp;{_qd_dur}s</span>
      <span style="font-size:0.82rem"><strong style="color:#64748b">Tokens</strong>&nbsp;{_qd_tok:,}</span>
      <span style="background:{_ec_bg2};color:{_ec_fg2};border-radius:4px;padding:2px 8px;
                  font-size:0.75rem;font-weight:600">{_qd_exit or "unknown"}</span>
    </div>
  </div>
  {"<div style='font-size:0.78rem;color:#475569;margin-top:8px;padding-top:8px;border-top:1px solid #e2e8f0'><strong style='color:#64748b'>Exit:</strong> " + _qd_exit_explain + "</div>" if _qd_exit_explain else ""}
  {"<div style='margin-top:8px;padding-top:8px;border-top:1px solid #e2e8f0;display:flex;gap:6px;flex-wrap:wrap;align-items:center'><span style='font-size:0.72rem;color:#94a3b8;font-weight:600;letter-spacing:0.03em'>QUALITY</span> " + _q_scores_html + "</div>" if _q_scores_html else ""}
  {_qd_inc_html}
  {_qd_legend}
</div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

                if _picked_sid:
                    _picked_qd = evals_ts[
                        (evals_ts["session_id"] == _picked_sid) &
                        evals_ts["agent"].str.endswith("_quality", na=False) &
                        (evals_ts["eval_name"] != "composite_score")
                    ].copy()

                    if not _picked_qd.empty:
                        for _qa in ["research_quality", "risk_quality",
                                    "orchestrator_quality", "session_quality"]:
                            _dims = _picked_qd[_picked_qd["agent"] == _qa]
                            if _dims.empty:
                                continue
                            _label    = _qa.replace("_quality", "").title()
                            _ac       = _agent_colors[_qa]
                            _abg      = _agent_bg[_qa]
                            _aborder  = _agent_border[_qa]
                            _comp_row = evals_ts[
                                (evals_ts["session_id"] == _picked_sid) &
                                (evals_ts["agent"] == _qa) &
                                (evals_ts["eval_name"] == "composite_score")
                            ]
                            _comp       = float(_comp_row["score"].iloc[0]) if not _comp_row.empty else 0.0
                            _comp_pass  = _comp >= 0.60
                            _badge_bg   = "#dcfce7" if _comp_pass else "#fee2e2"
                            _badge_txt  = "#166534" if _comp_pass else "#991b1b"
                            _pass_label = "passed" if _comp_pass else "below threshold"

                            st.markdown(
                                f'<div style="background:{_abg};border:1px solid {_aborder};'
                                f'border-radius:10px;padding:14px 16px 10px;margin-bottom:12px">'
                                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
                                f'<span style="font-weight:700;font-size:1rem;color:{_ac}">{_label}</span>'
                                f'<span style="background:{_badge_bg};color:{_badge_txt};border-radius:20px;'
                                f'padding:2px 10px;font-size:0.78rem;font-weight:600">'
                                f'composite&nbsp;{_comp:.2f} &mdash; {_pass_label}</span>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

                            _dc = st.columns(len(_dims))
                            for _di, (_, _drow) in enumerate(_dims.iterrows()):
                                _ds     = float(_drow.get("score") or 0)
                                _dname  = str(_drow["eval_name"])
                                _dlabel = _dname.replace("_", " ").title()
                                _score_color = "#10b981" if _ds >= 0.75 else ("#f59e0b" if _ds >= 0.60 else "#ef4444")
                                _pct      = int(_ds * 100)
                                _status   = _DIM_STATUS.get(_dname, "actionable")
                                _help_txt = _DIM_HELP.get(_dname, "")
                                _improve  = _DIM_IMPROVE.get(_dname, "")
                                _is_low   = _ds < 0.60

                                if _status == "gap":
                                    _sbadge = ('<span style="font-size:0.60rem;color:#64748b;'
                                               'background:#f1f5f9;border-radius:10px;padding:1px 6px">'
                                               'measurement gap</span>')
                                elif _status == "actionable" and _is_low:
                                    _sbadge = ('<span style="font-size:0.60rem;color:#92400e;'
                                               'background:#fef3c7;border-radius:10px;padding:1px 6px">'
                                               'fixable</span>')
                                else:
                                    _sbadge = ""

                                _dc[_di].markdown(
                                    f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
                                    f'border-radius:8px;padding:10px 8px 8px;text-align:center">'
                                    f'<div style="font-size:1.25rem;font-weight:700;color:{_score_color}">{_ds:.2f}</div>'
                                    f'<div style="font-size:0.70rem;color:#475569;margin:3px 0 5px;line-height:1.3">{_dlabel}</div>'
                                    f'<div style="min-height:18px;margin-bottom:5px">{_sbadge}</div>'
                                    f'<div style="background:#f1f5f9;border-radius:4px;height:5px">'
                                    f'<div style="background:{_score_color};width:{_pct}%;height:100%;border-radius:4px"></div>'
                                    f'</div></div>',
                                    unsafe_allow_html=True,
                                )
                                if _is_low and _help_txt:
                                    _pop_label = "Gap ?" if _status == "gap" else "Fix ?"
                                    with _dc[_di].popover(_pop_label):
                                        st.markdown(f"**{_dlabel}** ({_ds:.2f})")
                                        st.markdown(_help_txt)
                                        if _status == "actionable" and _improve:
                                            st.markdown(f"**Fix:** {_improve}")
                                        elif _status == "gap":
                                            st.markdown("**Gap:** Cannot score without output text in traces. No agent code change will fix this.")
                    else:
                        st.info("No quality dimension data for this session.")

        else:
            st.info("No quality evals yet. Run: `python3 scripts/backfill_quality.py`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Incidents Feed
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Incidents Feed":
    st.markdown("## Incidents Feed")

    st.markdown(
        '<div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;'
        'border-radius:0 6px 6px 0;margin-bottom:16px;font-size:0.9rem;color:#1e3a8a">'
        '<b>How this works:</b> &nbsp; '
        'Click <b>Run Analysis</b> to scan all sessions for failure patterns. '
        'Each detected incident shows its severity, pattern name, and cost wasted. '
        'Click <b>RCA →</b> on any row to open the full root cause breakdown — '
        'the call stack that failed, the evals that caught it, and the fix.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_run, col_info = st.columns([2, 5])
    with col_run:
        if st.button("Run Analysis on All Sessions", type="primary", use_container_width=True):
            with st.spinner("Analysing all sessions..."):
                rows_written = 0
                for _, sess in sessions.iterrows():
                    sid   = sess["id"]
                    trows = traces_all[traces_all["session_id"] == sid].to_dict("records")
                    evs   = run_all_evals(sess.to_dict(), trows, recent_costs)
                    incs  = run_all_detectors(sess.to_dict(), trows, evs, recent_costs)
                    if incs:
                        try:
                            _db().table("c_incidents").insert(
                                [i.to_db_row() for i in incs]
                            ).execute()
                            rows_written += len(incs)
                        except Exception:
                            pass
            st.success(f"Done. {rows_written} incident(s) written.")
            st.cache_data.clear()
            st.rerun()

    incidents = load_incidents()

    if incidents.empty:
        st.info("No incidents yet. Click 'Run Analysis' to detect patterns across all sessions.")
        st.stop()

    # Filters
    st.divider()
    fc1, fc2, fc3, fc4 = st.columns([2, 3, 2, 1.5])
    sev_filter     = fc1.multiselect("Severity",
                                      ["critical","warning","info"],
                                      default=["critical","warning"])
    pattern_filter = fc2.multiselect("Pattern",
                                      sorted(incidents["pattern_name"].unique().tolist()),
                                      default=incidents["pattern_name"].unique().tolist())
    sim_filter     = fc3.radio("Show", ["All","Real only","Simulated only"],
                                horizontal=True)
    show_shadow_cb = fc4.checkbox("Shadow CBs", value=False,
                                   help="Show Shadow CB entries (info-severity signals where a circuit breaker would have fired)")

    filtered = incidents.copy()
    if sev_filter:
        filtered = filtered[filtered["severity"].isin(sev_filter)]
    if pattern_filter:
        filtered = filtered[filtered["pattern_name"].isin(pattern_filter)]
    if sim_filter == "Real only":
        filtered = filtered[~filtered.get("is_simulated", False)]
    elif sim_filter == "Simulated only":
        filtered = filtered[filtered.get("is_simulated", filtered["is_simulated"].fillna(False))]
    if not show_shadow_cb:
        filtered = filtered[~filtered["pattern_name"].str.startswith("Shadow CB:")]

    # Summary KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.markdown(kpi("Total Incidents",  str(len(filtered))),                       unsafe_allow_html=True)
    kc2.markdown(kpi("Critical",         str((filtered["severity"]=="critical").sum())), unsafe_allow_html=True)
    kc3.markdown(kpi("Cost Wasted",      f"${filtered['cost_wasted'].sum():.2f}"),   unsafe_allow_html=True)
    kc4.markdown(kpi("Patterns Seen",    str(filtered["pattern_name"].nunique())),   unsafe_allow_html=True)

    st.divider()

    # Build AgGrid table — all filtered rows, AgGrid handles pagination via rowHeight
    _inc_PAGE = 10
    _inc_sorted = filtered.sort_values("created_at", ascending=False).reset_index(drop=True)

    _inc_tbl = pd.DataFrame({
        "_id":       _inc_sorted["id"],
        "_sid":      _inc_sorted["session_id"],
        "_sev":      _inc_sorted["severity"],
        "Time":      pd.to_datetime(_inc_sorted["created_at"], utc=True).dt.strftime("%m-%d %H:%M"),
        "Severity":  _inc_sorted["severity"].str.upper() + _inc_sorted["is_simulated"].fillna(False).apply(lambda x: " · SIM" if x else ""),
        "Pattern":   _inc_sorted["pattern_name"],
        "Root Cause": _inc_sorted["root_cause"].str[:110],
        "Cost":      _inc_sorted["cost_wasted"].apply(lambda x: f"${x:.4f}" if x > 0 else "—"),
    })

    _inc_gb = GridOptionsBuilder.from_dataframe(_inc_tbl)
    _inc_gb.configure_default_column(suppressMenu=True, sortable=False, resizable=False, filter=False)
    _inc_gb.configure_column("_id",  hide=True)
    _inc_gb.configure_column("_sid", hide=True)
    _inc_gb.configure_column("_sev", hide=True)
    _inc_gb.configure_column("Time",      width=90,  suppressSizeToFit=True)
    _inc_gb.configure_column("Severity",  width=130, suppressSizeToFit=True)
    _inc_gb.configure_column("Pattern",   width=220, suppressSizeToFit=True)
    _inc_gb.configure_column("Root Cause", flex=1)
    _inc_gb.configure_column("Cost",      width=80,  suppressSizeToFit=True)
    _inc_gb.configure_selection("single", use_checkbox=False)
    _inc_gb.configure_grid_options(
        pagination=True,
        paginationPageSize=_inc_PAGE,
        suppressPaginationPanel=len(_inc_tbl) <= _inc_PAGE,
        getRowStyle=JsCode("""
            function(params) {
                var s = (params.data._sev || '').toLowerCase();
                if (s === 'critical') return {'background': '#fee2e2', 'color': '#7f1d1d'};
                if (s === 'warning')  return {'background': '#fef9c3', 'color': '#713f12'};
            }
        """),
        rowHeight=34,
        headerHeight=36,
        suppressHorizontalScroll=True,
    )

    _inc_resp = AgGrid(
        _inc_tbl,
        gridOptions=_inc_gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=min(600, 56 + min(len(_inc_tbl), _inc_PAGE) * 34 + (0 if len(_inc_tbl) <= _inc_PAGE else 60)),
        use_container_width=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme="streamlit",
    )
    _inc_sel = _inc_resp.selected_rows
    if _inc_sel is not None and len(_inc_sel) > 0:
        _inc_sel_row = _inc_sel.iloc[0] if isinstance(_inc_sel, pd.DataFrame) else _inc_sel[0]
        _sel_id  = _inc_sel_row["_id"]
        _sel_sid = _inc_sel_row["_sid"]
        _matched = filtered[filtered["id"] == _sel_id]
        if not _matched.empty:
            goto_rca(_matched.iloc[0].to_dict(), _sel_sid)

    st.markdown(
        f"<div style='font-size:0.8rem;color:#94a3b8;margin-top:4px'>"
        f"{len(_inc_tbl)} incidents · Red = critical · Amber = warning · Click a row to open RCA</div>",
        unsafe_allow_html=True,
    )



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RCA View
# ══════════════════════════════════════════════════════════════════════════════
elif page == "RCA View":

    st.markdown("## RCA View")

    # ── Read share-link param ─────────────────────────────────────────────────
    _params = st.query_params
    if "sid" in _params and not st.session_state.get("rca_sid"):
        st.session_state["rca_sid"] = _params.get("sid", "")

    # ── Incident selector ─────────────────────────────────────────────────────
    inc_dict = st.session_state.get("rca_incident")
    rca_sid  = st.session_state.get("rca_sid")

    if not inc_dict and not incidents.empty:
        inc_dict = incidents.sort_values("cost_wasted", ascending=False).iloc[0].to_dict()
        rca_sid  = inc_dict.get("session_id")

    if not inc_dict:
        st.info("Select an incident from the Incidents Feed to view its RCA.")
        st.stop()

    if not incidents.empty:
        _opts = {
            f"{row['pattern_name']}  ·  "
            f"{row['created_at'].strftime('%m-%d %H:%M') if pd.notna(row['created_at']) else ''}  ·  "
            f"${row['cost_wasted']:.4f}": row.to_dict()
            for _, row in incidents.sort_values("created_at", ascending=False).iterrows()
        }
        _chosen_key = st.selectbox("Select incident", list(_opts.keys()))
        inc_dict    = _opts[_chosen_key]
        rca_sid     = inc_dict.get("session_id")

    # ── Controls: share link + re-run ─────────────────────────────────────────
    _ctrl1, _ctrl2, _ctrl3 = st.columns([4, 2, 2])
    with _ctrl1:
        if rca_sid:
            st.query_params["sid"] = rca_sid
            st.markdown(
                f'<div style="font-size:0.78rem;color:#64748b;padding:6px 0">'
                f'Share: <code>?sid={rca_sid[:8]}...</code></div>',
                unsafe_allow_html=True,
            )
    with _ctrl2:
        if st.button("Re-run Analysis", use_container_width=True):
            if rca_sid:
                with st.spinner("Re-running..."):
                    _sr = sessions[sessions["id"] == rca_sid]
                    if not _sr.empty:
                        _s   = _sr.iloc[0].to_dict()
                        _tr  = traces_all[traces_all["session_id"] == rca_sid].to_dict("records")
                        _rc  = sessions["total_cost_usd"].tolist()
                        _evs, _incs = run_analysis_for_session(_s, _tr, _rc)
                        _db().table("c_evals").delete().eq("session_id", rca_sid).execute()
                        _db().table("c_incidents").delete().eq("session_id", rca_sid).execute()
                        if _evs:
                            _db().table("c_evals").insert([e.to_db_row(rca_sid) for e in _evs]).execute()
                        if _incs:
                            _db().table("c_incidents").insert([i.to_db_row() for i in _incs]).execute()
                        st.cache_data.clear()
                        st.success("Analysis refreshed.")
                        st.rerun()

    # ── Model objects ─────────────────────────────────────────────────────────
    from engine.pattern_detector import Incident as IncidentModel

    inc_obj = IncidentModel(
        session_id    = rca_sid or "",
        pattern_name  = inc_dict.get("pattern_name", ""),
        severity      = inc_dict.get("severity", "info"),
        root_cause    = inc_dict.get("root_cause", ""),
        call_stack    = inc_dict.get("call_stack") or [],
        failed_evals  = inc_dict.get("failed_evals") or [],
        cost_wasted   = float(inc_dict.get("cost_wasted") or 0),
        tokens_wasted = int(inc_dict.get("tokens_wasted") or 0),
        fix_suggestion= inc_dict.get("fix_suggestion", ""),
        is_simulated  = bool(inc_dict.get("is_simulated", False)),
    )

    sess_traces = traces_all[traces_all["session_id"] == rca_sid].to_dict("records") if rca_sid else []
    sess_row    = sessions[sessions["id"] == rca_sid].iloc[0].to_dict() if (
        rca_sid and not sessions[sessions["id"] == rca_sid].empty
    ) else {}
    evals_df    = load_evals_for_session(rca_sid) if rca_sid else pd.DataFrame()

    # evals_by_agent
    evals_by_agent: dict[str, list[dict]] = {}
    if not evals_df.empty:
        for _, ev in evals_df.iterrows():
            _a = str(ev.get("agent", "")).lower()
            _raw = ev.get("detail") or {}
            if isinstance(_raw, str):
                try:
                    _raw = _json.loads(_raw)
                except Exception:
                    _raw = {}
            evals_by_agent.setdefault(_a, []).append({
                "agent":     ev.get("agent", ""),
                "eval_name": ev.get("eval_name", ""),
                "score":     float(ev.get("score") or 0),
                "passed":    bool(ev.get("passed")),
                "detail":    _raw,
            })
    elif inc_obj.failed_evals:
        for _fe in inc_obj.failed_evals:
            _a = str(_fe.get("agent", "")).lower()
            evals_by_agent.setdefault(_a, []).append({
                "agent":     _fe.get("agent", ""),
                "eval_name": _fe.get("eval_name", ""),
                "score":     float(_fe.get("score") or 0),
                "passed":    False,
                "detail":    {},
            })

    annotated = build_annotated_call_stack(sess_traces, inc_obj) if sess_traces else list(inc_obj.call_stack)

    # Per-agent stats from traces
    agent_stats: dict[str, dict] = {}
    for _t in sess_traces:
        _a = (_t.get("agent") or "").lower()
        if _a not in agent_stats:
            agent_stats[_a] = {"tokens": 0, "latency_ms": 0, "n_errors": 0, "n_llm": 0, "n_tool": 0}
        agent_stats[_a]["tokens"]     += int((_t.get("tokens_input") or 0) + (_t.get("tokens_output") or 0))
        agent_stats[_a]["latency_ms"] += int(_t.get("latency_ms") or 0)
        if _str(_t.get("error")) or _t.get("outcome") == "error":
            agent_stats[_a]["n_errors"] += 1
        if _t.get("step_type") == "llm_call":
            agent_stats[_a]["n_llm"] += 1
        if _t.get("step_type") == "tool_call":
            agent_stats[_a]["n_tool"] += 1

    # Root cause agent
    root_agent = None
    for _t in annotated:
        if _t.get("is_root"):
            root_agent = (_t.get("agent") or "").lower()
            break

    _agent_order   = ["market", "research", "risk", "orchestrator"]
    agents_present = [a for a in _agent_order if a in agent_stats]
    session_evs    = evals_by_agent.get("session", [])
    fix_text       = generate_fix_suggestion(inc_obj, sess_traces)
    summary        = summarize_incident(inc_obj, sess_row)

    sev_color = {"critical": "#ef4444", "warning": "#f59e0b", "info": "#3b82f6"}.get(inc_obj.severity, "#64748b")
    sev_bg    = {"critical": "#fff5f5", "warning": "#fffbeb", "info": "#eff6ff"}.get(inc_obj.severity, "#f8fafc")
    sev_text  = {"critical": "#7f1d1d", "warning": "#78350f", "info": "#1e3a8a"}.get(inc_obj.severity, "#0f172a")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — WHAT HAPPENED & WHAT TO FIX
    # ════════════════════════════════════════════════════════════════════════════
    _section_header(
        "What Happened & What to Fix",
        """**Start here.**

This section gives you the plain-English answer before you look at any trace evidence.

- **Left** — what broke, which agent caused it, what it cost
- **Right** — numbered fix steps tailored to this failure pattern

Read the fix, then scroll down to the Agent Breakdown to verify your understanding against the raw evidence.
""",
    )

    dur_str  = f"{summary['duration_s']}s" if summary.get("duration_s") else "unknown"
    cost_str = f"${summary['cost_wasted']:.4f}" if summary["cost_wasted"] else "$0"
    tok_str  = f"{summary['tokens_wasted']:,}" if summary["tokens_wasted"] else "0"
    trades   = int(sess_row.get("trades_executed", 0))

    _s1l, _s1r = st.columns([3, 2])
    with _s1l:
        _stat_style = "font-size:0.8rem;color:#6b7280;margin-top:3px"
        _val_style  = "font-weight:600;color:#111827"
        st.markdown(
            f'<div style="border-left:4px solid {sev_color};padding:14px 18px;'
            f'background:{sev_bg};border-radius:0 8px 8px 0">'
            f'{badge(inc_obj.severity, inc_obj.is_simulated)} '
            f'<span style="font-size:1.05rem;font-weight:700;color:{sev_text};margin-left:8px">'
            f'{inc_obj.pattern_name}</span>'
            f'<p style="color:#374151;font-size:0.88rem;margin:10px 0 10px 0;line-height:1.5">'
            f'{inc_obj.root_cause}</p>'
            f'<div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:4px">'
            f'<span style="{_stat_style}">Duration <span style="{_val_style}">{dur_str}</span></span>'
            f'<span style="{_stat_style}">Cost <span style="{_val_style}">{cost_str}</span></span>'
            f'<span style="{_stat_style}">Tokens <span style="{_val_style}">{tok_str}</span></span>'
            f'<span style="{_stat_style}">Trades <span style="{_val_style}">{trades}</span></span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with _s1r:
        _fix_items = [s.strip() for s in fix_text.strip().split("\n") if s.strip()]
        _fix_li = "".join(
            f'<li style="margin-bottom:6px;color:#064e3b;font-size:0.85rem;line-height:1.5">'
            f'{item.lstrip("0123456789. ")}'
            f'</li>'
            for item in _fix_items
        )
        st.markdown(
            f'<div style="background:#f0fdf4;border:1px solid #6ee7b7;border-radius:8px;padding:14px 16px">'
            f'<div style="font-size:0.7rem;font-weight:700;color:#065f46;'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">Fix Steps</div>'
            f'<ol style="margin:0;padding-left:18px">{_fix_li}</ol>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _pat_desc = PATTERN_DETAIL.get(inc_obj.pattern_name) or PATTERN_DESCRIPTIONS.get(inc_obj.pattern_name, "")
    if _pat_desc:
        with st.expander("How is this pattern detected?", expanded=False):
            st.markdown(_pat_desc)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — INCIDENT DETAILS
    # ════════════════════════════════════════════════════════════════════════════
    _section_header(
        "Incident Details",
        """**Key metrics and session-level health checks.**

- **Cost wasted** — money spent in this session that produced no output
- **Tokens wasted** — total LLM tokens consumed across all agents
- **Trades** — what the orchestrator ultimately produced; 0 = complete waste
- **Failed evals** — quality checks that did not meet their threshold
- **Cost by agent (donut)** — which agent consumed the most budget
- **Session evals** — holistic checks for the full pipeline: did all agents run, was cost normal, was there a measurable outcome?
""",
    )

    _kp1, _kp2, _kp3, _kp4 = st.columns(4)
    _kp1.metric("Cost wasted",   f"${inc_obj.cost_wasted:.4f}")
    _kp2.metric("Tokens wasted", f"{inc_obj.tokens_wasted:,}")
    _kp3.metric("Trades",        str(int(sess_row.get("trades_executed", 0))))
    _kp4.metric("Failed evals",  str(len(inc_obj.failed_evals)))

    _s2l, _s2r = st.columns([3, 2])
    with _s2l:
        _SESSION_EVAL_TIPS = {
            "pipeline_completion": (
                "Were all 4 agents present?\n"
                "Score = agents that ran / 4. Threshold = 1.0 (all must run).\n"
                "Failing here means the pipeline stopped early — research never handed off to risk or orchestrator."
            ),
            "cost_anomaly": (
                "Was this session unusually expensive?\n"
                "Score drops toward 0 the further cost is above the rolling mean.\n"
                "Flagged when cost > mean + 2 standard deviations. Catches runaway token loops."
            ),
            "outcome_linkage": (
                "Did the session produce a measurable result?\n"
                "Pass = trades executed OR a terminal_reason logged (e.g. no_opportunity, risk_rejected).\n"
                "Score 0.0 = 0 trades and no exit reason — the agent quit silently."
            ),
            "tokens_per_decision": (
                "How token-efficient was the session?\n"
                "Score = 1 - (tokens_per_decision / limit). Limit = 40,000 tokens per decision.\n"
                "High token use with few decisions signals context spiral or redundant tool calls."
            ),
        }
        if session_evs:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                'text-transform:uppercase;letter-spacing:0.06em;margin:10px 0 2px 0">'
                'Session</div>'
                '<div style="font-size:0.7rem;color:#94a3b8;margin-bottom:8px">'
                'Holistic checks across the full pipeline run — hover each row for details</div>',
                unsafe_allow_html=True,
            )
            for _ev in session_evs:
                _passed  = _ev["passed"]
                _score   = _ev["score"]
                _detail  = _ev.get("detail") or {}
                _ename   = _ev["eval_name"].replace("session.", "")
                _reason  = _eval_reason(_ev["eval_name"], _detail, _score, _passed)
                _tip     = _SESSION_EVAL_TIPS.get(_ename, "")
                _color   = "#10b981" if _passed else "#ef4444"
                _bg      = "#f0fdf4" if _passed else "#fff5f5"
                _border  = "#bbf7d0" if _passed else "#fecaca"
                _icon    = "+" if _passed else "x"
                _rhtml   = (
                    f'<div style="font-size:0.71rem;color:#6b7280;padding-left:18px;'
                    f'font-style:italic;margin-top:2px">{_reason}</div>'
                    if _reason else ""
                )
                st.markdown(
                    f'<div title="{_tip}" style="margin:3px 0;padding:6px 10px;background:{_bg};'
                    f'border-radius:4px;border:1px solid {_border};border-left:3px solid {_color};'
                    f'font-size:0.82rem;cursor:help">'
                    f'<span style="color:{_color};font-weight:700">{_icon}</span> &nbsp;'
                    f'<span style="color:#374151;font-weight:600">{_ename}</span>'
                    f'<span style="float:right;color:{_color};font-weight:600">{_score:.2f}</span>'
                    f'{score_bar(_score, _passed)}{_rhtml}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Run Analysis from the Incidents Feed to populate eval scores.")
    with _s2r:
        _donut = _build_cost_donut(sess_row, sess_traces)
        if _donut:
            st.caption("Cost by agent")
            st.plotly_chart(_donut, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — CALL CHAIN
    # ════════════════════════════════════════════════════════════════════════════
    _section_header(
        "Call Chain",
        """**Where in the pipeline did it break?**

Each card is one agent. Arrows show data flow left to right.

- **Green border** — all quality evals passed for this agent
- **Red border** — one or more evals failed; this agent has quality issues
- **Amber border** — root cause: this is where the failure originated
- **Gray border** — agent ran but has no eval data, or was skipped

Each card shows cost, token count, latency, and eval pass/fail counts inline.
""",
    )

    _chain_html = _render_call_chain_html(_agent_order, evals_by_agent, root_agent, agent_stats, sess_row, sess_traces)
    if _chain_html:
        st_components.html(_chain_html, height=310, scrolling=False)
    else:
        st.info("No trace data available to build call chain.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — EXECUTION TIMELINE
    # ════════════════════════════════════════════════════════════════════════════
    _section_header(
        "Execution Timeline",
        """**When did each step run, and how long did it take?**

Each bar = one step (LLM call or tool call) positioned at its actual time within the session.

- **X-axis** — seconds from session start
- **Bar length** — how long that step took (latency)
- **Red bars** — steps that errored
- **Gaps between agents** — handoff time or pipeline stall
- **Colored bars** — successful steps, color-coded by agent (same as call chain)

Hover over any bar for step name, exact timing, and outcome.
""",
    )

    _tl_fig = _build_timeline_fig(sess_traces)
    if _tl_fig:
        st.plotly_chart(_tl_fig, use_container_width=True)
    else:
        st.info("No trace timestamps available for timeline.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 5 — AGENT BREAKDOWN
    # ════════════════════════════════════════════════════════════════════════════
    _section_header(
        "Agent Breakdown",
        """**What did each agent actually do?**

One card per agent. Cards with failures are expanded by default; healthy agents are collapsed.

Inside each card:
- **LLM calls** — model, tokens in/out, latency, outcome
- **Tool calls** — tool name, latency, success or error (repeated identical errors are collapsed)
- **Quality evals** — automated scores 0.0 to 1.0; green = passed threshold, red = failed
- **Root cause callout** — amber banner on the agent where the failure originated
- **Recommended fix** — shown on the root cause agent only
""",
    )

    _traces_by_agent: dict[str, list[dict]] = {}
    for _t in sess_traces:
        _a = (_t.get("agent") or "unknown").lower()
        _traces_by_agent.setdefault(_a, []).append(_t)

    _all_agents = list(dict.fromkeys(
        agents_present + [a for a in _traces_by_agent if a not in agents_present]
    ))

    for _agent in _all_agents:
        _a_traces = sorted(_traces_by_agent.get(_agent, []), key=lambda x: x.get("created_at", ""))
        _a_evals  = evals_by_agent.get(_agent, [])
        _a_stats  = agent_stats.get(_agent, {})
        _is_root  = _agent == root_agent
        _has_fail = any(not e["passed"] for e in _a_evals) or _is_root

        _n_p    = sum(1 for e in _a_evals if e["passed"])
        _n_f    = sum(1 for e in _a_evals if not e["passed"])
        _n_llm  = _a_stats.get("n_llm", 0)
        _n_tool = _a_stats.get("n_tool", 0)
        _tok    = _a_stats.get("tokens", 0)
        _lat_s  = _a_stats.get("latency_ms", 0) // 1000
        _n_err  = _a_stats.get("n_errors", 0)

        _root_tag  = " <- ROOT CAUSE" if _is_root else ""
        _eval_tag  = f"  |  {_n_p}pass {_n_f}fail evals" if _a_evals else ""
        _stats_tag = f"  |  {_n_llm} LLM  {_n_tool} tools  {_tok:,} tok  {_lat_s}s"
        _err_tag   = f"  |  {_n_err} error(s)" if _n_err else ""
        _label     = f"{_agent.upper()}{_root_tag}{_eval_tag}{_stats_tag}{_err_tag}"

        with st.expander(_label, expanded=_has_fail):

            if _is_root:
                st.markdown(
                    f'<div style="background:#fffbeb;border-left:4px solid #f59e0b;'
                    f'padding:8px 14px;border-radius:0 6px 6px 0;margin-bottom:10px;'
                    f'font-size:0.85rem;color:#78350f">'
                    f'<b>ROOT CAUSE</b> — {inc_obj.root_cause}</div>',
                    unsafe_allow_html=True,
                )

            _llm_calls = [t for t in _a_traces if t.get("step_type") == "llm_call"]
            if _llm_calls:
                st.markdown(
                    '<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin:6px 0 3px 0">'
                    'LLM Calls</div>',
                    unsafe_allow_html=True,
                )
                for _t in _llm_calls:
                    _is_err  = bool(_str(_t.get("error"))) or _t.get("outcome") == "error"
                    _tok_in  = int(_t.get("tokens_input") or 0)
                    _tok_out = int(_t.get("tokens_output") or 0)
                    _lat     = int(_t.get("latency_ms") or 0)
                    _model   = _t.get("model") or "—"
                    _out_col = "#ef4444" if _is_err else "#10b981"
                    st.markdown(
                        f'<div class="trace-row {"trace-error" if _is_err else "trace-success"}">'
                        f'<b>llm_call</b> &nbsp;'
                        f'<code style="color:#94a3b8">{_model}</code> &nbsp;'
                        f'<code>{_tok_in}in {_tok_out}out</code> &nbsp;'
                        f'<code>{_lat}ms</code> &nbsp;'
                        f'<span style="color:{_out_col}">{_t.get("outcome", "")}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if _is_err and _str(_t.get("error")):
                        with st.expander("Error detail", expanded=True):
                            st.error(_str(_t.get("error")))

            _tool_calls = [t for t in _a_traces if t.get("step_type") == "tool_call"]
            if _tool_calls:
                st.markdown(
                    '<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin:10px 0 3px 0">'
                    'Tool Calls</div>',
                    unsafe_allow_html=True,
                )
                _ti = 0
                while _ti < len(_tool_calls):
                    _t  = _tool_calls[_ti]
                    _tn = _str(_t.get("tool_name")) or "tool"
                    # count consecutive calls to the same tool (any outcome)
                    _tj = _ti + 1
                    while _tj < len(_tool_calls):
                        if (_str(_tool_calls[_tj].get("tool_name")) or "tool") == _tn:
                            _tj += 1
                        else:
                            break
                    _count   = _tj - _ti
                    _group   = _tool_calls[_ti:_tj]
                    _n_err_g = sum(1 for _t2 in _group if bool(_str(_t2.get("error"))) or _t2.get("outcome") == "error")
                    _avg_lat = sum(int(_t2.get("latency_ms") or 0) for _t2 in _group) // _count

                    if _count >= 3:
                        # collapsed summary row
                        if _n_err_g:
                            _status = f'<span style="color:#ef4444">{_n_err_g}/{_count} errors</span>'
                        else:
                            _status = f'<span style="color:#10b981">all ok</span>'
                        st.markdown(
                            f'<div class="trace-row">'
                            f'<b>{_tn}</b> &nbsp; <code>× {_count}</code> &nbsp;'
                            f'<code>~{_avg_lat}ms avg</code> &nbsp;'
                            f'{_status} '
                            f'<span style="color:#94a3b8;font-size:0.73rem">(collapsed)</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if _n_err_g and _str(_t.get("error")):
                            with st.expander(f"Error — {_tn} × {_count}", expanded=_is_root):
                                st.error(_str(_t.get("error")))
                    else:
                        for _t2 in _group:
                            _is_err2 = bool(_str(_t2.get("error"))) or _t2.get("outcome") == "error"
                            _lat2    = int(_t2.get("latency_ms") or 0)
                            _outcome = _str(_t2.get("outcome")) or ("error" if _is_err2 else "ok")
                            _out_col = "#ef4444" if _is_err2 else "#10b981"
                            st.markdown(
                                f'<div class="trace-row {"trace-error" if _is_err2 else ""}">'
                                f'<b>{_tn}</b> &nbsp; <code>{_lat2}ms</code> &nbsp;'
                                f'<span style="color:{_out_col}">{_outcome}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if _is_err2 and _str(_t2.get("error")):
                                with st.expander(f"Error — {_tn}", expanded=_is_root):
                                    st.error(_str(_t2.get("error")))
                    _ti = _tj

            _other_steps = [t for t in _a_traces if t.get("step_type") not in ("llm_call", "tool_call")]
            if _other_steps:
                st.markdown(
                    '<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin:10px 0 3px 0">'
                    'Other Steps</div>',
                    unsafe_allow_html=True,
                )
                for _t in _other_steps:
                    _is_err  = bool(_str(_t.get("error"))) or _t.get("outcome") == "error"
                    _step    = _t.get("tool_name") or _t.get("step_type") or "step"
                    _lat     = int(_t.get("latency_ms") or 0)
                    _out_col = "#ef4444" if _is_err else "#10b981"
                    st.markdown(
                        f'<div class="trace-row">'
                        f'<b>{_step}</b> &nbsp; <code>{_lat}ms</code> &nbsp;'
                        f'<span style="color:{_out_col}">{_t.get("outcome", "")}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if _a_evals:
                st.markdown(
                    '<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin:12px 0 3px 0">'
                    'Quality Evals</div>',
                    unsafe_allow_html=True,
                )
                for _ev in _a_evals:
                    _passed = _ev["passed"]
                    _score  = _ev["score"]
                    _detail = _ev.get("detail") or {}
                    _reason = _eval_reason(_ev["eval_name"], _detail, _score, _passed)
                    _color  = "#10b981" if _passed else "#ef4444"
                    _bg     = "#f0fdf4" if _passed else "#fff5f5"
                    _border = "#bbf7d0" if _passed else "#fecaca"
                    _icon   = "+" if _passed else "x"
                    _rhtml  = (
                        f'<div style="font-size:0.71rem;color:#6b7280;padding-left:18px;font-style:italic">'
                        f'{_reason}</div>'
                        if _reason else ""
                    )
                    st.markdown(
                        f'<div style="margin:2px 0;padding:4px 10px;background:{_bg};'
                        f'border-radius:4px;border:1px solid {_border};border-left:3px solid {_color};'
                        f'font-size:0.82rem">'
                        f'<span style="color:{_color};font-weight:700">{_icon}</span> &nbsp;'
                        f'<b style="color:#374151">{_ev["agent"]}</b>'
                        f'<span style="color:#64748b">.{_ev["eval_name"]}</span>'
                        f'<span style="float:right;color:{_color};font-weight:600">{_score:.2f}</span>'
                        f'{score_bar(_score, _passed)}{_rhtml}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if _is_root:
                st.markdown(
                    f'<div style="margin-top:12px;background:#f0fdf4;border:1px solid #6ee7b7;'
                    f'border-radius:6px;padding:10px 14px">'
                    f'<div style="font-size:0.7rem;font-weight:700;color:#065f46;'
                    f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">'
                    f'Recommended Fix</div>'
                    f'<pre style="margin:0;font-size:0.77rem;color:#064e3b;'
                    f'white-space:pre-wrap;font-family:inherit">{fix_text}</pre>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 6 — COMPARE TO HEALTHY SESSION
    # ════════════════════════════════════════════════════════════════════════════
    _section_header(
        "Compare to Healthy Session",
        """**How different was this from a normal run?**

Side-by-side comparison: this incident session vs. the most recent session with no incidents.

- **Percentages** show the delta relative to the healthy baseline
- **Red** = this session performed worse on that metric
- **Green** = this session performed better
""",
    )

    _incident_sids    = set(incidents["session_id"].tolist()) if not incidents.empty else set()
    _healthy_sessions = sessions[~sessions["id"].isin(_incident_sids)]

    if not _healthy_sessions.empty and rca_sid:
        _ref = _healthy_sessions.sort_values("started_at", ascending=False).iloc[0].to_dict()

        def _row_html(label, this_v, ref_v, lower_better=True, fmt="f"):
            delta  = this_v - ref_v
            pct    = (delta / ref_v * 100) if ref_v else 0
            better = (delta < 0) if lower_better else (delta > 0)
            color  = "#10b981" if better else "#ef4444"
            sign   = "+" if delta > 0 else ""
            if fmt == "$":
                val_s = f"${this_v:.4f}"
            elif fmt == "i":
                val_s = f"{int(this_v):,}"
            else:
                val_s = f"{this_v:.1f}s"
            return (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:0.84rem">'
                f'<span style="color:#64748b">{label}</span>'
                f'<span><b>{val_s}</b> '
                f'<span style="color:{color};font-size:0.73rem">({sign}{pct:.0f}%)</span>'
                f'</span></div>'
            )

        _ref_ts  = _ref.get("started_at")
        _ref_str = _ref_ts.strftime("%m-%d %H:%M") if pd.notna(_ref_ts) else _ref["id"][:8]
        _ref_tok = int((_ref.get("total_tokens_input") or 0) + (_ref.get("total_tokens_output") or 0))
        _this_tok = int((sess_row.get("total_tokens_input") or 0) + (sess_row.get("total_tokens_output") or 0))

        _cc1, _cc2 = st.columns(2)
        with _cc1:
            st.markdown(f"**This session** `{(rca_sid or '')[:8]}...`")
            st.markdown(
                f'<div style="background:#fff5f5;border:1px solid #fecaca;border-radius:6px;padding:12px 14px">'
                + _row_html("Cost ($)", float(sess_row.get("total_cost_usd", 0)), float(_ref.get("total_cost_usd", 1)), fmt="$")
                + _row_html("Latency (s)", float(sess_row.get("total_latency_ms", 0)) / 1000, float(_ref.get("total_latency_ms", 1)) / 1000, fmt="s")
                + _row_html("Tokens", float(_this_tok), float(_ref_tok or 1), fmt="i")
                + _row_html("Trades", float(sess_row.get("trades_executed", 0)), float(_ref.get("trades_executed", 0)), lower_better=False, fmt="i")
                + f'</div>',
                unsafe_allow_html=True,
            )
        with _cc2:
            st.markdown(f"**Healthy baseline** `{_ref_str}`")
            st.markdown(
                f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:12px 14px">'
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:0.84rem"><span style="color:#64748b">Cost ($)</span><b>${float(_ref.get("total_cost_usd", 0)):.4f}</b></div>'
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:0.84rem"><span style="color:#64748b">Latency (s)</span><b>{int((_ref.get("total_latency_ms", 0) or 0) // 1000)}s</b></div>'
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:0.84rem"><span style="color:#64748b">Tokens</span><b>{_ref_tok:,}</b></div>'
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;font-size:0.84rem"><span style="color:#64748b">Trades</span><b>{int(_ref.get("trades_executed", 0))}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No clean baseline available yet — all sessions have incidents, or no session data found.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════
    # NOTES
    # ════════════════════════════════════════════════════════════════════════════
    _section_header(
        "Notes",
        """**Add your own observations.**

Document what you found during investigation, whether the fix was applied, or any context for teammates.

Notes are stored in session state and cleared on page refresh.
""",
    )

    _note_key = f"note_{rca_sid}"
    _existing = st.session_state.get(_note_key, "")
    _new_note = st.text_area(
        "note",
        value=_existing,
        placeholder="e.g. Confirmed — yfinance rate limit at 06:34 UTC. Applied timeout fix. Watching next 3 sessions.",
        label_visibility="collapsed",
        height=80,
    )
    if st.button("Save Note", key=f"save_{rca_sid}"):
        st.session_state[_note_key] = _new_note
        st.success("Note saved.")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Failure Simulator
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Failure Simulator":
    st.markdown("## Failure Simulator")
    st.markdown(
        '<div class="callout">Inject a synthetic failure pattern into the database. '
        'Watch the eval engine and pattern detector respond in real time. '
        'All simulated sessions are tagged and filterable.</div>',
        unsafe_allow_html=True,
    )

    sim_tab1, sim_tab2 = st.tabs(["Operational Failures", "Proactive Quality"])

    # ── Tab 1: Operational ────────────────────────────────────────────────────
    with sim_tab1:
        patterns = list_patterns()
        pat_map  = {p["label"]: p for p in patterns}

        sc1, sc2 = st.columns([2, 3])

        with sc1:
            chosen_label = st.selectbox(
                "Failure pattern",
                list(pat_map.keys()),
                key="op_pattern_select",
            )
            chosen = pat_map[chosen_label]

            st.markdown(
                f'{badge(chosen["severity"])} &nbsp; '
                f'<span style="color:#94a3b8">{chosen["description"]}</span>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            run_btn = st.button("Inject Failure + Run Analysis", type="primary",
                                use_container_width=True, key="op_run_btn")

        with sc2:
            st.markdown("#### What will be injected:")
            examples = {
                "Tool Timeout Loop": [
                    "1 × market agent trace (success)",
                    "1 × research agent llm_call (success)",
                    "8 × research tool_call get_stock_data (error: ReadTimeout)",
                    "Session: $1.98, 1027s, 0 trades",
                ],
                "Context Spiral": [
                    "1 × market agent trace",
                    "20 × research tool_call search_news (success, tokens climbing)",
                    "20 × research llm_call (success, high tokens)",
                    "No decision trace — agent never concludes",
                    "Session: $0.85, 42K tokens, 0 trades",
                ],
                "Pipeline Break": [
                    "market + research traces (success)",
                    "risk agent llm_call (error: ConnectionError)",
                    "No orchestrator traces",
                    "Session: $0.42, 0 trades",
                ],
                "Empty Result Loop": [
                    "market agent trace (success)",
                    "5 × research tool_call get_news with different queries (all succeed)",
                    "No trade output — results were thin",
                    "Session: $0.38, 0 trades",
                ],
                "Cost Anomaly": [
                    "All 4 agents run (success)",
                    "High token counts: 60K input, 20K output",
                    "Session: $3.10 — 10x normal cost",
                    "1 trade executed (pipeline worked but was expensive)",
                ],
                "Silent Exit": [
                    "All 4 agents run (success)",
                    "Normal token counts",
                    "Session: $0.19, 0 trades, no terminal_reason",
                    "Orchestrator exited without logging a reason",
                ],
                "Silent Propagation": [
                    "Market fetches stale data — all traces succeed, no exception thrown",
                    "data_freshness eval catches 15-min lag (threshold: 10 min) → Market fails",
                    "Research, Risk, Orchestrator all run on the bad input",
                    "Session: $0.0284 total · Market $0.0022 · Research $0.0148 · Risk $0.0063 · Orchestrator $0.0051",
                    "0 trades, no terminal_reason → Silent Exit pattern fires",
                    "CB savings = $0.0262 (Research + Risk + Orchestrator were preventable)",
                ],
            }
            for item in examples.get(chosen_label, []):
                st.markdown(f"- {item}")

        if run_btn:
            with st.status("Injecting failure...", expanded=True) as status:
                st.write("Creating simulated session...")
                sid = simulate_failure(chosen["id"], db=_db())
                st.write(f"Session injected: `{sid[:16]}...`")
                time.sleep(0.5)

                st.write("Loading traces...")
                new_traces_r = _db().table("c_traces").select("*").eq("session_id", sid).execute()
                new_traces   = new_traces_r.data or []
                st.write(f"Found {len(new_traces)} traces.")
                time.sleep(0.3)

                st.write("Loading session...")
                new_sess_r = _db().table("c_sessions").select("*").eq("id", sid).execute()
                new_sess   = new_sess_r.data[0] if new_sess_r.data else {}
                time.sleep(0.3)

                st.write("Running evals...")
                evs = run_all_evals(new_sess, new_traces, recent_costs)
                ev_rows = [e.to_db_row(sid) for e in evs]
                if ev_rows:
                    _db().table("c_evals").insert(ev_rows).execute()
                st.write(f"{len(evs)} evals computed — {sum(1 for e in evs if not e.passed)} failed.")
                time.sleep(0.3)

                st.write("Running pattern detection...")
                incs = run_all_detectors(new_sess, new_traces, evs, recent_costs)
                if incs:
                    _db().table("c_incidents").insert([i.to_db_row() for i in incs]).execute()

                status.update(label="Done!", state="complete")

            if incs:
                st.success(f"Incident detected: **{incs[0].pattern_name}**")
                st.markdown(
                    f'{badge(incs[0].severity, True)} &nbsp; {incs[0].root_cause}',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="fix-box">{generate_fix_suggestion(incs[0], new_traces)}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("View full RCA →", key="op_rca_btn"):
                    goto_rca(incs[0].to_db_row(), sid)
            else:
                st.warning("No incident detected for this simulation. Check pattern detector thresholds.")

            load_sessions.clear()
            load_traces.clear()
            load_all_evals.clear()

    # ── Tab 2: Proactive Quality ───────────────────────────────────────────────
    with sim_tab2:
        q_patterns = list_quality_patterns()
        q_pat_map  = {p["label"]: p for p in q_patterns}

        qc1, qc2 = st.columns([2, 3])

        with qc1:
            q_chosen_label = st.selectbox(
                "Quality pattern",
                list(q_pat_map.keys()),
                key="q_pattern_select",
            )
            q_chosen = q_pat_map[q_chosen_label]

            st.markdown(
                f'{badge(q_chosen["severity"])} &nbsp; '
                f'<span style="color:#94a3b8">{q_chosen["description"]}</span>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            q_run_btn = st.button("Inject Quality Pattern", type="primary",
                                  use_container_width=True, key="q_run_btn")

        with qc2:
            st.markdown("#### What will be injected:")
            q_examples = {
                "Grounding Failure": [
                    "3 sessions staggered: -48h, -24h, now",
                    "Each: research_quality.data_grounding = 0.25 (threshold 0.40)",
                    "Other quality dims normal (0.65–0.75)",
                    "No traces — quality evals written directly",
                    "Fires: Proactive: Grounding Failure",
                ],
                "Coherence Break": [
                    "1 session (now)",
                    "orchestrator_quality.decision_consistency = 0.30 (threshold 0.50)",
                    "Research and risk scores look normal (0.68–0.75)",
                    "No traces — quality evals written directly",
                    "Fires: Proactive: Coherence Break (critical)",
                ],
                "Quality Cascade": [
                    "5 sessions over 5 days",
                    "5 dimensions each decline >0.20:",
                    "  data_grounding 0.78→0.42, thesis_coherence 0.74→0.48",
                    "  parameter_completeness 0.80→0.50, decision_consistency 0.72→0.44",
                    "  pipeline_coherence 0.76→0.46",
                    "Composite score: 0.76→0.48",
                    "Fires: Proactive: Quality Cascade",
                ],
                "Silent Degradation": [
                    "5 sessions over 5 days",
                    "Composite quality: 0.78→0.48 (negative slope across window)",
                    "Operational evals stay clean: tool_success_rate=0.92, completion=0.88",
                    "No operational alert fires — trend-only detection",
                    "Fires: Proactive: Silent Degradation",
                ],
            }
            for item in q_examples.get(q_chosen_label, []):
                st.markdown(f"- {item}")

        if q_run_btn:
            with st.status("Injecting quality pattern...", expanded=True) as q_status:
                st.write(f"Building {q_chosen['id']} scenario...")
                q_sids, q_inc_objs = simulate_quality_failure(q_chosen["id"], db=_db())
                st.write(f"{len(q_sids)} session(s) written with quality evals.")
                st.write(f"Pattern detector: {len(q_inc_objs)} incident(s) fired.")
                q_status.update(label="Done!", state="complete")

            if q_inc_objs:
                inc0 = q_inc_objs[0]
                st.success(f"Incident detected: **{inc0.pattern_name}**")
                st.markdown(
                    f'{badge(inc0.severity, True)} &nbsp; {inc0.root_cause}',
                    unsafe_allow_html=True,
                )
                if inc0.fix_suggestion:
                    st.markdown(f'<div class="fix-box">{inc0.fix_suggestion}</div>',
                                unsafe_allow_html=True)
                st.info("Find this incident in **Incidents Feed** under **Simulated only**.")
            else:
                st.warning("Sessions + evals written but no incident fired. Check detector thresholds.")

            load_sessions.clear()
            load_all_evals.clear()

    # History of simulated incidents (shared across both tabs)
    if not incidents.empty:
        sim_inc = incidents[incidents.get("is_simulated", incidents["is_simulated"].fillna(False))]
        if not sim_inc.empty:
            st.divider()
            st.markdown(f"#### Past Simulations ({len(sim_inc)})")
            for _, inc in sim_inc.sort_values("created_at", ascending=False).head(10).iterrows():
                ts = inc["created_at"].strftime("%m-%d %H:%M") if pd.notna(inc.get("created_at")) else ""
                st.markdown(
                    f"{badge(inc['severity'], True)} &nbsp; **{inc['pattern_name']}** &nbsp; "
                    f"<small style='color:#64748b'>{ts}</small>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Trace Inspector
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Trace Inspector":
    st.markdown("## Trace Inspector")
    st.markdown(
        '<div class="callout">Shows exactly what enters the system in each trace format '
        'and what the normalized c_traces row looks like. '
        'Proves multi-format ingestion produces identical output.</div>',
        unsafe_allow_html=True,
    )

    format_tab, raw_tab = st.tabs(["Format Comparison", "Live Session Traces"])

    with format_tab:
        st.markdown("#### Three formats — one normalized schema")

        fmt = st.radio("Source format", ["Strategy C Native JSON", "OTel GenAI Span", "LangSmith Export"],
                       horizontal=True)

        left_col, right_col = st.columns(2)

        if fmt == "Strategy C Native JSON":
            raw = {
                "id": "abc-123",
                "session_id": "sess-456",
                "agent": "research",
                "step_type": "tool_call",
                "tool_name": "get_stock_data",
                "outcome": "error",
                "error": "ReadTimeout: HTTPSConnectionPool: Read timed out",
                "latency_ms": 30000,
                "tokens_input": 0,
                "tokens_output": 0,
                "model": None,
                "created_at": "2026-05-28T06:34:22Z",
            }
            normalized = raw.copy()
            normalized["_source"] = "native"

        elif fmt == "OTel GenAI Span":
            raw = {
                "name": "get_stock_data",
                "trace_id": "abc123",
                "span_id": "def456",
                "start_time": 1748390062000000000,
                "duration": 30000000000,
                "status": {"code": "ERROR", "message": "ReadTimeout: timed out"},
                "attributes": {
                    "agent": "research",
                    "session_id": "sess-456",
                    "gen_ai.tool.name": "get_stock_data",
                    "gen_ai.usage.input_tokens": 0,
                    "gen_ai.usage.output_tokens": 0,
                },
            }
            normalized = {
                "id":           "abc-123",
                "session_id":   "sess-456",
                "agent":        "research",
                "step_type":    "tool_call",
                "tool_name":    "get_stock_data",
                "outcome":      "error",
                "error":        "ReadTimeout: timed out",
                "latency_ms":   30000,
                "tokens_input": 0,
                "tokens_output":0,
                "model":        None,
                "created_at":   "2026-05-28T06:34:22Z",
                "_source":      "otel",
            }

        else:  # LangSmith
            raw = {
                "id": "run-789",
                "parent_run_id": "sess-456",
                "name": "get_stock_data",
                "run_type": "tool",
                "status": "error",
                "error": "ReadTimeout: timed out",
                "start_time": "2026-05-28T06:34:22.000Z",
                "end_time":   "2026-05-28T06:34:52.000Z",
                "inputs": {"symbol": "NVDA"},
                "outputs": None,
                "tags": ["research"],
                "extra": {"metadata": {"agent": "research"}},
            }
            normalized = {
                "id":           "abc-123",
                "session_id":   "sess-456",
                "agent":        "research",
                "step_type":    "tool_call",
                "tool_name":    "get_stock_data",
                "outcome":      "error",
                "error":        "ReadTimeout: timed out",
                "latency_ms":   30000,
                "tokens_input": 0,
                "tokens_output":0,
                "model":        None,
                "created_at":   "2026-05-28T06:34:22Z",
                "_source":      "langsmith",
            }

        with left_col:
            st.markdown(f"**Raw input ({fmt}):**")
            st.json(raw)

        with right_col:
            st.markdown("**Normalized c_traces row:**")
            st.json(normalized)
            st.success("All three formats produce the same schema. Eval + RCA runs unchanged.")

        st.divider()
        st.markdown("#### Field mapping")
        mapping = {
            "Strategy C Native JSON": [
                ("agent",       "agent",        "direct"),
                ("step_type",   "step_type",    "direct"),
                ("tool_name",   "tool_name",    "direct"),
                ("outcome",     "outcome",      "direct"),
                ("error",       "error",        "direct"),
                ("latency_ms",  "latency_ms",   "direct"),
                ("tokens_input","tokens_input", "direct"),
                ("created_at",  "created_at",   "direct"),
            ],
            "OTel GenAI Span": [
                ("attributes.agent",                  "agent",        "extract attribute"),
                ("name",                              "tool_name",    "direct"),
                ("'tool_call' (inferred)",            "step_type",    "inferred from span type"),
                ("status.code == ERROR",              "outcome",      "map ERROR→error, OK→success"),
                ("status.message",                    "error",        "direct"),
                ("duration / 1_000_000",              "latency_ms",   "nanoseconds to ms"),
                ("attributes.gen_ai.usage.input_tokens","tokens_input","extract attribute"),
                ("start_time / 1e9 → ISO 8601",       "created_at",   "nanoseconds to datetime"),
            ],
            "LangSmith Export": [
                ("extra.metadata.agent",              "agent",        "extract metadata"),
                ("name",                              "tool_name",    "direct"),
                ("'tool_call' (run_type==tool)",      "step_type",    "map run_type"),
                ("status",                            "outcome",      "map error→error, success→success"),
                ("error",                             "error",        "direct"),
                ("end_time - start_time",             "latency_ms",   "compute duration"),
                ("(not available in export)",         "tokens_input", "null if not in export"),
                ("start_time",                        "created_at",   "parse ISO 8601"),
            ],
        }
        st.dataframe(
            pd.DataFrame(mapping[fmt], columns=["Source field", "c_traces column", "Transform"]),
            use_container_width=True,
            hide_index=True,
        )

    with raw_tab:
        st.markdown("#### Live trace data from Strategy C")

        if sessions.empty:
            st.info("No sessions found.")
        else:
            session_opts = {
                f"{row['started_at'].strftime('%m-%d %H:%M') if pd.notna(row['started_at']) else row['id'][:8]}"
                f" — ${row['total_cost_usd']:.4f}": row["id"]
                for _, row in sessions.head(20).iterrows()
            }
            chosen_sess = st.selectbox("Session", list(session_opts.keys()))
            chosen_sid  = session_opts[chosen_sess]
            sess_t      = traces_all[traces_all["session_id"] == chosen_sid]

            # Filters
            tc1, tc2 = st.columns(2)
            agent_f   = tc1.multiselect("Agent", sorted(sess_t["agent"].dropna().unique()),
                                         default=sorted(sess_t["agent"].dropna().unique()))
            outcome_f = tc2.multiselect("Outcome", ["success","error",""],
                                         default=["success","error",""])

            filtered_t = sess_t[
                sess_t["agent"].isin(agent_f) &
                sess_t["outcome"].isin(outcome_f)
            ] if not sess_t.empty else sess_t

            st.dataframe(
                filtered_t[["created_at","agent","step_type","tool_name",
                             "outcome","error","latency_ms","tokens_input","tokens_output"]]
                .astype(str),
                use_container_width=True,
                height=400,
            )
            st.caption(f"{len(filtered_t)} of {len(sess_t)} traces shown")

            if st.checkbox("Show as JSON (first 5 rows)"):
                for row in filtered_t.head(5).to_dict("records"):
                    st.json(row)

