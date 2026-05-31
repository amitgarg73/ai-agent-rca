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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client

from engine.eval_engine   import run_all_evals
from engine.pattern_detector import run_all_detectors
from engine.rca_engine    import build_annotated_call_stack, generate_fix_suggestion, summarize_incident
from simulator.failure_sim import simulate_failure, list_patterns

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
</style>
""", unsafe_allow_html=True)


# ── DB ────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _db():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    return create_client(url, key)


@st.cache_data(ttl=60)
def load_sessions() -> pd.DataFrame:
    r = _db().table("c_sessions").select(
        "id,date,total_cost_usd,total_latency_ms,total_tokens_input,"
        "total_tokens_output,trades_proposed,trades_executed,agents_invoked,"
        "terminal_reason,started_at,completed_at,cost_breakdown"
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
        r = _db().table("c_evals").select(
            "session_id,agent,eval_name,score,passed"
        ).execute()
        df = pd.DataFrame(r.data or [])
        if not df.empty:
            df["score"]  = pd.to_numeric(df["score"],  errors="coerce").fillna(0)
            df["passed"] = df["passed"].astype(bool)
        return df
    except Exception:
        return pd.DataFrame()


def goto_rca(incident_dict: dict, session_id: str) -> None:
    """Navigate to RCA View with the given incident pre-selected."""
    st.session_state["rca_incident"] = incident_dict
    st.session_state["rca_sid"]      = session_id
    st.session_state["_page"]        = "RCA View"
    st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    "Ledger", "Quality Drift",
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
    page = st.query_params.get("page", "Ledger")
    if page not in _NAV_PAGES:
        page = "Ledger"

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

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total Sessions", str(len(sessions))), unsafe_allow_html=True)
    c2.markdown(kpi("Total Spend", f"${total_cost:.2f}"), unsafe_allow_html=True)
    c3.markdown(kpi("Trades Executed", str(int(total_trade))), unsafe_allow_html=True)
    c4.markdown(kpi("Wasted Sessions",
        f"{len(wasted)} · ${wasted_cost:.2f}",
        f"{wasted_pct:.0f}% of spend" if total_cost else ""),
        unsafe_allow_html=True)
    c5.markdown(kpi("Incidents Detected", str(inc_count)), unsafe_allow_html=True)

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
    else:
        display["Incidents"] = 0

    display["Wasted"] = (display["trades_executed"] == 0) & (display["total_cost_usd"] > 0.01)

    cols = ["started_at", "total_cost_usd", "Duration (s)", "trades_executed",
            "Tokens", "Incidents", "terminal_reason"]
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
            st.markdown("**Evals**")
            _ec = st.columns(2)
            for _i, (_, _ev) in enumerate(_dev.iterrows()):
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
    st.markdown("## Quality Drift")
    st.caption(
        "Are the agents getting worse over time? "
        "Each chart answers a different dimension: output decisions, eval health, "
        "critical per-agent scores, and pipeline completion."
    )

    if sessions.empty:
        st.info("No sessions found.")
        st.stop()

    s = sessions.sort_values("started_at").copy()
    s["label"] = s["started_at"].dt.strftime("%m-%d %H:%M")
    all_evals  = load_all_evals()

    # Join evals with session timestamps
    if not all_evals.empty and not s.empty:
        evals_ts = all_evals.merge(
            s[["id", "started_at", "label"]],
            left_on="session_id", right_on="id", how="left",
        ).sort_values("started_at")
    else:
        evals_ts = pd.DataFrame()

    # ── Section 1: Health KPIs ────────────────────────────────────────────────
    _n_recent = min(7, len(s))
    _n_prev   = min(7, max(0, len(s) - _n_recent))
    if not evals_ts.empty:
        _recent_sids = s.iloc[-_n_recent:]["id"].tolist()
        _prev_sids   = s.iloc[-_n_recent - _n_prev : -_n_recent]["id"].tolist() if _n_prev else []
        _recent_pass = evals_ts[evals_ts["session_id"].isin(_recent_sids)]["passed"].mean()
        _prev_pass   = evals_ts[evals_ts["session_id"].isin(_prev_sids)]["passed"].mean() if _prev_sids else None
        _pass_delta  = (_recent_pass - _prev_pass) if _prev_pass is not None else None
    else:
        _recent_pass = _pass_delta = None

    _recent_trades    = s.iloc[-_n_recent:]["trades_executed"].mean() if _n_recent else 0
    _prev_trades      = s.iloc[-_n_recent - _n_prev : -_n_recent]["trades_executed"].mean() if _n_prev else None
    _trades_delta     = (_recent_trades - _prev_trades) if _prev_trades is not None else None

    _recent_inc_count = len(incidents[incidents["session_id"].isin(s.iloc[-_n_recent:]["id"])]) if not incidents.empty else 0

    kc1, kc2, kc3 = st.columns(3)
    kc1.markdown(
        kpi("Eval Pass Rate (last 7)",
            f"{_recent_pass*100:.0f}%" if _recent_pass is not None else "—",
            (f"{'▲' if _pass_delta >= 0 else '▼'} {abs(_pass_delta)*100:.0f}pp vs prior 7"
             if _pass_delta is not None else ""),
        ), unsafe_allow_html=True,
    )
    kc2.markdown(
        kpi("Avg Trades / Session (last 7)",
            f"{_recent_trades:.1f}",
            (f"{'▲' if _trades_delta >= 0 else '▼'} {abs(_trades_delta):.1f} vs prior 7"
             if _trades_delta is not None else ""),
        ), unsafe_allow_html=True,
    )
    kc3.markdown(
        kpi("Incidents (last 7 sessions)", str(_recent_inc_count)),
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Section 2: Rolling eval pass rate by agent ────────────────────────────
    st.markdown("#### Eval Pass Rate by Agent")
    st.caption("Rolling 5-session average. A declining line means that agent's quality is degrading.")

    if not evals_ts.empty:
        _agent_pass = (
            evals_ts.groupby(["session_id", "agent", "started_at"])["passed"]
            .mean().reset_index()
            .sort_values("started_at")
        )
        fig_pass = go.Figure()
        for _ag, _clr in AGENT_COLORS.items():
            _ag_data = _agent_pass[_agent_pass["agent"] == _ag].copy()
            if len(_ag_data) < 2:
                continue
            _ag_data["rolling"] = _ag_data["passed"].rolling(5, min_periods=1).mean()
            _ag_data["lbl"] = _ag_data["started_at"].dt.strftime("%m-%d %H:%M")
            fig_pass.add_trace(go.Scatter(
                x=_ag_data["lbl"], y=_ag_data["rolling"],
                mode="lines+markers", name=_ag,
                line=dict(color=_clr, width=2),
                marker=dict(size=5),
            ))
        # Incident markers
        if not incidents.empty:
            for _, _inc in incidents.iterrows():
                _match = s[s["id"] == _inc["session_id"]]
                if not _match.empty:
                    _x = _match.iloc[0]["label"]
                    _c = "#ef4444" if _inc["severity"] == "critical" else "#f59e0b"
                    fig_pass.add_shape(type="line", x0=_x, x1=_x, y0=0, y1=1,
                        xref="x", yref="paper", line=dict(color=_c, dash="dot", width=1))
        fig_pass.update_layout(
            paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
            font_color="#1e293b", height=300,
            yaxis=dict(title="Pass rate", tickformat=".0%", range=[0, 1.05]),
            xaxis_tickangle=-30, legend=dict(orientation="h", y=1.08),
            margin=dict(t=40, b=60),
        )
        st.plotly_chart(fig_pass, use_container_width=True)
    else:
        st.info("No eval data found. Run the backfill script to populate c_evals.")

    st.divider()

    # ── Section 3: Decision rate + cost trend ─────────────────────────────────
    st.markdown("#### Decision Rate vs Cost")
    st.caption(
        "Cost staying flat while trades/session drops = agent running but not deciding. "
        "Cost rising while trades stay flat = waste."
    )

    s["rolling_trades"] = s["trades_executed"].rolling(7, min_periods=1).mean()
    s["rolling_cost"]   = s["total_cost_usd"].rolling(7, min_periods=1).mean()

    fig_dr = go.Figure()
    fig_dr.add_trace(go.Bar(
        x=s["label"], y=s["trades_executed"],
        name="Trades", marker_color="#10b981", opacity=0.6,
        yaxis="y",
    ))
    fig_dr.add_trace(go.Scatter(
        x=s["label"], y=s["rolling_trades"],
        mode="lines", name="7-session avg trades",
        line=dict(color="#064e3b", width=2, dash="dash"),
        yaxis="y",
    ))
    fig_dr.add_trace(go.Scatter(
        x=s["label"], y=s["total_cost_usd"],
        mode="lines+markers", name="Cost ($)",
        line=dict(color="#f59e0b", width=1.5),
        marker=dict(size=5),
        yaxis="y2",
    ))
    if not incidents.empty:
        for _, _inc in incidents.iterrows():
            _match = s[s["id"] == _inc["session_id"]]
            if not _match.empty:
                _x = _match.iloc[0]["label"]
                _c = "#ef4444" if _inc["severity"] == "critical" else "#f59e0b"
                fig_dr.add_shape(type="line", x0=_x, x1=_x, y0=0, y1=1,
                    xref="x", yref="paper", line=dict(color=_c, dash="dot", width=1))
    fig_dr.update_layout(
        paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
        font_color="#1e293b", height=300,
        xaxis_tickangle=-30,
        yaxis=dict(title="Trades", side="left"),
        yaxis2=dict(title="Cost (USD)", side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig_dr, use_container_width=True)

    st.divider()

    # ── Section 4: Critical eval scores over time ─────────────────────────────
    st.markdown("#### Critical Eval Scores Over Time")
    st.caption(
        "These three evals are leading indicators of failure. "
        "Scores below their thresholds are where incidents originate."
    )

    _KEY_EVALS = {
        "research.tool_success_rate":    ("#f59e0b", 0.80),
        "orchestrator.exit_quality":     ("#3b82f6", 0.70),
        "risk.assessment_complete":      ("#10b981", 1.00),
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
            # Threshold line
            fig_ev.add_hline(
                y=_thr, line_dash="dot", line_color=_clr,
                annotation_text=f"{_key} threshold",
                annotation_position="bottom right",
                annotation_font_size=9,
            )
        if not incidents.empty:
            for _, _inc in incidents.iterrows():
                _match = s[s["id"] == _inc["session_id"]]
                if not _match.empty:
                    _x = _match.iloc[0]["label"]
                    _c = "#ef4444" if _inc["severity"] == "critical" else "#f59e0b"
                    fig_ev.add_shape(type="line", x0=_x, x1=_x, y0=0, y1=1,
                        xref="x", yref="paper", line=dict(color=_c, dash="dot", width=1))
        fig_ev.update_layout(
            paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
            font_color="#1e293b", height=320,
            yaxis=dict(title="Score", range=[-0.05, 1.1]),
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=40, b=60),
        )
        st.plotly_chart(fig_ev, use_container_width=True)
    else:
        st.info("No eval data found.")

    st.divider()

    # ── Section 5: Pipeline completion rate ───────────────────────────────────
    st.markdown("#### Pipeline Completion Rate")
    st.caption("Fraction of sessions where all 4 agents ran. Drops signal systemic pipeline breaks.")

    if not evals_ts.empty:
        _pc = evals_ts[evals_ts["eval_name"] == "pipeline_completion"].copy()
        if not _pc.empty:
            _pc = _pc.sort_values("started_at")
            _pc["lbl"] = _pc["started_at"].dt.strftime("%m-%d %H:%M")
            _pc["rolling_pc"] = _pc["score"].rolling(7, min_periods=1).mean()

            fig_pc = go.Figure()
            fig_pc.add_trace(go.Bar(
                x=_pc["lbl"], y=_pc["score"],
                name="Completed",
                marker_color=["#10b981" if v == 1.0 else "#ef4444" for v in _pc["score"]],
                opacity=0.7,
            ))
            fig_pc.add_trace(go.Scatter(
                x=_pc["lbl"], y=_pc["rolling_pc"],
                mode="lines", name="7-session avg",
                line=dict(color="#0f172a", width=2, dash="dash"),
            ))
            fig_pc.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                font_color="#1e293b", height=250,
                yaxis=dict(title="Score", range=[0, 1.1]),
                xaxis_tickangle=-30,
                legend=dict(orientation="h", y=1.08),
                margin=dict(t=40, b=60),
            )
            st.plotly_chart(fig_pc, use_container_width=True)
        else:
            st.info("No pipeline_completion eval data found.")


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
    fc1, fc2, fc3 = st.columns(3)
    sev_filter     = fc1.multiselect("Severity",
                                      ["critical","warning","info"],
                                      default=["critical","warning","info"])
    pattern_filter = fc2.multiselect("Pattern",
                                      sorted(incidents["pattern_name"].unique().tolist()),
                                      default=incidents["pattern_name"].unique().tolist())
    sim_filter     = fc3.radio("Show", ["All","Real only","Simulated only"],
                                horizontal=True)

    filtered = incidents.copy()
    if sev_filter:
        filtered = filtered[filtered["severity"].isin(sev_filter)]
    if pattern_filter:
        filtered = filtered[filtered["pattern_name"].isin(pattern_filter)]
    if sim_filter == "Real only":
        filtered = filtered[~filtered.get("is_simulated", False)]
    elif sim_filter == "Simulated only":
        filtered = filtered[filtered.get("is_simulated", filtered["is_simulated"].fillna(False))]

    # Summary KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.markdown(kpi("Total Incidents",  str(len(filtered))),                       unsafe_allow_html=True)
    kc2.markdown(kpi("Critical",         str((filtered["severity"]=="critical").sum())), unsafe_allow_html=True)
    kc3.markdown(kpi("Cost Wasted",      f"${filtered['cost_wasted'].sum():.2f}"),   unsafe_allow_html=True)
    kc4.markdown(kpi("Patterns Seen",    str(filtered["pattern_name"].nunique())),   unsafe_allow_html=True)

    st.divider()

    # Incident rows
    for _, inc in filtered.sort_values("created_at", ascending=False).iterrows():
        sim  = bool(inc.get("is_simulated", False))
        ts   = inc["created_at"].strftime("%b %d %H:%M") if pd.notna(inc.get("created_at")) else ""
        cost = f"${inc['cost_wasted']:.4f}" if inc["cost_wasted"] > 0 else "—"
        tok  = f"{int(inc.get('tokens_wasted',0)):,}" if inc.get("tokens_wasted") else "—"

        with st.container():
            hc1, hc2, hc3, hc4, hc5 = st.columns([1.5, 2.5, 3.5, 1, 1])
            hc1.markdown(badge(inc["severity"], sim), unsafe_allow_html=True)
            hc2.markdown(f"**{inc['pattern_name']}**")
            hc3.markdown(f"<small style='color:#94a3b8'>{inc['root_cause'][:90]}...</small>",
                         unsafe_allow_html=True)
            hc4.markdown(f"<small>{cost}</small>", unsafe_allow_html=True)
            if hc5.button("RCA →", key=f"go_{inc['id']}"):
                goto_rca(inc.to_dict(), inc.get("session_id"))
        st.divider()



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

    patterns = list_patterns()
    pat_map  = {p["label"]: p for p in patterns}

    sc1, sc2 = st.columns([2, 3])

    with sc1:
        chosen_label = st.selectbox(
            "Failure pattern",
            list(pat_map.keys()),
        )
        chosen = pat_map[chosen_label]

        st.markdown(
            f'{badge(chosen["severity"])} &nbsp; '
            f'<span style="color:#94a3b8">{chosen["description"]}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        run_btn = st.button("Inject Failure + Run Analysis", type="primary",
                            use_container_width=True)

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
            if st.button("View full RCA →"):
                goto_rca(incs[0].to_db_row(), sid)
        else:
            st.warning("No incident detected for this simulation. Check pattern detector thresholds.")

        st.cache_data.clear()

    # History of simulated incidents
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

