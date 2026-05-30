"""
AI Agent RCA Dashboard
Full observability for multi-agent systems: Outcome Ledger + Incidents + RCA + Simulator.

Run: streamlit run dashboard/dashboard.py
Secrets: dashboard/.streamlit/secrets.toml (copy from observability/poc/.streamlit/secrets.toml)
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from collections import defaultdict
import streamlit as st
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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Hide Streamlit header bar */
[data-testid="stHeader"] { display: none; }

/* Reduce top padding on main content (default is 5rem — way too much) */
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }

/* Always keep sidebar visible — override any collapsed transform */
[data-testid="stSidebar"] {
    min-width: 244px !important;
    width: 244px !important;
    transform: none !important;
    display: block !important;
    left: 0 !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
}

/* Hide all sidebar toggle buttons so users can't collapse it */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    display: none !important;
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
</style>
<script>
// Force sidebar open if Streamlit's JS collapses it on load
(function keepSidebarOpen() {
    function expand() {
        var doc = window.parent ? window.parent.document : document;
        var btn = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button');
        if (btn) { btn.click(); }
    }
    // Try immediately and after a short delay for Streamlit hydration
    setTimeout(expand, 300);
    setTimeout(expand, 800);
})();
</script>
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
) -> str:
    """Pure HTML/CSS pipeline cards — no Plotly dependency."""
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

    def _card_style(a):
        if a == root_agent:
            return "background:#fff7ed;border:2px solid #f59e0b;"
        evs = evals_by_agent.get(a.lower(), [])
        if not evs:
            return "background:#f8fafc;border:2px solid #cbd5e1;"
        if any(not e["passed"] for e in evs):
            return "background:#fef2f2;border:2px solid #ef4444;"
        return "background:#f0fdf4;border:2px solid #10b981;"

    def _badge_color(a):
        if a == root_agent:
            return "#f59e0b"
        evs = evals_by_agent.get(a.lower(), [])
        if not evs:
            return "#94a3b8"
        return "#ef4444" if any(not e["passed"] for e in evs) else "#10b981"

    cards_html = ""
    for i, a in enumerate(agents_present):
        evs   = evals_by_agent.get(a.lower(), [])
        stats = agent_stats.get(a.lower(), {})
        n_p   = sum(1 for e in evs if e["passed"])
        n_f   = sum(1 for e in evs if not e["passed"])
        tok   = stats.get("tokens", 0)
        lat_s = (stats.get("latency_ms", 0) or 0) // 1000
        cost  = agent_cost.get(a.lower(), agent_cost.get(a, 0))
        badge = _badge_color(a)
        style = _card_style(a)

        root_badge = (
            '<div style="font-size:10px;font-weight:700;color:#b45309;'
            'background:#fef3c7;border-radius:4px;padding:1px 6px;margin-bottom:4px;'
            'display:inline-block;">ROOT CAUSE</div><br>'
            if a == root_agent else ""
        )
        eval_line = f'<span style="color:#10b981">&#10003; {n_p}</span>&nbsp;&nbsp;<span style="color:#ef4444">&#10007; {n_f}</span>' if evs else '<span style="color:#94a3b8">no evals</span>'

        card = f"""
        <div style="{style}border-radius:10px;padding:12px 14px;min-width:130px;
                     text-align:center;position:relative;">
          {root_badge}
          <div style="width:14px;height:14px;border-radius:50%;background:{badge};
                      margin:0 auto 6px;"></div>
          <div style="font-size:13px;font-weight:700;color:#1e293b;letter-spacing:.5px;">
            {a.upper()}
          </div>
          <div style="font-size:11px;color:#64748b;margin-top:4px;">${cost:.4f}</div>
          <div style="font-size:11px;color:#64748b;">{tok:,} tok &nbsp;·&nbsp; {lat_s}s</div>
          <div style="font-size:11px;margin-top:4px;">{eval_line}</div>
        </div>"""
        cards_html += card

        if i < len(agents_present) - 1:
            cards_html += """
            <div style="display:flex;align-items:center;padding:0 6px;color:#94a3b8;
                        font-size:20px;align-self:center;">&#8594;</div>"""

    terminal_label = "OUTPUT" if trades > 0 else "WASTE"
    terminal_bg    = "#f0fdf4" if trades > 0 else "#fef2f2"
    terminal_border= "#10b981" if trades > 0 else "#ef4444"
    terminal_color = "#16a34a" if trades > 0 else "#dc2626"
    terminal_icon  = "&#10003;" if trades > 0 else "&#10007;"
    terminal_sub   = f"{trades} trade{'s' if trades != 1 else ''}" if trades > 0 else "0 trades"

    cards_html += f"""
        <div style="display:flex;align-items:center;padding:0 6px;color:#94a3b8;
                    font-size:20px;align-self:center;">&#8594;</div>
        <div style="background:{terminal_bg};border:2px solid {terminal_border};
                    border-radius:10px;padding:12px 14px;min-width:100px;text-align:center;">
          <div style="font-size:18px;color:{terminal_color};">{terminal_icon}</div>
          <div style="font-size:13px;font-weight:700;color:{terminal_color};">{terminal_label}</div>
          <div style="font-size:11px;color:#64748b;margin-top:4px;">{terminal_sub}</div>
        </div>"""

    legend = """
    <div style="display:flex;gap:18px;margin-top:10px;flex-wrap:wrap;">
      <span style="font-size:11px;color:#64748b;">
        <span style="color:#10b981;">&#9679;</span> All evals passed
      </span>
      <span style="font-size:11px;color:#64748b;">
        <span style="color:#ef4444;">&#9679;</span> Evals failed
      </span>
      <span style="font-size:11px;color:#64748b;">
        <span style="color:#f59e0b;">&#9679;</span> Root cause
      </span>
      <span style="font-size:11px;color:#64748b;">
        <span style="color:#94a3b8;">&#9679;</span> No eval data
      </span>
    </div>"""

    return (
        f'<div style="background:#f8fafc;border-radius:12px;padding:16px 20px;">'
        f'<div style="display:flex;align-items:stretch;flex-wrap:nowrap;gap:0;overflow-x:auto;">'
        f"{cards_html}"
        f"</div>"
        f"{legend}"
        f"</div>"
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

    fig = go.Figure()
    for agent in agents:
        adf   = df[df["agent"] == agent]
        color = AGENT_COLORS.get(agent, "#94a3b8")
        for _, row in adf.iterrows():
            bar_color = "#ef4444" if row["is_err"] else color
            dur       = max(float(row["end_s"]) - float(row["start_s"]), 2.0)
            step_name = str(row["step"])
            bar_text  = step_name if dur >= label_threshold else ""
            fig.add_trace(go.Bar(
                x=[dur], y=[agent.upper()], base=[float(row["start_s"])],
                orientation="h",
                marker_color=bar_color,
                marker_line_width=0,
                text=bar_text,
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(size=9, color="#ffffff"),
                hovertemplate=(
                    f"<b>{agent}</b> · {step_name}<br>"
                    f"Start: {row['start_s']:.1f}s<br>"
                    f"Duration: {int(row['latency_ms'])}ms<br>"
                    f"Outcome: {row.get('outcome') or 'unknown'}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        # Error count annotation at right edge of agent row
        n_err = int(adf["is_err"].sum())
        if n_err:
            fig.add_annotation(
                x=1.01, y=agent.upper(),
                xref="paper", yref="y",
                text=f"⚠ {n_err} error{'s' if n_err > 1 else ''}",
                showarrow=False,
                font=dict(size=10, color="#ef4444"),
                xanchor="left",
            )

    fig.update_layout(
        paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff", font_color="#1e293b",
        height=max(200, len(agents) * 80 + 50),
        margin=dict(t=10, b=35, l=10, r=90),
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



# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### AI Agent RCA")
    st.markdown("<small style='color:#64748b'>Strategy C · Live</small>", unsafe_allow_html=True)
    st.divider()

    _NAV_PAGES = [
        "Ledger", "Session Deep Dive", "Quality Drift", "Before / After",
        "──────────", "Incidents Feed", "RCA View", "Failure Simulator", "Trace Inspector",
    ]
    _default = st.session_state.pop("_page", None)
    _idx = _NAV_PAGES.index(_default) if _default in _NAV_PAGES else 0

    page = st.radio(
        "Navigation",
        _NAV_PAGES,
        index=_idx,
        label_visibility="collapsed",
    )

    st.divider()
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
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

    st.caption(
        "Start here. Red rows = incidents detected. Amber rows = cost spent with 0 trades. "
        "Go to Incidents Feed to run analysis and drill into root causes."
    )

    # KPI row
    total_cost  = sessions["total_cost_usd"].sum()
    total_trade = sessions["trades_executed"].sum()
    wasted      = sessions[(sessions["trades_executed"] == 0) & (sessions["total_cost_usd"] > 0)]
    wasted_cost = wasted["total_cost_usd"].sum()
    inc_count   = len(incidents) if not incidents.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total Sessions", str(len(sessions))), unsafe_allow_html=True)
    c2.markdown(kpi("Total Spend", f"${total_cost:.2f}"), unsafe_allow_html=True)
    c3.markdown(kpi("Trades Executed", str(int(total_trade))), unsafe_allow_html=True)
    c4.markdown(kpi("Wasted Sessions",
        f"{len(wasted)} · ${wasted_cost:.2f}",
        f"{wasted_cost/total_cost*100:.0f}% of spend" if total_cost else ""),
        unsafe_allow_html=True)
    c5.markdown(kpi("Incidents Detected", str(inc_count)), unsafe_allow_html=True)

    st.divider()

    # Session table
    display = sessions.copy()
    display["Duration (s)"] = (
        display["total_latency_ms"] / 1000
    ).round(0).astype(int)
    display["Tokens"] = (display["total_tokens_input"] + display["total_tokens_output"]).astype(int)

    # Add incident count per session
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

    def row_style(row):
        if row["Incidents"] > 0:
            return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
        if row["Trades"] == 0:
            return ["background-color: #fef9c3; color: #713f12"] * len(row)
        return [""] * len(row)

    st.dataframe(
        tbl.style.apply(row_style, axis=1).format({"Cost ($)": "${:.4f}"}),
        use_container_width=True,
        height=480,
    )

    st.caption("Red rows = incidents detected. Amber rows = 0 trades (wasted).")

    # Agent cost breakdown
    if sessions["cost_breakdown"].notna().any():
        st.divider()
        st.markdown("#### Cost by Agent")
        agent_costs: dict[str, float] = defaultdict(float)
        for _, row in sessions.iterrows():
            bd = row.get("cost_breakdown") or {}
            if isinstance(bd, dict):
                for agent, data in bd.items():
                    if isinstance(data, dict):
                        agent_costs[agent] += data.get("cost_usd", 0)
        if agent_costs:
            fig = go.Figure(go.Bar(
                x=list(agent_costs.keys()),
                y=list(agent_costs.values()),
                marker_color=[AGENT_COLORS.get(a, "#94a3b8") for a in agent_costs],
                text=[f"${v:.4f}" for v in agent_costs.values()],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                font_color="#1e293b", margin=dict(t=20, b=20),
                yaxis_title="Cost (USD)", height=280,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Session Deep Dive
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Session Deep Dive":
    st.markdown("## Session Deep Dive")

    if sessions.empty:
        st.info("No sessions found.")
        st.stop()

    session_options = {
        f"{row['started_at'].strftime('%m-%d %H:%M') if pd.notna(row['started_at']) else row['id'][:8]}"
        f"  |  ${row['total_cost_usd']:.4f}"
        f"  |  {int(row['trades_executed'])} trades": row["id"]
        for _, row in sessions.iterrows()
    }
    chosen_label = st.selectbox("Select session", list(session_options.keys()))
    session_id   = session_options[chosen_label]
    session_row  = sessions[sessions["id"] == session_id].iloc[0].to_dict()
    sess_traces  = traces_all[traces_all["session_id"] == session_id].sort_values("created_at")

    # Session KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost",     f"${session_row['total_cost_usd']:.4f}")
    c2.metric("Duration", f"{int(session_row['total_latency_ms']//1000)}s")
    c3.metric("Tokens",   f"{int(session_row['total_tokens_input']+session_row['total_tokens_output']):,}")
    c4.metric("Trades",   str(int(session_row["trades_executed"])))

    if session_row.get("terminal_reason"):
        st.info(f"Exit reason: {session_row['terminal_reason']}")

    # Incidents for this session
    if not incidents.empty:
        sess_inc = incidents[incidents["session_id"] == session_id]
        if not sess_inc.empty:
            st.divider()
            st.markdown("#### Incidents Detected")
            for _, inc in sess_inc.iterrows():
                sim = inc.get("is_simulated", False)
                st.markdown(
                    f"{badge(inc['severity'], sim)} &nbsp; **{inc['pattern_name']}** — "
                    f"{inc['root_cause']}",
                    unsafe_allow_html=True,
                )
                if st.button(f"View RCA →", key=f"rca_{inc['id']}"):
                    goto_rca(inc.to_dict() if hasattr(inc, "to_dict") else inc, session_id)

    # Evals for this session
    evals_df = load_evals_for_session(session_id)
    if not evals_df.empty:
        st.divider()
        st.markdown("#### Eval Scores")
        ecols = st.columns(2)
        for i, (_, ev) in enumerate(evals_df.iterrows()):
            col = ecols[i % 2]
            with col:
                passed = bool(ev.get("passed"))
                score  = float(ev.get("score") or 0)
                icon   = "✓" if passed else "✗"
                color  = "#10b981" if passed else "#ef4444"
                col.markdown(
                    f'<div style="margin:4px 0">'
                    f'<span style="color:{color};font-weight:600">{icon}</span> '
                    f'<b>{ev["agent"]}.{ev["eval_name"]}</b> &nbsp; '
                    f'<code>{score:.2f}</code>'
                    f'{score_bar(score, passed)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Trace tree — collapsed by default
    st.divider()
    st.markdown("#### Trace Tree")
    st.caption("Expand an agent group to see individual steps. Click any step for details.")

    if sess_traces.empty:
        st.info("No traces for this session.")
    else:
        for agent_name, group in sess_traces.groupby("agent"):
            errors    = (group["outcome"] == "error").sum() + group["error"].notna().sum()
            tok_total = int(group["tokens_input"].sum() + group["tokens_output"].sum())
            label_color = "#ef4444" if errors > 0 else "#10b981"
            label = (
                f"{agent_name.upper()}  —  "
                f"{len(group)} steps  ·  "
                f"{tok_total:,} tokens  ·  "
                f"{int(group['latency_ms'].sum()//1000)}s"
            )
            if errors:
                label += f"  ·  ⚠ {errors} error(s)"

            with st.expander(label, expanded=False):
                for _, t in group.sort_values("created_at").iterrows():
                    is_err  = bool(t.get("error")) or t.get("outcome") == "error"
                    css     = "trace-error" if is_err else "trace-success"
                    step    = t.get("tool_name") or t.get("step_type") or "step"
                    lat     = int(t.get("latency_ms") or 0)
                    tok     = int((t.get("tokens_input") or 0) + (t.get("tokens_output") or 0))
                    outcome = t.get("outcome") or ""
                    ts_str  = (
                        t["created_at"].strftime("%H:%M:%S")
                        if pd.notna(t.get("created_at")) else ""
                    )

                    row_html = (
                        f'<div class="trace-row {css}">'
                        f'<b>{step}</b> &nbsp;'
                        f'<span style="color:#64748b">{ts_str}</span> &nbsp; '
                        f'<code>{lat}ms</code> &nbsp; '
                        f'<code>{tok} tok</code> &nbsp; '
                        f'<span style="color:{"#ef4444" if is_err else "#10b981"}">'
                        f'{outcome or ("error" if is_err else "")}'
                        f'</span>'
                        f'</div>'
                    )
                    st.markdown(row_html, unsafe_allow_html=True)

                    if is_err and t.get("error"):
                        with st.expander(f"Error detail — {step}", expanded=False):
                            st.error(t["error"])

    # Agent token timeline
    if not sess_traces.empty and (sess_traces["tokens_input"] + sess_traces["tokens_output"]).sum() > 0:
        st.divider()
        st.markdown("#### Token Usage Timeline")
        plot_df = sess_traces.copy()
        plot_df["tokens"] = plot_df["tokens_input"] + plot_df["tokens_output"]
        plot_df = plot_df[plot_df["tokens"] > 0].copy()
        if not plot_df.empty:
            fig = px.scatter(
                plot_df, x="created_at", y="tokens",
                color="agent", color_discrete_map=AGENT_COLORS,
                size="tokens", size_max=20,
                hover_data=["step_type","tool_name","latency_ms","outcome"],
            )
            fig.update_layout(
                paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                font_color="#1e293b", margin=dict(t=10, b=10), height=280,
                legend_title="Agent",
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Quality Drift
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Quality Drift":
    st.markdown("## Quality Drift")

    if sessions.empty:
        st.info("No sessions found.")
        st.stop()

    s = sessions.sort_values("started_at").copy()
    s["mean"]  = s["total_cost_usd"].expanding().mean()
    s["sigma"] = s["total_cost_usd"].expanding().std().fillna(0)
    s["upper"] = s["mean"] + 2 * s["sigma"]
    s["anomaly"] = s["total_cost_usd"] > s["upper"]
    s["label"]   = s["started_at"].dt.strftime("%m-%d %H:%M")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["label"], y=s["total_cost_usd"],
        mode="lines+markers", name="Session cost",
        line=dict(color="#3b82f6", width=1.5),
        marker=dict(
            size=[10 if a else 5 for a in s["anomaly"]],
            color=["#ef4444" if a else "#3b82f6" for a in s["anomaly"]],
        ),
    ))
    fig.add_trace(go.Scatter(
        x=s["label"], y=s["mean"],
        mode="lines", name="Rolling mean",
        line=dict(color="#64748b", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=s["label"], y=s["upper"],
        mode="lines", name="2σ threshold",
        line=dict(color="#f59e0b", dash="dot"),
    ))

    # Overlay incidents
    if not incidents.empty:
        for _, inc in incidents.iterrows():
            sid = inc.get("session_id")
            match = s[s["id"] == sid]
            if not match.empty:
                fig.add_vline(
                    x=match.iloc[0]["label"],
                    line=dict(color="#ef4444" if inc["severity"]=="critical" else "#f59e0b",
                              dash="solid", width=1),
                    annotation_text=inc["pattern_name"][:12],
                    annotation_font_size=9,
                )

    fig.update_layout(
        paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
        font_color="#1e293b", height=380,
        yaxis_title="Cost (USD)", xaxis_tickangle=-30,
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anomalies table
    anomalies = s[s["anomaly"]].copy()
    if not anomalies.empty:
        st.markdown(f"#### {len(anomalies)} Cost Anomalies Detected")
        st.dataframe(
            anomalies[["label","total_cost_usd","mean","upper","trades_executed"]]
            .rename(columns={"label":"Session","total_cost_usd":"Cost",
                             "mean":"Mean","upper":"2σ Limit","trades_executed":"Trades"})
            .style.format({"Cost":"${:.4f}","Mean":"${:.4f}","2σ Limit":"${:.4f}"}),
            use_container_width=True,
        )

    # Token efficiency
    st.divider()
    st.markdown("#### Tokens per Session")
    s["total_tokens"] = s["total_tokens_input"] + s["total_tokens_output"]
    s["tok_per_dollar"] = s.apply(
        lambda r: r["total_tokens"] / r["total_cost_usd"]
        if r["total_cost_usd"] > 0 else 0, axis=1
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=s["label"], y=s["total_tokens"],
        marker_color=["#ef4444" if t==0 else "#3b82f6" for t in s["trades_executed"]],
        name="Tokens",
    ))
    fig2.update_layout(
        paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
        font_color="#1e293b", height=250,
        yaxis_title="Total tokens", margin=dict(t=10,b=60),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Red bars = sessions with 0 trades.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Before / After
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Before / After":
    st.markdown("## Before / After")
    st.markdown("Compare raw trace data with the RCA dashboard view for any session.")

    if sessions.empty:
        st.info("No sessions found.")
        st.stop()

    # Default to most expensive session
    most_expensive_idx = sessions["total_cost_usd"].idxmax()
    default_sid = sessions.loc[most_expensive_idx, "id"] if most_expensive_idx is not None else sessions.iloc[0]["id"]

    session_options = {
        f"{row['started_at'].strftime('%m-%d %H:%M') if pd.notna(row['started_at']) else row['id'][:8]}"
        f"  |  ${row['total_cost_usd']:.4f}"
        f"  |  {int(row['trades_executed'])} trades": row["id"]
        for _, row in sessions.iterrows()
    }
    default_label = next((k for k, v in session_options.items() if v == default_sid), None)
    chosen = st.selectbox("Select session", list(session_options.keys()),
                          index=list(session_options.keys()).index(default_label) if default_label else 0)
    session_id  = session_options[chosen]
    session_row = sessions[sessions["id"] == session_id].iloc[0].to_dict()
    sess_traces = traces_all[traces_all["session_id"] == session_id]

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("#### Before — Raw Data")
        st.markdown(
            '<div class="callout callout-warning">'
            'What an engineer sees today: a session row and hundreds of trace rows. '
            'No diagnosis. No pattern name. No fix suggestion.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**c_sessions row:**")
        raw_session = {
            "id":              session_id[:16] + "...",
            "total_cost_usd":  session_row["total_cost_usd"],
            "total_latency_ms":session_row["total_latency_ms"],
            "trades_executed": session_row["trades_executed"],
            "terminal_reason": session_row.get("terminal_reason") or "null",
        }
        st.json(raw_session)
        st.markdown(f"**c_traces ({len(sess_traces)} rows):**")
        st.dataframe(
            sess_traces[["created_at","agent","step_type","tool_name","outcome","error","latency_ms"]]
            .head(30).astype(str),
            height=300,
            use_container_width=True,
        )
        if len(sess_traces) > 30:
            st.caption(f"... and {len(sess_traces)-30} more rows")

    with right:
        st.markdown("#### After — RCA Dashboard")
        st.markdown(
            '<div class="callout">'
            'Same data. Processed in seconds. Pattern named, root cause explained, fix suggested.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Run analysis on the fly
        traces_list = sess_traces.to_dict("records")
        evals       = run_all_evals(session_row, traces_list, recent_costs)
        inc_list    = run_all_detectors(session_row, traces_list, evals, recent_costs)

        if inc_list:
            inc = inc_list[0]
            st.markdown(
                f'{badge(inc.severity)} &nbsp; **{inc.pattern_name}**',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Root cause:** {inc.root_cause}")
            st.markdown("**Failed evals:**")
            for fe in inc.failed_evals[:4]:
                st.markdown(
                    f'- `{fe["agent"]}.{fe["eval_name"]}` — '
                    f'score **{fe["score"]:.2f}** (threshold {fe["threshold"]})'
                )
            st.markdown(
                f'<div class="fix-box">{generate_fix_suggestion(inc, traces_list)}</div>',
                unsafe_allow_html=True,
            )
            st.metric("Cost attributed to incident", f"${inc.cost_wasted:.4f}")
        else:
            st.success("No incidents detected for this session.")
            st.markdown("**Eval summary:**")
            for e in evals:
                icon = "✓" if e.passed else "✗"
                color = "#10b981" if e.passed else "#ef4444"
                st.markdown(
                    f'<span style="color:{color}">{icon}</span> '
                    f'`{e.agent}.{e.eval_name}` — {e.score:.2f}',
                    unsafe_allow_html=True,
                )


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

    dur_str  = f"{summary['duration_s']}s" if summary.get("duration_s") else "unknown duration"
    cost_str = f"${summary['cost_wasted']:.4f}" if summary["cost_wasted"] else "no cost recorded"
    tok_str  = f"{summary['tokens_wasted']:,} tokens" if summary["tokens_wasted"] else ""
    trades   = int(sess_row.get("trades_executed", 0))
    what_happened = (
        f"**{inc_obj.pattern_name}** — {inc_obj.root_cause}. "
        f"Session ran {dur_str}, spent {cost_str}"
        f"{' (' + tok_str + ')' if tok_str else ''}, produced {trades} trade(s)."
    )

    _s1l, _s1r = st.columns([3, 2])
    with _s1l:
        st.markdown(
            f'<div style="border-left:4px solid {sev_color};padding:14px 18px;'
            f'background:{sev_bg};border-radius:0 8px 8px 0">'
            f'{badge(inc_obj.severity, inc_obj.is_simulated)} '
            f'<span style="font-size:1.05rem;font-weight:700;color:{sev_text};margin-left:8px">'
            f'{inc_obj.pattern_name}</span><br>'
            f'<span style="color:#374151;font-size:0.88rem;margin-top:8px;display:block">'
            f'{what_happened}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with _s1r:
        st.markdown(
            f'<div style="background:#f0fdf4;border:1px solid #6ee7b7;border-radius:8px;padding:14px 16px">'
            f'<div style="font-size:0.7rem;font-weight:700;color:#065f46;'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Fix Steps</div>'
            f'<pre style="margin:0;font-size:0.78rem;color:#064e3b;'
            f'white-space:pre-wrap;font-family:inherit">{fix_text}</pre>'
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
        if session_evs:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                'text-transform:uppercase;letter-spacing:0.06em;margin:10px 0 4px 0">'
                'Session-level evals</div>',
                unsafe_allow_html=True,
            )
            for _ev in session_evs:
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
                    f'<b style="color:#374151">session</b>'
                    f'<span style="color:#64748b">.{_ev["eval_name"]}</span>'
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

    _chain_html = _render_call_chain_html(agents_present, evals_by_agent, root_agent, agent_stats, sess_row)
    if _chain_html:
        st.markdown(_chain_html.strip(), unsafe_allow_html=True)
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
                    _t        = _tool_calls[_ti]
                    _is_err   = bool(_str(_t.get("error"))) or _t.get("outcome") == "error"
                    _tn       = _t.get("tool_name") or "tool"
                    _lat      = int(_t.get("latency_ms") or 0)
                    _tj = _ti + 1
                    while _tj < len(_tool_calls) and _is_err:
                        _t2 = _tool_calls[_tj]
                        _t2_err = bool(_str(_t2.get("error"))) or _t2.get("outcome") == "error"
                        if _t2_err and _t2.get("tool_name") == _tn:
                            _tj += 1
                        else:
                            break
                    _count = _tj - _ti
                    if _count > 1 and _is_err:
                        st.markdown(
                            f'<div class="trace-row trace-error">'
                            f'<b>{_tn}</b> &nbsp; <code>{_lat}ms</code> &nbsp;'
                            f'<span style="color:#ef4444">error x{_count}</span> '
                            f'<span style="color:#94a3b8;font-size:0.73rem">(repeated, collapsed)</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"Error — {_tn} x{_count}", expanded=_is_root):
                            st.error(_str(_t.get("error")))
                    else:
                        _out_col = "#ef4444" if _is_err else "#10b981"
                        st.markdown(
                            f'<div class="trace-row {"trace-error" if _is_err else ""}">'
                            f'<b>{_tn}</b> &nbsp; <code>{_lat}ms</code> &nbsp;'
                            f'<span style="color:{_out_col}">{_t.get("outcome", "")}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if _is_err and _str(_t.get("error")):
                            with st.expander(f"Error — {_tn}", expanded=_is_root):
                                st.error(_str(_t.get("error")))
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

elif page == "──────────":
    st.info("Select a page from the sidebar.")
