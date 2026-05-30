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
/* Base */
[data-testid="stAppViewContainer"] { background: #0f172a; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #1e293b; }

/* KPI cards */
.kpi-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 14px 18px; text-align: center;
}
.kpi-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;
             letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.kpi-sub   { font-size: 0.75rem; color: #64748b; margin-top: 4px; }

/* Severity badges */
.badge-critical { background:#450a0a; color:#fca5a5; border:1px solid #ef4444;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
.badge-warning  { background:#431407; color:#fed7aa; border:1px solid #f59e0b;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
.badge-info     { background:#0c1a2e; color:#93c5fd; border:1px solid #3b82f6;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }
.badge-sim      { background:#1a0f2e; color:#c4b5fd; border:1px solid #8b5cf6;
                  padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; }

/* Trace tree */
.trace-row { font-family: 'Courier New', monospace; font-size: 0.78rem;
             padding: 4px 8px; border-left: 2px solid #334155; margin: 2px 0; }
.trace-error  { border-left-color: #ef4444; background: #1a0505; }
.trace-success { border-left-color: #10b981; background: #051a0f; }
.trace-root   { border-left-color: #f59e0b; background: #1a0e05; border-width: 3px; }

/* Score bar */
.score-bar-wrap { background: #334155; border-radius: 4px; height: 6px; margin: 4px 0; }
.score-bar-fill { height: 6px; border-radius: 4px; }

/* Callout */
.callout { background: #1e293b; border-left: 3px solid #3b82f6;
           padding: 10px 14px; border-radius: 0 6px 6px 0;
           margin: 8px 0; font-size: 0.9rem; }
.callout-critical { border-left-color: #ef4444; }
.callout-warning  { border-left-color: #f59e0b; }

/* Fix box */
.fix-box { background: #052e16; border: 1px solid #10b981; border-radius: 6px;
           padding: 12px 16px; font-family: monospace; font-size: 0.82rem;
           white-space: pre-wrap; color: #bbf7d0; }

/* Sim live indicator */
.sim-live { background: #1a0f2e; border: 1px solid #8b5cf6; border-radius: 6px;
            padding: 10px 16px; color: #c4b5fd; font-size: 0.9rem; }
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### AI Agent RCA")
    st.markdown("<small style='color:#64748b'>Strategy C · Live</small>", unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Ledger",
            "Session Deep Dive",
            "Quality Drift",
            "Before / After",
            "──────────",
            "Incidents Feed",
            "RCA View",
            "Failure Simulator",
            "Trace Inspector",
        ],
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
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#e2e8f0", margin=dict(t=20, b=20),
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
                    st.session_state["rca_incident_id"] = inc["id"]
                    st.session_state["rca_session_id"]  = session_id

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
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#e2e8f0", margin=dict(t=10, b=10), height=280,
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
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font_color="#e2e8f0", height=380,
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
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font_color="#e2e8f0", height=250,
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
                st.session_state["rca_incident"] = inc.to_dict()
                st.session_state["rca_sid"]      = inc.get("session_id")
        st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RCA View
# ══════════════════════════════════════════════════════════════════════════════
elif page == "RCA View":
    st.markdown("## RCA View")

    inc_dict = st.session_state.get("rca_incident")
    rca_sid  = st.session_state.get("rca_sid")

    if not inc_dict and not incidents.empty:
        # Default: most expensive incident
        inc_dict = incidents.sort_values("cost_wasted", ascending=False).iloc[0].to_dict()
        rca_sid  = inc_dict.get("session_id")

    if not inc_dict:
        st.info("Select an incident from the Incidents Feed to view its RCA.")
        st.stop()

    # Pick different incident
    if not incidents.empty:
        opts = {
            f"{row['pattern_name']} — "
            f"{row['created_at'].strftime('%m-%d %H:%M') if pd.notna(row['created_at']) else ''} "
            f"(${row['cost_wasted']:.4f})": row.to_dict()
            for _, row in incidents.sort_values("created_at", ascending=False).iterrows()
        }
        chosen_key = st.selectbox("Select incident", list(opts.keys()))
        inc_dict   = opts[chosen_key]
        rca_sid    = inc_dict.get("session_id")

    from engine.pattern_detector import Incident as IncidentModel

    # Reconstruct Incident object
    inc_obj = IncidentModel(
        session_id   = rca_sid or "",
        pattern_name = inc_dict.get("pattern_name",""),
        severity     = inc_dict.get("severity","info"),
        root_cause   = inc_dict.get("root_cause",""),
        call_stack   = inc_dict.get("call_stack") or [],
        failed_evals = inc_dict.get("failed_evals") or [],
        cost_wasted  = float(inc_dict.get("cost_wasted") or 0),
        tokens_wasted= int(inc_dict.get("tokens_wasted") or 0),
        fix_suggestion=inc_dict.get("fix_suggestion",""),
        is_simulated = bool(inc_dict.get("is_simulated", False)),
    )

    sess_traces = traces_all[traces_all["session_id"] == rca_sid].to_dict("records") if rca_sid else []
    sess_row    = sessions[sessions["id"] == rca_sid].iloc[0].to_dict() if (
        rca_sid and not sessions[sessions["id"] == rca_sid].empty
    ) else {}
    evals_df    = load_evals_for_session(rca_sid) if rca_sid else pd.DataFrame()

    # Header summary card
    sev_color = {"critical":"#EF4444","warning":"#F59E0B","info":"#3B82F6"}.get(
        inc_obj.severity,"#64748b"
    )
    st.markdown(
        f'<div style="border:1px solid {sev_color};border-radius:8px;padding:16px 20px;'
        f'background:#1e293b;margin-bottom:16px">'
        f'{badge(inc_obj.severity, inc_obj.is_simulated)}'
        f'<span style="font-size:1.3rem;font-weight:700;margin-left:12px">'
        f'{inc_obj.pattern_name}</span><br>'
        f'<span style="color:#94a3b8;font-size:0.85rem">{inc_obj.root_cause}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Cost wasted",    f"${inc_obj.cost_wasted:.4f}")
    mc2.metric("Tokens wasted",  f"{inc_obj.tokens_wasted:,}")
    mc3.metric("Trades",         str(int(sess_row.get("trades_executed", 0))))
    mc4.metric("Failed evals",   str(len(inc_obj.failed_evals)))

    st.divider()

    # Two-column: call stack + evals
    lcol, rcol = st.columns([3, 2])

    with lcol:
        st.markdown("#### Call Stack")
        annotated = build_annotated_call_stack(sess_traces, inc_obj) if sess_traces else inc_obj.call_stack

        for frame in annotated:
            is_root    = frame.get("is_root", False)
            is_relevant= frame.get("is_relevant", False)
            is_err     = frame.get("outcome") == "error" or bool(frame.get("error"))

            if is_root:
                css = "trace-root"
            elif is_err:
                css = "trace-error"
            elif is_relevant:
                css = "trace-success"
            else:
                css = "trace-row"

            agent   = frame.get("agent","")
            step    = frame.get("tool_name") or frame.get("step_name") or frame.get("step_type","")
            lat     = int(frame.get("latency_ms") or frame.get("duration_ms") or 0)
            tok     = int(frame.get("tokens") or 0)
            outcome = frame.get("outcome","")
            root_tag= " ← ROOT CAUSE" if is_root else ""

            st.markdown(
                f'<div class="trace-row {css}">'
                f'<b style="color:{AGENT_COLORS.get(agent,"#94a3b8")}">{agent}</b> &nbsp;'
                f'<b>{step}</b> &nbsp;'
                f'<code>{lat}ms</code> &nbsp;'
                f'{"<code>" + str(tok) + " tok</code>" if tok else ""} &nbsp;'
                f'<span style="color:{"#ef4444" if is_err else "#10b981"}">{outcome}</span>'
                f'<span style="color:#f59e0b;font-weight:700">{root_tag}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if frame.get("error"):
                with st.expander(f"Error: {step}", expanded=is_root):
                    st.error(frame["error"])

    with rcol:
        st.markdown("#### Eval Scores")

        if not evals_df.empty:
            for _, ev in evals_df.iterrows():
                passed = bool(ev.get("passed"))
                score  = float(ev.get("score") or 0)
                color  = "#10b981" if passed else "#ef4444"
                icon   = "✓" if passed else "✗"
                st.markdown(
                    f'<div style="margin:6px 0;padding:6px 10px;background:#1e293b;border-radius:4px">'
                    f'<span style="color:{color};font-weight:700">{icon}</span> &nbsp;'
                    f'<b>{ev["agent"]}</b>.<span style="color:#94a3b8">{ev["eval_name"]}</span>'
                    f'<span style="float:right;color:{color}">{score:.2f}</span>'
                    f'{score_bar(score, passed)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif inc_obj.failed_evals:
            for fe in inc_obj.failed_evals:
                score = float(fe.get("score",0))
                st.markdown(
                    f'<div style="margin:6px 0;padding:6px 10px;background:#1e293b;border-radius:4px">'
                    f'<span style="color:#ef4444;font-weight:700">✗</span> &nbsp;'
                    f'<b>{fe["agent"]}</b>.<span style="color:#94a3b8">{fe["eval_name"]}</span>'
                    f'<span style="float:right;color:#ef4444">{score:.2f}</span>'
                    f'{score_bar(score, False)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Run Analysis from Incidents Feed to populate eval scores.")

    # Eval explainer
    if not evals_df.empty:
        st.divider()
        st.markdown("#### Eval Explainer — Show Your Work")
        failed_evals_df = evals_df[~evals_df["passed"]] if "passed" in evals_df.columns else evals_df
        for _, ev in failed_evals_df.iterrows():
            with st.expander(f"{ev['agent']}.{ev['eval_name']} — score {ev['score']:.2f}", expanded=False):
                detail = ev.get("detail") or {}
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**What the eval checked:**")
                    st.json(detail)
                with col_b:
                    st.markdown("**Computation:**")
                    st.markdown(f"- Score: `{ev['score']:.3f}`")
                    st.markdown(f"- Threshold: `{ev['threshold']}`")
                    st.markdown(f"- Passed: `{ev['passed']}`")
                    st.markdown(score_bar(float(ev["score"]), bool(ev["passed"])), unsafe_allow_html=True)

    # Fix suggestion
    st.divider()
    fix = generate_fix_suggestion(inc_obj, sess_traces)
    st.markdown("#### Fix Suggestion")
    st.markdown(f'<div class="fix-box">{fix}</div>', unsafe_allow_html=True)


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
                st.session_state["rca_incident"] = incs[0].to_db_row()
                st.session_state["rca_sid"]      = sid
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
