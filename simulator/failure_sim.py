"""
Failure simulator — injects synthetic traces into Supabase for 6 named failure patterns.
Used for live demo and testing. All simulated rows are tagged is_simulated=True.

Run standalone:
    python3 -m simulator.failure_sim --pattern tool_timeout_loop
"""
from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timezone, timedelta

PATTERNS = [
    "tool_timeout_loop",
    "context_spiral",
    "pipeline_break",
    "empty_result_loop",
    "cost_anomaly",
    "silent_exit",
    "silent_propagation",
]

SEVERITY = {
    "tool_timeout_loop":   "critical",
    "context_spiral":      "warning",
    "pipeline_break":      "critical",
    "empty_result_loop":   "warning",
    "cost_anomaly":        "warning",
    "silent_exit":         "info",
    "silent_propagation":  "warning",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts(offset_seconds: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def _trace(session_id: str, agent: str, step_type: str, tool_name: str | None,
           outcome: str, error: str | None, latency_ms: int,
           tokens_in: int, tokens_out: int, offset_s: int = 0,
           sequence: int = 1) -> dict:
    return {
        "id":           str(uuid.uuid4()),
        "session_id":   session_id,
        "span_id":      str(uuid.uuid4()),
        "date":         datetime.now(timezone.utc).date().isoformat(),
        "sequence":     sequence,
        "agent":        agent,
        "step_type":    step_type,
        "tool_name":    tool_name,
        "outcome":      outcome,
        "error":        error,
        "latency_ms":   latency_ms,
        "tokens_input": tokens_in,
        "tokens_output":tokens_out,
        "model":        "claude-haiku-4-5-20251001",
        "created_at":   _ts(offset_s),
    }


def _session(label: list[str], cost: float, tokens_in: int, tokens_out: int,
             latency_ms: int, trades: int, reason: str | None) -> dict:
    return {
        "id":                  str(uuid.uuid4()),
        "date":                datetime.now(timezone.utc).date().isoformat(),
        "total_cost_usd":      cost,
        "total_latency_ms":    latency_ms,
        "total_tokens_input":  tokens_in,
        "total_tokens_output": tokens_out,
        "trades_proposed":     trades,
        "trades_executed":     trades,
        "agents_invoked":      label,
        "terminal_reason":     reason if reason is not None else "",
        "started_at":          _ts(-latency_ms // 1000),
        "completed_at":        _now(),
        "is_simulated":        True,
    }


# ── Pattern builders ──────────────────────────────────────────────────────────

def _build_tool_timeout_loop(session_id: str, retries: int = 8) -> list[dict]:
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 1200, 500, 200, offset_s=-1027, sequence=1))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 800, 400, 150, offset_s=-1020, sequence=2))
    for i in range(retries):
        traces.append(_trace(session_id, "research", "tool_call", "get_stock_data",
                             "error", "ReadTimeout: HTTPSConnectionPool: Read timed out",
                             30_000, 0, 0, offset_s=-1010 + i * 120, sequence=3 + i))
    return traces


def _build_context_spiral(session_id: str) -> list[dict]:
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 1100, 400, 180, offset_s=-900, sequence=1))
    seq = 2
    for i in range(20):
        tok_in = 1500 + i * 400
        traces.append(_trace(session_id, "research", "tool_call", "search_news",
                             "success", None, 2000, tok_in, 200,
                             offset_s=-890 + i * 40, sequence=seq))
        seq += 1
        traces.append(_trace(session_id, "research", "llm_call", None,
                             "success", None, 3000, tok_in + 200, 400,
                             offset_s=-885 + i * 40, sequence=seq))
        seq += 1
    return traces


def _build_pipeline_break(session_id: str) -> list[dict]:
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 900, 350, 150, offset_s=-300, sequence=1))
    traces.append(_trace(session_id, "research", "tool_call", "get_stock_data",
                         "success", None, 1800, 600, 250, offset_s=-280, sequence=2))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 2100, 800, 400, offset_s=-260, sequence=3))
    traces.append(_trace(session_id, "research", "decision", None,
                         "success", None, 500, 200, 100, offset_s=-240, sequence=4))
    traces.append(_trace(session_id, "risk", "llm_call", None,
                         "error", "ConnectionError: upstream service unavailable",
                         5000, 200, 0, offset_s=-230, sequence=5))
    return traces


def _build_empty_result_loop(session_id: str) -> list[dict]:
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 1000, 380, 160, offset_s=-600, sequence=1))
    for i in range(5):
        traces.append(_trace(session_id, "research", "tool_call", "get_news",
                             "success", None, 1500, 300, 50,
                             offset_s=-580 + i * 80, sequence=2 + i))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 2500, 3000, 600, offset_s=-170, sequence=7))
    return traces


def _build_cost_anomaly(session_id: str) -> list[dict]:
    traces = []
    for seq, (agent, tok_in, tok_out) in enumerate([
        ("market", 8000, 3000),
        ("research", 25000, 8000),
        ("risk", 12000, 4000),
        ("orchestrator", 15000, 5000),
    ], start=1):
        traces.append(_trace(session_id, agent, "llm_call", None,
                             "success", None, 8000, tok_in, tok_out,
                             offset_s=-400, sequence=seq))
    return traces


def _build_silent_exit(session_id: str) -> list[dict]:
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 900, 350, 140, offset_s=-120, sequence=1))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1800, 600, 280, offset_s=-100, sequence=2))
    traces.append(_trace(session_id, "risk", "llm_call", None,
                         "success", None, 1200, 400, 180, offset_s=-80, sequence=3))
    traces.append(_trace(session_id, "orchestrator", "llm_call", None,
                         "success", None, 2000, 800, 350, offset_s=-60, sequence=4))
    return traces


def _build_silent_propagation(session_id: str) -> list[dict]:
    """
    Market fetches stale data (no exception — all traces succeed operationally).
    data_freshness eval fires because first market trace is 15 min after session start.
    Research, Risk, and Orchestrator all run on the bad input anyway.
    Session ends with 0 trades and no terminal reason → detect_silent_exit fires.
    CB savings = research + risk + orchestrator cost ($0.0262).
    """
    traces = []
    # Market — operationally succeeds but data is 15 min stale
    # Session started 20 min ago (offset -1200); these traces are 5 min ago (offset -300)
    # → freshness eval: lag = 15 min > 10 min threshold → FAILS
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 820, 310, 140, offset_s=-300, sequence=1))
    traces.append(_trace(session_id, "market", "tool_call", "get_market_data",
                         None, None, 1100, 0, 0, offset_s=-290, sequence=2))
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 640, 280, 110, offset_s=-280, sequence=3))

    # Research — runs and succeeds on stale market data, never knowing it's bad
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1400, 620, 280, offset_s=-260, sequence=4))
    traces.append(_trace(session_id, "research", "tool_call", "get_fundamentals",
                         None, None, 1800, 0, 0, offset_s=-240, sequence=5))
    traces.append(_trace(session_id, "research", "tool_call", "get_earnings",
                         None, None, 1600, 0, 0, offset_s=-220, sequence=6))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 2200, 4800, 1100, offset_s=-200, sequence=7))

    # Risk — runs normally, assesses positions based on stale research
    traces.append(_trace(session_id, "risk", "llm_call", None,
                         "success", None, 1100, 1800, 620, offset_s=-170, sequence=8))
    traces.append(_trace(session_id, "risk", "agent_message", None,
                         "completed", None, 400, 0, 0, offset_s=-155, sequence=9))

    # Orchestrator — runs but produces 0 trades; no terminal_reason logged
    traces.append(_trace(session_id, "orchestrator", "llm_call", None,
                         "success", None, 1800, 2100, 780, offset_s=-130, sequence=10))

    return traces


PATTERN_BUILDERS = {
    "tool_timeout_loop": (
        lambda sid: _build_tool_timeout_loop(sid),
        lambda sid: _session(
            ["market", "research"], 1.98, 45000, 2000, 1_027_000, 0, None
        ),
    ),
    "context_spiral": (
        lambda sid: _build_context_spiral(sid),
        lambda sid: _session(
            ["market", "research"], 0.85, 42000, 8000, 900_000, 0, None
        ),
    ),
    "pipeline_break": (
        lambda sid: _build_pipeline_break(sid),
        lambda sid: _session(
            ["market", "research", "risk"], 0.42, 2150, 900, 300_000, 0, None
        ),
    ),
    "empty_result_loop": (
        lambda sid: _build_empty_result_loop(sid),
        lambda sid: _session(
            ["market", "research"], 0.38, 4430, 860, 600_000, 0, None
        ),
    ),
    "cost_anomaly": (
        lambda sid: _build_cost_anomaly(sid),
        lambda sid: _session(
            ["market", "research", "risk", "orchestrator"], 3.10, 60000, 20000, 400_000, 1, None
        ),
    ),
    "silent_exit": (
        lambda sid: _build_silent_exit(sid),
        lambda sid: _session(
            ["market", "research", "risk", "orchestrator"], 0.19, 2150, 950, 120_000, 0, None
        ),
    ),
    "silent_propagation": (
        lambda sid: _build_silent_propagation(sid),
        lambda sid: {
            **_session(
                ["market", "research", "risk", "orchestrator"],
                0.0284,          # total cost
                10_010, 2_030,   # tokens in/out
                1_200_000,       # latency 20 min — session started 20 min ago
                0, None,         # 0 trades, no terminal_reason
            ),
            "cost_breakdown": {
                "market":       {"cost_usd": 0.0022},
                "research":     {"cost_usd": 0.0148},
                "risk":         {"cost_usd": 0.0063},
                "orchestrator": {"cost_usd": 0.0051},
            },
        },
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def simulate_failure(pattern: str, db=None) -> str:
    """
    Inject synthetic traces for the given failure pattern.
    Returns the simulated session_id.
    If db is None, returns the session_id and prints what would be written.
    """
    if pattern not in PATTERN_BUILDERS:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose: {PATTERNS}")

    trace_builder, session_builder = PATTERN_BUILDERS[pattern]
    session_id = str(uuid.uuid4())
    session    = session_builder(session_id)
    session["id"] = session_id
    traces     = trace_builder(session_id)

    if db:
        db.table("c_sessions").insert(session).execute()
        if traces:
            db.table("c_traces").insert(traces).execute()
    else:
        print(f"[DRY RUN] Pattern: {pattern}")
        print(f"  session_id: {session_id}")
        print(f"  session cost: ${session['total_cost_usd']}")
        print(f"  traces: {len(traces)}")

    return session_id


def list_patterns() -> list[dict]:
    return [
        {
            "id":          p,
            "label":       p.replace("_", " ").title(),
            "severity":    SEVERITY[p],
            "description": _pattern_description(p),
        }
        for p in PATTERNS
    ]


def _pattern_description(pattern: str) -> str:
    descriptions = {
        "tool_timeout_loop": "External API times out. Agent retries with no limit. $1.98 burned, 0 trades.",
        "context_spiral":    "Research agent gathers endlessly, never decides. 40K+ tokens, 0 output.",
        "pipeline_break":    "Research completes, risk agent errors. Orchestrator never runs.",
        "empty_result_loop": "Tool returns thin results. Agent retries with variations. No outcome.",
        "cost_anomaly":        "Session burns 10x normal cost — model selection or runaway token usage.",
        "silent_exit":         "All agents run, session ends, 0 trades. No reason logged.",
        "silent_propagation":  "Market fetches stale data — no exception. All 4 agents run on bad input. $0.0262 preventable spend. CB savings demo.",
    }
    return descriptions.get(pattern, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=PATTERNS, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    if args.dry_run:
        sid = simulate_failure(args.pattern, db=None)
    else:
        from sdk.db import get_db
        sid = simulate_failure(args.pattern, db=get_db())
        print(f"Injected: {args.pattern} → session_id={sid}")
