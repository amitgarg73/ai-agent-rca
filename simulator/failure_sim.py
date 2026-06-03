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
    "hyperactive_polling",
    "tool_fabrication",
    "handoff_schema_break",
    "error_misinterpretation",
]

QUALITY_PATTERNS = [
    "grounding_failure",
    "coherence_break",
    "quality_cascade",
    "silent_degradation",
]

SEVERITY = {
    "tool_timeout_loop":     "critical",
    "context_spiral":        "warning",
    "pipeline_break":        "critical",
    "empty_result_loop":     "warning",
    "cost_anomaly":          "warning",
    "silent_exit":           "info",
    "silent_propagation":    "warning",
    "hyperactive_polling":   "warning",
    "tool_fabrication":      "critical",
    "handoff_schema_break":  "critical",
    "error_misinterpretation": "warning",
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


def _build_hyperactive_polling(session_id: str) -> list[dict]:
    """
    Research agent calls search_news 8 times successfully — agent is polling
    obsessively without reaching a decision threshold.
    """
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 920, 310, 130, offset_s=-480, sequence=1))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1100, 420, 180, offset_s=-460, sequence=2))
    for i in range(8):
        traces.append(_trace(session_id, "research", "tool_call", "search_news",
                             "success", None, 850, 0, 0,
                             offset_s=-450 + i * 45, sequence=3 + i))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 2100, 6400, 800, offset_s=-60, sequence=11))
    return traces


def _build_tool_fabrication(session_id: str) -> list[dict]:
    """
    Research agent produces tool_call traces with suspiciously low latency (15ms).
    Real external API calls take 100ms+. These traces suggest fabricated responses.
    """
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 880, 300, 120, offset_s=-300, sequence=1))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1200, 450, 200, offset_s=-280, sequence=2))
    traces.append(_trace(session_id, "research", "tool_call", "get_market_data",
                         None, None, 15, 0, 0, offset_s=-260, sequence=3))
    traces.append(_trace(session_id, "research", "tool_call", "get_earnings",
                         None, None, 12, 0, 0, offset_s=-250, sequence=4))
    traces.append(_trace(session_id, "research", "tool_call", "get_fundamentals",
                         None, None, 18, 0, 0, offset_s=-240, sequence=5))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1800, 2200, 600, offset_s=-220, sequence=6))
    return traces


def _build_handoff_schema_break(session_id: str) -> list[dict]:
    """
    Research agent completes successfully. Risk agent errors immediately on its
    first call — research output does not match the schema risk expects.
    """
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 900, 320, 140, offset_s=-300, sequence=1))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1400, 520, 220, offset_s=-280, sequence=2))
    traces.append(_trace(session_id, "research", "tool_call", "get_stock_data",
                         None, None, 1800, 0, 0, offset_s=-260, sequence=3))
    traces.append(_trace(session_id, "research", "tool_call", "get_news",
                         None, None, 1200, 0, 0, offset_s=-240, sequence=4))
    traces.append(_trace(session_id, "research", "agent_message", None,
                         "completed", None, 300, 0, 0, offset_s=-220, sequence=5))
    traces.append(_trace(session_id, "risk", "llm_call", None,
                         "error", "KeyError: 'analysis' missing from research output — "
                                  "expected field not provided by upstream agent",
                         200, 150, 0, offset_s=-200, sequence=6))
    return traces


def _build_error_misinterpretation(session_id: str) -> list[dict]:
    """
    Research agent receives HTTP 429 errors twice but continues processing
    with LLM calls instead of aborting or applying correct backoff logic.
    """
    traces = []
    traces.append(_trace(session_id, "market", "llm_call", None,
                         "success", None, 870, 290, 120, offset_s=-400, sequence=1))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1100, 400, 160, offset_s=-380, sequence=2))
    traces.append(_trace(session_id, "research", "tool_call", "get_prices",
                         "error", "HTTP 429: Too Many Requests — rate limit exceeded",
                         8000, 0, 0, offset_s=-360, sequence=3))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 1800, 1200, 400, offset_s=-340, sequence=4))
    traces.append(_trace(session_id, "research", "tool_call", "get_prices",
                         "error", "HTTP 429: Too Many Requests — rate limit exceeded",
                         8000, 0, 0, offset_s=-310, sequence=5))
    traces.append(_trace(session_id, "research", "llm_call", None,
                         "success", None, 2200, 2100, 580, offset_s=-280, sequence=6))
    traces.append(_trace(session_id, "orchestrator", "llm_call", None,
                         "success", None, 1400, 800, 300, offset_s=-240, sequence=7))
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
    "hyperactive_polling": (
        lambda sid: _build_hyperactive_polling(sid),
        lambda sid: _session(
            ["market", "research"], 0.31, 8430, 1110, 480_000, 0, None
        ),
    ),
    "tool_fabrication": (
        lambda sid: _build_tool_fabrication(sid),
        lambda sid: _session(
            ["market", "research"], 0.08, 2950, 920, 300_000, 0, None
        ),
    ),
    "handoff_schema_break": (
        lambda sid: _build_handoff_schema_break(sid),
        lambda sid: _session(
            ["market", "research", "risk"], 0.15, 990, 480, 300_000, 0, None
        ),
    ),
    "error_misinterpretation": (
        lambda sid: _build_error_misinterpretation(sid),
        lambda sid: _session(
            ["market", "research", "orchestrator"], 0.22, 4790, 1560, 400_000, 0, None
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


# ── Quality pattern builders (Phase Q3) ──────────────────────────────────────
# Each returns (sessions: list[dict], evals_by_session: dict[str, list]).
# Sessions are in chronological order with staggered started_at timestamps.

def _qual_eval_row(session_id: str, agent: str, eval_name: str,
                   score: float, threshold: float = 0.60) -> dict:
    return {
        "id":         str(uuid.uuid4()),
        "session_id": session_id,
        "agent":      agent,
        "eval_name":  eval_name,
        "score":      round(score, 3),
        "passed":     score >= threshold,
        "threshold":  threshold,
        "detail":     {},
    }


def _build_quality_grounding_failure() -> tuple[list[dict], dict[str, list]]:
    """3 sessions with research.data_grounding < 0.40 each."""
    sessions: list[dict] = []
    evals_by: dict[str, list] = {}
    for i in range(3):
        offset_h = -(2 - i) * 24
        sid  = str(uuid.uuid4())
        sess = _session(
            ["market", "research", "risk", "orchestrator"],
            0.12, 5000, 2000, 90_000, 1, "converged",
        )
        sess["id"]           = sid
        sess["started_at"]   = _ts(offset_h * 3600)
        sess["completed_at"] = _ts(offset_h * 3600 + 90)
        sessions.append(sess)
        evals_by[sid] = [
            _qual_eval_row(sid, "research_quality",     "data_grounding",         0.25),
            _qual_eval_row(sid, "research_quality",     "thesis_coherence",       0.70),
            _qual_eval_row(sid, "research_quality",     "composite_score",        0.55),
            _qual_eval_row(sid, "risk_quality",         "parameter_completeness", 0.75),
            _qual_eval_row(sid, "risk_quality",         "composite_score",        0.72),
            _qual_eval_row(sid, "orchestrator_quality", "decision_consistency",   0.68),
            _qual_eval_row(sid, "orchestrator_quality", "composite_score",        0.66),
            _qual_eval_row(sid, "session_quality",      "pipeline_coherence",     0.80),
            _qual_eval_row(sid, "session_quality",      "composite_score",        0.74),
        ]
    return sessions, evals_by


def _build_quality_coherence_break() -> tuple[list[dict], dict[str, list]]:
    """1 session with orchestrator.decision_consistency < 0.50."""
    sid  = str(uuid.uuid4())
    sess = _session(
        ["market", "research", "risk", "orchestrator"],
        0.15, 6000, 2500, 85_000, 0, "converged",
    )
    sess["id"] = sid
    evals = [
        _qual_eval_row(sid, "research_quality",     "data_grounding",          0.72),
        _qual_eval_row(sid, "research_quality",     "thesis_coherence",        0.68),
        _qual_eval_row(sid, "research_quality",     "composite_score",         0.68),
        _qual_eval_row(sid, "risk_quality",         "parameter_completeness",  0.75),
        _qual_eval_row(sid, "risk_quality",         "composite_score",         0.73),
        _qual_eval_row(sid, "orchestrator_quality", "decision_consistency",    0.30),
        _qual_eval_row(sid, "orchestrator_quality", "resolution_completeness", 0.65),
        _qual_eval_row(sid, "orchestrator_quality", "composite_score",         0.48),
        _qual_eval_row(sid, "session_quality",      "pipeline_coherence",      0.80),
        _qual_eval_row(sid, "session_quality",      "composite_score",         0.74),
    ]
    return [sess], {sid: evals}


def _build_quality_cascade() -> tuple[list[dict], dict[str, list]]:
    """5 sessions with 5 dimensions each declining >0.20."""
    sessions: list[dict] = []
    evals_by: dict[str, list] = {}
    declining = [
        ("research_quality",     "data_grounding",         0.78, 0.42),
        ("research_quality",     "thesis_coherence",       0.74, 0.48),
        ("risk_quality",         "parameter_completeness", 0.80, 0.50),
        ("orchestrator_quality", "decision_consistency",   0.72, 0.44),
        ("session_quality",      "pipeline_coherence",     0.76, 0.46),
    ]
    n = 5
    for i in range(n):
        offset_h = -(n - 1 - i) * 24
        sid  = str(uuid.uuid4())
        sess = _session(
            ["market", "research", "risk", "orchestrator"],
            0.12, 5000, 2000, 90_000, 1, "converged",
        )
        sess["id"]           = sid
        sess["started_at"]   = _ts(offset_h * 3600)
        sess["completed_at"] = _ts(offset_h * 3600 + 90)
        sessions.append(sess)
        rows = []
        for agent, dim, start, end in declining:
            score = start + (end - start) * (i / (n - 1))
            rows.append(_qual_eval_row(sid, agent, dim, round(score, 3)))
        comp = round(0.76 - 0.28 * (i / (n - 1)), 3)
        for agent in ["research_quality", "risk_quality",
                      "orchestrator_quality", "session_quality"]:
            rows.append(_qual_eval_row(sid, agent, "composite_score", comp))
        evals_by[sid] = rows
    return sessions, evals_by


def _build_quality_silent_degradation() -> tuple[list[dict], dict[str, list]]:
    """5 sessions: composite quality 0.78→0.48 while operational evals stay stable."""
    sessions: list[dict] = []
    evals_by: dict[str, list] = {}
    n = 5
    for i in range(n):
        offset_h = -(n - 1 - i) * 24
        sid  = str(uuid.uuid4())
        sess = _session(
            ["market", "research", "risk", "orchestrator"],
            0.12, 5000, 2000, 90_000, 1, "converged",
        )
        sess["id"]           = sid
        sess["started_at"]   = _ts(offset_h * 3600)
        sess["completed_at"] = _ts(offset_h * 3600 + 90)
        sessions.append(sess)
        comp = round(0.78 - 0.30 * (i / (n - 1)), 3)
        rows = []
        for agent in ["research_quality", "risk_quality",
                      "orchestrator_quality", "session_quality"]:
            rows.append(_qual_eval_row(sid, agent, "composite_score", comp))
        # Operational evals — clean and passing throughout
        rows += [
            _qual_eval_row(sid, "research", "tool_success_rate", 0.92, 0.80),
            _qual_eval_row(sid, "research", "completion",        0.88, 0.70),
        ]
        evals_by[sid] = rows
    return sessions, evals_by


_QUALITY_BUILDERS = {
    "grounding_failure":  _build_quality_grounding_failure,
    "coherence_break":    _build_quality_coherence_break,
    "quality_cascade":    _build_quality_cascade,
    "silent_degradation": _build_quality_silent_degradation,
}

_QUALITY_DESCRIPTIONS = {
    "grounding_failure":  "3 sessions of research.data_grounding < 0.40. Research making proposals without enough data sources.",
    "coherence_break":    "Orchestrator decision_consistency scored 0.30 — final decision does not match research and risk output.",
    "quality_cascade":    "5 quality dimensions each declined >0.20 over 5 sessions. Systemic degradation across agents.",
    "silent_degradation": "Composite quality 0.78→0.48 over 5 sessions while all operational evals stay clean. No alert fired.",
}


def simulate_quality_failure(pattern: str, db=None) -> tuple[list[str], list]:
    """
    Inject synthetic sessions + quality evals for a Proactive quality pattern.
    Writes sessions, evals, runs the detector in-memory, writes incidents.
    Returns (session_ids, incidents). Incidents have is_simulated=True.
    """
    if pattern not in _QUALITY_BUILDERS:
        raise ValueError(f"Unknown quality pattern '{pattern}'. Choose: {QUALITY_PATTERNS}")

    sessions, evals_by = _QUALITY_BUILDERS[pattern]()

    if not db:
        print(f"[DRY RUN] Proactive quality pattern: {pattern}")
        for s in sessions:
            print(f"  session_id={s['id']}  evals={len(evals_by.get(s['id'], []))}")
        return [s["id"] for s in sessions], []

    for sess in sessions:
        db.table("c_sessions").insert(sess).execute()

    all_eval_rows = [row for rows in evals_by.values() for row in rows]
    if all_eval_rows:
        db.table("c_evals").insert(all_eval_rows).execute()

    from engine.pattern_detector import run_quality_detectors
    incidents = run_quality_detectors(sessions[-1], sessions, evals_by)
    for inc in incidents:
        inc.is_simulated = True
    if incidents:
        db.table("c_incidents").insert([i.to_db_row() for i in incidents]).execute()
        for inc in incidents:
            print(f"  [{inc.severity}] {inc.pattern_name}")
    else:
        print(f"[WARN] {pattern}: sessions + evals written but no incident fired")

    return [s["id"] for s in sessions], incidents


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


def list_quality_patterns() -> list[dict]:
    return [
        {
            "id":          p,
            "label":       p.replace("_", " ").title(),
            "severity":    "warning" if p != "coherence_break" else "critical",
            "description": _QUALITY_DESCRIPTIONS[p],
        }
        for p in QUALITY_PATTERNS
    ]


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
        "tool_timeout_loop":       "External API times out. Agent retries with no limit. $1.98 burned, 0 trades.",
        "context_spiral":          "Research agent gathers endlessly, never decides. 40K+ tokens, 0 output.",
        "pipeline_break":          "Research completes, risk agent errors. Orchestrator never runs.",
        "empty_result_loop":       "Tool returns thin results. Agent retries with variations. No outcome.",
        "cost_anomaly":            "Session burns 10x normal cost — model selection or runaway token usage.",
        "silent_exit":             "All agents run, session ends, 0 trades. No reason logged.",
        "silent_propagation":      "Market fetches stale data — no exception. All 4 agents run on bad input. $0.0262 preventable spend.",
        "hyperactive_polling":     "Research calls search_news 8x successfully. No decision threshold. Polling loop burns 8+ tool call slots.",
        "tool_fabrication":        "Tool calls complete in 12-18ms. Real API calls take 100ms+. Agent fabricated responses.",
        "handoff_schema_break":    "Research succeeds. Risk errors immediately — output schema mismatch at handoff boundary.",
        "error_misinterpretation": "Two HTTP 429 errors received. Agent continues with LLM calls instead of backing off.",
    }
    return descriptions.get(pattern, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject synthetic failure patterns into Supabase")
    parser.add_argument("--pattern", choices=PATTERNS + QUALITY_PATTERNS, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    db = None if args.dry_run else __import__("sdk.db", fromlist=["get_db"]).get_db()

    if args.pattern in QUALITY_PATTERNS:
        sids, incs = simulate_quality_failure(args.pattern, db=db)
        print(f"Quality pattern '{args.pattern}': {len(sids)} session(s), {len(incs)} incident(s)")
    else:
        sid = simulate_failure(args.pattern, db=db)
        print(f"Pattern '{args.pattern}': session_id={sid}")
        print(f"Injected: {args.pattern} → session_id={sid}")
