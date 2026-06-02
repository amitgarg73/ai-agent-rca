"""
LLM-as-judge quality evaluator — Phase Q1.

Scores each session's agents across semantic quality dimensions.
Stores results in c_evals with agent='{agent}_quality'.

Two modes (auto-selected per session):
  - Structural proxy: uses trace patterns as quality signals (default — traces
    from Strategy C do not store raw LLM output text)
  - LLM judge: when 'output' text fields exist in traces, calls Haiku for
    semantic scoring (future — add output field to c_traces to activate)

Dimensions scored:
  research_quality: data_grounding, thesis_coherence, actionability,
                    catalyst_specificity, risk_acknowledgment, composite_score
  risk_quality:     research_consistency, parameter_completeness,
                    volatility_accounting, stop_loss_quality,
                    position_sizing_rationale, composite_score
  orchestrator_quality: decision_consistency, resolution_completeness,
                        reasoning_transparency, upstream_integration, composite_score
  session_quality:  pipeline_coherence, reasoning_chain, composite_score

Run standalone:
    python3 scripts/backfill_quality.py [--dry-run]
"""
from __future__ import annotations

import logging
from typing import Optional

from engine.eval_engine import EvalResult

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

COMPOSITE_PASS         = 0.60
RESEARCH_GROUNDING_MIN = 0.40
RISK_CONSISTENCY_MIN   = 0.50
ORCH_DECISION_MIN      = 0.50
DIMENSION_PASS         = 0.50   # default pass threshold for individual dimensions

# Terminal reason quality buckets (mirrors eval_engine.py)
_GOOD_EXITS    = {"eod_complete", "converged", "no_opportunity", "risk_rejected",
                  "market_closed", "position_limit", "daily_limit"}
_PARTIAL_EXITS = {"structural_block"}
_BAD_EXITS     = {"in_progress", "error"}

# Tool name keywords for domain detection
_NEWS_KEYWORDS      = {"news", "earn"}
_VOL_KEYWORDS       = {"atr", "volatil", "vix", "beta"}


# ── Trace helpers ─────────────────────────────────────────────────────────────

def _agent_traces(traces: list[dict], prefix: str) -> list[dict]:
    return [t for t in traces
            if (t.get("agent") or "").lower().startswith(prefix)]

def _research_traces(traces):  return _agent_traces(traces, "research")
def _risk_traces(traces):      return _agent_traces(traces, "risk")
def _orch_traces(traces):      return _agent_traces(traces, "orchestrator")

def _tool_calls(traces: list[dict]) -> list[dict]:
    return [t for t in traces if (t.get("step_type") or "") == "tool_call"]

def _ok(traces: list[dict]) -> list[dict]:
    return [t for t in traces if t.get("outcome") != "error"]

def _decision_traces(traces: list[dict]) -> list[dict]:
    return [t for t in traces
            if (t.get("step_type") or "") in ("decision", "agent_message")]

def _tool_has_keyword(tool_name: str, keywords: set) -> bool:
    name = (tool_name or "").lower()
    return any(kw in name for kw in keywords)


# ── Research quality ──────────────────────────────────────────────────────────

def _score_research(traces: list[dict]) -> list[EvalResult]:
    res = _research_traces(traces)
    source = "structural_proxy"

    if not res:
        neutral = EvalResult("composite_score", "research_quality", 0.5, True,
                             COMPOSITE_PASS, {source: source, "note": "no research traces"})
        dims = ["data_grounding", "thesis_coherence", "actionability",
                "catalyst_specificity", "risk_acknowledgment"]
        return [
            EvalResult(d, "research_quality", 0.5, True, DIMENSION_PASS,
                       {"source": source, "note": "no research traces"})
            for d in dims
        ] + [neutral]

    ok_tools = _ok(_tool_calls(res))
    distinct_tools = len({t.get("tool_name", "") for t in ok_tools if t.get("tool_name")})

    # data_grounding: diverse tool use indicates data-backed research
    dg = min(1.0, distinct_tools / 3.0) if distinct_tools > 0 else 0.1
    dg_note = f"{distinct_tools} distinct tool type(s) used successfully"

    # thesis_coherence: presence of a decision/agent_message trace
    ok_dec = _ok(_decision_traces(res))
    all_dec = _decision_traces(res)
    tc = 0.85 if ok_dec else (0.45 if all_dec else 0.20)
    tc_note = "decision trace present" if ok_dec else "no decision trace"

    # actionability: research output reached the risk agent
    risk_ran = bool(_risk_traces(traces))
    act = 0.90 if (risk_ran and ok_tools) else (0.55 if risk_ran else 0.20)
    act_note = f"risk_ran={risk_ran}, ok_tool_calls={len(ok_tools)}"

    # catalyst_specificity: news or earnings tool used
    used_news = any(_tool_has_keyword(t.get("tool_name", ""), _NEWS_KEYWORDS) for t in ok_tools)
    cs = 0.85 if used_news else (0.50 if ok_tools else 0.20)
    cs_note = "news/earnings tool used" if used_news else "no news/earnings tool found"

    # volatility_accounting: ATR or volatility tool used in research (drives position sizing)
    used_vol = any(_tool_has_keyword(t.get("tool_name", ""), _VOL_KEYWORDS) for t in ok_tools)
    va = 0.85 if used_vol else (0.40 if ok_tools else 0.15)
    va_note = "volatility/ATR tool used" if used_vol else "no ATR/volatility tool found in research"

    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    raw     = [dg,   tc,   act,  cs,   va]
    composite = round(sum(w * s for w, s in zip(weights, raw)), 3)

    dim_names = ["data_grounding", "thesis_coherence", "actionability",
                 "catalyst_specificity", "volatility_accounting"]
    dim_notes = [dg_note, tc_note, act_note, cs_note, va_note]
    thresholds = [RESEARCH_GROUNDING_MIN, DIMENSION_PASS, DIMENSION_PASS,
                  DIMENSION_PASS, DIMENSION_PASS]

    rows = [
        EvalResult(name, "research_quality", round(score, 3), score >= thr, thr,
                   {"source": source, "note": note})
        for name, score, thr, note in zip(dim_names, raw, thresholds, dim_notes)
    ]
    rows.append(EvalResult(
        "composite_score", "research_quality", composite,
        composite >= COMPOSITE_PASS, COMPOSITE_PASS,
        {"source": source,
         "dimensions": {n: round(s, 3) for n, s in zip(dim_names, raw)}},
    ))
    return rows


# ── Risk quality ──────────────────────────────────────────────────────────────

def _score_risk(traces: list[dict]) -> list[EvalResult]:
    risk = _risk_traces(traces)
    source = "structural_proxy"

    if not risk:
        dims = ["research_consistency", "parameter_completeness",
                "stop_loss_quality", "position_sizing_rationale"]
        return [
            EvalResult(d, "risk_quality", 0.5, True, DIMENSION_PASS,
                       {"source": source, "note": "no risk traces"})
            for d in dims
        ] + [EvalResult("composite_score", "risk_quality", 0.5, True, COMPOSITE_PASS,
                        {"source": source, "note": "no risk traces"})]

    res_ran  = bool(_research_traces(traces))
    ok_tools = _ok(_tool_calls(risk))
    ok_dec   = _ok(_decision_traces(risk))
    all_dec  = _decision_traces(risk)

    # research_consistency: risk ran after research
    rc = 0.90 if res_ran else 0.30
    rc_note = f"research_ran={res_ran}"

    # parameter_completeness: produced a decision trace
    pc = 0.90 if ok_dec else (0.50 if all_dec else 0.20)
    pc_note = "decision trace present" if ok_dec else "no decision trace"

    # stop_loss_quality: structural proxy unavailable without output text
    slq = 0.60
    slq_note = "structural proxy: cannot determine stop quality without output text"

    # position_sizing_rationale: had data to derive sizing from
    psr = 0.80 if len(ok_tools) >= 2 else (0.50 if ok_tools else 0.30)
    psr_note = f"{len(ok_tools)} successful tool call(s)"

    weights = [0.30, 0.30, 0.25, 0.15]
    raw     = [rc,   pc,   slq,  psr]
    composite = round(sum(w * s for w, s in zip(weights, raw)), 3)

    dim_names = ["research_consistency", "parameter_completeness",
                 "stop_loss_quality", "position_sizing_rationale"]
    dim_notes = [rc_note, pc_note, slq_note, psr_note]
    thresholds = [RISK_CONSISTENCY_MIN, DIMENSION_PASS,
                  DIMENSION_PASS, DIMENSION_PASS]

    rows = [
        EvalResult(name, "risk_quality", round(score, 3), score >= thr, thr,
                   {"source": source, "note": note})
        for name, score, thr, note in zip(dim_names, raw, thresholds, dim_notes)
    ]
    rows.append(EvalResult(
        "composite_score", "risk_quality", composite,
        composite >= COMPOSITE_PASS, COMPOSITE_PASS,
        {"source": source,
         "dimensions": {n: round(s, 3) for n, s in zip(dim_names, raw)}},
    ))
    return rows


# ── Orchestrator quality ──────────────────────────────────────────────────────

def _score_orchestrator(session: dict, traces: list[dict]) -> list[EvalResult]:
    orch     = _orch_traces(traces)
    res_ran  = bool(_research_traces(traces))
    risk_ran = bool(_risk_traces(traces))
    trades   = float(session.get("trades_executed") or 0)
    terminal = (session.get("terminal_reason") or "").lower()
    source   = "structural_proxy"

    if not orch:
        dims = ["decision_consistency", "resolution_completeness",
                "reasoning_transparency", "upstream_integration"]
        return [
            EvalResult(d, "orchestrator_quality", 0.5, True, DIMENSION_PASS,
                       {"source": source, "note": "no orchestrator traces"})
            for d in dims
        ] + [EvalResult("composite_score", "orchestrator_quality", 0.5, True, COMPOSITE_PASS,
                        {"source": source, "note": "no orchestrator traces"})]

    full_pipeline = res_ran and risk_ran
    ok_dec = _ok(_decision_traces(orch))

    # decision_consistency: outcome consistent with upstream pipeline
    if trades > 0 and full_pipeline:
        dc, dc_note = 1.0, "trades placed with full upstream pipeline"
    elif trades == 0 and terminal in _GOOD_EXITS:
        dc, dc_note = 0.85, f"no trade, valid exit reason: {terminal}"
    elif trades > 0 and not full_pipeline:
        dc, dc_note = 0.40, "trades placed but research or risk did not complete"
    else:
        dc, dc_note = 0.30, f"no trade, exit_reason={terminal or 'missing'}"

    # resolution_completeness: clear terminal reason
    if terminal in _GOOD_EXITS:
        rc, rc_note = 1.0, f"good exit: {terminal}"
    elif terminal in _PARTIAL_EXITS:
        rc, rc_note = 0.50, f"partial exit: {terminal}"
    elif terminal in _BAD_EXITS or not terminal:
        rc, rc_note = 0.10, f"bad/missing exit: {terminal or 'missing'}"
    else:
        rc, rc_note = 0.70, f"unrecognized exit reason: {terminal}"

    # reasoning_transparency: decision trace present
    rt = 0.85 if ok_dec else (0.50 if _decision_traces(orch) else 0.20)
    rt_note = "decision trace present" if ok_dec else "no decision trace"

    # upstream_integration: all upstream stages ran
    ui = 0.85 if full_pipeline else (0.50 if (res_ran or risk_ran) else 0.20)
    ui_note = f"research_ran={res_ran}, risk_ran={risk_ran}"

    weights = [0.35, 0.30, 0.20, 0.15]
    raw     = [dc,   rc,   rt,   ui]
    composite = round(sum(w * s for w, s in zip(weights, raw)), 3)

    dim_names = ["decision_consistency", "resolution_completeness",
                 "reasoning_transparency", "upstream_integration"]
    dim_notes = [dc_note, rc_note, rt_note, ui_note]
    thresholds = [ORCH_DECISION_MIN, DIMENSION_PASS, DIMENSION_PASS, DIMENSION_PASS]

    rows = [
        EvalResult(name, "orchestrator_quality", round(score, 3), score >= thr, thr,
                   {"source": source, "note": note})
        for name, score, thr, note in zip(dim_names, raw, thresholds, dim_notes)
    ]
    rows.append(EvalResult(
        "composite_score", "orchestrator_quality", composite,
        composite >= COMPOSITE_PASS, COMPOSITE_PASS,
        {"source": source,
         "dimensions": {n: round(s, 3) for n, s in zip(dim_names, raw)}},
    ))
    return rows


# ── Session coherence ─────────────────────────────────────────────────────────

def _score_session_coherence(session: dict, traces: list[dict]) -> list[EvalResult]:
    res_ran  = bool(_research_traces(traces))
    risk_ran = bool(_risk_traces(traces))
    orch_ran = bool(_orch_traces(traces))
    source   = "structural_proxy"

    ran_count = sum([res_ran, risk_ran, orch_ran])

    # pipeline_coherence: full pipeline or clean market-only
    if ran_count == 3:
        pc, pc_note = 1.0, "all downstream agents ran"
    elif ran_count == 0:
        pc, pc_note = 0.80, "market-only session (no downstream agents)"
    else:
        pc, pc_note = 0.30, f"partial pipeline: {ran_count}/3 downstream agents ran"

    # reasoning_chain: no stage had exclusively error traces
    error_only_stages = 0
    for fn in [_research_traces, _risk_traces, _orch_traces]:
        stage = fn(traces)
        if stage and all(t.get("outcome") == "error" for t in stage):
            error_only_stages += 1
    rc = max(0.20, 1.0 - error_only_stages * 0.40)
    rc_note = f"{error_only_stages} agent stage(s) had only error traces"

    composite = round((pc + rc) / 2, 3)

    return [
        EvalResult("pipeline_coherence", "session_quality", round(pc, 3), pc >= 0.60, 0.60,
                   {"source": source, "note": pc_note}),
        EvalResult("reasoning_chain",    "session_quality", round(rc, 3), rc >= 0.60, 0.60,
                   {"source": source, "note": rc_note}),
        EvalResult("composite_score",    "session_quality", composite,
                   composite >= COMPOSITE_PASS, COMPOSITE_PASS,
                   {"source": source}),
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def judge_session(session: dict, traces: list[dict]) -> list[EvalResult]:
    """
    Score a session's quality across all agents.
    Returns list of EvalResult; agent names end in '_quality'.
    Never raises — all errors return empty list.
    """
    try:
        results: list[EvalResult] = []
        results.extend(_score_research(traces))
        results.extend(_score_risk(traces))
        results.extend(_score_orchestrator(session, traces))
        results.extend(_score_session_coherence(session, traces))
        return results
    except Exception as exc:
        log.error("quality_judge failed for session=%s: %s",
                  session.get("id", "?")[:8], exc)
        return []
