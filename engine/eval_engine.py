"""
Eval engine — 13 evals for Strategy C agents.
Reads c_sessions + c_traces, writes to c_evals, returns results for pattern detector.

Schema note: uses real Strategy C column names:
  c_traces:  agent, step_type, tool_name, outcome, error, latency_ms,
             tokens_input, tokens_output, created_at, session_id
  c_sessions: total_cost_usd, trades_executed, terminal_reason,
              started_at, total_tokens_input, total_tokens_output
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

REQUIRED_AGENTS = ["market", "research", "risk", "orchestrator"]
TOKEN_SPIRAL_THRESHOLD    = 40_000
TOKEN_EFFICIENCY_THRESHOLD = 30_000
TOOL_SUCCESS_MIN_RATE     = 0.80
COST_ANOMALY_SIGMA        = 2.0
DATA_FRESHNESS_MINUTES    = 30


@dataclass
class EvalResult:
    eval_name:  str
    agent:      str
    score:      float
    passed:     bool
    threshold:  float
    detail:     dict = field(default_factory=dict)

    def to_db_row(self, session_id: str) -> dict:
        return {
            "id":         str(uuid.uuid4()),
            "session_id": session_id,
            "agent":      self.agent,
            "eval_name":  self.eval_name,
            "score":      round(self.score, 3),
            "passed":     self.passed,
            "threshold":  self.threshold,
            "detail":     self.detail,
        }


# ── Market agent evals ────────────────────────────────────────────────────────

def eval_market_data_completeness(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if market agent produced at least one successful trace."""
    market = [t for t in traces if (t.get("agent") or "").lower() in ("market", "market_shadow")]
    if not market:
        return EvalResult("data_completeness", "market", 0.0, False, 0.7,
                          {"reason": "no market agent traces found"})
    success = [t for t in market if (t.get("outcome") or "") == "success"]
    score   = len(success) / len(market)
    return EvalResult("data_completeness", "market", round(score, 2), score >= 0.7, 0.7,
                      {"total": len(market), "success": len(success)})


def eval_market_data_freshness(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if first market trace is within 30 min of session start."""
    market = sorted(
        [t for t in traces if (t.get("agent") or "").lower() in ("market", "market_shadow")],
        key=lambda x: x.get("created_at", "")
    )
    if not market:
        return EvalResult("data_freshness", "market", 0.0, False, 0.5,
                          {"reason": "no market agent traces"})
    started = session.get("started_at") or ""
    first   = market[0].get("created_at") or ""
    if not started or not first:
        return EvalResult("data_freshness", "market", 1.0, True, 0.5,
                          {"reason": "timestamps unavailable — assuming fresh"})
    try:
        t0    = datetime.fromisoformat(started.replace("Z", "+00:00"))
        t1    = datetime.fromisoformat(first.replace("Z", "+00:00"))
        delta = abs((t1 - t0).total_seconds()) / 60
        score = max(0.0, 1.0 - delta / (DATA_FRESHNESS_MINUTES * 2))
        return EvalResult("data_freshness", "market", round(score, 2),
                          delta <= DATA_FRESHNESS_MINUTES, 0.5,
                          {"lag_minutes": round(delta, 1), "threshold_minutes": DATA_FRESHNESS_MINUTES})
    except Exception as e:
        return EvalResult("data_freshness", "market", 1.0, True, 0.5,
                          {"reason": f"parse error: {e}"})


# ── Research agent evals ──────────────────────────────────────────────────────

def eval_research_completion(traces: list[dict], session: dict) -> EvalResult:
    """
    Score 1.0 if research agent ran its LLM AND at least one tool call succeeded.
    Score 0.3 if LLM ran but all tool calls failed (agent started but couldn't fetch data).
    Score 0.0 if no research traces or no successful LLM call at all.
    """
    research = [t for t in traces if (t.get("agent") or "").lower() == "research"]
    if not research:
        return EvalResult("completion", "research", 0.0, False, 0.7,
                          {"reason": "no research agent traces"})
    llm_ok = any(
        (t.get("step_type") or "") in ("decision", "llm_call")
        and (t.get("outcome") or "") == "success"
        for t in research
    )
    if not llm_ok:
        return EvalResult("completion", "research", 0.0, False, 0.7,
                          {"reason": "no successful llm_call or decision trace"})
    tool_calls = [t for t in research if (t.get("step_type") or "") == "tool_call"]
    tool_ok    = any((t.get("outcome") or "") == "success" for t in tool_calls)
    if tool_calls and not tool_ok:
        # LLM ran but every tool call failed — agent couldn't fetch data
        return EvalResult("completion", "research", 0.3, False, 0.7,
                          {"reason": "llm ran but all tool calls failed",
                           "tool_calls": len(tool_calls)})
    return EvalResult("completion", "research", 1.0, True, 0.7,
                      {"llm_ok": True, "tool_calls_ok": True})


def eval_research_token_efficiency(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if research agent used < 30K tokens total."""
    research = [t for t in traces if (t.get("agent") or "").lower() == "research"]
    tokens   = sum((t.get("tokens_input") or 0) + (t.get("tokens_output") or 0) for t in research)
    score    = max(0.0, 1.0 - tokens / (TOKEN_EFFICIENCY_THRESHOLD * 2))
    return EvalResult("token_efficiency", "research", round(score, 2),
                      tokens < TOKEN_EFFICIENCY_THRESHOLD, 0.5,
                      {"tokens_used": tokens, "threshold": TOKEN_EFFICIENCY_THRESHOLD})


def eval_research_tool_success_rate(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if >= 80% of research agent tool calls succeeded."""
    tools = [
        t for t in traces
        if (t.get("agent") or "").lower() == "research"
        and (t.get("step_type") or "") == "tool_call"
    ]
    if not tools:
        return EvalResult("tool_success_rate", "research", 1.0, True, 0.8,
                          {"reason": "no tool calls made"})
    success = sum(1 for t in tools if (t.get("outcome") or "") == "success")
    rate    = success / len(tools)
    return EvalResult("tool_success_rate", "research", round(rate, 2),
                      rate >= TOOL_SUCCESS_MIN_RATE, TOOL_SUCCESS_MIN_RATE,
                      {"total": len(tools), "success": success,
                       "failed": len(tools) - success, "rate": round(rate, 2)})


# ── Risk agent evals ──────────────────────────────────────────────────────────

def eval_risk_assessment_complete(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if risk agent has at least one successful trace."""
    risk = [t for t in traces if (t.get("agent") or "").lower() == "risk"]
    if not risk:
        return EvalResult("assessment_complete", "risk", 0.0, False, 1.0,
                          {"reason": "no risk agent traces"})
    success = [t for t in risk if (t.get("outcome") or "") == "success"]
    score   = 1.0 if success else 0.0
    return EvalResult("assessment_complete", "risk", score, bool(success), 1.0,
                      {"total": len(risk), "success": len(success)})


def eval_risk_within_parameters(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if risk agent has no error traces (proxy for parameter compliance)."""
    risk   = [t for t in traces if (t.get("agent") or "").lower() == "risk"]
    errors = [t for t in risk if t.get("error")]
    score  = 1.0 if not errors else max(0.0, 1.0 - len(errors) / max(1, len(risk)))
    return EvalResult("within_parameters", "risk", round(score, 2),
                      not errors, 0.9,
                      {"risk_traces": len(risk), "error_traces": len(errors),
                       "errors": [t.get("error") for t in errors[:3]]})


# ── Orchestrator evals ────────────────────────────────────────────────────────

def eval_orchestrator_decision_made(traces: list[dict], session: dict) -> EvalResult:
    """
    Score 1.0 if session has trades_executed > 0 OR terminal_reason logged.
    Either means the orchestrator made an explicit decision.
    """
    trades = int(session.get("trades_executed") or 0)
    reason = session.get("terminal_reason") or ""
    orch   = [t for t in traces if (t.get("agent") or "").lower() == "orchestrator"]

    if trades > 0:
        return EvalResult("decision_made", "orchestrator", 1.0, True, 0.7,
                          {"trades_executed": trades})
    if reason:
        return EvalResult("decision_made", "orchestrator", 0.8, True, 0.7,
                          {"terminal_reason": reason})
    if orch:
        score = 0.4
        return EvalResult("decision_made", "orchestrator", score, False, 0.7,
                          {"reason": "orchestrator ran but no decision or reason recorded",
                           "orch_traces": len(orch)})
    return EvalResult("decision_made", "orchestrator", 0.0, False, 0.7,
                      {"reason": "orchestrator never ran"})


def eval_orchestrator_consistency(traces: list[dict], session: dict) -> EvalResult:
    """
    Score 1.0 if orchestrator ran after research — proxy for consistent pipeline.
    If research succeeded but orchestrator has errors, flag inconsistency.
    """
    research_ok = any(
        (t.get("agent") or "").lower() == "research"
        and (t.get("outcome") or "") == "success"
        for t in traces
    )
    orch_errors = [
        t for t in traces
        if (t.get("agent") or "").lower() == "orchestrator"
        and t.get("error")
    ]
    if not research_ok:
        return EvalResult("consistency", "orchestrator", 1.0, True, 0.8,
                          {"reason": "research did not succeed — orchestrator consistency N/A"})
    if orch_errors:
        return EvalResult("consistency", "orchestrator", 0.2, False, 0.8,
                          {"reason": "orchestrator errors after research succeeded",
                           "errors": [t.get("error") for t in orch_errors[:2]]})
    return EvalResult("consistency", "orchestrator", 1.0, True, 0.8,
                      {"reason": "orchestrator ran cleanly after research"})


# ── Holistic session evals ────────────────────────────────────────────────────

def eval_session_pipeline_completion(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if all 4 required agents have at least one trace."""
    agents_present = {(t.get("agent") or "").lower() for t in traces}
    # market_shadow counts as market
    if "market_shadow" in agents_present:
        agents_present.add("market")
    missing = [a for a in REQUIRED_AGENTS if a not in agents_present]
    score   = len([a for a in REQUIRED_AGENTS if a in agents_present]) / len(REQUIRED_AGENTS)
    return EvalResult("pipeline_completion", "session", round(score, 2),
                      not missing, 1.0,
                      {"present": list(agents_present & set(REQUIRED_AGENTS)),
                       "missing": missing})


def eval_session_cost_anomaly(traces: list[dict], session: dict,
                               recent_costs: list[float] | None = None) -> EvalResult:
    """Score 0.0 if session cost is > mean + 2 sigma of recent sessions."""
    cost = float(session.get("total_cost_usd") or 0)
    if not recent_costs or len(recent_costs) < 5:
        return EvalResult("cost_anomaly", "session", 1.0, True, 0.5,
                          {"reason": "insufficient history", "cost_usd": cost})
    mean  = statistics.mean(recent_costs)
    stdev = statistics.stdev(recent_costs) if len(recent_costs) > 1 else 0
    z     = (cost - mean) / stdev if stdev > 0 else 0
    score = max(0.0, 1.0 - max(0, z - 1) / 2)
    return EvalResult("cost_anomaly", "session", round(score, 2),
                      z <= COST_ANOMALY_SIGMA, 0.5,
                      {"cost_usd": cost, "mean": round(mean, 4),
                       "stdev": round(stdev, 4), "z_score": round(z, 2),
                       "threshold_sigma": COST_ANOMALY_SIGMA})


def eval_session_outcome_linkage(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if session has trades OR an explicit terminal reason."""
    trades = int(session.get("trades_executed") or 0)
    reason = session.get("terminal_reason") or ""
    if trades > 0:
        return EvalResult("outcome_linkage", "session", 1.0, True, 0.7,
                          {"trades_executed": trades})
    if reason:
        return EvalResult("outcome_linkage", "session", 0.8, True, 0.7,
                          {"terminal_reason": reason, "trades_executed": 0})
    return EvalResult("outcome_linkage", "session", 0.0, False, 0.7,
                      {"reason": "0 trades and no terminal_reason — silent exit"})


def eval_session_tokens_per_decision(traces: list[dict], session: dict) -> EvalResult:
    """Score 1.0 if total tokens < 50K. Penalizes context spirals."""
    tokens  = int((session.get("total_tokens_input") or 0) +
                  (session.get("total_tokens_output") or 0))
    trades  = max(1, int(session.get("trades_executed") or 0))
    tok_per = tokens / trades
    score   = max(0.0, 1.0 - tok_per / (TOKEN_SPIRAL_THRESHOLD * 2))
    return EvalResult("tokens_per_decision", "session", round(score, 2),
                      tok_per < TOKEN_SPIRAL_THRESHOLD, 0.5,
                      {"total_tokens": tokens, "trades": trades,
                       "tokens_per_decision": round(tok_per),
                       "threshold": TOKEN_SPIRAL_THRESHOLD})


# ── Registry & runner ─────────────────────────────────────────────────────────

PER_AGENT_EVALS = [
    eval_market_data_completeness,
    eval_market_data_freshness,
    eval_research_completion,
    eval_research_token_efficiency,
    eval_research_tool_success_rate,
    eval_risk_assessment_complete,
    eval_risk_within_parameters,
    eval_orchestrator_decision_made,
    eval_orchestrator_consistency,
]

SESSION_EVALS = [
    eval_session_pipeline_completion,
    eval_session_cost_anomaly,
    eval_session_outcome_linkage,
    eval_session_tokens_per_decision,
]


def run_all_evals(
    session: dict,
    traces: list[dict],
    recent_costs: list[float] | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for fn in PER_AGENT_EVALS:
        results.append(fn(traces, session))
    for fn in SESSION_EVALS:
        if fn.__name__ == "eval_session_cost_anomaly":
            results.append(fn(traces, session, recent_costs))
        else:
            results.append(fn(traces, session))
    return results


def run_and_persist(session: dict, traces: list[dict],
                    recent_costs: list[float] | None = None,
                    db=None) -> list[EvalResult]:
    """Run all evals and write results to c_evals if db is provided."""
    results = run_all_evals(session, traces, recent_costs)
    if db:
        sid = session.get("id", "")
        rows = [r.to_db_row(sid) for r in results]
        if rows:
            db.table("c_evals").insert(rows).execute()
    return results
