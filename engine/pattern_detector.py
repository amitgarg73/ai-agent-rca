"""
Pattern detector — Phase 1 rule-based detection over traces + eval results.
Writes incidents to c_incidents. Each detector returns an Incident or None.
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter

from engine.eval_engine import EvalResult, REQUIRED_AGENTS, COST_ANOMALY_SIGMA

TOOL_RETRY_THRESHOLD   = 3
CONTEXT_SPIRAL_TOKENS  = 40_000
COST_ANOMALY_SIGMA_INC = COST_ANOMALY_SIGMA


@dataclass
class Incident:
    session_id:     str
    pattern_name:   str
    severity:       str          # critical | warning | info
    root_cause:     str
    call_stack:     list[dict]
    failed_evals:   list[dict]
    cost_wasted:    float = 0.0
    tokens_wasted:  int   = 0
    fix_suggestion: str   = ""
    is_simulated:   bool  = False

    def to_db_row(self) -> dict:
        return {
            "id":             str(uuid.uuid4()),
            "session_id":     self.session_id,
            "pattern_name":   self.pattern_name,
            "severity":       self.severity,
            "root_cause":     self.root_cause,
            "call_stack":     self.call_stack,
            "failed_evals":   self.failed_evals,
            "cost_wasted":    round(self.cost_wasted, 4),
            "tokens_wasted":  self.tokens_wasted,
            "fix_suggestion": self.fix_suggestion,
            "is_simulated":   self.is_simulated,
        }


def _failed_eval_rows(evals: list[EvalResult]) -> list[dict]:
    return [
        {"agent": e.agent, "eval_name": e.eval_name,
         "score": e.score, "threshold": e.threshold}
        for e in evals if not e.passed
    ]


def _trace_to_stack_frame(t: dict) -> dict:
    return {
        "agent":       t.get("agent", ""),
        "step_type":   t.get("step_type", ""),
        "tool_name":   t.get("tool_name", ""),
        "outcome":     t.get("outcome", ""),
        "error":       t.get("error", ""),
        "latency_ms":  t.get("latency_ms", 0),
        "tokens":      (t.get("tokens_input") or 0) + (t.get("tokens_output") or 0),
        "created_at":  t.get("created_at", ""),
    }


# ── Individual pattern detectors ──────────────────────────────────────────────

def detect_tool_timeout_loop(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when the same tool_name has 3+ error traces.
    Signature of: external API timeout with no retry limit.
    """
    tool_errors: dict[str, list[dict]] = {}
    for t in sorted(traces, key=lambda x: x.get("created_at", "")):
        if (t.get("step_type") or "") == "tool_call" and (
            t.get("outcome") == "error" or t.get("error")
        ):
            tool = t.get("tool_name") or "unknown_tool"
            tool_errors.setdefault(tool, []).append(t)

    for tool_name, errors in tool_errors.items():
        if len(errors) >= TOOL_RETRY_THRESHOLD:
            session_id = session.get("id", "")
            cost       = float(session.get("total_cost_usd") or 0)
            tokens     = int((session.get("total_tokens_input") or 0) +
                             (session.get("total_tokens_output") or 0))
            agent      = errors[0].get("agent", "unknown")
            err_msg    = errors[0].get("error") or "unknown error"
            return Incident(
                session_id   = session_id,
                pattern_name = "Tool Timeout Loop",
                severity     = "critical",
                root_cause   = (
                    f"{tool_name} failed {len(errors)} times in {agent} agent. "
                    f"Error: {err_msg[:120]}"
                ),
                call_stack   = [_trace_to_stack_frame(t) for t in errors],
                failed_evals = _failed_eval_rows(evals),
                cost_wasted  = cost,
                tokens_wasted= tokens,
                fix_suggestion = (
                    f"Add max_retries=2 and timeout=10s to {tool_name} "
                    f"in the {agent} agent tool configuration. "
                    f"Consider exponential backoff between retries."
                ),
            )
    return None


def detect_context_spiral(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when research agent burns > 40K tokens with 0 trades executed.
    Signature of: agent gathering endlessly without deciding.
    """
    research_tokens = sum(
        (t.get("tokens_input") or 0) + (t.get("tokens_output") or 0)
        for t in traces
        if (t.get("agent") or "").lower() == "research"
    )
    trades = int(session.get("trades_executed") or 0)

    if research_tokens > CONTEXT_SPIRAL_TOKENS and trades == 0:
        session_id = session.get("id", "")
        cost       = float(session.get("total_cost_usd") or 0)
        research_traces = sorted(
            [t for t in traces if (t.get("agent") or "").lower() == "research"],
            key=lambda x: x.get("created_at", "")
        )
        return Incident(
            session_id   = session_id,
            pattern_name = "Context Spiral",
            severity     = "warning",
            root_cause   = (
                f"Research agent consumed {research_tokens:,} tokens "
                f"with 0 trades produced. Agent gathered context "
                f"without reaching a decision."
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in research_traces[-20:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = cost,
            tokens_wasted= research_tokens,
            fix_suggestion = (
                "Add a hard token budget to the research agent "
                f"(e.g. max_tokens={CONTEXT_SPIRAL_TOKENS // 2}). "
                "Force a decision step if budget is reached without a recommendation."
            ),
        )
    return None


def detect_pipeline_break(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when research agent ran but orchestrator did not.
    Signature of: mid-pipeline failure or uncaught exception.
    """
    agents_seen = {(t.get("agent") or "").lower() for t in traces}
    if "market_shadow" in agents_seen:
        agents_seen.add("market")

    research_ran     = "research" in agents_seen
    orchestrator_ran = "orchestrator" in agents_seen

    if research_ran and not orchestrator_ran:
        session_id     = session.get("id", "")
        cost           = float(session.get("total_cost_usd") or 0)
        research_traces = sorted(
            [t for t in traces if (t.get("agent") or "").lower() == "research"],
            key=lambda x: x.get("created_at", "")
        )
        last_research = research_traces[-1] if research_traces else {}
        return Incident(
            session_id   = session_id,
            pattern_name = "Pipeline Break",
            severity     = "critical",
            root_cause   = (
                f"Research agent completed but orchestrator never started. "
                f"Last research step: {last_research.get('step_type', 'unknown')} "
                f"(outcome: {last_research.get('outcome', 'unknown')})"
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in research_traces[-10:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = cost,
            tokens_wasted= int((session.get("total_tokens_input") or 0) +
                               (session.get("total_tokens_output") or 0)),
            fix_suggestion = (
                "Check for uncaught exceptions between research and orchestrator steps. "
                "Add try/except around the research -> orchestrator handoff. "
                "Ensure research agent errors are propagated, not silently swallowed."
            ),
        )
    return None


def detect_cost_anomaly(
    session: dict, traces: list[dict], evals: list[EvalResult],
    recent_costs: list[float] | None = None,
) -> Incident | None:
    """
    Fires when session cost > mean + 2 sigma of recent sessions.
    """
    if not recent_costs or len(recent_costs) < 5:
        return None

    cost  = float(session.get("total_cost_usd") or 0)
    mean  = statistics.mean(recent_costs)
    stdev = statistics.stdev(recent_costs) if len(recent_costs) > 1 else 0
    z     = (cost - mean) / stdev if stdev > 0 else 0

    if z > COST_ANOMALY_SIGMA_INC:
        session_id = session.get("id", "")
        sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))
        return Incident(
            session_id   = session_id,
            pattern_name = "Cost Anomaly",
            severity     = "warning",
            root_cause   = (
                f"Session cost ${cost:.4f} is {z:.1f} standard deviations above "
                f"the mean (${mean:.4f}). Significantly higher than typical sessions."
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in sorted_traces[-15:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = max(0.0, cost - mean),
            tokens_wasted= 0,
            fix_suggestion = (
                f"Investigate which agent consumed the excess cost. "
                f"Mean session cost is ${mean:.4f}. "
                f"Check token usage by agent in Session Deep Dive."
            ),
        )
    return None


def detect_silent_exit(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when session ends with 0 trades and no terminal reason.
    Not caused by a tool error (that would be Tool Timeout Loop).
    """
    trades = int(session.get("trades_executed") or 0)
    reason = session.get("terminal_reason") or ""

    # Only fire if no other pattern already explains it
    has_tool_errors = any(
        (t.get("step_type") or "") == "tool_call" and (
            t.get("outcome") == "error" or t.get("error")
        )
        for t in traces
    )
    if trades == 0 and not reason and not has_tool_errors and traces:
        session_id = session.get("id", "")
        sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))
        return Incident(
            session_id   = session_id,
            pattern_name = "Silent Exit",
            severity     = "info",
            root_cause   = (
                "Session ended with 0 trades and no terminal reason logged. "
                "No tool errors detected. Likely a logic path that skips "
                "both trade entry and explicit rejection."
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in sorted_traces[-10:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = float(session.get("total_cost_usd") or 0),
            tokens_wasted= int((session.get("total_tokens_input") or 0) +
                               (session.get("total_tokens_output") or 0)),
            fix_suggestion = (
                "Add explicit terminal_reason logging for all exit paths "
                "in the orchestrator (e.g. 'no_opportunity', 'risk_rejected', "
                "'market_closed'). Every session should record why it ended."
            ),
        )
    return None


def detect_empty_result_loop(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when same tool is called 3+ times successfully but session still
    fails — suggesting the tool returns empty/useless results.
    """
    tool_calls = [
        t for t in traces
        if (t.get("step_type") or "") == "tool_call"
        and (t.get("outcome") or "") == "success"
    ]
    tool_counts = Counter(t.get("tool_name") or "unknown" for t in tool_calls)
    trades      = int(session.get("trades_executed") or 0)

    for tool_name, count in tool_counts.items():
        if count >= TOOL_RETRY_THRESHOLD and trades == 0:
            session_id = session.get("id", "")
            relevant   = [t for t in tool_calls if t.get("tool_name") == tool_name]
            return Incident(
                session_id   = session_id,
                pattern_name = "Empty Result Loop",
                severity     = "warning",
                root_cause   = (
                    f"{tool_name} called {count} times successfully but "
                    f"produced no actionable output (0 trades). "
                    f"Tool may be returning empty or low-quality results."
                ),
                call_stack   = [_trace_to_stack_frame(t) for t in relevant],
                failed_evals = _failed_eval_rows(evals),
                cost_wasted  = float(session.get("total_cost_usd") or 0),
                tokens_wasted= int((session.get("total_tokens_input") or 0) +
                                   (session.get("total_tokens_output") or 0)),
                fix_suggestion = (
                    f"Add result quality validation after {tool_name} calls. "
                    f"If result is empty or below quality threshold, "
                    f"fail fast rather than retrying with same parameters."
                ),
            )
    return None


# ── Main runner ───────────────────────────────────────────────────────────────

DETECTORS = [
    detect_tool_timeout_loop,
    detect_context_spiral,
    detect_pipeline_break,
    detect_empty_result_loop,
    detect_silent_exit,
]


def run_all_detectors(
    session: dict,
    traces: list[dict],
    evals: list[EvalResult],
    recent_costs: list[float] | None = None,
) -> list[Incident]:
    incidents: list[Incident] = []
    for detector in DETECTORS:
        try:
            result = detector(session, traces, evals)
            if result:
                incidents.append(result)
        except Exception:
            pass  # never let a detector crash the pipeline
    # Cost anomaly needs recent_costs
    try:
        result = detect_cost_anomaly(session, traces, evals, recent_costs)
        if result:
            incidents.append(result)
    except Exception:
        pass
    return incidents


def run_and_persist(
    session: dict,
    traces: list[dict],
    evals: list[EvalResult],
    recent_costs: list[float] | None = None,
    db=None,
) -> list[Incident]:
    incidents = run_all_detectors(session, traces, evals, recent_costs)
    if db and incidents:
        rows = [i.to_db_row() for i in incidents]
        db.table("c_incidents").insert(rows).execute()
    return incidents
