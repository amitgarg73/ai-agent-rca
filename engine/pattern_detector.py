"""
Pattern detector — Phase 1 rule-based detection over traces + eval results.
Writes incidents to c_incidents. Each detector returns an Incident or None.
"""
from __future__ import annotations

import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter

from engine.eval_engine import EvalResult, REQUIRED_AGENTS, COST_ANOMALY_SIGMA


def _real_error(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None

TOOL_RETRY_THRESHOLD      = 3
CONTEXT_SPIRAL_TOKENS     = 40_000
COST_ANOMALY_SIGMA_INC    = COST_ANOMALY_SIGMA
HYPERACTIVE_POLL_THRESHOLD = 6       # same tool called 6+ times successfully = polling loop
FABRICATION_LATENCY_MS    = 50       # tool_call faster than 50ms is suspicious
FABRICATION_MIN_COUNT     = 2        # need at least 2 suspect traces to fire
_HTTP_ERROR_CODES         = {"400", "401", "403", "429", "500", "502", "503"}

# Shadow circuit breaker config — which evals trigger a CB in shadow mode
# Format: { agent: [(eval_name, fail_threshold), ...] }
CB_CONFIG: dict[str, list[tuple[str, float]]] = {
    "research": [("tool_success_rate", 0.80), ("completion", 0.70)],
    "risk":     [("assessment_complete", 1.0)],
}


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
            t.get("outcome") == "error" or bool(_real_error(t.get("error")))
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
                f"Session cost USD{cost:.4f} is {z:.1f} standard deviations above "
                f"the mean (USD{mean:.4f}). Significantly higher than typical sessions."
            ),
            call_stack   = [_trace_to_stack_frame(t) for t in sorted_traces[-15:]],
            failed_evals = _failed_eval_rows(evals),
            cost_wasted  = max(0.0, cost - mean),
            tokens_wasted= 0,
            fix_suggestion = (
                f"Investigate which agent consumed the excess cost. "
                f"Mean session cost is USD{mean:.4f}. "
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
            t.get("outcome") == "error" or bool(_real_error(t.get("error")))
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


def detect_hyperactive_polling(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when the same tool is called 6+ times successfully in one session.
    Distinct from Tool Timeout Loop (errors) and Empty Result Loop (3+ + 0 trades).
    Signature of: agent polling obsessively without making a decision.
    """
    tool_success: dict[str, list[dict]] = {}
    for t in sorted(traces, key=lambda x: x.get("created_at", "")):
        if (t.get("step_type") or "") == "tool_call" and t.get("outcome") != "error":
            name = t.get("tool_name") or "unknown"
            tool_success.setdefault(name, []).append(t)

    for tool_name, calls in tool_success.items():
        if len(calls) >= HYPERACTIVE_POLL_THRESHOLD:
            session_id = session.get("id", "")
            agent = calls[0].get("agent", "unknown")
            total_latency = sum(c.get("latency_ms") or 0 for c in calls)
            return Incident(
                session_id    = session_id,
                pattern_name  = "Hyperactive Polling Loop",
                severity      = "warning",
                root_cause    = (
                    f"{tool_name} called {len(calls)} times successfully by {agent} agent. "
                    f"Agent is polling without reaching a decision threshold. "
                    f"Total time in tool calls: {total_latency:,}ms."
                ),
                call_stack    = [_trace_to_stack_frame(t) for t in calls],
                failed_evals  = _failed_eval_rows(evals),
                cost_wasted   = float(session.get("total_cost_usd") or 0),
                tokens_wasted = int((session.get("total_tokens_input") or 0) +
                                    (session.get("total_tokens_output") or 0)),
                fix_suggestion = (
                    f"Add a max_iterations cap to {agent} agent's {tool_name} loop "
                    f"(e.g. max_calls=3). Force a decision step once the cap is hit. "
                    f"Consider caching results between calls to avoid redundant fetches."
                ),
            )
    return None


def detect_tool_fabrication(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when 2+ tool_call traces complete in under 50ms.
    Real external API calls take 100ms+. Sub-50ms suggests the agent fabricated the response.
    """
    suspects = [
        t for t in traces
        if (t.get("step_type") or "") == "tool_call"
        and t.get("outcome") != "error"
        and 0 < (t.get("latency_ms") or 0) < FABRICATION_LATENCY_MS
    ]
    if len(suspects) >= FABRICATION_MIN_COUNT:
        session_id = session.get("id", "")
        agents = list({t.get("agent", "unknown") for t in suspects})
        tools  = list({t.get("tool_name", "unknown") for t in suspects})
        return Incident(
            session_id    = session_id,
            pattern_name  = "Tool Call Fabrication",
            severity      = "critical",
            root_cause    = (
                f"{len(suspects)} tool call(s) completed in under {FABRICATION_LATENCY_MS}ms "
                f"with success status. External API calls take 100ms+. "
                f"Affected agents: {agents}. Tools: {tools}."
            ),
            call_stack    = [_trace_to_stack_frame(t) for t in suspects],
            failed_evals  = _failed_eval_rows(evals),
            cost_wasted   = float(session.get("total_cost_usd") or 0),
            tokens_wasted = 0,
            fix_suggestion = (
                "Verify that tool calls are making real external requests. "
                "Add response validation: check that tool output is consistent with "
                "what the actual API returns (schema, latency, value ranges). "
                "Enable request logging in tool wrappers to confirm HTTP calls are made."
            ),
        )
    return None


def detect_handoff_schema_break(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when research completes successfully but the risk agent's first trace is an error.
    Signature of: research output not matching the schema risk agent expects at handoff.
    """
    sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))

    research_traces = [t for t in sorted_traces
                       if (t.get("agent") or "").lower().startswith("research")]
    risk_traces     = [t for t in sorted_traces
                       if (t.get("agent") or "").lower() == "risk"]

    if not research_traces or not risk_traces:
        return None

    research_ok = any(
        (t.get("step_type") or "") in ("llm_call", "agent_message", "decision")
        and t.get("outcome") != "error"
        for t in research_traces
    )
    if not research_ok:
        return None  # research itself failed — different pattern

    first_risk = risk_traces[0]
    risk_errored = (
        first_risk.get("outcome") == "error"
        or bool(_real_error(first_risk.get("error")))
    )
    if not risk_errored:
        return None

    session_id = session.get("id", "")
    err_msg    = first_risk.get("error") or "unknown schema error"
    return Incident(
        session_id    = session_id,
        pattern_name  = "Handoff Schema Break",
        severity      = "critical",
        root_cause    = (
            f"Research agent completed successfully but risk agent errored on its "
            f"first trace. Research output did not match risk agent's expected schema. "
            f"Error at handoff: {str(err_msg)[:120]}"
        ),
        call_stack    = (
            [_trace_to_stack_frame(t) for t in research_traces[-5:]]
            + [_trace_to_stack_frame(first_risk)]
        ),
        failed_evals  = _failed_eval_rows(evals),
        cost_wasted   = float(session.get("total_cost_usd") or 0),
        tokens_wasted = int((session.get("total_tokens_input") or 0) +
                            (session.get("total_tokens_output") or 0)),
        fix_suggestion = (
            "Add schema validation at the research -> risk handoff. "
            "Define a typed contract (Pydantic model or TypedDict) for what research "
            "must return before passing to risk. Fail fast with a clear error if the "
            "contract is violated, rather than letting risk agent crash on bad input."
        ),
    )


def detect_error_misinterpretation(
    session: dict, traces: list[dict], evals: list[EvalResult]
) -> Incident | None:
    """
    Fires when tool calls return HTTP error codes (400/401/429/503) but the same
    agent continues with an LLM call rather than aborting or retrying correctly.
    Signature of: agent ignoring or mishandling meaningful API error signals.
    """
    sorted_traces = sorted(traces, key=lambda x: x.get("created_at", ""))
    mishandled: list[dict] = []

    for i, t in enumerate(sorted_traces):
        if (t.get("step_type") or "") != "tool_call":
            continue
        err = str(t.get("error") or "")
        if not any(code in err for code in _HTTP_ERROR_CODES):
            continue
        # Check same agent continued with an LLM call after the error
        subsequent = sorted_traces[i + 1:]
        continued = any(
            s.get("agent") == t.get("agent")
            and (s.get("step_type") or "") == "llm_call"
            for s in subsequent
        )
        if continued:
            mishandled.append(t)

    if len(mishandled) >= 2:
        session_id   = session.get("id", "")
        agents_hit   = list({t.get("agent", "unknown") for t in mishandled})
        error_codes  = [str(t.get("error") or "")[:60] for t in mishandled]
        return Incident(
            session_id    = session_id,
            pattern_name  = "Error Misinterpretation",
            severity      = "warning",
            root_cause    = (
                f"{len(mishandled)} HTTP error(s) received but agent(s) {agents_hit} "
                f"continued processing instead of aborting or retrying correctly. "
                f"Errors: {error_codes[:3]}"
            ),
            call_stack    = [_trace_to_stack_frame(t) for t in mishandled],
            failed_evals  = _failed_eval_rows(evals),
            cost_wasted   = float(session.get("total_cost_usd") or 0),
            tokens_wasted = int((session.get("total_tokens_input") or 0) +
                                (session.get("total_tokens_output") or 0)),
            fix_suggestion = (
                "Add explicit HTTP error code handlers in tool wrappers: "
                "429 = exponential backoff + retry; "
                "401 = surface credential error to operator, do not continue; "
                "400 = log malformed request and abort the tool call chain; "
                "503 = circuit breaker open, fail fast."
            ),
        )
    return None


def compute_shadow_cb_fires(evals: list[EvalResult]) -> list[dict]:
    """
    Returns list of CB fire events based on CB_CONFIG thresholds.
    Each entry: {agent, eval_name, score, threshold, would_save_note}.
    Used for dashboard shadow CB markers — never aborts the pipeline.
    """
    fires = []
    for e in evals:
        if e.agent not in CB_CONFIG:
            continue
        for eval_name, threshold in CB_CONFIG[e.agent]:
            if e.eval_name == eval_name and e.score < threshold:
                fires.append({
                    "agent":      e.agent,
                    "eval_name":  e.eval_name,
                    "score":      round(e.score, 3),
                    "threshold":  threshold,
                })
    return fires


# ── Main runner ───────────────────────────────────────────────────────────────

DETECTORS = [
    detect_tool_timeout_loop,
    detect_context_spiral,
    detect_pipeline_break,
    detect_empty_result_loop,
    detect_silent_exit,
    detect_hyperactive_polling,
    detect_tool_fabrication,
    detect_handoff_schema_break,
    detect_error_misinterpretation,
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
