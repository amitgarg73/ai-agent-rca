"""
Tests for rca_engine.py — previously had zero coverage.
Covers build_annotated_call_stack, generate_fix_suggestion, summarize_incident.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.rca_engine import (
    build_annotated_call_stack,
    generate_fix_suggestion,
    summarize_incident,
    SEVERITY_COLORS,
)
from engine.pattern_detector import Incident


# ── Helpers ───────────────────────────────────────────────────────────────────

def _incident(pattern="Tool Timeout Loop", severity="critical",
              call_stack=None, failed_evals=None) -> Incident:
    return Incident(
        session_id="sess-001",
        pattern_name=pattern,
        severity=severity,
        root_cause="test root cause",
        call_stack=call_stack or [],
        failed_evals=failed_evals or [],
        cost_wasted=0.05,
        tokens_wasted=500,
        fix_suggestion="default fix",
    )


def _trace(agent: str, step_type: str = "tool_call", tool_name: str = "get_data",
           outcome: str = "success", error: str | None = None,
           created_at: str = "2026-06-08T10:00:00", tokens_in: int = 100,
           tokens_out: int = 50) -> dict:
    return {
        "agent":        agent,
        "step_type":    step_type,
        "tool_name":    tool_name,
        "outcome":      outcome,
        "error":        error,
        "created_at":   created_at,
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
    }


def _session(started: str = "2026-06-08T10:00:00",
             completed: str = "2026-06-08T10:05:00",
             trades: int = 1) -> dict:
    return {
        "started_at":   started,
        "completed_at": completed,
        "trades_executed": trades,
    }


# ── build_annotated_call_stack ────────────────────────────────────────────────

class TestBuildAnnotatedCallStack:

    def test_returns_same_count_as_input(self):
        traces = [_trace("market"), _trace("research"), _trace("risk")]
        inc    = _incident("Pipeline Break")
        result = build_annotated_call_stack(traces, inc)
        assert len(result) == 3

    def test_each_trace_gets_is_root_and_is_relevant(self):
        traces = [_trace("research")]
        inc    = _incident("Context Spiral")
        result = build_annotated_call_stack(traces, inc)
        assert "is_root"     in result[0]
        assert "is_relevant" in result[0]
        assert "tokens"      in result[0]

    def test_tokens_field_is_sum_of_input_and_output(self):
        traces = [_trace("market", tokens_in=200, tokens_out=80)]
        inc    = _incident("Cost Anomaly")
        result = build_annotated_call_stack(traces, inc)
        assert result[0]["tokens"] == 280

    def test_sorted_by_created_at(self):
        traces = [
            _trace("risk",       created_at="2026-06-08T10:02:00"),
            _trace("market",     created_at="2026-06-08T10:00:00"),
            _trace("research",   created_at="2026-06-08T10:01:00"),
        ]
        inc    = _incident("Pipeline Break")
        result = build_annotated_call_stack(traces, inc)
        agents = [r["agent"] for r in result]
        assert agents == ["market", "research", "risk"]

    def test_empty_traces_returns_empty(self):
        result = build_annotated_call_stack([], _incident())
        assert result == []

    # Tool Timeout Loop
    def test_tool_timeout_marks_first_error_as_root(self):
        stack = [{"agent": "research", "tool_name": "get_data",
                  "step_type": "tool_call", "outcome": "error", "error": "timeout"}]
        inc    = _incident("Tool Timeout Loop", call_stack=stack)
        traces = [
            _trace("research", tool_name="get_data", outcome="error",
                   error="timeout", created_at="2026-06-08T10:00:01"),
            _trace("research", tool_name="get_data", outcome="error",
                   error="timeout", created_at="2026-06-08T10:00:02"),
        ]
        result = build_annotated_call_stack(traces, inc)
        roots  = [r for r in result if r["is_root"]]
        assert len(roots) == 1
        assert roots[0]["created_at"] == "2026-06-08T10:00:01"

    def test_tool_timeout_marks_all_matching_tool_traces_relevant(self):
        stack = [{"agent": "research", "tool_name": "get_data",
                  "step_type": "tool_call", "outcome": "error"}]
        inc    = _incident("Tool Timeout Loop", call_stack=stack)
        traces = [
            _trace("research", tool_name="get_data",  outcome="error"),
            _trace("research", tool_name="other_tool",outcome="success"),
            _trace("research", tool_name="get_data",  outcome="error"),
        ]
        result = build_annotated_call_stack(traces, inc)
        relevant = [r for r in result if r["is_relevant"]]
        assert all(r["tool_name"] == "get_data" for r in relevant)
        assert len(relevant) == 2

    def test_tool_timeout_no_root_without_call_stack(self):
        inc    = _incident("Tool Timeout Loop", call_stack=[])
        traces = [_trace("research", outcome="error", error="timeout")]
        result = build_annotated_call_stack(traces, inc)
        assert not any(r["is_root"] for r in result)

    # Context Spiral
    def test_context_spiral_marks_research_traces_relevant(self):
        traces = [
            _trace("market"),
            _trace("research"),
            _trace("research", step_type="llm_call"),
            _trace("risk"),
        ]
        inc    = _incident("Context Spiral")
        result = build_annotated_call_stack(traces, inc)
        for r in result:
            if r["agent"] == "research":
                assert r["is_relevant"]
            else:
                assert not r["is_relevant"]

    # Pipeline Break
    def test_pipeline_break_marks_research_and_orchestrator_relevant(self):
        traces = [
            _trace("market"),
            _trace("research"),
            _trace("orchestrator"),
        ]
        inc    = _incident("Pipeline Break")
        result = build_annotated_call_stack(traces, inc)
        relevant_agents = {r["agent"] for r in result if r["is_relevant"]}
        assert relevant_agents == {"research", "orchestrator"}
        assert not any(r["is_relevant"] for r in result if r["agent"] == "market")

    # Cost Anomaly / Silent Exit / Empty Result Loop
    def test_cost_anomaly_marks_error_traces_relevant(self):
        traces = [
            _trace("market", outcome="success"),
            _trace("research", outcome="error", error="some error"),
        ]
        inc    = _incident("Cost Anomaly")
        result = build_annotated_call_stack(traces, inc)
        relevant = [r for r in result if r["is_relevant"]]
        assert len(relevant) == 1
        assert relevant[0]["agent"] == "research"

    def test_unknown_pattern_marks_nothing_relevant(self):
        traces = [_trace("market"), _trace("research")]
        inc    = _incident("Unknown Pattern")
        result = build_annotated_call_stack(traces, inc)
        assert not any(r["is_relevant"] for r in result)
        assert not any(r["is_root"] for r in result)


# ── generate_fix_suggestion ───────────────────────────────────────────────────

class TestGenerateFixSuggestion:

    def test_tool_timeout_mentions_timeout_param(self):
        stack = [{"agent": "research", "tool_name": "get_news"}]
        inc   = _incident("Tool Timeout Loop", call_stack=stack)
        fix   = generate_fix_suggestion(inc, [])
        assert "timeout" in fix.lower()
        assert "get_news" in fix

    def test_tool_timeout_mentions_agent_name(self):
        stack = [{"agent": "market", "tool_name": "get_data"}]
        inc   = _incident("Tool Timeout Loop", call_stack=stack)
        fix   = generate_fix_suggestion(inc, [])
        assert "market" in fix

    def test_tool_timeout_mentions_retries(self):
        stack = [{"agent": "research", "tool_name": "get_data"}]
        inc   = _incident("Tool Timeout Loop", call_stack=stack)
        fix   = generate_fix_suggestion(inc, [])
        assert "retr" in fix.lower()

    def test_context_spiral_mentions_token_budget(self):
        inc = Incident(
            session_id="s", pattern_name="Context Spiral", severity="warning",
            root_cause="", call_stack=[], failed_evals=[],
            tokens_wasted=80_000, fix_suggestion="",
        )
        fix = generate_fix_suggestion(inc, [])
        assert "token" in fix.lower()

    def test_pipeline_break_mentions_handoff(self):
        inc = _incident("Pipeline Break")
        fix = generate_fix_suggestion(inc, [])
        assert "handoff" in fix.lower() or "orchestrator" in fix.lower()

    def test_cost_anomaly_mentions_budget(self):
        inc = _incident("Cost Anomaly")
        fix = generate_fix_suggestion(inc, [])
        assert "cost" in fix.lower() or "budget" in fix.lower()

    def test_silent_exit_mentions_terminal_reason(self):
        inc = _incident("Silent Exit")
        fix = generate_fix_suggestion(inc, [])
        assert "terminal_reason" in fix

    def test_empty_result_loop_mentions_tool(self):
        stack = [{"agent": "research", "tool_name": "get_news", "step_type": "tool_call"}]
        inc   = _incident("Empty Result Loop", call_stack=stack)
        fix   = generate_fix_suggestion(inc, [])
        assert "get_news" in fix

    def test_unknown_pattern_returns_existing_fix_suggestion(self):
        inc = _incident("Some Unknown Pattern")
        inc.fix_suggestion = "my custom fix"
        fix = generate_fix_suggestion(inc, [])
        assert fix == "my custom fix"

    def test_all_known_patterns_return_non_empty_string(self):
        patterns = [
            ("Tool Timeout Loop",  [{"agent": "a", "tool_name": "t"}]),
            ("Context Spiral",     []),
            ("Pipeline Break",     []),
            ("Cost Anomaly",       []),
            ("Silent Exit",        []),
            ("Empty Result Loop",  [{"agent": "a", "tool_name": "t"}]),
        ]
        for pattern, stack in patterns:
            inc = Incident(
                session_id="s", pattern_name=pattern, severity="warning",
                root_cause="", call_stack=stack, failed_evals=[],
                tokens_wasted=1000, fix_suggestion="fallback",
            )
            fix = generate_fix_suggestion(inc, [])
            assert isinstance(fix, str) and len(fix) > 0, f"Empty fix for {pattern}"


# ── summarize_incident ────────────────────────────────────────────────────────

class TestSummarizeIncident:

    def test_returns_all_required_keys(self):
        inc  = _incident()
        sess = _session()
        summary = summarize_incident(inc, sess)
        required = {"pattern", "severity", "severity_color", "root_cause",
                    "fix_suggestion", "cost_wasted", "tokens_wasted",
                    "duration_s", "trades", "agents", "failed_eval_count"}
        assert required.issubset(summary.keys())

    def test_duration_computed_from_timestamps(self):
        inc  = _incident()
        sess = _session(started="2026-06-08T10:00:00", completed="2026-06-08T10:05:00")
        summary = summarize_incident(inc, sess)
        assert summary["duration_s"] == 300

    def test_duration_zero_on_missing_timestamps(self):
        inc  = _incident()
        summary = summarize_incident(inc, {})
        assert summary["duration_s"] == 0

    def test_trades_from_session(self):
        inc  = _incident()
        sess = _session(trades=3)
        assert summarize_incident(inc, sess)["trades"] == 3

    def test_severity_color_maps_known_severities(self):
        for sev, color in SEVERITY_COLORS.items():
            inc = _incident(severity=sev)
            assert summarize_incident(inc, {})["severity_color"] == color

    def test_agents_extracted_from_call_stack(self):
        stack = [
            {"agent": "market",   "tool_name": "x"},
            {"agent": "research", "tool_name": "y"},
        ]
        inc = _incident(call_stack=stack)
        agents = set(summarize_incident(inc, {})["agents"])
        assert agents == {"market", "research"}

    def test_failed_eval_count(self):
        failed = [
            {"agent": "research", "eval_name": "tool_success_rate", "score": 0.5},
            {"agent": "risk",     "eval_name": "assessment_complete", "score": 0.0},
        ]
        inc = _incident(failed_evals=failed)
        assert summarize_incident(inc, {})["failed_eval_count"] == 2

    def test_pattern_and_severity_passed_through(self):
        inc = _incident("Context Spiral", severity="warning")
        summary = summarize_incident(inc, {})
        assert summary["pattern"]  == "Context Spiral"
        assert summary["severity"] == "warning"
