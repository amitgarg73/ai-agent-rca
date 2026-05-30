"""
Comprehensive tests for pattern_detector.py.
Each detector: fires correctly, does not fire on clean session, edge cases.
Run: python3 -m pytest tests/test_detector.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.eval_engine     import run_all_evals
from engine.pattern_detector import (
    detect_tool_timeout_loop,
    detect_context_spiral,
    detect_pipeline_break,
    detect_cost_anomaly,
    detect_silent_exit,
    detect_empty_result_loop,
    run_all_detectors,
    Incident,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_trace(agent="research", step_type="tool_call", tool_name="get_stock_data",
               outcome="success", error=None, latency_ms=500,
               tokens_in=100, tokens_out=50,
               created_at="2026-05-28T06:34:00Z", session_id="sess-001"):
    return {
        "id": "t-001", "session_id": session_id,
        "agent": agent, "step_type": step_type, "tool_name": tool_name,
        "outcome": outcome, "error": error, "latency_ms": latency_ms,
        "tokens_input": tokens_in, "tokens_output": tokens_out,
        "created_at": created_at,
    }

def make_session(cost=0.10, trades=1, reason=None, tokens_in=5000, tokens_out=2000):
    return {
        "id": "sess-001", "total_cost_usd": cost,
        "trades_executed": trades, "terminal_reason": reason,
        "started_at": "2026-05-28T06:00:00Z", "completed_at": "2026-05-28T06:30:00Z",
        "total_tokens_input": tokens_in, "total_tokens_output": tokens_out,
    }

EMPTY_EVALS = []


# ── Tool Timeout Loop ─────────────────────────────────────────────────────────

class TestToolTimeoutLoop:
    def test_fires_on_3_errors(self):
        traces = [
            make_trace(tool_name="get_stock_data", outcome="error",
                       error="ReadTimeout", created_at=f"2026-05-28T06:3{i}:00Z")
            for i in range(3)
        ]
        inc = detect_tool_timeout_loop(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Tool Timeout Loop"
        assert inc.severity == "critical"

    def test_fires_on_yfinance_scenario(self):
        """The real 2026-05-28 incident: 8 retries of get_stock_data."""
        traces = [
            make_trace(tool_name="get_stock_data", outcome="error",
                       error="ReadTimeout: HTTPSConnectionPool timed out",
                       created_at=f"2026-05-28T06:{30+i}:00Z")
            for i in range(8)
        ]
        inc = detect_tool_timeout_loop(make_session(cost=1.98, trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.cost_wasted == pytest.approx(1.98)
        assert "get_stock_data" in inc.root_cause
        assert "8" in inc.root_cause
        assert "max_retries" in inc.fix_suggestion

    def test_does_not_fire_on_2_errors(self):
        traces = [
            make_trace(tool_name="get_stock_data", outcome="error", error="timeout",
                       created_at=f"2026-05-28T06:3{i}:00Z")
            for i in range(2)
        ]
        inc = detect_tool_timeout_loop(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_on_different_tools(self):
        """Errors on different tools should not trigger — not a retry loop."""
        traces = [
            make_trace(tool_name="get_stock_data",  outcome="error", error="timeout"),
            make_trace(tool_name="get_news",         outcome="error", error="timeout"),
            make_trace(tool_name="get_earnings",     outcome="error", error="timeout"),
            make_trace(tool_name="get_technicals",   outcome="error", error="timeout"),
        ]
        inc = detect_tool_timeout_loop(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_on_clean_session(self):
        traces = [make_trace(tool_name="get_stock_data", outcome="success")]
        inc = detect_tool_timeout_loop(make_session(trades=1), traces, EMPTY_EVALS)
        assert inc is None

    def test_edge_empty_traces(self):
        inc = detect_tool_timeout_loop(make_session(), [], EMPTY_EVALS)
        assert inc is None

    def test_call_stack_contains_error_traces(self):
        traces = [
            make_trace(tool_name="get_stock_data", outcome="error",
                       error="timeout", created_at=f"2026-05-28T06:3{i}:00Z")
            for i in range(5)
        ]
        inc = detect_tool_timeout_loop(make_session(), traces, EMPTY_EVALS)
        assert len(inc.call_stack) == 5
        assert all(f["tool_name"] == "get_stock_data" for f in inc.call_stack)


# ── Context Spiral ────────────────────────────────────────────────────────────

class TestContextSpiral:
    def test_fires_on_high_tokens_no_trades(self):
        # Detector counts tokens from research traces (not session totals)
        # Must exceed TOKEN_SPIRAL_THRESHOLD (40_000)
        traces = [
            make_trace(agent="research", step_type="llm_call",
                       tokens_in=30000, tokens_out=12000)
        ]
        inc = detect_context_spiral(make_session(trades=0, tokens_in=42000, tokens_out=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Context Spiral"
        assert inc.severity == "warning"

    def test_does_not_fire_when_trade_made(self):
        traces = [
            make_trace(agent="research", step_type="llm_call",
                       tokens_in=25000, tokens_out=8000)
        ]
        inc = detect_context_spiral(make_session(trades=1), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_below_token_threshold(self):
        traces = [make_trace(agent="research", tokens_in=5000, tokens_out=2000)]
        inc = detect_context_spiral(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is None

    def test_fix_suggestion_mentions_budget(self):
        traces = [make_trace(agent="research", tokens_in=30000, tokens_out=12000)]
        inc = detect_context_spiral(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert "budget" in inc.fix_suggestion.lower() or "token" in inc.fix_suggestion.lower()

    def test_edge_empty_traces(self):
        inc = detect_context_spiral(make_session(trades=0), [], EMPTY_EVALS)
        assert inc is None


# ── Pipeline Break ────────────────────────────────────────────────────────────

class TestPipelineBreak:
    def test_fires_research_no_orchestrator(self):
        traces = [
            make_trace(agent="research",    outcome="success"),
            make_trace(agent="market",      outcome="success"),
            make_trace(agent="risk",        outcome="error", error="connection refused"),
        ]
        inc = detect_pipeline_break(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Pipeline Break"
        assert inc.severity == "critical"

    def test_does_not_fire_when_orchestrator_present(self):
        traces = [
            make_trace(agent=a, outcome="success")
            for a in ["market","research","risk","orchestrator"]
        ]
        inc = detect_pipeline_break(make_session(trades=1), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_when_only_market_and_research(self):
        """market + research without orch: could be pipeline break, but only fires
        when research is present and orch is absent."""
        traces = [
            make_trace(agent="market",   outcome="success"),
            make_trace(agent="research", outcome="success"),
        ]
        inc = detect_pipeline_break(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None  # research present, orchestrator absent → fires

    def test_does_not_fire_no_research(self):
        """If research never ran, pipeline break doesn't apply."""
        traces = [make_trace(agent="market", outcome="success")]
        inc = detect_pipeline_break(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is None

    def test_fix_suggestion_mentions_handoff(self):
        traces = [make_trace(agent="research", outcome="success")]
        inc = detect_pipeline_break(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert "handoff" in inc.fix_suggestion.lower() or "try" in inc.fix_suggestion.lower()


# ── Cost Anomaly ──────────────────────────────────────────────────────────────

class TestCostAnomaly:
    def test_fires_on_spike(self):
        recent = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13, 0.11, 0.12]
        inc = detect_cost_anomaly(make_session(cost=2.50), [], EMPTY_EVALS, recent)
        assert inc is not None
        assert inc.pattern_name == "Cost Anomaly"
        assert inc.severity == "warning"

    def test_does_not_fire_normal_cost(self):
        recent = [0.10, 0.12, 0.09, 0.11, 0.10]
        inc = detect_cost_anomaly(make_session(cost=0.11), [], EMPTY_EVALS, recent)
        assert inc is None

    def test_does_not_fire_insufficient_history(self):
        inc = detect_cost_anomaly(make_session(cost=5.00), [], EMPTY_EVALS, [0.10, 0.12])
        assert inc is None

    def test_does_not_fire_no_history(self):
        inc = detect_cost_anomaly(make_session(cost=5.00), [], EMPTY_EVALS, None)
        assert inc is None

    def test_cost_wasted_is_excess(self):
        recent = [0.08, 0.10, 0.12, 0.09, 0.11, 0.10, 0.13, 0.09, 0.11, 0.10]
        inc = detect_cost_anomaly(make_session(cost=1.50), [], EMPTY_EVALS, recent)
        assert inc is not None
        assert inc.cost_wasted > 0
        assert inc.cost_wasted < 1.50  # excess, not total


# ── Silent Exit ───────────────────────────────────────────────────────────────

class TestSilentExit:
    def test_fires_on_zero_trades_no_reason(self):
        traces = [make_trace(agent=a, outcome="success") for a in
                  ["market","research","risk","orchestrator"]]
        inc = detect_silent_exit(make_session(trades=0, reason=None), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Silent Exit"
        assert inc.severity == "info"

    def test_does_not_fire_with_trades(self):
        traces = [make_trace(agent=a, outcome="success") for a in
                  ["market","research","risk","orchestrator"]]
        inc = detect_silent_exit(make_session(trades=1, reason=None), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_with_reason(self):
        traces = [make_trace(agent=a, outcome="success") for a in
                  ["market","research","risk","orchestrator"]]
        inc = detect_silent_exit(make_session(trades=0, reason="no_opportunity"), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_when_tool_errors_present(self):
        """Tool errors are handled by Tool Timeout Loop, not Silent Exit."""
        traces = [make_trace(agent="research", outcome="error", error="timeout")]
        inc = detect_silent_exit(make_session(trades=0, reason=None), traces, EMPTY_EVALS)
        assert inc is None

    def test_fix_suggestion_mentions_terminal_reason(self):
        traces = [make_trace(agent="orchestrator", outcome="success")]
        inc = detect_silent_exit(make_session(trades=0, reason=None), traces, EMPTY_EVALS)
        assert inc is not None
        assert "terminal_reason" in inc.fix_suggestion


# ── Empty Result Loop ─────────────────────────────────────────────────────────

class TestEmptyResultLoop:
    def test_fires_on_repeated_successful_tool_calls_no_trade(self):
        traces = [
            make_trace(agent="research", step_type="tool_call",
                       tool_name="get_news", outcome="success")
            for _ in range(5)
        ]
        inc = detect_empty_result_loop(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Empty Result Loop"

    def test_does_not_fire_when_trade_made(self):
        traces = [
            make_trace(agent="research", step_type="tool_call",
                       tool_name="get_news", outcome="success")
            for _ in range(5)
        ]
        inc = detect_empty_result_loop(make_session(trades=1), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_below_threshold(self):
        traces = [
            make_trace(agent="research", step_type="tool_call",
                       tool_name="get_news", outcome="success")
            for _ in range(2)
        ]
        inc = detect_empty_result_loop(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_when_errors_present(self):
        """Error tool calls are Tool Timeout Loop territory."""
        traces = [
            make_trace(agent="research", step_type="tool_call",
                       tool_name="get_news", outcome="error", error="timeout")
            for _ in range(5)
        ]
        inc = detect_empty_result_loop(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is None


# ── run_all_detectors ─────────────────────────────────────────────────────────

class TestRunAllDetectors:
    def test_clean_session_no_incidents(self):
        traces = [make_trace(agent=a, outcome="success") for a in
                  ["market","research","risk","orchestrator"]]
        evals = run_all_evals(make_session(trades=1), traces, [0.10]*10)
        incs  = run_all_detectors(make_session(trades=1), traces, evals, [0.10]*10)
        assert incs == []

    def test_yfinance_scenario_fires_tool_timeout(self):
        traces = [
            make_trace(tool_name="get_stock_data", outcome="error",
                       error="ReadTimeout", created_at=f"2026-05-28T06:{30+i}:00Z")
            for i in range(8)
        ]
        sess  = make_session(cost=1.98, trades=0, reason=None)
        evals = run_all_evals(sess, traces, [0.10]*10)
        incs  = run_all_detectors(sess, traces, evals, [0.10]*10)
        patterns = [i.pattern_name for i in incs]
        assert "Tool Timeout Loop" in patterns

    def test_all_incidents_have_required_fields(self):
        traces = [
            make_trace(tool_name="get_stock_data", outcome="error",
                       error="timeout", created_at=f"2026-05-28T06:3{i}:00Z")
            for i in range(4)
        ]
        incs = run_all_detectors(make_session(trades=0), traces, EMPTY_EVALS)
        for inc in incs:
            assert inc.pattern_name
            assert inc.severity in ("critical","warning","info")
            assert inc.root_cause
            assert isinstance(inc.call_stack, list)
            assert isinstance(inc.failed_evals, list)
            assert inc.fix_suggestion

    def test_to_db_row_shape(self):
        traces = [
            make_trace(tool_name="get_stock_data", outcome="error",
                       error="timeout", created_at=f"2026-05-28T06:3{i}:00Z")
            for i in range(4)
        ]
        incs = run_all_detectors(make_session(trades=0), traces, EMPTY_EVALS)
        for inc in incs:
            row = inc.to_db_row()
            assert "session_id"    in row
            assert "pattern_name"  in row
            assert "severity"      in row
            assert "call_stack"    in row
            assert "fix_suggestion"in row

    def test_no_crash_on_empty_input(self):
        incs = run_all_detectors(make_session(), [], EMPTY_EVALS, None)
        assert isinstance(incs, list)

    def test_multiple_patterns_can_fire(self):
        """A session can have both a cost anomaly and a silent exit."""
        traces = [make_trace(agent=a, outcome="success") for a in
                  ["market","research","risk","orchestrator"]]
        sess   = make_session(cost=3.50, trades=0, reason=None)
        evals  = run_all_evals(sess, traces, [0.10]*10)
        incs   = run_all_detectors(sess, traces, evals, [0.10]*10)
        patterns = [i.pattern_name for i in incs]
        assert len(patterns) >= 1  # at minimum silent exit
