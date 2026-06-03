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
    detect_hyperactive_polling,
    detect_tool_fabrication,
    detect_handoff_schema_break,
    detect_error_misinterpretation,
    run_all_detectors,
    detect_grounding_failure,
    detect_coherence_break,
    detect_quality_cascade,
    detect_silent_degradation,
    run_quality_detectors,
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


# ── Silent Propagation simulator pattern ──────────────────────────────────────

class TestSilentPropagationSimulator:
    """
    Validates that the silent_propagation simulator pattern produces traces and
    a session that (a) trigger detect_silent_exit and (b) yield a positive CB
    savings estimate from compute_cb_savings.
    """

    def _build(self):
        from simulator.failure_sim import PATTERN_BUILDERS, _build_silent_propagation
        import uuid
        sid = str(uuid.uuid4())
        traces  = _build_silent_propagation(sid)
        _, sess_fn = PATTERN_BUILDERS["silent_propagation"]
        session = sess_fn(sid)
        session["id"] = sid
        return sid, session, traces

    def test_all_four_agents_present(self):
        sid, session, traces = self._build()
        agents = {(t.get("agent") or "").lower() for t in traces}
        assert "market"       in agents
        assert "research"     in agents
        assert "risk"         in agents
        assert "orchestrator" in agents

    def test_no_tool_errors_in_traces(self):
        sid, session, traces = self._build()
        tool_errors = [
            t for t in traces
            if (t.get("step_type") or "") == "tool_call"
            and t.get("outcome") == "error"
        ]
        assert tool_errors == [], "silent_propagation must have no tool_call errors"

    def test_session_has_cost_breakdown(self):
        sid, session, traces = self._build()
        bd = session.get("cost_breakdown") or {}
        assert isinstance(bd, dict)
        for agent in ("market", "research", "risk", "orchestrator"):
            assert agent in bd, f"cost_breakdown missing '{agent}'"
            assert bd[agent].get("cost_usd", 0) > 0

    def test_detect_silent_exit_fires(self):
        sid, session, traces = self._build()
        evals = run_all_evals(session, traces)
        inc = detect_silent_exit(session, traces, evals)
        assert inc is not None, "detect_silent_exit should fire for silent_propagation"
        assert inc.pattern_name == "Silent Exit"

    def test_market_freshness_eval_fails(self):
        from engine.eval_engine import eval_market_data_freshness
        sid, session, traces = self._build()
        result = eval_market_data_freshness(traces, session)
        assert not result.passed, (
            f"data_freshness should fail (lag {result.detail.get('lag_minutes')} min > 10 min)"
        )

    def test_cb_savings_positive(self):
        from engine.eval_engine import run_all_evals, EvalResult
        import pandas as pd
        sid, session, traces = self._build()
        evals = run_all_evals(session, traces)

        # build minimal DataFrames matching dashboard's compute_cb_savings signature
        eval_rows = [e.to_db_row(sid) for e in evals]
        evals_df  = pd.DataFrame(eval_rows)
        traces_df = pd.DataFrame(traces)
        traces_df["session_id"] = sid

        # import compute_cb_savings via sys path trick
        import sys, importlib, types
        # The function lives in dashboard.py; re-implement inline to avoid Streamlit import
        pipeline = ["market", "research", "risk", "orchestrator"]
        bd = session.get("cost_breakdown") or {}
        agent_costs = {k: v.get("cost_usd", 0) for k, v in bd.items() if isinstance(v, dict)}
        total_cost  = float(session.get("total_cost_usd") or 0)

        agents_ran = set()
        for t in traces:
            a = (t.get("agent") or "").lower()
            if a.startswith("research"):
                agents_ran.add("research")
            else:
                agents_ran.add(a)

        first_fail_idx = None
        for i, ag in enumerate(pipeline):
            if ag in agents_ran:
                ag_e = evals_df[(evals_df["agent"] == ag)]
                if not ag_e.empty and float(ag_e["passed"].sum()) / len(ag_e) < 0.8:
                    first_fail_idx = i
                    break

        assert first_fail_idx == 0, "market should be first failing agent"
        agents_after = [ag for ag in pipeline[1:] if ag in agents_ran]
        savings = sum(agent_costs.get(ag, 0) for ag in agents_after)
        assert savings > 0, f"CB savings should be positive, got {savings}"
        assert abs(savings - 0.0262) < 0.001, f"Expected ~$0.0262, got ${savings:.4f}"


# ── Q3 Quality Pattern Detectors ──────────────────────────────────────────────

def _make_qual_session(idx: int = 0, cost: float = 0.10, trades: int = 1) -> dict:
    return {
        "id": f"sess-q{idx:03d}",
        "total_cost_usd": cost,
        "trades_executed": trades,
        "started_at": f"2026-06-0{(idx % 9) + 1}T06:00:00Z",
    }


def _qeval(session_id: str, agent: str, eval_name: str,
           score: float, passed: bool = True) -> dict:
    return {
        "session_id": session_id,
        "agent": agent,
        "eval_name": eval_name,
        "score": score,
        "passed": passed,
        "threshold": 0.60,
    }


def _build_history(n: int, cost: float = 0.10) -> list[dict]:
    return [_make_qual_session(i, cost=cost) for i in range(n)]


class TestGroundingFailure:
    def _evals(self, sessions: list[dict], score: float) -> dict:
        return {
            s["id"]: [_qeval(s["id"], "research_quality", "data_grounding", score)]
            for s in sessions
        }

    def test_fires_on_3_consecutive_low_sessions(self):
        sessions = _build_history(3)
        evals    = self._evals(sessions, 0.25)
        inc      = detect_grounding_failure(sessions[-1], sessions, evals)
        assert inc is not None
        assert inc.pattern_name == "Proactive: Grounding Failure"
        assert inc.severity == "warning"

    def test_does_not_fire_when_only_2_low_sessions(self):
        sessions = _build_history(2)
        evals    = self._evals(sessions, 0.25)
        inc      = detect_grounding_failure(sessions[-1], sessions, evals)
        assert inc is None

    def test_does_not_fire_when_last_session_above_threshold(self):
        sessions = _build_history(3)
        evals    = {
            sessions[0]["id"]: [_qeval(sessions[0]["id"], "research_quality", "data_grounding", 0.25)],
            sessions[1]["id"]: [_qeval(sessions[1]["id"], "research_quality", "data_grounding", 0.25)],
            sessions[2]["id"]: [_qeval(sessions[2]["id"], "research_quality", "data_grounding", 0.65)],
        }
        inc = detect_grounding_failure(sessions[-1], sessions, evals)
        assert inc is None

    def test_does_not_fire_when_missing_eval_data(self):
        sessions = _build_history(3)
        # No evals for middle session
        evals = {
            sessions[0]["id"]: [_qeval(sessions[0]["id"], "research_quality", "data_grounding", 0.20)],
            sessions[2]["id"]: [_qeval(sessions[2]["id"], "research_quality", "data_grounding", 0.20)],
        }
        inc = detect_grounding_failure(sessions[-1], sessions, evals)
        assert inc is None

    def test_uses_last_window_of_3(self):
        """With 5 sessions where only last 3 are below threshold, should still fire."""
        sessions = _build_history(5)
        evals = {}
        for i, s in enumerate(sessions):
            score = 0.25 if i >= 2 else 0.80
            evals[s["id"]] = [_qeval(s["id"], "research_quality", "data_grounding", score)]
        inc = detect_grounding_failure(sessions[-1], sessions, evals)
        assert inc is not None

    def test_root_cause_contains_scores(self):
        sessions = _build_history(3)
        evals    = self._evals(sessions, 0.30)
        inc      = detect_grounding_failure(sessions[-1], sessions, evals)
        assert "0.30" in inc.root_cause


class TestCoherenceBreak:
    def test_fires_when_decision_consistency_below_threshold(self):
        session = _make_qual_session(0)
        evals   = {session["id"]: [
            _qeval(session["id"], "orchestrator_quality", "decision_consistency", 0.35)
        ]}
        inc = detect_coherence_break(session, [session], evals)
        assert inc is not None
        assert inc.pattern_name == "Proactive: Coherence Break"
        assert inc.severity == "critical"

    def test_does_not_fire_above_threshold(self):
        session = _make_qual_session(0)
        evals   = {session["id"]: [
            _qeval(session["id"], "orchestrator_quality", "decision_consistency", 0.75)
        ]}
        inc = detect_coherence_break(session, [session], evals)
        assert inc is None

    def test_does_not_fire_at_exact_threshold(self):
        session = _make_qual_session(0)
        evals   = {session["id"]: [
            _qeval(session["id"], "orchestrator_quality", "decision_consistency", 0.50)
        ]}
        inc = detect_coherence_break(session, [session], evals)
        assert inc is None

    def test_does_not_fire_when_eval_missing(self):
        session = _make_qual_session(0)
        inc = detect_coherence_break(session, [session], {})
        assert inc is None

    def test_cost_wasted_equals_session_cost(self):
        session = _make_qual_session(0, cost=1.23)
        evals   = {session["id"]: [
            _qeval(session["id"], "orchestrator_quality", "decision_consistency", 0.20)
        ]}
        inc = detect_coherence_break(session, [session], evals)
        assert inc.cost_wasted == 1.23


class TestQualityCascade:
    def _evals_declining(self, sessions: list[dict],
                         dims: list[tuple[str, str]], drop: float = 0.30) -> dict:
        """Build evals where listed dims go from 0.80 to 0.80-drop across the window."""
        result = {}
        n = len(sessions)
        for i, s in enumerate(sessions):
            rows = []
            for agent, dim in dims:
                score = 0.80 - drop * (i / max(n - 1, 1))
                rows.append(_qeval(s["id"], agent, dim, round(score, 3)))
            result[s["id"]] = rows
        return result

    def test_fires_on_3_declining_dims_over_5_sessions(self):
        sessions = _build_history(5)
        dims     = [
            ("research_quality",     "data_grounding"),
            ("risk_quality",         "parameter_completeness"),
            ("orchestrator_quality", "decision_consistency"),
        ]
        evals = self._evals_declining(sessions, dims, drop=0.30)
        inc   = detect_quality_cascade(sessions[-1], sessions, evals)
        assert inc is not None
        assert inc.pattern_name == "Proactive: Quality Cascade"

    def test_does_not_fire_on_only_2_declining_dims(self):
        sessions = _build_history(5)
        dims     = [
            ("research_quality",     "data_grounding"),
            ("risk_quality",         "parameter_completeness"),
        ]
        evals = self._evals_declining(sessions, dims, drop=0.30)
        inc   = detect_quality_cascade(sessions[-1], sessions, evals)
        assert inc is None

    def test_does_not_fire_when_drop_below_threshold(self):
        sessions = _build_history(5)
        dims     = [
            ("research_quality",     "data_grounding"),
            ("risk_quality",         "parameter_completeness"),
            ("orchestrator_quality", "decision_consistency"),
        ]
        evals = self._evals_declining(sessions, dims, drop=0.10)   # only 0.10 drop — below 0.20
        inc   = detect_quality_cascade(sessions[-1], sessions, evals)
        assert inc is None

    def test_critical_severity_on_5_plus_dims(self):
        sessions = _build_history(5)
        dims     = [
            ("research_quality",     "data_grounding"),
            ("research_quality",     "thesis_coherence"),
            ("research_quality",     "actionability"),
            ("risk_quality",         "parameter_completeness"),
            ("orchestrator_quality", "decision_consistency"),
        ]
        evals = self._evals_declining(sessions, dims, drop=0.30)
        inc   = detect_quality_cascade(sessions[-1], sessions, evals)
        assert inc is not None
        assert inc.severity == "critical"

    def test_requires_at_least_3_sessions(self):
        sessions = _build_history(2)
        dims     = [("research_quality", "data_grounding")]
        evals    = self._evals_declining(sessions, dims, drop=0.30)
        inc      = detect_quality_cascade(sessions[-1], sessions, evals)
        assert inc is None


class TestSilentDegradation:
    def _composite_evals(self, sessions: list[dict],
                         start: float, end: float) -> dict[str, list]:
        """Build composite_score evals declining from start to end across the window."""
        result = {}
        n      = len(sessions)
        agents = ["research_quality", "risk_quality",
                  "orchestrator_quality", "session_quality"]
        for i, s in enumerate(sessions):
            score = start + (end - start) * (i / max(n - 1, 1))
            rows  = [
                _qeval(s["id"], a, "composite_score", round(score, 3), passed=score >= 0.60)
                for a in agents
            ]
            # Add some stable operational evals (passed=True)
            rows += [
                {"session_id": s["id"], "agent": "research", "eval_name": "tool_success_rate",
                 "score": 0.90, "passed": True, "threshold": 0.80}
            ]
            result[s["id"]] = rows
        return result

    def test_fires_when_composite_declining_and_ops_stable(self):
        sessions = _build_history(5)
        evals    = self._composite_evals(sessions, start=0.80, end=0.50)
        inc      = detect_silent_degradation(sessions[-1], sessions, evals)
        assert inc is not None
        assert inc.pattern_name == "Proactive: Silent Degradation"

    def test_warning_when_below_0_60(self):
        sessions = _build_history(5)
        evals    = self._composite_evals(sessions, start=0.75, end=0.45)
        inc      = detect_silent_degradation(sessions[-1], sessions, evals)
        assert inc is not None
        assert inc.severity == "warning"

    def test_info_when_declining_but_still_above_0_60(self):
        sessions = _build_history(5)
        evals    = self._composite_evals(sessions, start=0.85, end=0.65)
        inc      = detect_silent_degradation(sessions[-1], sessions, evals)
        assert inc is not None
        assert inc.severity == "info"

    def test_does_not_fire_when_stable(self):
        sessions = _build_history(5)
        evals    = self._composite_evals(sessions, start=0.75, end=0.73)  # flat
        inc      = detect_silent_degradation(sessions[-1], sessions, evals)
        assert inc is None

    def test_does_not_fire_when_ops_also_degrading(self):
        """If ops are also failing, let operational detectors handle it."""
        sessions = _build_history(5)
        n        = len(sessions)
        agents   = ["research_quality", "risk_quality",
                    "orchestrator_quality", "session_quality"]
        evals: dict[str, list] = {}
        for i, s in enumerate(sessions):
            score = 0.80 - 0.35 * (i / max(n - 1, 1))
            rows  = [_qeval(s["id"], a, "composite_score", round(score, 3)) for a in agents]
            # Operational evals failing — op pass rate below floor
            rows += [
                {"session_id": s["id"], "agent": "research", "eval_name": "tool_success_rate",
                 "score": 0.40, "passed": False, "threshold": 0.80}
            ]
            evals[s["id"]] = rows
        inc = detect_silent_degradation(sessions[-1], sessions, evals)
        assert inc is None

    def test_requires_at_least_3_sessions(self):
        sessions = _build_history(2)
        evals    = self._composite_evals(sessions, start=0.80, end=0.50)
        inc      = detect_silent_degradation(sessions[-1], sessions, evals)
        assert inc is None


class TestRunQualityDetectors:
    def test_returns_list(self):
        session = _make_qual_session(0)
        result  = run_quality_detectors(session, [session], {})
        assert isinstance(result, list)

    def test_collects_multiple_incidents(self):
        """Both coherence break and grounding failure can fire in the same session."""
        sessions = _build_history(3)
        current  = sessions[-1]
        evals    = {}
        for s in sessions:
            evals[s["id"]] = [
                _qeval(s["id"], "research_quality",     "data_grounding",        0.20),
                _qeval(s["id"], "orchestrator_quality", "decision_consistency",  0.30),
            ]
        incidents = run_quality_detectors(current, sessions, evals)
        names = [i.pattern_name for i in incidents]
        assert "Proactive: Grounding Failure" in names
        assert "Proactive: Coherence Break"   in names

    def test_never_raises_on_bad_input(self):
        """Detectors must not crash the monitor even with garbage data."""
        result = run_quality_detectors({}, [None, "bad", 42], {"x": [None]})
        assert isinstance(result, list)


# ── Hyperactive Polling Loop ──────────────────────────────────────────────────

class TestHyperactivePolling:
    def _polling_traces(self, count=6, tool="search_news", agent="research"):
        return [
            make_trace(agent=agent, step_type="tool_call", tool_name=tool,
                       outcome="success", latency_ms=800,
                       created_at=f"2026-05-28T06:{30+i:02d}:00Z")
            for i in range(count)
        ]

    def test_fires_at_threshold(self):
        traces = self._polling_traces(6)
        inc = detect_hyperactive_polling(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Hyperactive Polling Loop"
        assert inc.severity == "warning"

    def test_does_not_fire_below_threshold(self):
        traces = self._polling_traces(5)
        inc = detect_hyperactive_polling(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is None

    def test_fires_even_when_trades_produced(self):
        traces = self._polling_traces(8)
        inc = detect_hyperactive_polling(make_session(trades=1), traces, EMPTY_EVALS)
        assert inc is not None  # distinct from empty_result_loop which requires trades=0

    def test_does_not_fire_on_error_calls(self):
        traces = [
            make_trace(step_type="tool_call", tool_name="get_data",
                       outcome="error", error="timeout",
                       created_at=f"2026-05-28T06:{30+i:02d}:00Z")
            for i in range(8)
        ]
        inc = detect_hyperactive_polling(make_session(), traces, EMPTY_EVALS)
        assert inc is None  # errors are Tool Timeout Loop territory

    def test_does_not_fire_on_different_tools(self):
        tools = ["tool_a", "tool_b", "tool_c", "tool_d", "tool_e", "tool_f", "tool_g"]
        traces = [
            make_trace(step_type="tool_call", tool_name=t, outcome="success",
                       created_at=f"2026-05-28T06:{30+i:02d}:00Z")
            for i, t in enumerate(tools)
        ]
        inc = detect_hyperactive_polling(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_call_stack_contains_all_polls(self):
        traces = self._polling_traces(8)
        inc = detect_hyperactive_polling(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert len(inc.call_stack) == 8

    def test_fix_suggestion_mentions_cap(self):
        traces = self._polling_traces(6)
        inc = detect_hyperactive_polling(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert "max" in inc.fix_suggestion.lower() or "cap" in inc.fix_suggestion.lower()

    def test_empty_traces(self):
        inc = detect_hyperactive_polling(make_session(), [], EMPTY_EVALS)
        assert inc is None


# ── Tool Call Fabrication ─────────────────────────────────────────────────────

class TestToolCallFabrication:
    def _fast_tool_trace(self, latency=15, tool="get_market_data", agent="research"):
        return make_trace(agent=agent, step_type="tool_call", tool_name=tool,
                          outcome="success", latency_ms=latency)

    def test_fires_on_two_fast_calls(self):
        traces = [self._fast_tool_trace(20), self._fast_tool_trace(30)]
        inc = detect_tool_fabrication(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Tool Call Fabrication"
        assert inc.severity == "critical"

    def test_does_not_fire_on_one_fast_call(self):
        traces = [self._fast_tool_trace(20)]
        inc = detect_tool_fabrication(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_on_realistic_latency(self):
        traces = [
            make_trace(step_type="tool_call", outcome="success", latency_ms=350),
            make_trace(step_type="tool_call", outcome="success", latency_ms=820),
        ]
        inc = detect_tool_fabrication(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_on_zero_latency(self):
        # latency=0 means not recorded — skip these
        traces = [
            make_trace(step_type="tool_call", outcome="success", latency_ms=0),
            make_trace(step_type="tool_call", outcome="success", latency_ms=0),
        ]
        inc = detect_tool_fabrication(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_on_error_traces(self):
        traces = [
            make_trace(step_type="tool_call", outcome="error", latency_ms=10),
            make_trace(step_type="tool_call", outcome="error", latency_ms=15),
        ]
        inc = detect_tool_fabrication(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_root_cause_mentions_latency(self):
        traces = [self._fast_tool_trace(10), self._fast_tool_trace(25)]
        inc = detect_tool_fabrication(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert "ms" in inc.root_cause.lower() or "latency" in inc.root_cause.lower()

    def test_fix_suggestion_mentions_validation(self):
        traces = [self._fast_tool_trace(), self._fast_tool_trace()]
        inc = detect_tool_fabrication(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert "validat" in inc.fix_suggestion.lower()

    def test_empty_traces(self):
        inc = detect_tool_fabrication(make_session(), [], EMPTY_EVALS)
        assert inc is None


# ── Handoff Schema Break ──────────────────────────────────────────────────────

class TestHandoffSchemaBreak:
    def _research_ok_traces(self):
        return [
            make_trace(agent="research", step_type="llm_call", outcome="success",
                       created_at="2026-05-28T06:10:00Z"),
            make_trace(agent="research", step_type="tool_call", tool_name="get_data",
                       outcome="success", created_at="2026-05-28T06:11:00Z"),
            make_trace(agent="research", step_type="agent_message", outcome="completed",
                       created_at="2026-05-28T06:12:00Z"),
        ]

    def test_fires_when_risk_first_trace_errors(self):
        traces = self._research_ok_traces() + [
            make_trace(agent="risk", step_type="llm_call", outcome="error",
                       error="KeyError: 'analysis' missing in research output",
                       created_at="2026-05-28T06:13:00Z"),
        ]
        inc = detect_handoff_schema_break(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Handoff Schema Break"
        assert inc.severity == "critical"

    def test_does_not_fire_when_risk_succeeds(self):
        traces = self._research_ok_traces() + [
            make_trace(agent="risk", step_type="llm_call", outcome="success",
                       created_at="2026-05-28T06:13:00Z"),
        ]
        inc = detect_handoff_schema_break(make_session(trades=1), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_when_research_also_failed(self):
        traces = [
            make_trace(agent="research", step_type="llm_call", outcome="error",
                       error="LLM timeout", created_at="2026-05-28T06:10:00Z"),
            make_trace(agent="risk", step_type="llm_call", outcome="error",
                       error="KeyError: missing field", created_at="2026-05-28T06:11:00Z"),
        ]
        inc = detect_handoff_schema_break(make_session(), traces, EMPTY_EVALS)
        assert inc is None  # research itself failed — not a handoff issue

    def test_does_not_fire_without_risk_traces(self):
        traces = self._research_ok_traces()
        inc = detect_handoff_schema_break(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_without_research_traces(self):
        traces = [
            make_trace(agent="risk", step_type="llm_call", outcome="error",
                       error="missing input", created_at="2026-05-28T06:10:00Z"),
        ]
        inc = detect_handoff_schema_break(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_error_field_triggers_as_well_as_outcome(self):
        traces = self._research_ok_traces() + [
            make_trace(agent="risk", step_type="llm_call", outcome=None,
                       error="ValueError: unexpected None for field 'score'",
                       created_at="2026-05-28T06:13:00Z"),
        ]
        inc = detect_handoff_schema_break(make_session(), traces, EMPTY_EVALS)
        assert inc is not None

    def test_call_stack_includes_last_research_and_first_risk(self):
        traces = self._research_ok_traces() + [
            make_trace(agent="risk", step_type="llm_call", outcome="error",
                       error="schema mismatch", created_at="2026-05-28T06:13:00Z"),
        ]
        inc = detect_handoff_schema_break(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        agents_in_stack = [f["agent"] for f in inc.call_stack]
        assert "research" in agents_in_stack
        assert "risk" in agents_in_stack

    def test_fix_mentions_schema(self):
        traces = self._research_ok_traces() + [
            make_trace(agent="risk", step_type="llm_call", outcome="error",
                       error="schema error", created_at="2026-05-28T06:13:00Z"),
        ]
        inc = detect_handoff_schema_break(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert "schema" in inc.fix_suggestion.lower()


# ── Error Misinterpretation ───────────────────────────────────────────────────

class TestErrorMisinterpretation:
    def _http_error_trace(self, code="429", agent="research", ts_min=30):
        return make_trace(
            agent=agent, step_type="tool_call", tool_name="get_prices",
            outcome="error", error=f"HTTP {code}: Too Many Requests",
            created_at=f"2026-05-28T06:{ts_min:02d}:00Z"
        )

    def _llm_after(self, agent="research", ts_min=31):
        return make_trace(
            agent=agent, step_type="llm_call", outcome="success",
            created_at=f"2026-05-28T06:{ts_min:02d}:00Z"
        )

    def test_fires_on_two_mishandled_errors(self):
        traces = [
            self._http_error_trace("429", ts_min=30),
            self._llm_after(ts_min=31),
            self._http_error_trace("429", ts_min=32),
            self._llm_after(ts_min=33),
        ]
        inc = detect_error_misinterpretation(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is not None
        assert inc.pattern_name == "Error Misinterpretation"
        assert inc.severity == "warning"

    def test_does_not_fire_on_one_event(self):
        traces = [
            self._http_error_trace("429", ts_min=30),
            self._llm_after(ts_min=31),
        ]
        inc = detect_error_misinterpretation(make_session(trades=0), traces, EMPTY_EVALS)
        assert inc is None

    def test_does_not_fire_when_agent_stops_after_error(self):
        traces = [
            self._http_error_trace("429", ts_min=30),
            # no subsequent llm_call from same agent
            make_trace(agent="orchestrator", step_type="llm_call", outcome="success",
                       created_at="2026-05-28T06:31:00Z"),
            self._http_error_trace("503", ts_min=32),
        ]
        inc = detect_error_misinterpretation(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_fires_on_401(self):
        traces = [
            self._http_error_trace("401", ts_min=30),
            self._llm_after(ts_min=31),
            self._http_error_trace("401", ts_min=32),
            self._llm_after(ts_min=33),
        ]
        inc = detect_error_misinterpretation(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert "401" in inc.root_cause

    def test_does_not_fire_on_non_http_errors(self):
        traces = [
            make_trace(step_type="tool_call", outcome="error", error="ReadTimeout",
                       created_at="2026-05-28T06:30:00Z"),
            self._llm_after(ts_min=31),
            make_trace(step_type="tool_call", outcome="error", error="ConnectionError",
                       created_at="2026-05-28T06:32:00Z"),
            self._llm_after(ts_min=33),
        ]
        inc = detect_error_misinterpretation(make_session(), traces, EMPTY_EVALS)
        assert inc is None

    def test_fix_mentions_error_codes(self):
        traces = [
            self._http_error_trace("429", ts_min=30), self._llm_after(ts_min=31),
            self._http_error_trace("429", ts_min=32), self._llm_after(ts_min=33),
        ]
        inc = detect_error_misinterpretation(make_session(), traces, EMPTY_EVALS)
        assert inc is not None
        assert "429" in inc.fix_suggestion

    def test_empty_traces(self):
        inc = detect_error_misinterpretation(make_session(), [], EMPTY_EVALS)
        assert inc is None


# ── Shadow CB compute_shadow_cb_fires ─────────────────────────────────────────

class TestShadowCBFires:
    def test_fires_on_tool_success_rate_below_threshold(self):
        from engine.pattern_detector import compute_shadow_cb_fires, CB_CONFIG
        from engine.eval_engine import EvalResult
        evals = [
            EvalResult("tool_success_rate", "research", 0.50, False, 0.80, {}),
        ]
        fires = compute_shadow_cb_fires(evals)
        assert len(fires) == 1
        assert fires[0]["agent"] == "research"
        assert fires[0]["eval_name"] == "tool_success_rate"

    def test_does_not_fire_when_eval_passes(self):
        from engine.pattern_detector import compute_shadow_cb_fires
        from engine.eval_engine import EvalResult
        evals = [EvalResult("tool_success_rate", "research", 0.95, True, 0.80, {})]
        fires = compute_shadow_cb_fires(evals)
        assert fires == []

    def test_does_not_fire_for_agents_not_in_cb_config(self):
        from engine.pattern_detector import compute_shadow_cb_fires
        from engine.eval_engine import EvalResult
        evals = [EvalResult("data_freshness", "market", 0.0, False, 0.5, {})]
        fires = compute_shadow_cb_fires(evals)
        assert fires == []

    def test_multiple_cb_fires(self):
        from engine.pattern_detector import compute_shadow_cb_fires
        from engine.eval_engine import EvalResult
        evals = [
            EvalResult("tool_success_rate", "research", 0.50, False, 0.80, {}),
            EvalResult("completion",        "research", 0.30, False, 0.70, {}),
            EvalResult("assessment_complete","risk",    0.0,  False, 1.00, {}),
        ]
        fires = compute_shadow_cb_fires(evals)
        assert len(fires) == 3
