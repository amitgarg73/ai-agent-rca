"""
Tests for engine/quality_judge.py.
Covers structural proxy scoring for all 4 quality categories.
Run: python3 -m pytest tests/test_quality_judge.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.quality_judge import (
    judge_session,
    _score_research,
    _score_risk,
    _score_orchestrator,
    _score_session_coherence,
)
from engine.eval_engine import EvalResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def trace(agent="research", step_type="tool_call", tool_name="get_stock_data",
          outcome=None, error=None, latency_ms=500,
          tokens_in=1000, tokens_out=500, session_id="sess-q01"):
    return {
        "id": "t-x", "session_id": session_id,
        "agent": agent, "step_type": step_type, "tool_name": tool_name,
        "outcome": outcome, "error": error, "latency_ms": latency_ms,
        "tokens_input": tokens_in, "tokens_output": tokens_out,
        "created_at": "2026-06-01T14:00:00Z",
    }

def session(trades=1, terminal="eod_complete", cost=0.20):
    return {
        "id": "sess-q01",
        "trades_executed": trades,
        "terminal_reason": terminal,
        "total_cost_usd": cost,
        "total_tokens_input": 10000,
        "total_tokens_output": 3000,
    }

def llm(agent="research", tokens_in=5000, tokens_out=1000):
    return trace(agent=agent, step_type="llm_call", tool_name=None,
                 outcome="success", tokens_in=tokens_in, tokens_out=tokens_out)

def decision(agent="orchestrator"):
    return trace(agent=agent, step_type="decision", tool_name=None, outcome="success")

def tool(agent="research", name="get_stock_data", outcome=None):
    return trace(agent=agent, step_type="tool_call", tool_name=name, outcome=outcome)


# ── judge_session: output shape ───────────────────────────────────────────────

class TestJudgeSessionShape:
    def test_returns_list_of_eval_results(self):
        results = judge_session(session(), [])
        assert isinstance(results, list)
        assert all(isinstance(r, EvalResult) for r in results)

    def test_all_agents_have_quality_suffix(self):
        results = judge_session(session(), [])
        agents = {r.agent for r in results}
        for a in agents:
            assert a.endswith("_quality"), f"agent '{a}' missing _quality suffix"

    def test_four_quality_categories_present(self):
        results = judge_session(session(), [])
        agents = {r.agent for r in results}
        assert "research_quality"    in agents
        assert "risk_quality"        in agents
        assert "orchestrator_quality" in agents
        assert "session_quality"     in agents

    def test_composite_score_per_category(self):
        results = judge_session(session(), [])
        composites = [r for r in results if r.eval_name == "composite_score"]
        assert len(composites) == 4

    def test_scores_in_range(self):
        traces = [
            llm("research"), tool("research", "get_news"), tool("research", "get_stock_data"),
            llm("risk"), decision("risk"),
            llm("orchestrator"), decision("orchestrator"),
        ]
        results = judge_session(session(trades=1, terminal="eod_complete"), traces)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"{r.agent}.{r.eval_name} score {r.score} out of range"

    def test_no_traces_agent_quality_returns_neutrals(self):
        results = judge_session(session(), [])
        # Per-agent quality evals (research/risk/orchestrator) return 0.5 when no traces.
        # session_quality uses pipeline state, so it's allowed to differ.
        agent_quality = [r for r in results if r.agent in
                         ("research_quality", "risk_quality", "orchestrator_quality")]
        for r in agent_quality:
            assert r.score == 0.5, (
                f"expected 0.5 neutral for {r.agent}.{r.eval_name}, got {r.score}"
            )

    def test_never_raises_on_bad_session(self):
        results = judge_session({}, [{"bad": "data"}])
        assert isinstance(results, list)


# ── Research quality ──────────────────────────────────────────────────────────

class TestResearchQuality:
    def test_no_traces_returns_neutrals(self):
        results = _score_research([])
        for r in results:
            assert r.score == 0.5
            assert r.agent == "research_quality"

    def test_three_distinct_tools_raises_grounding(self):
        traces = [
            tool("research", "get_stock_data"),
            tool("research", "get_news"),
            tool("research", "get_atr"),
        ]
        results = _score_research(traces)
        dg = next(r for r in results if r.eval_name == "data_grounding")
        assert dg.score >= 0.9

    def test_single_tool_lowers_grounding(self):
        traces = [tool("research", "get_stock_data")]
        results = _score_research(traces)
        dg = next(r for r in results if r.eval_name == "data_grounding")
        assert dg.score < 0.5

    def test_news_tool_raises_catalyst_specificity(self):
        traces = [tool("research", "search_news"), tool("research", "get_stock_data")]
        results = _score_research(traces)
        cs = next(r for r in results if r.eval_name == "catalyst_specificity")
        assert cs.score >= 0.80

    def test_no_news_tool_lowers_catalyst_specificity(self):
        traces = [tool("research", "get_stock_data"), tool("research", "get_atr")]
        results = _score_research(traces)
        cs = next(r for r in results if r.eval_name == "catalyst_specificity")
        assert cs.score <= 0.55

    def test_decision_trace_raises_thesis_coherence(self):
        traces = [
            tool("research", "get_stock_data"),
            trace("research", "decision", outcome="success"),
        ]
        results = _score_research(traces)
        tc = next(r for r in results if r.eval_name == "thesis_coherence")
        assert tc.score >= 0.80

    def test_risk_ran_raises_actionability(self):
        traces = [
            tool("research", "get_stock_data"),
            tool("risk", "get_atr"),
        ]
        results = _score_research(traces)
        act = next(r for r in results if r.eval_name == "actionability")
        assert act.score >= 0.85

    def test_no_risk_ran_lowers_actionability(self):
        traces = [tool("research", "get_stock_data")]
        results = _score_research(traces)
        act = next(r for r in results if r.eval_name == "actionability")
        assert act.score <= 0.25

    def test_atr_tool_raises_volatility_accounting(self):
        traces = [tool("research", "get_atr"), tool("research", "get_stock_data")]
        results = _score_research(traces)
        va = next(r for r in results if r.eval_name == "volatility_accounting")
        assert va.score >= 0.80

    def test_no_vol_tool_lowers_volatility_accounting(self):
        traces = [tool("research", "get_news")]
        results = _score_research(traces)
        va = next(r for r in results if r.eval_name == "volatility_accounting")
        assert va.score <= 0.45

    def test_composite_computed_correctly(self):
        traces = [
            tool("research", "get_stock_data"),
            tool("research", "get_news"),
            tool("research", "get_atr"),
            trace("research", "decision", outcome="success"),
            tool("risk", "get_atr"),
            llm("research", tokens_in=8000, tokens_out=3000),
        ]
        results = _score_research(traces)
        composite = next(r for r in results if r.eval_name == "composite_score")
        dim_scores = composite.detail.get("dimensions", {})
        assert len(dim_scores) == 5
        assert 0.0 <= composite.score <= 1.0

    def test_error_tool_calls_not_counted_as_distinct(self):
        traces = [
            tool("research", "get_stock_data", outcome="error"),
            tool("research", "get_news", outcome="error"),
        ]
        results = _score_research(traces)
        dg = next(r for r in results if r.eval_name == "data_grounding")
        assert dg.score < 0.3  # error calls give no distinct tool credit


# ── Risk quality ──────────────────────────────────────────────────────────────

class TestRiskQuality:
    def test_no_traces_returns_neutrals(self):
        results = _score_risk([])
        for r in results:
            assert r.score == 0.5
            assert r.agent == "risk_quality"

    def test_research_ran_raises_consistency(self):
        traces = [llm("research"), llm("risk")]
        results = _score_risk(traces)
        rc = next(r for r in results if r.eval_name == "research_consistency")
        assert rc.score >= 0.85

    def test_no_research_lowers_consistency(self):
        traces = [llm("risk")]
        results = _score_risk(traces)
        rc = next(r for r in results if r.eval_name == "research_consistency")
        assert rc.score <= 0.35

    def test_decision_trace_raises_parameter_completeness(self):
        traces = [llm("risk"), decision("risk")]
        results = _score_risk(traces)
        pc = next(r for r in results if r.eval_name == "parameter_completeness")
        assert pc.score >= 0.85

    def test_two_ok_tools_raises_position_sizing(self):
        traces = [tool("risk", "get_atr"), tool("risk", "get_stock_data")]
        results = _score_risk(traces)
        psr = next(r for r in results if r.eval_name == "position_sizing_rationale")
        assert psr.score >= 0.75

    def test_composite_present(self):
        traces = [llm("research"), llm("risk"), decision("risk"), tool("risk", "get_atr")]
        results = _score_risk(traces)
        composite = next(r for r in results if r.eval_name == "composite_score")
        assert 0.0 <= composite.score <= 1.0


# ── Orchestrator quality ──────────────────────────────────────────────────────

class TestOrchestratorQuality:
    def test_no_traces_returns_neutrals(self):
        results = _score_orchestrator(session(), [])
        for r in results:
            assert r.score == 0.5
            assert r.agent == "orchestrator_quality"

    def test_trades_with_full_pipeline_max_consistency(self):
        traces = [llm("research"), llm("risk"), llm("orchestrator"), decision("orchestrator")]
        results = _score_orchestrator(session(trades=1, terminal="eod_complete"), traces)
        dc = next(r for r in results if r.eval_name == "decision_consistency")
        assert dc.score == 1.0

    def test_no_trade_good_exit_moderate_consistency(self):
        traces = [llm("research"), llm("risk"), llm("orchestrator")]
        results = _score_orchestrator(session(trades=0, terminal="no_opportunity"), traces)
        dc = next(r for r in results if r.eval_name == "decision_consistency")
        assert 0.80 <= dc.score <= 0.90

    def test_trades_without_pipeline_lowers_consistency(self):
        traces = [llm("orchestrator")]  # no research or risk
        results = _score_orchestrator(session(trades=2, terminal="eod_complete"), traces)
        dc = next(r for r in results if r.eval_name == "decision_consistency")
        assert dc.score <= 0.45

    def test_good_terminal_reason_max_resolution(self):
        for reason in ["eod_complete", "no_opportunity", "risk_rejected"]:
            traces = [llm("orchestrator")]
            results = _score_orchestrator(session(trades=0, terminal=reason), traces)
            rc = next(r for r in results if r.eval_name == "resolution_completeness")
            assert rc.score == 1.0, f"expected 1.0 for terminal_reason={reason}"

    def test_bad_terminal_reason_lowers_resolution(self):
        traces = [llm("orchestrator")]
        results = _score_orchestrator(session(trades=0, terminal="error"), traces)
        rc = next(r for r in results if r.eval_name == "resolution_completeness")
        assert rc.score <= 0.15

    def test_decision_trace_raises_transparency(self):
        traces = [llm("orchestrator"), decision("orchestrator")]
        results = _score_orchestrator(session(), traces)
        rt = next(r for r in results if r.eval_name == "reasoning_transparency")
        assert rt.score >= 0.80

    def test_full_pipeline_raises_upstream_integration(self):
        traces = [llm("research"), llm("risk"), llm("orchestrator")]
        results = _score_orchestrator(session(), traces)
        ui = next(r for r in results if r.eval_name == "upstream_integration")
        assert ui.score >= 0.80

    def test_no_upstream_lowers_integration(self):
        traces = [llm("orchestrator")]
        results = _score_orchestrator(session(), traces)
        ui = next(r for r in results if r.eval_name == "upstream_integration")
        assert ui.score <= 0.25


# ── Session coherence ─────────────────────────────────────────────────────────

class TestSessionCoherence:
    def test_full_pipeline_max_coherence(self):
        traces = [llm("research"), llm("risk"), llm("orchestrator")]
        results = _score_session_coherence(session(), traces)
        pc = next(r for r in results if r.eval_name == "pipeline_coherence")
        assert pc.score == 1.0

    def test_partial_pipeline_low_coherence(self):
        traces = [llm("research")]  # no risk or orchestrator
        results = _score_session_coherence(session(), traces)
        pc = next(r for r in results if r.eval_name == "pipeline_coherence")
        assert pc.score <= 0.35

    def test_empty_traces_market_only_score(self):
        results = _score_session_coherence(session(), [])
        pc = next(r for r in results if r.eval_name == "pipeline_coherence")
        assert pc.score == 0.80

    def test_error_only_stage_lowers_reasoning_chain(self):
        traces = [
            trace("research", "tool_call", outcome="error"),
            trace("research", "tool_call", outcome="error"),
            llm("risk"),
            llm("orchestrator"),
        ]
        results = _score_session_coherence(session(), traces)
        rc = next(r for r in results if r.eval_name == "reasoning_chain")
        assert rc.score < 1.0

    def test_no_error_only_stages_max_reasoning_chain(self):
        traces = [llm("research"), llm("risk"), llm("orchestrator")]
        results = _score_session_coherence(session(), traces)
        rc = next(r for r in results if r.eval_name == "reasoning_chain")
        assert rc.score == 1.0

    def test_composite_is_mean_of_two_dims(self):
        traces = [llm("research"), llm("risk"), llm("orchestrator")]
        results = _score_session_coherence(session(), traces)
        pc = next(r for r in results if r.eval_name == "pipeline_coherence").score
        rc = next(r for r in results if r.eval_name == "reasoning_chain").score
        comp = next(r for r in results if r.eval_name == "composite_score").score
        assert abs(comp - round((pc + rc) / 2, 3)) < 0.001
