"""
Comprehensive tests for all 17 evals in eval_engine.py.
Each eval has: pass case, fail case, edge case (empty/missing data).
Run: python3 -m pytest tests/test_evals.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.eval_engine import (
    eval_market_data_completeness,
    eval_market_data_freshness,
    eval_research_completion,
    eval_research_token_efficiency,
    eval_research_tool_success_rate,
    eval_research_tool_diversity,
    eval_risk_assessment_complete,
    eval_risk_within_parameters,
    eval_orchestrator_decision_made,
    eval_orchestrator_exit_quality,
    eval_session_pipeline_completion,
    eval_session_cost_anomaly,
    eval_session_outcome_linkage,
    eval_session_tokens_per_decision,
    eval_cost_per_trade,
    eval_research_conversion,
    eval_proposal_acceptance,
    run_all_evals,
    REQUIRED_AGENTS,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_trace(agent="research", step_type="tool_call", tool_name="get_stock_data",
               outcome="success", error=None, latency_ms=500,
               tokens_in=100, tokens_out=50, created_at="2026-05-28T06:34:00Z",
               session_id="sess-001"):
    return {
        "id": "trace-001", "session_id": session_id,
        "agent": agent, "step_type": step_type, "tool_name": tool_name,
        "outcome": outcome, "error": error, "latency_ms": latency_ms,
        "tokens_input": tokens_in, "tokens_output": tokens_out,
        "created_at": created_at,
    }

def make_session(cost=0.10, trades=1, reason=None, started_at="2026-05-28T06:00:00Z",
                 tokens_in=5000, tokens_out=2000, proposed=1):
    return {
        "id": "sess-001",
        "total_cost_usd":      cost,
        "trades_executed":     trades,
        "trades_proposed":     proposed,
        "terminal_reason":     reason,
        "started_at":          started_at,
        "completed_at":        "2026-05-28T06:30:00Z",
        "total_tokens_input":  tokens_in,
        "total_tokens_output": tokens_out,
    }


# ── Market: data_completeness ─────────────────────────────────────────────────

class TestMarketDataCompleteness:
    def test_pass_with_success_trace(self):
        traces = [make_trace(agent="market", outcome="success")]
        r = eval_market_data_completeness(traces, make_session())
        assert r.passed
        assert r.score == 1.0

    def test_pass_with_market_shadow(self):
        traces = [make_trace(agent="market_shadow", outcome="success")]
        r = eval_market_data_completeness(traces, make_session())
        assert r.passed

    def test_fail_all_errors(self):
        traces = [make_trace(agent="market", outcome="error"),
                  make_trace(agent="market", outcome="error")]
        r = eval_market_data_completeness(traces, make_session())
        assert not r.passed
        assert r.score == 0.0

    def test_fail_no_market_traces(self):
        traces = [make_trace(agent="research", outcome="success")]
        r = eval_market_data_completeness(traces, make_session())
        assert not r.passed
        assert r.score == 0.0
        assert "no market" in r.detail["reason"]

    def test_edge_empty_traces(self):
        r = eval_market_data_completeness([], make_session())
        assert not r.passed

    def test_partial_score(self):
        traces = [make_trace(agent="market", outcome="success"),
                  make_trace(agent="market", outcome="error")]
        r = eval_market_data_completeness(traces, make_session())
        assert 0 < r.score < 1


# ── Market: data_freshness ────────────────────────────────────────────────────

class TestMarketDataFreshness:
    def test_pass_fresh_data(self):
        traces = [make_trace(agent="market", created_at="2026-05-28T06:05:00Z")]
        sess   = make_session(started_at="2026-05-28T06:00:00Z")
        r = eval_market_data_freshness(traces, sess)
        assert r.passed
        assert r.detail["lag_minutes"] < 10

    def test_fail_stale_data(self):
        traces = [make_trace(agent="market", created_at="2026-05-28T08:00:00Z")]
        sess   = make_session(started_at="2026-05-28T06:00:00Z")
        r = eval_market_data_freshness(traces, sess)
        assert not r.passed
        assert r.detail["lag_minutes"] > 10

    def test_edge_no_traces(self):
        r = eval_market_data_freshness([], make_session())
        assert not r.passed

    def test_edge_missing_timestamps(self):
        traces = [make_trace(agent="market", created_at=None)]
        sess   = make_session(started_at=None)
        r = eval_market_data_freshness(traces, sess)
        assert r.passed  # assumes fresh when timestamps unavailable


# ── Research: completion ──────────────────────────────────────────────────────

class TestResearchCompletion:
    def test_pass_decision_trace(self):
        traces = [make_trace(agent="research", step_type="decision", outcome="success")]
        r = eval_research_completion(traces, make_session())
        assert r.passed
        assert r.score == 1.0

    def test_pass_llm_call_success(self):
        traces = [make_trace(agent="research", step_type="llm_call", outcome="success")]
        r = eval_research_completion(traces, make_session())
        assert r.passed

    def test_fail_no_research_traces(self):
        traces = [make_trace(agent="market", outcome="success")]
        r = eval_research_completion(traces, make_session())
        assert not r.passed
        assert r.score == 0.0

    def test_fail_only_errors(self):
        traces = [make_trace(agent="research", step_type="tool_call", outcome="error")]
        r = eval_research_completion(traces, make_session())
        assert not r.passed

    def test_no_llm_call_scores_zero(self):
        # Only tool calls, no LLM ran — score 0.0 (agent never started reasoning)
        traces = [make_trace(agent="research", step_type="tool_call", outcome="success")]
        r = eval_research_completion(traces, make_session())
        assert not r.passed
        assert r.score == 0.0

    def test_partial_credit_llm_ran_all_tools_failed(self):
        # LLM ran but every tool call failed — score 0.3 (started but couldn't fetch data)
        traces = [
            make_trace(agent="research", step_type="llm_call", outcome="success"),
            make_trace(agent="research", step_type="tool_call", outcome="error"),
            make_trace(agent="research", step_type="tool_call", outcome="error"),
        ]
        r = eval_research_completion(traces, make_session())
        assert not r.passed
        assert r.score == 0.3

    def test_edge_empty_traces(self):
        r = eval_research_completion([], make_session())
        assert not r.passed


# ── Research: token_efficiency ────────────────────────────────────────────────

class TestResearchTokenEfficiency:
    def test_pass_low_tokens(self):
        traces = [make_trace(agent="research", tokens_in=5000, tokens_out=2000)]
        r = eval_research_token_efficiency(traces, make_session())
        assert r.passed

    def test_fail_high_tokens(self):
        traces = [make_trace(agent="research", tokens_in=25000, tokens_out=10000)]
        r = eval_research_token_efficiency(traces, make_session())
        assert not r.passed

    def test_edge_no_research_traces(self):
        r = eval_research_token_efficiency([], make_session())
        assert r.passed  # no research = trivially passes
        assert r.detail["tokens_used"] == 0

    def test_score_decreases_with_tokens(self):
        t_low  = [make_trace(agent="research", tokens_in=1000,  tokens_out=500)]
        t_high = [make_trace(agent="research", tokens_in=20000, tokens_out=8000)]
        r_low  = eval_research_token_efficiency(t_low,  make_session())
        r_high = eval_research_token_efficiency(t_high, make_session())
        assert r_low.score > r_high.score


# ── Research: tool_success_rate ───────────────────────────────────────────────

class TestResearchToolSuccessRate:
    def test_pass_all_success(self):
        traces = [make_trace(agent="research", step_type="tool_call", outcome="success"),
                  make_trace(agent="research", step_type="tool_call", outcome="success")]
        r = eval_research_tool_success_rate(traces, make_session())
        assert r.passed
        assert r.score == 1.0

    def test_fail_all_errors(self):
        traces = [make_trace(agent="research", step_type="tool_call", outcome="error")] * 8
        r = eval_research_tool_success_rate(traces, make_session())
        assert not r.passed
        assert r.score == 0.0
        assert r.detail["rate"] == 0.0

    def test_pass_no_tool_calls(self):
        traces = [make_trace(agent="research", step_type="llm_call", outcome="success")]
        r = eval_research_tool_success_rate(traces, make_session())
        assert r.passed
        assert "no tool calls" in r.detail["reason"]

    def test_below_threshold(self):
        traces = [
            make_trace(agent="research", step_type="tool_call", outcome="success"),
            make_trace(agent="research", step_type="tool_call", outcome="error"),
            make_trace(agent="research", step_type="tool_call", outcome="error"),
            make_trace(agent="research", step_type="tool_call", outcome="error"),
        ]
        r = eval_research_tool_success_rate(traces, make_session())
        assert not r.passed
        assert r.score == pytest.approx(0.25)


# ── Research: tool_diversity ──────────────────────────────────────────────────

class TestResearchToolDiversity:
    def test_pass_three_distinct_tools(self):
        traces = [
            make_trace(agent="research", step_type="tool_call", tool_name="get_stock_data"),
            make_trace(agent="research", step_type="tool_call", tool_name="get_news"),
            make_trace(agent="research", step_type="tool_call", tool_name="get_financials"),
        ]
        r = eval_research_tool_diversity(traces, make_session())
        assert r.passed
        assert r.score == 1.0

    def test_pass_two_distinct_tools(self):
        traces = [
            make_trace(agent="research", step_type="tool_call", tool_name="get_stock_data"),
            make_trace(agent="research", step_type="tool_call", tool_name="get_news"),
            make_trace(agent="research", step_type="tool_call", tool_name="get_stock_data"),
        ]
        r = eval_research_tool_diversity(traces, make_session())
        assert r.passed
        assert r.score == 0.7

    def test_fail_single_tool_loop(self):
        traces = [
            make_trace(agent="research", step_type="tool_call", tool_name="get_stock_data")
        ] * 8
        r = eval_research_tool_diversity(traces, make_session())
        assert not r.passed
        assert r.score == 0.3
        assert r.detail["distinct_tools"] == 1

    def test_fail_no_tool_calls(self):
        traces = [make_trace(agent="research", step_type="llm_call", outcome="success")]
        r = eval_research_tool_diversity(traces, make_session())
        assert not r.passed
        assert r.score == 0.0

    def test_edge_empty_traces(self):
        r = eval_research_tool_diversity([], make_session())
        assert not r.passed
        assert r.score == 0.0

    def test_detail_lists_tool_names(self):
        traces = [
            make_trace(agent="research", step_type="tool_call", tool_name="get_stock_data"),
            make_trace(agent="research", step_type="tool_call", tool_name="get_news"),
        ]
        r = eval_research_tool_diversity(traces, make_session())
        assert "tool_names" in r.detail
        assert set(r.detail["tool_names"]) == {"get_stock_data", "get_news"}


# ── Risk agent evals ──────────────────────────────────────────────────────────

class TestRiskEvals:
    def test_assessment_complete_pass(self):
        traces = [make_trace(agent="risk", outcome="success")]
        r = eval_risk_assessment_complete(traces, make_session())
        assert r.passed

    def test_assessment_complete_fail_no_traces(self):
        r = eval_risk_assessment_complete([], make_session())
        assert not r.passed

    def test_assessment_complete_fail_all_errors(self):
        traces = [make_trace(agent="risk", outcome="error")]
        r = eval_risk_assessment_complete(traces, make_session())
        assert not r.passed

    def test_within_parameters_pass(self):
        traces = [make_trace(agent="risk", outcome="success", error=None)]
        r = eval_risk_within_parameters(traces, make_session())
        assert r.passed
        assert r.score == 1.0

    def test_within_parameters_fail(self):
        traces = [make_trace(agent="risk", error="PositionSizeLimitExceeded: 15% > max 10%")]
        r = eval_risk_within_parameters(traces, make_session())
        assert not r.passed
        assert r.detail["error_traces"] == 1


# ── Orchestrator evals ────────────────────────────────────────────────────────

class TestOrchestratorDecisionMade:
    def test_pass_trades(self):
        r = eval_orchestrator_decision_made([], make_session(trades=2))
        assert r.passed
        assert r.score == 1.0

    def test_pass_good_named_exit(self):
        r = eval_orchestrator_decision_made([], make_session(trades=0, reason="no_opportunity"))
        assert r.passed
        assert r.score == 1.0

    def test_fail_structural_block(self):
        r = eval_orchestrator_decision_made([], make_session(trades=0, reason="structural_block"))
        assert not r.passed
        assert r.score == 0.5

    def test_fail_bad_exit(self):
        r = eval_orchestrator_decision_made([], make_session(trades=0, reason="in_progress"))
        assert not r.passed
        assert r.score == 0.2

    def test_fail_orch_ran_no_output(self):
        traces = [make_trace(agent="orchestrator", step_type="llm_call", outcome="success")]
        r = eval_orchestrator_decision_made(traces, make_session(trades=0, reason=None))
        assert not r.passed
        assert r.score == 0.2

    def test_fail_silent(self):
        r = eval_orchestrator_decision_made([], make_session(trades=0, reason=None))
        assert not r.passed
        assert r.score == 0.0


class TestOrchestratorExitQuality:
    def test_pass_good_named_exit(self):
        r = eval_orchestrator_exit_quality([], make_session(trades=0, reason="converged"))
        assert r.passed
        assert r.score == 1.0

    def test_pass_all_good_exit_reasons(self):
        good = ["eod_complete", "converged", "no_opportunity", "risk_rejected",
                "market_closed", "position_limit", "daily_limit"]
        for reason in good:
            r = eval_orchestrator_exit_quality([], make_session(trades=0, reason=reason))
            assert r.passed, f"Expected pass for reason={reason}"

    def test_partial_structural_block(self):
        r = eval_orchestrator_exit_quality([], make_session(trades=0, reason="structural_block"))
        assert not r.passed
        assert r.score == 0.5

    def test_fail_bad_exit_in_progress(self):
        r = eval_orchestrator_exit_quality([], make_session(trades=0, reason="in_progress"))
        assert not r.passed
        assert r.score == 0.2

    def test_fail_bad_exit_error(self):
        r = eval_orchestrator_exit_quality([], make_session(trades=0, reason="error"))
        assert not r.passed
        assert r.score == 0.2

    def test_pass_trades_no_reason(self):
        # Trades produced but no terminal_reason — partial pass (0.8)
        r = eval_orchestrator_exit_quality([], make_session(trades=3, reason=None))
        assert r.passed
        assert r.score == 0.8

    def test_fail_silent_exit(self):
        r = eval_orchestrator_exit_quality([], make_session(trades=0, reason=None))
        assert not r.passed
        assert r.score == 0.0


# ── Holistic session evals ────────────────────────────────────────────────────

class TestSessionEvals:
    def test_pipeline_completion_all_agents(self):
        traces = [make_trace(agent=a) for a in REQUIRED_AGENTS]
        r = eval_session_pipeline_completion(traces, make_session())
        assert r.passed
        assert r.score == 1.0

    def test_pipeline_completion_missing_orchestrator(self):
        traces = [make_trace(agent=a) for a in ["market","research","risk"]]
        r = eval_session_pipeline_completion(traces, make_session())
        assert not r.passed
        assert "orchestrator" in r.detail["missing"]

    def test_pipeline_completion_market_shadow_counts(self):
        traces = [make_trace(agent=a) for a in ["market_shadow","research","risk","orchestrator"]]
        r = eval_session_pipeline_completion(traces, make_session())
        assert r.passed

    def test_pipeline_completion_empty(self):
        r = eval_session_pipeline_completion([], make_session())
        assert not r.passed

    def test_cost_anomaly_pass_normal(self):
        recent = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13]
        r = eval_session_cost_anomaly([], make_session(cost=0.11), recent)
        assert r.passed

    def test_cost_anomaly_fail_spike(self):
        recent = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13]
        r = eval_session_cost_anomaly([], make_session(cost=2.50), recent)
        assert not r.passed
        assert r.detail["z_score"] > 2

    def test_cost_anomaly_insufficient_history(self):
        r = eval_session_cost_anomaly([], make_session(), recent_costs=[0.10, 0.12])
        assert r.passed
        assert "insufficient" in r.detail["reason"]

    def test_outcome_linkage_with_trade(self):
        r = eval_session_outcome_linkage([], make_session(trades=1))
        assert r.passed
        assert r.score == 1.0

    def test_outcome_linkage_good_named_exit(self):
        r = eval_session_outcome_linkage([], make_session(trades=0, reason="market_closed"))
        assert r.passed
        assert r.score == 1.0

    def test_outcome_linkage_structural_block(self):
        r = eval_session_outcome_linkage([], make_session(trades=0, reason="structural_block"))
        assert not r.passed
        assert r.score == 0.5

    def test_outcome_linkage_bad_exit(self):
        r = eval_session_outcome_linkage([], make_session(trades=0, reason="in_progress"))
        assert not r.passed
        assert r.score == 0.2

    def test_outcome_linkage_fail_silent(self):
        r = eval_session_outcome_linkage([], make_session(trades=0, reason=None))
        assert not r.passed
        assert r.score == 0.0

    def test_tokens_per_decision_pass_low(self):
        r = eval_session_tokens_per_decision([], make_session(trades=1, tokens_in=5000, tokens_out=2000))
        assert r.passed

    def test_tokens_per_decision_fail_high(self):
        r = eval_session_tokens_per_decision([], make_session(trades=0, tokens_in=40000, tokens_out=5000))
        assert not r.passed

    def test_tokens_per_decision_zero_trades_uses_raw_tokens(self):
        # 0 trades: raw token spend used, not divided by 1
        r_zero  = eval_session_tokens_per_decision([], make_session(trades=0, tokens_in=1000, tokens_out=500))
        r_one   = eval_session_tokens_per_decision([], make_session(trades=1, tokens_in=1000, tokens_out=500))
        # Both should score the same (1500 tokens / 1 trade = 1500 tokens raw)
        assert r_zero.score == r_one.score
        assert r_zero.detail["tokens_per_decision"] == r_one.detail["tokens_per_decision"]


# ── run_all_evals ─────────────────────────────────────────────────────────────

class TestRunAllEvals:
    def test_returns_17_results(self):
        traces = [make_trace(agent=a) for a in REQUIRED_AGENTS]
        results = run_all_evals(make_session(), traces, [0.10]*10)
        assert len(results) == 17

    def test_all_results_have_required_fields(self):
        results = run_all_evals(make_session(), [], [0.10]*10)
        for r in results:
            assert hasattr(r, "eval_name")
            assert hasattr(r, "agent")
            assert 0.0 <= r.score <= 1.0
            assert isinstance(r.passed, bool)

    def test_to_db_row_shape(self):
        results = run_all_evals(make_session(), [], [0.10]*10)
        for r in results:
            row = r.to_db_row("sess-001")
            assert "session_id" in row
            assert "eval_name" in row
            assert "score" in row
            assert "passed" in row
            assert "detail" in row

    def test_yfinance_scenario_fails_evals(self):
        """The known yfinance hang should fail research evals."""
        traces = []
        for i in range(8):
            traces.append(make_trace(
                agent="research", step_type="tool_call",
                tool_name="get_stock_data", outcome="error",
                error="ReadTimeout: timed out", tokens_in=0, tokens_out=0,
            ))
        sess    = make_session(cost=1.98, trades=0, reason=None, tokens_in=45000, tokens_out=2000)
        results = run_all_evals(sess, traces, [0.10]*10)
        by_name = {r.eval_name: r for r in results}

        assert not by_name["tool_success_rate"].passed
        assert by_name["tool_success_rate"].score == 0.0
        assert not by_name["tool_diversity"].passed   # single tool looped
        assert not by_name["outcome_linkage"].passed
        assert not by_name["pipeline_completion"].passed


# ── Business outcome evals ────────────────────────────────────────────────────

class TestCostPerTrade:
    def test_pass_low_cost(self):
        r = eval_cost_per_trade([], make_session(cost=0.30, trades=1))
        assert r.passed
        assert r.detail["cost_per_trade"] == 0.30

    def test_fail_high_cost(self):
        r = eval_cost_per_trade([], make_session(cost=1.98, trades=0))
        assert not r.passed
        assert r.detail["cost_per_trade"] == 1.98

    def test_divides_by_trades(self):
        # $0.60 total, 2 trades = $0.30/trade → should pass
        r = eval_cost_per_trade([], make_session(cost=0.60, trades=2))
        assert r.passed
        assert r.detail["cost_per_trade"] == 0.30

    def test_score_zero_when_very_expensive(self):
        r = eval_cost_per_trade([], make_session(cost=5.0, trades=0))
        assert r.score == 0.0
        assert not r.passed

    def test_eval_name_and_agent(self):
        r = eval_cost_per_trade([], make_session())
        assert r.eval_name == "cost_per_trade"
        assert r.agent == "business"


class TestResearchConversion:
    def _make_research_llm_trace(self):
        return make_trace(agent="research", step_type="llm_call", outcome="success")

    def test_pass_with_trade(self):
        traces = [self._make_research_llm_trace()]
        r = eval_research_conversion(traces, make_session(trades=1))
        assert r.passed
        assert r.detail["research_runs"] == 1

    def test_fail_no_trade_from_research(self):
        traces = [self._make_research_llm_trace(), self._make_research_llm_trace(),
                  self._make_research_llm_trace(), self._make_research_llm_trace()]
        r = eval_research_conversion(traces, make_session(trades=0))
        assert not r.passed
        assert r.detail["conversion_rate"] == 0.0

    def test_fail_no_research_traces(self):
        r = eval_research_conversion([], make_session(trades=1))
        assert not r.passed
        assert "no successful research" in r.detail["reason"]

    def test_score_clipped_at_1(self):
        traces = [self._make_research_llm_trace()]
        r = eval_research_conversion(traces, make_session(trades=5))
        assert r.score == 1.0

    def test_eval_name_and_agent(self):
        r = eval_research_conversion([], make_session())
        assert r.eval_name == "research_conversion"
        assert r.agent == "business"


class TestProposalAcceptance:
    def test_pass_all_proposed_executed(self):
        r = eval_proposal_acceptance([], make_session(trades=2, proposed=2))
        assert r.passed
        assert r.detail["acceptance_rate"] == 1.0

    def test_fail_low_acceptance(self):
        r = eval_proposal_acceptance([], make_session(trades=0, proposed=5))
        assert not r.passed
        assert r.detail["acceptance_rate"] == 0.0

    def test_pass_above_threshold(self):
        # 2 of 3 proposed = 66% > 40% threshold
        r = eval_proposal_acceptance([], make_session(trades=2, proposed=3))
        assert r.passed

    def test_zero_proposed_zero_trades_passes(self):
        r = eval_proposal_acceptance([], make_session(trades=0, proposed=0))
        assert r.passed

    def test_zero_proposed_with_trades_fails(self):
        r = eval_proposal_acceptance([], make_session(trades=1, proposed=0))
        assert not r.passed

    def test_eval_name_and_agent(self):
        r = eval_proposal_acceptance([], make_session())
        assert r.eval_name == "proposal_acceptance"
        assert r.agent == "business"
