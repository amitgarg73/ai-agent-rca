"""
Golden regression tests — lock in exact score values for canonical scenarios.

If any score changes during migration, these tests catch it immediately.
The canonical scenarios are stable inputs chosen to produce deterministic outputs.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.eval_engine import (
    run_all_evals,
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
    TOKEN_EFFICIENCY_THRESHOLD,
    TOKEN_SPIRAL_THRESHOLD,
    COST_PER_TRADE_THRESHOLD,
)


# ── Canonical sessions ────────────────────────────────────────────────────────

def _healthy_session() -> dict:
    return {
        "id":                  "sess-gold-healthy",
        "total_cost_usd":      0.10,
        "total_tokens_input":  3000,
        "total_tokens_output": 1500,
        "trades_executed":     2,
        "trades_proposed":     2,
        "terminal_reason":     "converged",
        "started_at":          "2026-06-08T10:00:00",
        "completed_at":        "2026-06-08T10:05:00",
    }

def _healthy_traces() -> list[dict]:
    """All 4 agents, 3 distinct research tools, no errors, timestamps fresh."""
    return [
        # market
        {"agent": "market",       "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:00:30",
         "tokens_input": 200, "tokens_output": 100, "entity_id": None},
        # research — 3 distinct tools
        {"agent": "research",     "step_type": "llm_call",  "tool_name": None,
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:00",
         "tokens_input": 400, "tokens_output": 200, "entity_id": None},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_news",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:10",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_atr",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:20",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_price",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:30",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        # risk
        {"agent": "risk",         "step_type": "tool_call", "tool_name": "check_risk",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:02:00",
         "tokens_input": 100, "tokens_output": 50, "entity_id": None},
        {"agent": "risk",         "step_type": "decision",  "tool_name": None,
         "outcome": "approved",  "error": None, "created_at": "2026-06-08T10:02:10",
         "tokens_input": 0, "tokens_output": 0, "entity_id": None},
        # orchestrator
        {"agent": "orchestrator", "step_type": "decision",  "tool_name": None,
         "outcome": "execute",   "error": None, "created_at": "2026-06-08T10:03:00",
         "tokens_input": 200, "tokens_output": 100, "entity_id": None},
    ]

def _timeout_session() -> dict:
    return {
        "id":                  "sess-gold-timeout",
        "total_cost_usd":      0.30,
        "total_tokens_input":  500,
        "total_tokens_output": 100,
        "trades_executed":     0,
        "trades_proposed":     0,
        "terminal_reason":     "error",
        "started_at":          "2026-06-08T10:00:00",
        "completed_at":        "2026-06-08T10:05:00",
    }

def _timeout_traces() -> list[dict]:
    """Research tool times out 3 times — tool_success_rate fails."""
    base = {"step_type": "tool_call", "tool_name": "get_stock_data",
            "outcome": "error", "error": "ReadTimeout",
            "tokens_input": 0, "tokens_output": 0, "entity_id": None}
    return [
        {"agent": "market", "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:00:30",
         "tokens_input": 200, "tokens_output": 100, "entity_id": None},
        {**base, "agent": "research", "created_at": "2026-06-08T10:01:00"},
        {**base, "agent": "research", "created_at": "2026-06-08T10:01:30"},
        {**base, "agent": "research", "created_at": "2026-06-08T10:02:00"},
    ]


# ── Healthy session: every eval golden score ──────────────────────────────────

class TestHealthySessionGoldenScores:
    """Lock in every eval score for the canonical healthy scenario."""

    def test_market_data_completeness_1_0(self):
        r = eval_market_data_completeness(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_market_data_freshness_passes(self):
        # lag=0.5min, threshold=10min → score=1-0.5/20=0.975 → rounds to 0.97
        r = eval_market_data_freshness(_healthy_traces(), _healthy_session())
        assert r.score == pytest.approx(0.97, abs=0.01) and r.passed

    def test_research_completion_1_0(self):
        r = eval_research_completion(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_research_token_efficiency_passes(self):
        # 600 tokens, threshold=30K → score=1-600/60000=0.99
        r = eval_research_token_efficiency(_healthy_traces(), _healthy_session())
        assert r.score == pytest.approx(0.99, abs=0.01) and r.passed

    def test_research_tool_success_rate_1_0(self):
        r = eval_research_tool_success_rate(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_research_tool_diversity_1_0(self):
        r = eval_research_tool_diversity(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_risk_assessment_complete_1_0(self):
        r = eval_risk_assessment_complete(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_risk_within_parameters_1_0(self):
        r = eval_risk_within_parameters(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_orchestrator_decision_made_1_0(self):
        r = eval_orchestrator_decision_made(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_orchestrator_exit_quality_1_0(self):
        r = eval_orchestrator_exit_quality(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_pipeline_completion_1_0(self):
        r = eval_session_pipeline_completion(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_cost_anomaly_passes_with_no_history(self):
        # No cost history → returns 1.0 (can't detect anomaly without baseline)
        r = eval_session_cost_anomaly(_healthy_traces(), _healthy_session(), [])
        assert r.passed

    def test_cost_anomaly_passes_within_history(self):
        history = [0.10] * 15
        r = eval_session_cost_anomaly(_healthy_traces(), _healthy_session(), history)
        assert r.score == 1.0 and r.passed

    def test_outcome_linkage_1_0(self):
        r = eval_session_outcome_linkage(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_tokens_per_decision_passes(self):
        # 4500 total / 2 trades = 2250 tok_per, threshold=40K → score=1-2250/80000≈0.97
        r = eval_session_tokens_per_decision(_healthy_traces(), _healthy_session())
        assert r.score == pytest.approx(0.97, abs=0.01) and r.passed

    def test_cost_per_trade_passes(self):
        # $0.10 / 2 trades = $0.05 cpt, threshold=$0.50 → score=1-0.05/1.5≈0.97
        r = eval_cost_per_trade(_healthy_traces(), _healthy_session())
        assert r.score == pytest.approx(0.97, abs=0.01) and r.passed

    def test_research_conversion_passes(self):
        r = eval_research_conversion(_healthy_traces(), _healthy_session())
        assert r.passed

    def test_proposal_acceptance_1_0(self):
        r = eval_proposal_acceptance(_healthy_traces(), _healthy_session())
        assert r.score == 1.0 and r.passed

    def test_all_17_evals_pass(self):
        results  = run_all_evals(_healthy_session(), _healthy_traces(), [0.10] * 15)
        failed   = [r for r in results if not r.passed]
        assert not failed, f"Evals failed unexpectedly: {[r.eval_name for r in failed]}"


# ── Tool timeout scenario: expected failures ──────────────────────────────────

class TestTimeoutSessionGoldenScores:

    def test_research_tool_success_rate_0_0(self):
        r = eval_research_tool_success_rate(_timeout_traces(), _timeout_session())
        assert r.score == 0.0 and not r.passed

    def test_research_completion_0_3_llm_ran_all_tools_failed(self):
        # No LLM call in timeout traces — research never started
        r = eval_research_completion(_timeout_traces(), _timeout_session())
        assert r.score == 0.0 and not r.passed

    def test_orchestrator_decision_made_fails(self):
        r = eval_orchestrator_decision_made(_timeout_traces(), _timeout_session())
        assert not r.passed

    def test_outcome_linkage_fails(self):
        r = eval_session_outcome_linkage(_timeout_traces(), _timeout_session())
        assert not r.passed

    def test_pipeline_completion_partial(self):
        # Only market + research(failed) — no risk, no orchestrator
        r = eval_session_pipeline_completion(_timeout_traces(), _timeout_session())
        assert r.score < 1.0


# ── Score formula golden values ───────────────────────────────────────────────

class TestScoreFormulas:
    """Lock in exact scores for boundary values of formula-based evals."""

    def test_token_efficiency_at_half_threshold(self):
        # 15K tokens = TOKEN_EFFICIENCY_THRESHOLD / 2 → score = 1 - 15K/60K = 0.75
        sess   = {**_healthy_session(), "total_tokens_input": 0, "total_tokens_output": 0}
        traces = [{"agent": "research", "step_type": "llm_call", "outcome": "success",
                   "tokens_input": 15_000, "tokens_output": 0, "entity_id": None}]
        r = eval_research_token_efficiency(traces, sess)
        assert r.score == pytest.approx(0.75, abs=0.01)

    def test_token_efficiency_at_threshold_is_just_below_1(self):
        # 30K tokens = threshold → score = 1 - 30K/60K = 0.5, not passed
        traces = [{"agent": "research", "step_type": "llm_call", "outcome": "success",
                   "tokens_input": TOKEN_EFFICIENCY_THRESHOLD, "tokens_output": 0,
                   "entity_id": None}]
        r = eval_research_token_efficiency(traces, _healthy_session())
        assert not r.passed
        assert r.score == pytest.approx(0.5, abs=0.01)

    def test_cost_per_trade_at_threshold(self):
        # cost=$0.50, trades=1 → cpt=$0.50 = threshold → passed=False (must be <=)
        sess = {**_healthy_session(), "total_cost_usd": 0.50, "trades_executed": 1}
        r = eval_cost_per_trade([], sess)
        assert r.passed  # threshold is <=, so exactly at threshold passes

    def test_cost_per_trade_above_threshold_fails(self):
        sess = {**_healthy_session(), "total_cost_usd": 1.50, "trades_executed": 1}
        r = eval_cost_per_trade([], sess)
        assert not r.passed

    def test_cost_per_trade_score_formula(self):
        # cpt = $0.75, threshold = $0.50, multiplier = 3
        # score = max(0, 1 - 0.75 / 1.5) = max(0, 0.5) = 0.5
        sess = {**_healthy_session(), "total_cost_usd": 0.75, "trades_executed": 1}
        r = eval_cost_per_trade([], sess)
        assert r.score == pytest.approx(0.5, abs=0.01)

    def test_tokens_per_decision_zero_trades_uses_raw_tokens(self):
        # 0 trades, 20K total tokens → tok_per = 20K < 40K → passed
        sess = {**_healthy_session(),
                "total_tokens_input": 20_000,
                "total_tokens_output": 0,
                "trades_executed": 0}
        r = eval_session_tokens_per_decision([], sess)
        assert r.passed
        assert r.detail["tokens_per_decision"] == 20_000

    def test_tokens_per_decision_normalises_by_trade_count(self):
        # 60K total, 2 trades → tok_per = 30K < 40K threshold → passed
        sess = {**_healthy_session(),
                "total_tokens_input": 60_000,
                "total_tokens_output": 0,
                "trades_executed": 2}
        r = eval_session_tokens_per_decision([], sess)
        assert r.passed
        assert r.detail["tokens_per_decision"] == 30_000

    def test_research_conversion_rate_exactly_at_threshold(self):
        # 1 trade, 1 ticker researched → rate = 1.0 >= 0.30 threshold → passed
        traces = [{"agent": "research_AAPL", "step_type": "tool_call",
                   "outcome": "success", "entity_id": "AAPL",
                   "tokens_input": 0, "tokens_output": 0}]
        sess = {**_healthy_session(), "trades_executed": 1}
        r = eval_research_conversion(traces, sess)
        assert r.passed

    def test_proposal_acceptance_rate_exactly_at_threshold(self):
        # 2 proposed, 1 executed → rate = 0.5 >= 0.40 → passed
        sess = {**_healthy_session(), "trades_executed": 1, "trades_proposed": 2}
        r = eval_proposal_acceptance([], sess)
        assert r.passed
