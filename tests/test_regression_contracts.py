"""
Data contract regression tests — lock in the exact field names and types of
EvalResult.to_db_row() and Incident.to_db_row().

These are the rows written to c_evals / c_incidents (and ag_evals / ag_incidents
after migration). Any field rename or type change will break these tests.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import pytest

from engine.eval_engine import EvalResult, run_all_evals, REQUIRED_AGENTS
from engine.pattern_detector import Incident, run_all_detectors


# ── EvalResult contract ───────────────────────────────────────────────────────

EVAL_DB_ROW_FIELDS = {"id", "session_id", "agent", "eval_name",
                      "score", "passed", "threshold", "detail"}

class TestEvalResultContract:

    def _row(self, **kwargs) -> dict:
        r = EvalResult(
            eval_name="tool_success_rate",
            agent="research",
            score=0.9,
            passed=True,
            threshold=0.8,
            detail={"total": 5, "success": 5},
        )
        return r.to_db_row("sess-abc")

    def test_to_db_row_has_all_required_fields(self):
        assert EVAL_DB_ROW_FIELDS.issubset(self._row().keys())

    def test_to_db_row_has_no_extra_fields(self):
        assert set(self._row().keys()) == EVAL_DB_ROW_FIELDS

    def test_id_is_valid_uuid(self):
        row = self._row()
        uuid.UUID(row["id"])  # raises if invalid

    def test_session_id_matches_argument(self):
        r = EvalResult("x", "y", 1.0, True, 0.5)
        assert r.to_db_row("my-session")["session_id"] == "my-session"

    def test_score_is_rounded_to_3_decimals(self):
        r = EvalResult("x", "y", 0.666666, True, 0.5)
        assert r.to_db_row("s")["score"] == 0.667

    def test_score_type_is_float(self):
        assert isinstance(self._row()["score"], float)

    def test_passed_type_is_bool(self):
        assert isinstance(self._row()["passed"], bool)

    def test_threshold_type_is_float(self):
        assert isinstance(self._row()["threshold"], float)

    def test_detail_type_is_dict(self):
        assert isinstance(self._row()["detail"], dict)

    def test_eval_name_and_agent_preserved(self):
        r   = EvalResult("my_eval", "my_agent", 0.5, False, 0.8)
        row = r.to_db_row("s")
        assert row["eval_name"] == "my_eval"
        assert row["agent"]     == "my_agent"


# ── Incident contract ─────────────────────────────────────────────────────────

INCIDENT_DB_ROW_FIELDS = {"id", "session_id", "pattern_name", "severity",
                           "root_cause", "call_stack", "failed_evals",
                           "cost_wasted", "tokens_wasted", "fix_suggestion",
                           "is_simulated"}

class TestIncidentContract:

    def _inc(self) -> Incident:
        return Incident(
            session_id="sess-001",
            pattern_name="Tool Timeout Loop",
            severity="critical",
            root_cause="tool timed out 3 times",
            call_stack=[{"agent": "research", "tool_name": "get_data"}],
            failed_evals=[{"eval_name": "tool_success_rate", "score": 0.2}],
            cost_wasted=0.08,
            tokens_wasted=1500,
            fix_suggestion="add timeout",
        )

    def test_to_db_row_has_all_required_fields(self):
        assert INCIDENT_DB_ROW_FIELDS.issubset(self._inc().to_db_row().keys())

    def test_to_db_row_has_no_extra_fields(self):
        assert set(self._inc().to_db_row().keys()) == INCIDENT_DB_ROW_FIELDS

    def test_id_is_valid_uuid(self):
        uuid.UUID(self._inc().to_db_row()["id"])

    def test_cost_wasted_rounded_to_4_decimals(self):
        inc = self._inc()
        inc.cost_wasted = 0.123456789
        assert inc.to_db_row()["cost_wasted"] == 0.1235

    def test_cost_wasted_type_is_float(self):
        assert isinstance(self._inc().to_db_row()["cost_wasted"], float)

    def test_tokens_wasted_type_is_int(self):
        assert isinstance(self._inc().to_db_row()["tokens_wasted"], int)

    def test_call_stack_type_is_list(self):
        assert isinstance(self._inc().to_db_row()["call_stack"], list)

    def test_failed_evals_type_is_list(self):
        assert isinstance(self._inc().to_db_row()["failed_evals"], list)

    def test_is_simulated_defaults_false(self):
        assert self._inc().to_db_row()["is_simulated"] is False

    def test_is_simulated_propagated(self):
        inc = self._inc()
        inc.is_simulated = True
        assert inc.to_db_row()["is_simulated"] is True

    def test_session_id_preserved(self):
        assert self._inc().to_db_row()["session_id"] == "sess-001"


# ── run_all_evals output contract ─────────────────────────────────────────────

EXPECTED_EVAL_NAMES = {
    # per-agent
    "data_completeness", "data_freshness",              # market
    "completion", "token_efficiency",                   # research
    "tool_success_rate", "tool_diversity",              # research
    "assessment_complete", "within_parameters",         # risk
    "decision_made", "exit_quality",                    # orchestrator
    # session
    "pipeline_completion", "cost_anomaly",
    "outcome_linkage", "tokens_per_decision",
    # business
    "cost_per_trade", "research_conversion", "proposal_acceptance",
}

EXPECTED_EVAL_AGENTS = {
    "market", "research", "risk", "orchestrator", "session", "business",
}

def _minimal_session() -> dict:
    return {
        "id":                  "sess-min",
        "total_cost_usd":      0.05,
        "total_tokens_input":  1000,
        "total_tokens_output": 500,
        "trades_executed":     1,
        "trades_proposed":     1,
        "terminal_reason":     "converged",
        "started_at":          "2026-06-08T10:00:00",
        "completed_at":        "2026-06-08T10:05:00",
    }

def _minimal_traces() -> list[dict]:
    return [
        {"agent": "market",       "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:00:30",
         "tokens_input": 100, "tokens_output": 50},
        {"agent": "research",     "step_type": "llm_call",  "tool_name": None,
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:01:00",
         "tokens_input": 200, "tokens_output": 100},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_news",
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:01:10",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_atr",
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:01:20",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "risk",         "step_type": "tool_call", "tool_name": "check_exposure",
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:02:00",
         "tokens_input": 50, "tokens_output": 30},
        {"agent": "risk",         "step_type": "decision",  "tool_name": None,
         "outcome": "approved",  "error": None, "created_at": "2026-06-08T10:02:10",
         "tokens_input": 0, "tokens_output": 0},
        {"agent": "orchestrator", "step_type": "decision",  "tool_name": None,
         "outcome": "execute",   "error": None, "created_at": "2026-06-08T10:03:00",
         "tokens_input": 100, "tokens_output": 80},
    ]


class TestRunAllEvalsContract:

    def test_returns_exactly_17_results(self):
        results = run_all_evals(_minimal_session(), _minimal_traces(), [0.05])
        assert len(results) == 17

    def test_all_eval_names_present(self):
        results = run_all_evals(_minimal_session(), _minimal_traces(), [0.05])
        names = {r.eval_name for r in results}
        assert names == EXPECTED_EVAL_NAMES

    def test_all_agents_present(self):
        results = run_all_evals(_minimal_session(), _minimal_traces(), [0.05])
        agents = {r.agent for r in results}
        assert agents == EXPECTED_EVAL_AGENTS

    def test_every_result_is_eval_result(self):
        from engine.eval_engine import EvalResult
        results = run_all_evals(_minimal_session(), _minimal_traces())
        assert all(isinstance(r, EvalResult) for r in results)

    def test_all_scores_in_0_to_1_range(self):
        results = run_all_evals(_minimal_session(), _minimal_traces(), [0.05] * 10)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"{r.eval_name} score {r.score} out of range"

    def test_passed_consistent_with_score_and_threshold(self):
        results = run_all_evals(_minimal_session(), _minimal_traces(), [0.05])
        for r in results:
            if r.score >= r.threshold:
                assert r.passed, f"{r.eval_name}: score={r.score} >= threshold={r.threshold} but passed=False"

    def test_to_db_row_all_results_valid(self):
        results = run_all_evals(_minimal_session(), _minimal_traces())
        for r in results:
            row = r.to_db_row("sess-test")
            assert EVAL_DB_ROW_FIELDS.issubset(row.keys())
            assert 0.0 <= row["score"] <= 1.0
