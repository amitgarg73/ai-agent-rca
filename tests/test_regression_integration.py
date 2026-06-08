"""
End-to-end integration regression tests.

Validates the full pipeline:
  session + traces
    → run_all_evals (17 EvalResults)
    → run_all_detectors (Incidents)
    → build_annotated_call_stack (annotated traces)
    → generate_fix_suggestion (fix string)
    → summarize_incident (display dict)

These tests confirm that the data flows correctly through every layer with
no coupling to c_* table names. All inputs are plain dicts — the same shape
as ag_sessions / ag_traces rows.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.eval_engine    import run_all_evals
from engine.pattern_detector import run_all_detectors, run_quality_detectors, Incident
from engine.quality_judge  import judge_session
from engine.rca_engine     import (build_annotated_call_stack,
                                   generate_fix_suggestion,
                                   summarize_incident)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _session(sid="sess-e2e", trades=1, cost=0.10, tokens=4500,
             reason="converged", started="2026-06-08T10:00:00",
             completed="2026-06-08T10:05:00") -> dict:
    return {
        "id":                  sid,
        "total_cost_usd":      cost,
        "total_tokens_input":  int(tokens * 0.67),
        "total_tokens_output": int(tokens * 0.33),
        "trades_executed":     trades,
        "trades_proposed":     trades,
        "terminal_reason":     reason,
        "started_at":          started,
        "completed_at":        completed,
        "is_simulated":        False,
    }

def _full_healthy_traces() -> list[dict]:
    return [
        {"agent": "market",       "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:00:30",
         "tokens_input": 200, "tokens_output": 100, "entity_id": None},
        {"agent": "research",     "step_type": "llm_call",  "tool_name": None,
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:00",
         "tokens_input": 300, "tokens_output": 150, "entity_id": None},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_news",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:10",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "research",     "step_type": "tool_call", "tool_name": "get_atr",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:01:20",
         "tokens_input": 0, "tokens_output": 0, "entity_id": "AAPL"},
        {"agent": "risk",         "step_type": "tool_call", "tool_name": "check_risk",
         "outcome": "success",   "error": None, "created_at": "2026-06-08T10:02:00",
         "tokens_input": 50, "tokens_output": 30, "entity_id": None},
        {"agent": "risk",         "step_type": "decision",  "tool_name": None,
         "outcome": "approved",  "error": None, "created_at": "2026-06-08T10:02:10",
         "tokens_input": 0, "tokens_output": 0, "entity_id": None},
        {"agent": "orchestrator", "step_type": "decision",  "tool_name": None,
         "outcome": "execute",   "error": None, "created_at": "2026-06-08T10:03:00",
         "tokens_input": 150, "tokens_output": 80, "entity_id": None},
    ]

def _timeout_traces() -> list[dict]:
    return [
        {"agent": "market",   "step_type": "tool_call", "tool_name": "get_market",
         "outcome": "success", "error": None, "created_at": "2026-06-08T10:00:30",
         "tokens_input": 200, "tokens_output": 100, "entity_id": None},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out",
         "created_at": "2026-06-08T10:01:00",
         "tokens_input": 0, "tokens_output": 0, "entity_id": None},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out",
         "created_at": "2026-06-08T10:01:30",
         "tokens_input": 0, "tokens_output": 0, "entity_id": None},
        {"agent": "research", "step_type": "tool_call", "tool_name": "get_stock_data",
         "outcome": "error",   "error": "ReadTimeout: timed out",
         "created_at": "2026-06-08T10:02:00",
         "tokens_input": 0, "tokens_output": 0, "entity_id": None},
    ]


# ── Clean session: no incidents ───────────────────────────────────────────────

class TestCleanSessionEndToEnd:

    def test_no_incidents_for_healthy_session(self):
        sess   = _session()
        traces = _full_healthy_traces()
        evals  = run_all_evals(sess, traces, [0.10] * 15)
        incidents = run_all_detectors(sess, traces, evals, [0.10] * 15)
        assert incidents == []

    def test_17_evals_all_pass_for_healthy_session(self):
        sess   = _session()
        traces = _full_healthy_traces()
        evals  = run_all_evals(sess, traces, [0.10] * 15)
        assert len(evals) == 17
        failed = [r for r in evals if not r.passed]
        assert not failed, f"Unexpected failures: {[r.eval_name for r in failed]}"

    def test_quality_judge_returns_results_for_healthy_session(self):
        sess   = _session()
        traces = _full_healthy_traces()
        results = judge_session(sess, traces)
        assert len(results) > 0
        agents  = {r.agent for r in results}
        assert "research_quality" in agents

    def test_quality_judge_composite_above_zero(self):
        sess   = _session()
        traces = _full_healthy_traces()
        results = judge_session(sess, traces)
        composites = [r for r in results if r.eval_name == "composite_score"]
        assert all(r.score > 0 for r in composites)


# ── Timeout session: full incident pipeline ───────────────────────────────────

class TestTimeoutIncidentEndToEnd:

    def _run(self):
        sess      = _session(trades=0, reason="error", cost=0.30)
        traces    = _timeout_traces()
        costs     = [0.10] * 10
        evals     = run_all_evals(sess, traces, costs)
        incidents = run_all_detectors(sess, traces, evals, costs)
        return sess, traces, evals, incidents

    def test_tool_timeout_incident_fires(self):
        _, _, _, incidents = self._run()
        names = [i.pattern_name for i in incidents]
        assert "Tool Timeout Loop" in names

    def test_incident_has_all_required_fields(self):
        _, _, _, incidents = self._run()
        timeout_inc = next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")
        row = timeout_inc.to_db_row()
        required = {"id", "session_id", "pattern_name", "severity", "root_cause",
                    "call_stack", "failed_evals", "cost_wasted", "tokens_wasted",
                    "fix_suggestion", "is_simulated"}
        assert required.issubset(row.keys())

    def test_incident_severity_is_critical(self):
        _, _, _, incidents = self._run()
        timeout_inc = next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")
        assert timeout_inc.severity == "critical"

    def test_annotated_call_stack_marks_root(self):
        sess, traces, evals, incidents = self._run()
        timeout_inc = next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")
        annotated   = build_annotated_call_stack(traces, timeout_inc)
        roots       = [a for a in annotated if a["is_root"]]
        assert len(roots) == 1
        assert roots[0]["error"] is not None

    def test_annotated_call_stack_marks_relevant_traces(self):
        sess, traces, evals, incidents = self._run()
        timeout_inc = next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")
        annotated   = build_annotated_call_stack(traces, timeout_inc)
        relevant    = [a for a in annotated if a["is_relevant"]]
        assert len(relevant) == 3  # all 3 timeout tool calls
        assert all(a["tool_name"] == "get_stock_data" for a in relevant)

    def test_fix_suggestion_mentions_tool(self):
        _, traces, evals, incidents = self._run()
        timeout_inc = next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")
        fix = generate_fix_suggestion(timeout_inc, traces)
        assert "get_stock_data" in fix
        assert "timeout" in fix.lower()

    def test_summarize_incident_complete(self):
        sess, traces, evals, incidents = self._run()
        timeout_inc = next(i for i in incidents if i.pattern_name == "Tool Timeout Loop")
        summary = summarize_incident(timeout_inc, sess)
        assert summary["pattern"]   == "Tool Timeout Loop"
        assert summary["severity"]  == "critical"
        assert summary["duration_s"] == 300
        assert summary["trades"]     == 0


# ── Data flows through without table references ───────────────────────────────

class TestSchemaAgnosticism:
    """Verifies that engine functions operate purely on dict fields —
    no c_* or ag_* table name is referenced inside any engine function."""

    def test_run_all_evals_accepts_arbitrary_dict_keys(self):
        # Extra fields that aren't in c_sessions schema — should be silently ignored
        sess = {
            **_session(),
            "tenant_id":   "t-abc",
            "workflow_id": "wf-123",
            "ag_extra":    "should be ignored",
        }
        results = run_all_evals(sess, _full_healthy_traces())
        assert len(results) == 17

    def test_run_all_detectors_accepts_arbitrary_dict_keys(self):
        sess   = {**_session(), "tenant_id": "t-abc", "workflow_id": "wf-123"}
        traces = _full_healthy_traces()
        evals  = run_all_evals(sess, traces)
        # Should not raise
        incidents = run_all_detectors(sess, traces, evals, [])
        assert isinstance(incidents, list)

    def test_judge_session_accepts_arbitrary_dict_keys(self):
        sess   = {**_session(), "tenant_id": "t-abc"}
        traces = _full_healthy_traces()
        results = judge_session(sess, traces)
        assert isinstance(results, list)

    def test_build_annotated_call_stack_works_with_ag_traces_shape(self):
        """ag_traces has tenant_id, workflow_id extra — engine must tolerate them."""
        traces = [
            {**t, "tenant_id": "t-abc", "workflow_id": "wf-123"}
            for t in _full_healthy_traces()
        ]
        inc = Incident(
            session_id="s", pattern_name="Context Spiral", severity="warning",
            root_cause="", call_stack=[], failed_evals=[],
        )
        result = build_annotated_call_stack(traces, inc)
        assert len(result) == len(traces)
        # Extra fields are passed through
        assert all("tenant_id" in r for r in result)

    def test_eval_result_to_db_row_produces_stable_schema(self):
        """Same inputs must always produce the same field set — migration guarantee."""
        results = run_all_evals(_session(), _full_healthy_traces())
        field_sets = [set(r.to_db_row("s").keys()) for r in results]
        # All 17 evals produce identical field sets
        assert len(set(frozenset(fs) for fs in field_sets)) == 1

    def test_incident_to_db_row_produces_stable_schema(self):
        sess, traces, evals, incidents = (
            _session(trades=0, reason="error"),
            _timeout_traces(),
            run_all_evals(_session(trades=0, reason="error"), _timeout_traces(), [0.1]*10),
            None,
        )
        incidents = run_all_detectors(
            _session(trades=0, reason="error"), _timeout_traces(), evals, [0.1]*10
        )
        if incidents:
            row = incidents[0].to_db_row()
            required = {"id", "session_id", "pattern_name", "severity", "root_cause",
                        "call_stack", "failed_evals", "cost_wasted", "tokens_wasted",
                        "fix_suggestion", "is_simulated"}
            assert required.issubset(row.keys())
