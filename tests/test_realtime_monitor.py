"""
Tests for evaluate_session() in realtime_monitor.py.
Verifies that both operational evals and quality evals are run and written
for each new session.

Run: python3 -m pytest tests/test_realtime_monitor.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch, call
import pytest

from engine.realtime_monitor import evaluate_session


def _session(sid="sess-001"):
    return {
        "id": sid,
        "started_at": "2026-06-02T10:00:00",
        "completed_at": "2026-06-02T10:05:00",
        "total_cost_usd": 0.05,
        "terminal_reason": "completed",
        "trades_executed": 1,
        "is_simulated": False,
    }


def _make_eval_result(agent="market", name="data_completeness", score=1.0):
    r = MagicMock()
    r.to_db_row.return_value = {
        "session_id": "sess-001",
        "agent": agent,
        "eval_name": name,
        "score": score,
    }
    return r


class TestEvaluateSessionQualityIntegration:

    def test_judge_session_called_with_traces(self):
        """Quality judge is called with the traces loaded for the session."""
        session = _session()
        traces = [{"agent": "research", "step_type": "tool_call"}]

        op_eval = _make_eval_result("market", "data_completeness")
        q_eval  = _make_eval_result("research_quality", "composite_score", 0.8)

        mock_db = MagicMock()

        with patch("sdk.db.load_session_traces", return_value=traces), \
             patch("sdk.db.load_recent_session_costs", return_value=[0.05]), \
             patch("engine.eval_engine.run_all_evals", return_value=[op_eval]), \
             patch("engine.pattern_detector.run_all_detectors", return_value=[]), \
             patch("engine.pattern_detector.compute_shadow_cb_fires", return_value=[]), \
             patch("engine.quality_judge.judge_session", return_value=[q_eval]) as mock_judge:

            result = evaluate_session(session, db=mock_db, dry_run=True)

        mock_judge.assert_called_once_with(session, traces)
        assert result["quality_evals"] == 1
        assert result["evals"] == 1
        assert result["error"] is None

    def test_both_evals_written_to_db(self):
        """Operational and quality eval rows are upserted together."""
        session = _session()

        op_eval = _make_eval_result("market", "data_completeness")
        q_eval  = _make_eval_result("research_quality", "composite_score", 0.75)

        mock_db = MagicMock()
        mock_db.table.return_value.upsert.return_value.execute.return_value = None

        with patch("sdk.db.load_session_traces", return_value=[]), \
             patch("sdk.db.load_recent_session_costs", return_value=[]), \
             patch("engine.eval_engine.run_all_evals", return_value=[op_eval]), \
             patch("engine.pattern_detector.run_all_detectors", return_value=[]), \
             patch("engine.pattern_detector.compute_shadow_cb_fires", return_value=[]), \
             patch("engine.quality_judge.judge_session", return_value=[q_eval]):

            result = evaluate_session(session, db=mock_db, dry_run=False)

        upsert_call = mock_db.table.return_value.upsert
        assert upsert_call.called
        written_rows = upsert_call.call_args[0][0]
        agents_written = {r["agent"] for r in written_rows}
        assert "market" in agents_written
        assert "research_quality" in agents_written

    def test_quality_eval_count_in_result(self):
        """Result dict includes quality_evals key with correct count."""
        session = _session()
        q_evals = [
            _make_eval_result("research_quality", "composite_score"),
            _make_eval_result("risk_quality", "composite_score"),
        ]

        with patch("sdk.db.load_session_traces", return_value=[]), \
             patch("sdk.db.load_recent_session_costs", return_value=[]), \
             patch("engine.eval_engine.run_all_evals", return_value=[]), \
             patch("engine.pattern_detector.run_all_detectors", return_value=[]), \
             patch("engine.pattern_detector.compute_shadow_cb_fires", return_value=[]), \
             patch("engine.quality_judge.judge_session", return_value=q_evals):

            result = evaluate_session(session, db=None, dry_run=True)

        assert result["quality_evals"] == 2

    def test_quality_judge_failure_does_not_crash_session(self):
        """If judge_session returns empty (error case), evaluate_session still completes."""
        session = _session()
        op_eval = _make_eval_result("market", "data_completeness")

        with patch("sdk.db.load_session_traces", return_value=[]), \
             patch("sdk.db.load_recent_session_costs", return_value=[]), \
             patch("engine.eval_engine.run_all_evals", return_value=[op_eval]), \
             patch("engine.pattern_detector.run_all_detectors", return_value=[]), \
             patch("engine.pattern_detector.compute_shadow_cb_fires", return_value=[]), \
             patch("engine.quality_judge.judge_session", return_value=[]):

            result = evaluate_session(session, db=None, dry_run=True)

        assert result["error"] is None
        assert result["evals"] == 1
        assert result["quality_evals"] == 0

    def test_no_db_writes_in_dry_run(self):
        """dry_run=True skips all DB upserts."""
        session = _session()
        op_eval = _make_eval_result("market", "data_completeness")
        q_eval  = _make_eval_result("research_quality", "composite_score")

        mock_db = MagicMock()

        with patch("sdk.db.load_session_traces", return_value=[]), \
             patch("sdk.db.load_recent_session_costs", return_value=[]), \
             patch("sdk.db.load_recent_sessions", return_value=[]), \
             patch("sdk.db.load_evals_by_session", return_value={}), \
             patch("engine.eval_engine.run_all_evals", return_value=[op_eval]), \
             patch("engine.pattern_detector.run_all_detectors", return_value=[]), \
             patch("engine.pattern_detector.compute_shadow_cb_fires", return_value=[]), \
             patch("engine.pattern_detector.run_quality_detectors", return_value=[]), \
             patch("engine.quality_judge.judge_session", return_value=[q_eval]):

            evaluate_session(session, db=mock_db, dry_run=True)

        mock_db.table.assert_not_called()
