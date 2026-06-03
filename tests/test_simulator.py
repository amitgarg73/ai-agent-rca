"""
Tests for simulator/failure_sim.py — quality pattern builders and simulate_quality_failure().
Run: python3 -m pytest tests/test_simulator.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from simulator.failure_sim import (
    PATTERNS,
    QUALITY_PATTERNS,
    simulate_failure,
    simulate_quality_failure,
    _build_quality_grounding_failure,
    _build_quality_coherence_break,
    _build_quality_cascade,
    _build_quality_silent_degradation,
    _QUALITY_BUILDERS,
    _QUALITY_DESCRIPTIONS,
)


# ── Builder shape tests ────────────────────────────────────────────────────────

class TestGroundingFailureBuilder:
    def test_returns_three_sessions(self):
        sessions, evals_by = _build_quality_grounding_failure()
        assert len(sessions) == 3

    def test_each_session_has_evals(self):
        sessions, evals_by = _build_quality_grounding_failure()
        for s in sessions:
            assert s["id"] in evals_by
            assert len(evals_by[s["id"]]) > 0

    def test_data_grounding_below_threshold(self):
        sessions, evals_by = _build_quality_grounding_failure()
        for s in sessions:
            rows = evals_by[s["id"]]
            dg = next((r for r in rows if r["eval_name"] == "data_grounding"), None)
            assert dg is not None
            assert dg["score"] < 0.40

    def test_sessions_are_simulated(self):
        sessions, _ = _build_quality_grounding_failure()
        for s in sessions:
            assert s.get("is_simulated") is True

    def test_staggered_timestamps(self):
        sessions, _ = _build_quality_grounding_failure()
        times = [s["started_at"] for s in sessions]
        assert times[0] < times[1] < times[2]


class TestCoherenceBreakBuilder:
    def test_returns_one_session(self):
        sessions, evals_by = _build_quality_coherence_break()
        assert len(sessions) == 1

    def test_decision_consistency_below_threshold(self):
        sessions, evals_by = _build_quality_coherence_break()
        sid = sessions[0]["id"]
        rows = evals_by[sid]
        dc = next((r for r in rows if r["eval_name"] == "decision_consistency"), None)
        assert dc is not None
        assert dc["score"] < 0.50

    def test_session_is_simulated(self):
        sessions, _ = _build_quality_coherence_break()
        assert sessions[0].get("is_simulated") is True


class TestQualityCascadeBuilder:
    def test_returns_five_sessions(self):
        sessions, evals_by = _build_quality_cascade()
        assert len(sessions) == 5

    def test_dimensions_decline(self):
        sessions, evals_by = _build_quality_cascade()
        dims = ["data_grounding", "thesis_coherence", "parameter_completeness",
                "decision_consistency", "pipeline_coherence"]
        for dim in dims:
            scores = []
            for s in sessions:
                row = next((r for r in evals_by[s["id"]] if r["eval_name"] == dim), None)
                if row:
                    scores.append(row["score"])
            assert len(scores) == 5
            assert scores[0] > scores[-1], f"{dim} should decline"
            assert scores[0] - scores[-1] >= 0.20, f"{dim} drop should be >= 0.20"

    def test_sessions_are_simulated(self):
        sessions, _ = _build_quality_cascade()
        for s in sessions:
            assert s.get("is_simulated") is True


class TestSilentDegradationBuilder:
    def test_returns_five_sessions(self):
        sessions, evals_by = _build_quality_silent_degradation()
        assert len(sessions) == 5

    def test_composite_declines(self):
        sessions, evals_by = _build_quality_silent_degradation()
        comps = []
        for s in sessions:
            rows = evals_by[s["id"]]
            c = next((r["score"] for r in rows
                      if r["eval_name"] == "composite_score"
                      and r["agent"] == "research_quality"), None)
            if c is not None:
                comps.append(c)
        assert comps[0] > comps[-1]
        assert comps[0] - comps[-1] >= 0.20

    def test_operational_evals_pass(self):
        sessions, evals_by = _build_quality_silent_degradation()
        for s in sessions:
            rows = evals_by[s["id"]]
            op = [r for r in rows if r["agent"] == "research" and r["eval_name"] == "tool_success_rate"]
            if op:
                assert op[0]["passed"] is True

    def test_sessions_are_simulated(self):
        sessions, _ = _build_quality_silent_degradation()
        for s in sessions:
            assert s.get("is_simulated") is True


# ── Detector fires from builder output ────────────────────────────────────────

class TestQualityDetectorFiringFromBuilders:
    """Verify each builder produces data that fires its corresponding detector."""

    def test_grounding_failure_fires(self):
        from engine.pattern_detector import detect_grounding_failure, run_quality_detectors
        sessions, evals_by = _build_quality_grounding_failure()
        inc = detect_grounding_failure(sessions[-1], sessions, evals_by)
        assert inc is not None
        assert "Grounding Failure" in inc.pattern_name

    def test_coherence_break_fires(self):
        from engine.pattern_detector import detect_coherence_break
        sessions, evals_by = _build_quality_coherence_break()
        inc = detect_coherence_break(sessions[-1], sessions, evals_by)
        assert inc is not None
        assert "Coherence Break" in inc.pattern_name

    def test_quality_cascade_fires(self):
        from engine.pattern_detector import detect_quality_cascade
        sessions, evals_by = _build_quality_cascade()
        inc = detect_quality_cascade(sessions[-1], sessions, evals_by)
        assert inc is not None
        assert "Quality Cascade" in inc.pattern_name

    def test_silent_degradation_fires(self):
        from engine.pattern_detector import detect_silent_degradation
        sessions, evals_by = _build_quality_silent_degradation()
        inc = detect_silent_degradation(sessions[-1], sessions, evals_by)
        assert inc is not None
        assert "Silent Degradation" in inc.pattern_name


# ── simulate_quality_failure dry-run ──────────────────────────────────────────

class TestSimulateQualityFailure:
    def test_grounding_failure_dry_run(self):
        sids, incs = simulate_quality_failure("grounding_failure", db=None)
        assert len(sids) == 3
        assert all(isinstance(s, str) for s in sids)
        assert incs == []  # dry-run returns no incidents

    def test_coherence_break_dry_run(self):
        sids, incs = simulate_quality_failure("coherence_break", db=None)
        assert len(sids) == 1

    def test_quality_cascade_dry_run(self):
        sids, incs = simulate_quality_failure("quality_cascade", db=None)
        assert len(sids) == 5

    def test_silent_degradation_dry_run(self):
        sids, incs = simulate_quality_failure("silent_degradation", db=None)
        assert len(sids) == 5

    def test_unknown_pattern_raises(self):
        with pytest.raises(ValueError, match="Unknown quality pattern"):
            simulate_quality_failure("made_up_pattern", db=None)

    def test_returns_unique_ids(self):
        sids, _ = simulate_quality_failure("quality_cascade", db=None)
        assert len(set(sids)) == len(sids)


# ── Metadata completeness ──────────────────────────────────────────────────────

class TestSimulatorMetadata:
    def test_all_quality_patterns_have_builder(self):
        for p in QUALITY_PATTERNS:
            assert p in _QUALITY_BUILDERS

    def test_all_quality_patterns_have_description(self):
        for p in QUALITY_PATTERNS:
            assert p in _QUALITY_DESCRIPTIONS
            assert len(_QUALITY_DESCRIPTIONS[p]) > 20

    def test_no_overlap_between_pattern_lists(self):
        assert not set(PATTERNS) & set(QUALITY_PATTERNS)
