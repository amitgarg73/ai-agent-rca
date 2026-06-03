"""
Isolation Forest anomaly detector — Phase D1.

Learns what "normal" looks like from real session data.
Scores each new session against that baseline.
Anomaly score > threshold fires an "Isolation Forest Anomaly" incident — no named pattern needed.

Design decisions:
  - Per-pipeline model (single model for Strategy C; multi-tenant = per-customer models)
  - Trained on first N sessions with evals; retrained incrementally as sessions accumulate
  - Model persists to disk (pickle) — survives dashboard/monitor restarts
  - Threshold 0.65: tuned to ~10% false positive rate on clean sessions
  - Never runs in the agent pipeline — batch only

Data requirement: 20+ sessions with eval scores populated (D1 gate).

Run standalone:
    python3 -m engine.anomaly_detector --fit     # train on current sessions
    python3 -m engine.anomaly_detector --score   # score all sessions, print report
    python3 -m engine.anomaly_detector --fit --score   # train then score
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH        = os.path.join(os.path.dirname(__file__), "if_model.pkl")
ANOMALY_THRESHOLD = 0.65   # score above this → flag as unknown anomaly
MIN_SESSIONS      = 20     # minimum sessions before model is meaningful
N_ESTIMATORS      = 100
CONTAMINATION     = 0.10   # expected fraction of anomalous sessions in training data

FEATURE_NAMES = [
    "total_cost_usd",
    "total_tokens_input",
    "total_tokens_output",
    "total_latency_ms",
    "trades_executed",
    "agent_count",
    "op_pass_rate",      # fraction of operational evals that passed
    "tool_error_rate",   # fraction of tool_call traces that errored
]


def _extract_features(sessions: list[dict], evals_by_session: dict[str, list] | None = None,
                       traces_by_session: dict[str, list] | None = None) -> np.ndarray:
    rows = []
    for s in sessions:
        sid = s.get("id", "")

        # Eval pass rate — operational evals only (exclude business evals)
        op_pass_rate = 0.5  # default if no evals
        if evals_by_session and sid in evals_by_session:
            op_evals = [e for e in evals_by_session[sid]
                        if e.get("agent") not in ("business",)]
            if op_evals:
                op_pass_rate = sum(1 for e in op_evals if e.get("passed")) / len(op_evals)

        # Tool error rate from traces
        tool_error_rate = 0.0
        if traces_by_session and sid in traces_by_session:
            tool_traces = [t for t in traces_by_session[sid]
                           if t.get("step_type") == "tool_call"]
            if tool_traces:
                tool_error_rate = sum(
                    1 for t in tool_traces if t.get("outcome") == "error"
                ) / len(tool_traces)

        rows.append([
            float(s.get("total_cost_usd")      or 0),
            float(s.get("total_tokens_input")   or 0),
            float(s.get("total_tokens_output")  or 0),
            float(s.get("total_latency_ms")     or 0),
            float(s.get("trades_executed")      or 0),
            float(len(s.get("agents_invoked")   or [])),
            float(op_pass_rate),
            float(tool_error_rate),
        ])
    return np.array(rows, dtype=float) if rows else np.zeros((0, len(FEATURE_NAMES)))


class IsolationForestDetector:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._trained_on = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, sessions: list[dict],
            evals_by_session: dict[str, list] | None = None,
            traces_by_session: dict[str, list] | None = None) -> bool:
        """
        Train on session data. Returns True if model was updated.
        Requires at least MIN_SESSIONS rows.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        if len(sessions) < MIN_SESSIONS:
            log.warning("Need %d sessions to train; have %d — skipping fit",
                        MIN_SESSIONS, len(sessions))
            return False

        X = _extract_features(sessions, evals_by_session, traces_by_session)
        if X.shape[0] < MIN_SESSIONS:
            return False

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled)

        # Store training decision_function range for score normalization
        df_train = model.decision_function(X_scaled)
        # _df_scale: use p10 of training decisions as the "most anomalous in training"
        # Sessions with df < 0 are outliers; we scale -df by abs(p10) so that
        # sessions as anomalous as training outliers get a score ~0.7
        df_p10 = float(np.percentile(df_train, 10))
        self._df_scale = max(abs(df_p10), 0.01)

        self._model      = (scaler, model)
        self._trained_on = len(sessions)
        self._save()
        log.info("Isolation Forest trained on %d sessions", len(sessions))
        return True

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, session: dict,
              evals: list | None = None,
              traces: list | None = None) -> float:
        """
        Returns anomaly score in [0, 1]. Higher = more anomalous.
        Returns 0.0 if model not loaded.
        """
        if self._model is None:
            self._load()
        if self._model is None:
            return 0.0

        scaler, model = self._model
        evals_map  = {session.get("id", ""): evals}  if evals  else None
        traces_map = {session.get("id", ""): traces} if traces else None

        X = _extract_features([session], evals_map, traces_map)
        if X.shape[0] == 0:
            return 0.0

        try:
            X_scaled = scaler.transform(X)
            # decision_function: positive = inlier, negative = outlier
            # More negative = more anomalous.
            df = float(model.decision_function(X_scaled)[0])
            # Scale so training-level outliers score ~0.7; beyond that → approaching 1.0
            df_scale = getattr(self, "_df_scale", 0.05)
            score = max(0.0, min(1.0, -df / df_scale * 0.7))
            return round(score, 3)
        except Exception as exc:
            log.debug("Scoring error: %s", exc)
            return 0.0

    def detect(self, session: dict,
               evals: list | None = None,
               traces: list | None = None,
               threshold: float = ANOMALY_THRESHOLD):
        """
        Run anomaly detection on a session.
        Returns an Incident if anomalous, None otherwise.
        Import is deferred to avoid circular imports.
        """
        from engine.pattern_detector import Incident, _failed_eval_rows
        from engine.eval_engine import EvalResult

        score = self.score(session, evals, traces)
        if score < threshold:
            return None

        evals_typed = [e for e in (evals or []) if isinstance(e, EvalResult)]
        return Incident(
            session_id    = session.get("id", ""),
            pattern_name  = "Isolation Forest Anomaly",
            severity      = "warning",
            root_cause    = (
                f"Session anomaly score {score:.2f} exceeds threshold {threshold}. "
                f"No named pattern matches — statistical outlier vs. {self._trained_on} "
                f"baseline sessions. Manual review recommended."
            ),
            call_stack    = [],
            failed_evals  = _failed_eval_rows(evals_typed),
            cost_wasted   = float(session.get("total_cost_usd") or 0),
            tokens_wasted = 0,
            fix_suggestion = (
                "Review session traces manually. "
                "If a recurring failure mode is found, consider adding it as a named pattern. "
                "If this is a legitimate edge case, no action needed."
            ),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "model":      self._model,
                    "trained_on": self._trained_on,
                    "df_scale":   getattr(self, "_df_scale", 0.05),
                }, f)
        except Exception as exc:
            log.warning("Could not save IF model: %s", exc)

    def _load(self) -> bool:
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self._model      = data["model"]
                self._trained_on = data.get("trained_on", 0)
                self._df_scale   = data.get("df_scale", 0.05)
                log.info("Loaded IF model (trained on %d sessions)", self._trained_on)
                return True
        except Exception as exc:
            log.warning("Could not load IF model: %s", exc)
        return False

    @property
    def is_ready(self) -> bool:
        if self._model is None:
            self._load()
        return self._model is not None


# ── Standalone runner ─────────────────────────────────────────────────────────

def _load_training_data(db):
    """Load all sessions + their evals and traces from Supabase."""
    sessions_r = db.table("c_sessions").select("*").execute()
    sessions = sessions_r.data or []

    evals_r   = db.table("c_evals").select("*").execute()
    evals_raw = evals_r.data or []

    # Group evals by session_id
    evals_by_session: dict[str, list] = {}
    for e in evals_raw:
        sid = e.get("session_id", "")
        evals_by_session.setdefault(sid, []).append(e)

    return sessions, evals_by_session


def _run_fit(db):
    sessions, evals_by_session = _load_training_data(db)
    detector = IsolationForestDetector()
    ok = detector.fit(sessions, evals_by_session)
    if ok:
        print(f"Model trained on {len(sessions)} sessions → {MODEL_PATH}")
    else:
        print(f"Need {MIN_SESSIONS} sessions; have {len(sessions)} — skipping")
    return detector


def _run_score(db, detector: IsolationForestDetector | None = None, persist: bool = True):
    if detector is None:
        detector = IsolationForestDetector()
    if not detector.is_ready:
        print("No model found — run with --fit first")
        return

    sessions_r = db.table("c_sessions").select("*").order("started_at", desc=True).execute()
    sessions   = sessions_r.data or []
    evals_r    = db.table("c_evals").select("*").execute()
    evals_by: dict[str, list] = {}
    for e in (evals_r.data or []):
        evals_by.setdefault(e["session_id"], []).append(e)

    # Clear existing Isolation Forest Anomaly incidents so re-runs stay idempotent
    if persist:
        db.table("c_incidents").delete().eq("pattern_name", "Isolation Forest Anomaly").execute()

    print(f"\n{'Session':<38} {'Score':>6} {'Flag'}")
    print("-" * 55)
    flagged = 0
    for s in sessions:
        sid      = s["id"]
        incident = detector.detect(s, evals=evals_by.get(sid, []))
        score    = detector.score(s, evals=evals_by.get(sid, []))
        flag     = "ANOMALY" if incident else ""
        if incident:
            flagged += 1
            if persist:
                try:
                    db.table("c_incidents").insert(incident.to_db_row()).execute()
                except Exception as exc:
                    print(f"  [warn] could not write incident for {sid[:8]}: {exc}")
        print(f"{sid:<38} {score:>6.3f} {flag}")

    print(f"\n{flagged}/{len(sessions)} sessions flagged as anomalous "
          f"(threshold={ANOMALY_THRESHOLD})")
    if persist and flagged:
        print(f"{flagged} Isolation Forest Anomaly incidents written to c_incidents.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit",      action="store_true", help="Train model on current sessions")
    parser.add_argument("--score",    action="store_true", help="Score all sessions")
    parser.add_argument("--dry-run",  action="store_true", help="Print scores but do not write incidents")
    args = parser.parse_args()

    if not args.fit and not args.score:
        parser.print_help()
        raise SystemExit(0)

    # Load credentials from secrets.toml so env vars aren't required
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, _root)
    try:
        import tomllib as _tl
    except ImportError:
        import tomli as _tl  # type: ignore[no-redef]
    _secrets_path = os.path.join(_root, "dashboard", ".streamlit", "secrets.toml")
    with open(_secrets_path, "rb") as _f:
        _sec = _tl.load(_f)
    from supabase import create_client as _cc
    db = _cc(_sec["SUPABASE_URL"], _sec["SUPABASE_KEY"])

    detector = None
    if args.fit:
        detector = _run_fit(db)
    if args.score:
        _run_score(db, detector, persist=not args.dry_run)
