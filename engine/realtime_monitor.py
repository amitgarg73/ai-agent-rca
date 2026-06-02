"""
Realtime monitor — listens for new sessions and fires evals + pattern detection
without touching the trading-agent-c pipeline.

Architecture:
  trading-agent-c writes c_sessions/c_traces as it always has.
  This process subscribes externally and reacts — the agent is unaware.

Two modes:
  1. Supabase Realtime (websocket) — ~150ms reaction time after session insert
  2. Polling fallback — polls every POLL_INTERVAL_S seconds (safer, no websocket)

Shadow CBs: logs "would_trigger_cb" to c_incidents. Never aborts anything.

Run standalone:
    python3 -m engine.realtime_monitor           # realtime mode
    python3 -m engine.realtime_monitor --poll    # polling mode
    python3 -m engine.realtime_monitor --dry-run # no DB writes
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [monitor] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

POLL_INTERVAL_S   = 30    # polling fallback: check every 30s
LOOKBACK_MINUTES  = 5     # on each poll, evaluate sessions completed in the last N minutes
CB_INCIDENT_LABEL = "Shadow CB"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _minutes_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=n)).isoformat()


# ── Core evaluation pipeline ───────────────────────────────────────────────────

def evaluate_session(session: dict, db=None, dry_run: bool = False) -> dict:
    """
    Run evals + pattern detection on a single session.
    Writes to c_evals and c_incidents (shadow CB entries included).
    Returns summary dict. Never raises — all errors are caught and logged.
    """
    sid = session.get("id", "unknown")
    result = {"session_id": sid, "evals": 0, "incidents": 0, "shadow_cb_fires": 0,
              "error": None}
    try:
        from sdk.db import load_session_traces, load_recent_session_costs
        from engine.eval_engine import run_all_evals
        from engine.pattern_detector import run_all_detectors, compute_shadow_cb_fires
        from engine.quality_judge import judge_session

        traces = []
        recent_costs: list[float] = []

        if db:
            traces       = load_session_traces(db, sid)
            recent_costs = load_recent_session_costs(db, limit=30)

        evals         = run_all_evals(session, traces, recent_costs)
        quality_evals = judge_session(session, traces)
        incidents     = run_all_detectors(session, traces, evals, recent_costs)
        cb_fires      = compute_shadow_cb_fires(evals)

        result["evals"]            = len(evals)
        result["quality_evals"]    = len(quality_evals)
        result["incidents"]        = len(incidents)
        result["shadow_cb_fires"]  = len(cb_fires)

        if not dry_run and db:
            # Write operational + quality evals together
            all_eval_rows = [e.to_db_row(sid) for e in evals + quality_evals]
            if all_eval_rows:
                db.table("c_evals").upsert(
                    all_eval_rows,
                    on_conflict="session_id,agent,eval_name"
                ).execute()

            # Write pattern incidents
            if incidents:
                db.table("c_incidents").insert(
                    [i.to_db_row() for i in incidents]
                ).execute()

            # Write shadow CB fire records as info-severity incidents
            for fire in cb_fires:
                row = {
                    "session_id":    sid,
                    "pattern_name":  f"{CB_INCIDENT_LABEL}: {fire['eval_name']}",
                    "severity":      "info",
                    "root_cause":    (
                        f"{fire['agent']}.{fire['eval_name']} scored {fire['score']:.3f} "
                        f"(threshold {fire['threshold']}). CB would have fired here in real mode."
                    ),
                    "call_stack":    [],
                    "failed_evals":  [fire],
                    "cost_wasted":   0.0,
                    "tokens_wasted": 0,
                    "fix_suggestion": (
                        f"Enable real circuit breaker for {fire['agent']}.{fire['eval_name']} "
                        f"when Strategy C testing is complete."
                    ),
                    "is_simulated":  bool(session.get("is_simulated", False)),
                }
                import uuid
                row["id"] = str(uuid.uuid4())
                db.table("c_incidents").insert(row).execute()

        log.info(
            "session=%s evals=%d quality=%d incidents=%d shadow_cb=%d%s",
            sid[:8], result["evals"], result["quality_evals"],
            result["incidents"], result["shadow_cb_fires"],
            " [DRY RUN]" if dry_run else "",
        )

    except Exception as exc:
        result["error"] = str(exc)
        log.error("session=%s error=%s", sid[:8], exc)

    return result


# ── Polling mode ───────────────────────────────────────────────────────────────

def _already_evaluated(db, session_id: str) -> bool:
    """Return True if c_evals already has rows for this session."""
    try:
        r = (db.table("c_evals")
               .select("id", count="exact")
               .eq("session_id", session_id)
               .limit(1)
               .execute())
        return (r.count or 0) > 0
    except Exception:
        return False


def run_polling(db=None, dry_run: bool = False, interval_s: int = POLL_INTERVAL_S):
    log.info("Starting polling monitor (interval=%ds, dry_run=%s)", interval_s, dry_run)
    seen: set[str] = set()

    while True:
        try:
            cutoff = _minutes_ago(LOOKBACK_MINUTES)
            q = db.table("c_sessions").select("*").gte("completed_at", cutoff)
            rows = q.execute().data or []

            new = [r for r in rows if r["id"] not in seen]
            for session in new:
                sid = session["id"]
                if not _already_evaluated(db, sid):
                    evaluate_session(session, db=db, dry_run=dry_run)
                seen.add(sid)

            if new:
                log.info("Polled %d new session(s) in last %dm", len(new), LOOKBACK_MINUTES)

        except Exception as exc:
            log.error("Poll cycle error: %s", exc)

        time.sleep(interval_s)


# ── Realtime mode ──────────────────────────────────────────────────────────────

def run_realtime(db=None, dry_run: bool = False):
    """
    Subscribe to c_sessions INSERT events via Supabase Realtime.
    Falls back to polling if the websocket cannot be established.
    """
    log.info("Starting realtime monitor (dry_run=%s)", dry_run)

    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

        rt_client = create_client(url, key)

        def _on_insert(payload):
            session = payload.get("record") or {}
            if session.get("id"):
                evaluate_session(session, db=db, dry_run=dry_run)

        channel = (
            rt_client
            .channel("session-inserts")
            .on_postgres_changes(
                event="INSERT",
                schema="public",
                table="c_sessions",
                callback=_on_insert,
            )
            .subscribe()
        )

        log.info("Realtime subscription active — waiting for session inserts...")

        # Keep alive in main thread
        while True:
            time.sleep(60)

    except Exception as exc:
        log.warning("Realtime setup failed (%s) — falling back to polling", exc)
        run_polling(db=db, dry_run=dry_run)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent RCA realtime monitor")
    parser.add_argument("--poll",    action="store_true", help="Use polling instead of realtime")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without writing to DB")
    parser.add_argument("--interval",type=int, default=POLL_INTERVAL_S,
                        help=f"Polling interval in seconds (default {POLL_INTERVAL_S})")
    args = parser.parse_args()

    db = None
    if not args.dry_run:
        try:
            from sdk.db import get_db
            db = get_db()
            log.info("Connected to Supabase")
        except Exception as e:
            log.error("Cannot connect to DB: %s", e)
            sys.exit(1)

    if args.poll:
        run_polling(db=db, dry_run=args.dry_run, interval_s=args.interval)
    else:
        run_realtime(db=db, dry_run=args.dry_run)
