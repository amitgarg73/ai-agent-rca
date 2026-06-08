"""
Backfill evals + incidents for all sessions in c_sessions.
Deletes existing c_evals + c_incidents rows first, then re-inserts.

Usage:
    python3 scripts/backfill_evals.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import os

# Resolve project root regardless of where the script is invoked from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
from supabase import create_client

from engine.eval_engine       import run_all_evals
from engine.pattern_detector  import run_all_detectors, Incident
from engine.rca_engine        import build_annotated_call_stack, generate_fix_suggestion


def load_secrets():
    path = os.path.join(ROOT, "dashboard", ".streamlit", "secrets.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_analysis(session: dict, traces: list[dict], recent_costs: list[float]):
    evals     = run_all_evals(session, traces, recent_costs)
    incidents = run_all_detectors(session, traces, evals, recent_costs)
    # Annotate each incident with call_stack + fix_suggestion if not already set
    for inc in incidents:
        if not inc.call_stack:
            inc.call_stack = build_annotated_call_stack(session, traces, evals)
        if not inc.fix_suggestion:
            inc.fix_suggestion = generate_fix_suggestion(inc, traces)
    return evals, incidents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run evals but do not write to DB")
    args = parser.parse_args()

    secrets = load_secrets()
    db      = create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])

    print("Loading sessions...")
    sessions_resp = db.table("c_sessions").select("*").eq("is_simulated", False).order("started_at").execute()
    sessions      = sessions_resp.data or []
    print(f"  {len(sessions)} sessions found")

    print("Loading traces...")
    all_traces: list[dict] = []
    page_size = 1000
    offset    = 0
    while True:
        batch = db.table("c_traces").select("*").range(offset, offset + page_size - 1).execute()
        rows  = batch.data or []
        all_traces.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    print(f"  {len(all_traces)} traces found")

    # Index traces by session_id
    traces_by_session: dict[str, list[dict]] = {}
    for t in all_traces:
        sid = t.get("session_id") or ""
        traces_by_session.setdefault(sid, []).append(t)

    all_costs = [float(s.get("total_cost_usd") or 0) for s in sessions]

    total_evals     = 0
    total_incidents = 0
    incident_sessions: list[str] = []

    for idx, sess in enumerate(sessions):
        sid    = sess["id"]
        traces = traces_by_session.get(sid, [])
        evals, incidents = run_analysis(sess, traces, all_costs)

        total_evals     += len(evals)
        total_incidents += len(incidents)
        if incidents:
            incident_sessions.append(sid[:8])

        summary = f"[{idx+1:3d}/{len(sessions)}] {sid[:8]}  evals={len(evals)}  incidents={len(incidents)}"
        if incidents:
            names = ", ".join(i.pattern_name for i in incidents)
            summary += f"  [{names}]"
        print(summary)

        if not args.dry_run:
            db.table("c_evals").delete().eq("session_id", sid).execute()
            db.table("c_incidents").delete().eq("session_id", sid).execute()
            if evals:
                db.table("c_evals").insert([e.to_db_row(sid) for e in evals]).execute()
            if incidents:
                db.table("c_incidents").insert([i.to_db_row() for i in incidents]).execute()

    print()
    print(f"Done. {total_evals} evals, {total_incidents} incidents across {len(sessions)} sessions.")
    if incident_sessions:
        print(f"Sessions with incidents: {incident_sessions}")
    if args.dry_run:
        print("DRY RUN — nothing written to DB.")


if __name__ == "__main__":
    main()
