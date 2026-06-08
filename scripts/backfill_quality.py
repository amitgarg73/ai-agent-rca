"""
Backfill quality evals for all sessions in c_sessions.
Deletes existing quality evals (agent LIKE '%_quality') per session, then re-inserts.

Usage:
    python3 scripts/backfill_quality.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
from supabase import create_client

from engine.quality_judge import judge_session


def load_secrets():
    path = os.path.join(ROOT, "dashboard", ".streamlit", "secrets.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run judge but do not write to DB")
    args = parser.parse_args()

    secrets = load_secrets()
    db      = create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])

    print("Loading sessions...")
    sessions = db.table("c_sessions").select("*").eq("is_simulated", False).order("started_at").execute().data or []
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

    traces_by_session: dict[str, list[dict]] = {}
    for t in all_traces:
        sid = t.get("session_id") or ""
        traces_by_session.setdefault(sid, []).append(t)

    total_evals = 0
    low_quality: list[str] = []

    for idx, sess in enumerate(sessions):
        sid    = sess["id"]
        traces = traces_by_session.get(sid, [])
        evals  = judge_session(sess, traces)

        total_evals += len(evals)

        composite_scores = {e.agent: e.score for e in evals if e.eval_name == "composite_score"}
        failing = [f"{a}={s:.2f}" for a, s in composite_scores.items() if s < 0.60]
        if failing:
            low_quality.append(sid[:8])

        summary = (f"[{idx+1:3d}/{len(sessions)}] {sid[:8]}"
                   f"  evals={len(evals)}"
                   f"  composites={composite_scores}")
        if failing:
            summary += f"  LOW: {failing}"
        print(summary)

        if not args.dry_run:
            # Delete existing quality evals for this session
            db.table("c_evals").delete() \
              .eq("session_id", sid) \
              .like("agent", "%_quality") \
              .execute()
            if evals:
                db.table("c_evals").insert([e.to_db_row(sid) for e in evals]).execute()

    print()
    print(f"Done. {total_evals} quality evals across {len(sessions)} sessions.")
    if low_quality:
        print(f"Sessions with low composite quality (<0.60): {low_quality}")
    if args.dry_run:
        print("DRY RUN — nothing written to DB.")


if __name__ == "__main__":
    main()
