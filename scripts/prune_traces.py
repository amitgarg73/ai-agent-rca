"""
Prune raw payload columns from old traces to save space.

Retention policy:
  tool_input + tool_output  — null after 7 days  (no ML/eval use case)
  agent_reasoning           — null after 30 days (needed by Phase Q1 quality judge)

All other trace columns (agent, step_type, tool_name, outcome, entity_id,
tokens_input, tokens_output, latency_ms, error, etc.) are kept indefinitely —
they are required by the eval engine, pattern detector, and sequence miner.

Usage:
    python3 scripts/prune_traces.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
from supabase import create_client


def load_secrets():
    path = os.path.join(ROOT, "dashboard", ".streamlit", "secrets.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


def count_non_null(db, field: str, before_date: str) -> int:
    res = (
        db.table("c_traces")
        .select("id", count="exact")
        .lt("date", before_date)
        .not_.is_(field, "null")
        .execute()
    )
    return res.count or 0


def prune_field(db, field: str, before_date: str, dry_run: bool) -> int:
    count = count_non_null(db, field, before_date)
    if count == 0:
        print(f"  {field}: nothing to prune (already null or no rows before {before_date})")
        return 0
    print(f"  {field}: {count} rows to null (date < {before_date})")
    if not dry_run:
        db.table("c_traces").update({field: None}).lt("date", before_date).execute()
        print(f"    done.")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be pruned without writing")
    args = parser.parse_args()

    secrets = load_secrets()
    db      = create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])

    today         = date.today()
    cutoff_7d     = str(today - timedelta(days=7))
    cutoff_30d    = str(today - timedelta(days=30))

    print(f"Prune policy as of {today}:")
    print(f"  tool_input + tool_output : null where date < {cutoff_7d}  (7-day window)")
    print(f"  agent_reasoning          : null where date < {cutoff_30d} (30-day window)")
    if args.dry_run:
        print("  DRY RUN — no writes\n")
    else:
        print()

    total = 0
    total += prune_field(db, "tool_input",      cutoff_7d,  args.dry_run)
    total += prune_field(db, "tool_output",     cutoff_7d,  args.dry_run)
    total += prune_field(db, "agent_reasoning", cutoff_30d, args.dry_run)

    print(f"\nTotal rows affected: {total}")
    if args.dry_run:
        print("DRY RUN — nothing written.")


if __name__ == "__main__":
    main()
