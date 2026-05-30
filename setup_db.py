"""
Creates c_evals and c_incidents tables in Strategy C Supabase.

Run once: python3 setup_db.py

If the RPC is not available, the script prints the SQL to run manually in
the Supabase SQL Editor:
  https://supabase.com/dashboard/project/fpuyabfxtrzwciehfetk/sql
"""
import os, sys
from supabase import create_client


def get_client():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open("dashboard/.streamlit/secrets.toml", "rb") as f:
                s = tomllib.load(f)
            url = s["SUPABASE_URL"]
            key = s["SUPABASE_KEY"]
        except Exception:
            print("Set SUPABASE_URL and SUPABASE_KEY env vars, or create dashboard/.streamlit/secrets.toml")
            sys.exit(1)
    return create_client(url, key)


C_EVALS_DDL = """
create table if not exists c_evals (
    id          uuid primary key default gen_random_uuid(),
    session_id  uuid,
    agent       text,
    eval_name   text,
    score       float,
    passed      boolean,
    threshold   float,
    detail      jsonb,
    created_at  timestamptz default now()
);
create index if not exists c_evals_session_idx on c_evals(session_id);
"""

C_INCIDENTS_DDL = """
create table if not exists c_incidents (
    id              uuid primary key default gen_random_uuid(),
    session_id      uuid,
    pattern_name    text,
    severity        text,
    root_cause      text,
    call_stack      jsonb,
    failed_evals    jsonb,
    cost_wasted     float default 0,
    tokens_wasted   int   default 0,
    fix_suggestion  text,
    is_simulated    boolean default false,
    created_at      timestamptz default now()
);
create index if not exists c_incidents_session_idx on c_incidents(session_id);
create index if not exists c_incidents_severity_idx on c_incidents(severity);
"""

COMBINED_SQL = C_EVALS_DDL.strip() + "\n\n" + C_INCIDENTS_DDL.strip()


def print_manual_instructions():
    print("\n" + "="*60)
    print("Run this SQL in the Supabase SQL Editor:")
    print("  https://supabase.com/dashboard/project/fpuyabfxtrzwciehfetk/sql")
    print("="*60)
    print(COMBINED_SQL)
    print("="*60 + "\n")


def run():
    db = get_client()

    # Try exec_sql RPC first
    try:
        for ddl in [C_EVALS_DDL, C_INCIDENTS_DDL]:
            db.rpc("exec_sql", {"sql": ddl}).execute()
        print("Tables created: c_evals, c_incidents")
        return
    except Exception:
        pass

    # Try verifying tables already exist
    missing = []
    for table in ["c_evals", "c_incidents"]:
        try:
            db.table(table).select("id").limit(1).execute()
        except Exception:
            missing.append(table)

    if not missing:
        print("Tables already exist: c_evals, c_incidents")
        return

    print(f"Missing tables: {missing}")
    print("Could not create tables automatically (exec_sql RPC not available).")
    print_manual_instructions()


if __name__ == "__main__":
    run()
