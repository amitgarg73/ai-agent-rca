"""Supabase connection helper shared across engine, simulator, and dashboard."""
import os
from functools import lru_cache
from supabase import create_client, Client


@lru_cache(maxsize=1)
def get_db() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_KEY env vars, "
            "or copy observability/poc/.streamlit/secrets.toml to "
            "dashboard/.streamlit/secrets.toml"
        )
    return create_client(url, key)


def load_session_traces(db: Client, session_id: str) -> list[dict]:
    r = db.table("c_traces").select("*").eq("session_id", session_id).order("created_at").execute()
    return r.data or []


def load_recent_session_costs(db: Client, limit: int = 30) -> list[float]:
    r = (
        db.table("c_sessions")
        .select("total_cost_usd")
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [float(row["total_cost_usd"] or 0) for row in (r.data or [])]


def write_eval(db: Client, row: dict) -> None:
    db.table("c_evals").insert(row).execute()


def write_incident(db: Client, row: dict) -> None:
    db.table("c_incidents").insert(row).execute()


def write_trace(db: Client, row: dict) -> None:
    db.table("c_traces").insert(row).execute()


def write_session(db: Client, row: dict) -> str:
    r = db.table("c_sessions").insert(row).execute()
    return r.data[0]["id"]
