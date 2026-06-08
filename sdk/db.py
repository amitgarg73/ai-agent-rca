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


def load_recent_sessions(db: Client, limit: int = 10, exclude_simulated: bool = True) -> list[dict]:
    """Return up to `limit` most recent sessions in chronological order (oldest first).

    Excludes simulated sessions by default so quality pattern detectors don't
    fire on injected demo data.
    """
    q = db.table("c_sessions").select("*")
    if exclude_simulated:
        q = q.eq("is_simulated", False)
    r = q.order("started_at", desc=True).limit(limit).execute()
    return list(reversed(r.data or []))


def load_evals_by_session(db: Client, session_ids: list[str]) -> dict[str, list]:
    """Return all c_evals rows for the given session IDs, keyed by session_id."""
    if not session_ids:
        return {}
    r = (
        db.table("c_evals")
        .select("*")
        .in_("session_id", session_ids)
        .execute()
    )
    result: dict[str, list] = {}
    for e in (r.data or []):
        result.setdefault(e["session_id"], []).append(e)
    return result


def write_eval(db: Client, row: dict) -> None:
    db.table("c_evals").insert(row).execute()


def write_incident(db: Client, row: dict) -> None:
    db.table("c_incidents").insert(row).execute()


def write_trace(db: Client, row: dict) -> None:
    db.table("c_traces").insert(row).execute()


def write_session(db: Client, row: dict) -> str:
    r = db.table("c_sessions").insert(row).execute()
    return r.data[0]["id"]
