"""
Calibrate eval thresholds from real session data.

Splits sessions into clean (converged + trades + no incidents) vs degraded,
then computes score distributions per eval and recommends thresholds at the
p20 of clean sessions.

Usage:
    python3 scripts/calibrate_thresholds.py
    python3 scripts/calibrate_thresholds.py --percentile 10   # more permissive
    python3 scripts/calibrate_thresholds.py --min-sessions 5  # require more clean sessions
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

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


def percentile(values: list[float], p: int) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (k - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def classify_session(sess: dict, incident_session_ids: set[str]) -> str:
    sid = sess["id"]
    terminal = sess.get("terminal_reason") or ""
    trades = int(sess.get("trades_executed") or 0)

    # Structural non-events: market said skip or no opportunity
    if terminal in ("skip_propagated", "no_opportunity", "market_closed"):
        return "structural"

    # Still running or errored — exclude
    if terminal in ("in_progress", "error", ""):
        return "excluded"

    has_incident = sid in incident_session_ids

    # Clean: converged with trades, no incidents
    if terminal == "converged" and trades > 0 and not has_incident:
        return "clean"

    # Degraded: has incidents, or converged with 0 trades unexpectedly
    if has_incident:
        return "degraded"

    if terminal == "converged" and trades == 0:
        return "degraded"

    # structural_block with no incidents: pipeline worked, just nothing to trade
    if terminal == "structural_block" and not has_incident:
        return "structural"

    return "degraded"


def compute_tpr_fpr(clean: list[float], degraded: list[float], threshold: float):
    """True positive rate (degraded below threshold) and false positive rate (clean below threshold)."""
    tpr = sum(1 for v in degraded if v < threshold) / len(degraded) if degraded else float("nan")
    fpr = sum(1 for v in clean if v < threshold) / len(clean) if clean else float("nan")
    return tpr, fpr


CURRENT_THRESHOLDS = {
    "data_completeness":      0.70,
    "data_freshness":         0.50,
    "completion":             0.50,
    "token_efficiency":       0.50,
    "tool_success_rate":      0.50,
    "tool_diversity":         0.50,
    "assessment_complete":    0.50,
    "within_parameters":      0.50,
    "decision_made":          0.50,
    "exit_quality":           0.50,
    "pipeline_completion":    0.50,
    "cost_anomaly":           0.50,
    "outcome_linkage":        0.50,
    "tokens_per_decision":    0.50,
    "cost_per_trade":         0.50,
    "research_conversion":    0.50,
    "proposal_acceptance":    0.50,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile", type=int, default=20,
                        help="Percentile of clean sessions to use as threshold (default: 20)")
    parser.add_argument("--min-sessions", type=int, default=3,
                        help="Minimum clean sessions required to recommend a threshold (default: 3)")
    args = parser.parse_args()

    secrets = load_secrets()
    db = create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])

    print("Loading sessions...")
    sessions = db.table("c_sessions").select("*").order("started_at").execute().data or []
    print(f"  {len(sessions)} sessions")

    print("Loading incidents...")
    incidents = db.table("c_incidents").select("session_id").execute().data or []
    incident_sids = {r["session_id"] for r in incidents}

    print("Loading evals...")
    evals = db.table("c_evals").select("session_id,eval_name,score").execute().data or []
    print(f"  {len(evals)} eval rows")

    # Classify sessions
    labels: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    for sess in sessions:
        label = classify_session(sess, incident_sids)
        labels[sess["id"]] = label
        counts[label] += 1

    print(f"\nSession breakdown:")
    print(f"  clean      : {counts['clean']}")
    print(f"  degraded   : {counts['degraded']}")
    print(f"  structural : {counts['structural']}")
    print(f"  excluded   : {counts['excluded']}")

    if counts["clean"] < args.min_sessions:
        print(f"\nNot enough clean sessions ({counts['clean']} < {args.min_sessions}). "
              "Run more sessions before calibrating.")
        return

    # Bucket eval scores by label
    clean_scores:    dict[str, list[float]] = defaultdict(list)
    degraded_scores: dict[str, list[float]] = defaultdict(list)

    for row in evals:
        sid = row["session_id"]
        label = labels.get(sid)
        if label == "clean":
            clean_scores[row["eval_name"]].append(float(row["score"]))
        elif label == "degraded":
            degraded_scores[row["eval_name"]].append(float(row["score"]))

    # Build recommendations
    all_evals = sorted(set(list(clean_scores.keys()) + list(degraded_scores.keys())))

    col_w = [28, 8, 8, 10, 10, 10, 9, 9, 8]
    header = (
        f"{'eval':<{col_w[0]}} {'c_p20':>{col_w[1]}} {'c_mean':>{col_w[2]}} "
        f"{'d_mean':>{col_w[3]}} {'gap':>{col_w[4]}} {'current':>{col_w[5]}} "
        f"{'rec':>{col_w[6]}} {'TPR':>{col_w[7]}} {'FPR':>{col_w[8]}}"
    )
    divider = "-" * len(header)

    print(f"\n{'Threshold Calibration':^{len(header)}}")
    print(f"(p{args.percentile} of clean sessions | TPR=degraded below threshold | FPR=clean below threshold)")
    print(divider)
    print(header)
    print(divider)

    changed: list[str] = []
    insufficient: list[str] = []

    for eval_name in all_evals:
        clean = clean_scores[eval_name]
        degraded = degraded_scores[eval_name]
        current = CURRENT_THRESHOLDS.get(eval_name, 0.50)

        if len(clean) < args.min_sessions:
            insufficient.append(eval_name)
            print(f"  {eval_name:<{col_w[0]-2}} -- insufficient clean data ({len(clean)} sessions)")
            continue

        rec = percentile(clean, args.percentile)
        c_mean = mean(clean)
        d_mean = mean(degraded) if degraded else float("nan")
        gap = c_mean - d_mean if degraded else float("nan")

        tpr, fpr = compute_tpr_fpr(clean, degraded, rec) if degraded else (float("nan"), float("nan"))

        flag = ""
        if abs(rec - current) > 0.05:
            flag = " <"
            changed.append(eval_name)

        nan_fmt = lambda v: f"{v:.2f}" if v == v else "  n/a"  # noqa: E731

        print(
            f"  {eval_name:<{col_w[0]-2}} "
            f"{rec:>{col_w[1]}.2f} "
            f"{c_mean:>{col_w[2]}.2f} "
            f"{nan_fmt(d_mean):>{col_w[3]}} "
            f"{nan_fmt(gap):>{col_w[4]}} "
            f"{current:>{col_w[5]}.2f} "
            f"{rec:>{col_w[6]}.2f}"
            f"  {nan_fmt(tpr):>6}"
            f"  {nan_fmt(fpr):>5}"
            f"{flag}"
        )

    print(divider)

    if changed:
        print(f"\n{len(changed)} threshold(s) differ from current by >0.05 (marked <):")
        for name in changed:
            clean = clean_scores[name]
            rec = percentile(clean, args.percentile)
            current = CURRENT_THRESHOLDS.get(name, 0.50)
            direction = "raise" if rec > current else "lower"
            print(f"  {name}: {current:.2f} -> {rec:.2f}  ({direction})")

    if insufficient:
        print(f"\n{len(insufficient)} eval(s) skipped (< {args.min_sessions} clean sessions):")
        print(f"  {', '.join(insufficient)}")

    print(f"\nNote: recommendations based on p{args.percentile} of {counts['clean']} clean sessions.")
    print("To apply: update constants in engine/eval_engine.py and re-run backfill_evals.py.")


if __name__ == "__main__":
    main()
