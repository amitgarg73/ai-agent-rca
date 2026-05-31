# CLAUDE.md — AI Agent RCA

## What This Is

A product prototype: real-time failure detection, semantic quality evaluation, and root cause
analysis for multi-agent AI systems. Born from the AI Agent Reliability observability initiative.

## The Problem

AI agents fail silently. HTTP 200, no exception, but $1.98 burned and 0 trades produced.
Current tools show traces (what happened). None tell you why, which step caused it, or what
it cost. Quality degrades quietly before operational metrics move. Evals are disconnected from
execution. Every incident is debugged from scratch.

## What This Builds

Four layers working together:
1. **Operational evals** — 17 rule-based evals (<1ms) across 4 agents + session holistic
2. **Semantic quality evals** — LLM-as-judge per agent per dimension, async post-session
3. **Pattern detector** — 10 named patterns (6 operational + 4 quality) + Isolation Forest
4. **Dashboard** — 6-page Streamlit app with Quality Drift, Incidents Feed, RCA View

## Full Design

See `design/product-design.md` — single source of truth covering all phases, rubrics,
circuit breaker config, LLM judge architecture, pattern library, and product positioning.

## Test Bed

Strategy C — 6-agent live trading system (trading-agent-c Supabase project).
27+ real sessions, 900+ traces. Real data from day one.

## Supabase

Same project as trading-agent-c. Credentials in `.streamlit/secrets.toml` (do not commit).
Tables used: `c_sessions`, `c_traces`, `c_positions` (existing), `c_evals`, `c_incidents` (added).

## Folder Structure

```
engine/
  eval_engine.py      # 17 evals: 3 registries (PER_AGENT, SESSION, BUSINESS)
  pattern_detector.py # 6 operational patterns; quality patterns in Phase Q3
  rca_engine.py       # call stack builder + fix suggestion writer
  quality_judge.py    # [Phase Q1] LLM-as-judge, Haiku, per-agent quality scoring
  anomaly_detector.py # [Phase D1] Isolation Forest, unknown pattern detection
simulator/
  failure_sim.py      # inject synthetic traces for 6 named failure patterns
dashboard/
  dashboard.py        # 6-page Streamlit app, password protected
  .streamlit/
    secrets.toml      # NOT committed
design/
  product-design.md   # single source of truth — all phases, rubrics, arch
  AI_Agent_RCA_Prototype_Design.docx  # Word version (external sharing)
scripts/
  backfill_evals.py   # re-run operational evals on historical sessions
  backfill_quality.py # [Phase Q1] run quality judge on historical sessions
tests/
  test_evals.py       # 84 tests for all 17 evals
  test_detector.py    # pattern detection rules
  test_simulator.py   # synthetic trace injection
```

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| 1-3 | Data foundation, RCA engine, dashboard (6 pages) | DONE |
| Q1 | LLM quality judge + backfill | NOT STARTED |
| Q2 | Quality dashboard integration | NOT STARTED |
| Q3 | Quality patterns (Grounding Failure, Coherence Break, etc.) | NOT STARTED |
| Q4 | Shadow quality circuit breakers | NOT STARTED |
| 4a-4c | In-process operational CBs + realtime subscription | NOT STARTED |
| D1 | Isolation Forest anomaly detection (needs 20+ sessions) | NOT STARTED |
| D2 | Sequence pattern mining (needs 50+ sessions) | NOT STARTED |

## Key Design Decisions

- Operational evals: pure Python, <1ms, zero LLM cost, run synchronously in backfill + on-demand
- Quality evals: Haiku, ~$0.002/session, post-session async only, never in agent critical path
- Business evals: computed on-the-fly in dashboard, not stored in c_evals
- Circuit breakers: shadow mode while Strategy C is in testing; flip one flag for real mode
- In-process CBs: use in-memory traces — zero DB round trips, ~0.5ms compute
- Isolation Forest: per-customer models at productization; single model for Strategy C now
- Pattern detection: rules first (known), Isolation Forest (unknown), sequence mining at 50+

## Relationship to Other Projects

- `observability/` — research docs, market analysis, Outcome Ledger POC. Separate concern.
- `trading-agent-c/` — the test bed. Shares Supabase. Do not modify its schema.
- This project eventually becomes a standalone product / open-source SDK.

## What NOT to Do

- Do not commit secrets.toml
- Do not modify c_sessions, c_traces, c_positions schema in trading-agent-c
- Do not run quality judge in the agent pipeline critical path
- Do not skip tests — all evals and detectors must have passing tests before commit
- Do not enable real circuit breakers while Strategy C is still in testing

## CLI

```bash
streamlit run dashboard/dashboard.py          # local dev
python3 scripts/backfill_evals.py             # re-run operational evals
python3 scripts/backfill_evals.py --dry-run   # preview only
python3 -m pytest tests/ -v                   # all tests (84 passing)
```
