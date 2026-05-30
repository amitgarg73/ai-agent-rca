# CLAUDE.md — AI Agent RCA

## What This Is

A product prototype: real-time failure detection, eval linkage, and root cause analysis
for multi-agent AI systems. Born from the AI Agent Reliability observability initiative.

## The Problem

AI agents fail silently. HTTP 200, no exception, but $1.98 burned and 0 trades produced.
Current tools show traces (what happened). None tell you why, which step caused it,
or what it cost. Evals are disconnected from execution. Every incident is debugged from scratch.

## What This Builds

Three things working together:
1. SDK — @observe decorator instruments any agent, any framework
2. Engine — evals run per-agent + holistically; pattern detector names failures; RCA walks the call stack
3. Dashboard — Incidents Feed + Interactive RCA View on top of existing Outcome Ledger

## Test Bed

Strategy C — 6-agent live trading system (trading-agent-c Supabase project).
22 real sessions, 883 traces, 5 known incidents. Real data from day one.

## Supabase

Same project as trading-agent-c. Credentials in .streamlit/secrets.toml (copy from
observability/poc/.streamlit/secrets.toml — do not commit).

New tables to create: c_evals, c_incidents (DDL in design/prototype-design.md).
Existing tables unchanged: c_sessions, c_traces, c_positions.

## Folder Structure

```
sdk/
  observe.py          # @observe + @observe_tool decorators
  normalizers/
    otel.py           # OTel GenAI spans -> c_traces
    langsmith.py      # LangSmith export format -> c_traces
engine/
  eval_engine.py      # 13 evals: per-agent + holistic
  pattern_detector.py # Phase 1: rule-based; Phase 2: Isolation Forest
  rca_engine.py       # call stack builder + fix suggestion writer
simulator/
  failure_sim.py      # inject synthetic traces for 6 named failure patterns
dashboard/
  dashboard.py        # Streamlit: Incidents Feed + RCA View
  .streamlit/
    secrets.toml      # NOT committed — copy from observability/poc/
design/
  prototype-design.md       # full design (16 sections)
  rca-product-idea.md       # product brainstorm + secret sauce debate
  AI_Agent_RCA_Prototype_Design.docx  # readable Word version
tests/
  test_evals.py       # unit tests for all 13 evals
  test_detector.py    # unit tests for pattern detection rules
  test_simulator.py   # confirms simulator injects expected trace shapes
```

## Build Sequence

Phase 1 — Data (no UI): c_evals + c_incidents tables, eval_engine.py,
           pattern_detector.py, backfill 22 sessions, validate 5 known incidents fire.

Phase 2 — RCA (no UI): rca_engine.py, call stack construction, fix suggestions,
           validate against yfinance hang.

Phase 3 — Dashboard: Incidents Feed tab, Interactive RCA View drill-down,
           eval scores in Session Deep Dive.

Phase 4 — Real-time: Supabase realtime subscription, live detection on session close.

Demo: ingestion flexibility (3 trace formats), failure simulator, before/after flow.
See design/prototype-design.md Section 13 for full demo script.

## Key Design Decisions

- Eval logic runs synchronously (~0.5ms); DB writes fire async (non-blocking)
- LLM-as-judge (Phase 4) runs post-session only, never in agent critical path
- Prompt caching on judge system prompt; batch 4 agents in 1 Haiku call
- Circuit breaker: eval fails between agent handoffs -> pipeline aborted, downstream cost saved
- Pattern detection phases: rules now, Isolation Forest at 20+ sessions, sequence mining at 50+
- Multi-tenant: per-customer Isolation Forest; shared rule library + sequence trie

## Relationship to Other Projects

- observability/ — research docs, market analysis, Outcome Ledger dashboard. Separate concern.
- trading-agent-c/ — the test bed. Shares Supabase. Do not modify its schema.
- This project eventually becomes a standalone product / open-source SDK.

## What NOT to Do

- Do not commit secrets.toml
- Do not modify c_sessions, c_traces, c_positions schema in trading-agent-c
- Do not run LLM-as-judge in the agent pipeline critical path
- Do not skip tests — all evals and detectors must have passing tests before commit
