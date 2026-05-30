# AI Agent RCA — Prototype Design

## Problem Statement

AI agents fail silently. When a session burns $1.98 over 1027 seconds and produces zero
trades, the system returns HTTP 200. No exception. No alert. The only way to know something
went wrong is to manually read hundreds of trace rows and reconstruct what happened.

Three specific gaps:

**1. Traces without diagnosis**
Observability tools (LangSmith, Langfuse, our own dashboard) show you what happened. None
of them tell you why. Reading raw traces to find a root cause is manual, slow, and requires
the person who built the system. It does not scale.

**2. Evals disconnected from execution**
Teams run evals to validate output quality. But evals sit in a separate tool, disconnected
from the trace that produced the output. You know the answer was wrong. You do not know
which agent step caused it or what it cost you.

**3. No named failure taxonomy**
Every incident is bespoke. Teams debug from scratch each time. There is no shared vocabulary
for "Tool Timeout Loop" or "Context Spiral" — no pattern library that lets you recognize
a failure type the moment you see its signature, and no accumulated learning across sessions.

**What we are building:**
A prototype that connects Strategy C trace data to per-agent and holistic evals in real
time, detects named failure patterns automatically, and generates an interactive RCA with
full call stack — so any engineer can understand what went wrong without reading raw traces.

---

## System Flow

```
┌────────────────────────────────────────────────────────────────┐
│                   STRATEGY C (Live / Sim)                      │
│   Market → Research → Risk → Orchestrator → Trade Execution    │
└──────────────────────┬─────────────────────────────────────────┘
                       │ writes traces, sessions, positions
                       ▼
┌────────────────────────────────────────────────────────────────┐
│                        SUPABASE                                │
│   c_sessions | c_traces | c_positions                          │
│   [NEW] c_evals | c_incidents                                  │
└──────────┬─────────────────────────────────────────────────────┘
           │ poll / realtime subscription
     ┌─────┴──────────────────────┐
     │                            │
     ▼                            ▼
┌──────────────┐         ┌─────────────────────┐
│  Eval Engine │         │   Pattern Detector  │
│              │         │                     │
│  Per-agent:  │─evals──▶│  Tool Timeout Loop  │
│  market      │         │  Context Spiral     │
│  research    │         │  Pipeline Break     │
│  risk        │         │  Empty Result Loop  │
│  orchestrator│         │  Eval Fail Cascade  │
│              │         │  Cost Anomaly       │
│  Holistic:   │         └──────────┬──────────┘
│  efficiency  │                    │ incident record
│  completion  │                    ▼
│  outcome     │         ┌─────────────────────┐
└──────────────┘         │     RCA Engine      │
                         │                     │
                         │  walk trace stack   │
                         │  find root step     │
                         │  attach eval scores │
                         │  name the pattern   │
                         │  suggest fix        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         DASHBOARD             │
                    │                               │
                    │  [existing] Ledger            │
                    │  [existing] Session Deep Dive │
                    │  [existing] Quality Drift     │
                    │  [existing] Before/After      │
                    │                               │
                    │  [NEW] Incidents Feed         │
                    │  [NEW] Interactive RCA View   │
                    └───────────────────────────────┘
```

---

## New Data Tables

### c_evals
```sql
CREATE TABLE c_evals (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  uuid REFERENCES c_sessions(id),
    agent       text,          -- 'research' | 'market' | 'risk' | 'orchestrator' | 'session'
    eval_name   text,          -- 'completion' | 'efficiency' | 'output_quality' | 'outcome'
    score       float,         -- 0.0 to 1.0
    passed      boolean,       -- score >= threshold
    threshold   float,
    detail      jsonb,         -- raw evidence used to compute score
    created_at  timestamptz DEFAULT now()
);
```

### c_incidents
```sql
CREATE TABLE c_incidents (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    uuid REFERENCES c_sessions(id),
    pattern_name  text,         -- 'Tool Timeout Loop' | 'Context Spiral' | etc.
    severity      text,         -- 'critical' | 'warning' | 'info'
    root_cause    text,         -- human-readable one-liner
    call_stack    jsonb,        -- ordered trace steps leading to failure
    failed_evals  jsonb,        -- which evals failed and their scores
    cost_wasted   float,        -- USD attributed to this incident
    fix_suggestion text,
    created_at    timestamptz DEFAULT now()
);
```

---

## Eval Definitions

### Per-Agent Evals

**Market Agent**
| Eval | Pass condition | Fail signal |
|---|---|---|
| data_completeness | All required fields non-null (price, volume, sector) | Null price or missing symbol data |
| data_freshness | Timestamp within 30 min of session start | Stale or missing market data |

**Research Agent**
| Eval | Pass condition | Fail signal |
|---|---|---|
| completion | Returns structured recommendation (symbol, direction, rationale) | Empty or null recommendation |
| token_efficiency | tokens_used < 30K for a recommendation | Excessive tokens with no output |
| tool_success_rate | >80% of tool calls return non-error response | High tool error rate |

**Risk Agent**
| Eval | Pass condition | Fail signal |
|---|---|---|
| assessment_complete | Returns risk score + position size + stop loss | Any field null |
| within_parameters | Position size <= max from config | Risk parameters violated |

**Orchestrator Agent**
| Eval | Pass condition | Fail signal |
|---|---|---|
| decision_made | Returns trade or explicit no-trade with reason | No decision logged |
| consistency | Decision consistent with research + risk inputs | Contradicts upstream agent output |

### Holistic Session Evals

| Eval | Pass condition | Fail signal |
|---|---|---|
| pipeline_completion | All 4 agents completed their stage | Any agent short-circuited |
| cost_efficiency | Session cost within 2 sigma of rolling mean | Cost spike |
| outcome_linkage | If trade made, position recorded; if no trade, reason logged | Silent exit |
| tokens_per_decision | Tokens < 50K per trade decision | Context spiral |

---

## Pattern Library

| Pattern | Detection rule | Example |
|---|---|---|
| Tool Timeout Loop | tool_call error → same tool retried 3+ times | yfinance timeout → 8 retries, $1.98 wasted |
| Context Spiral | token count > 40K with no output step recorded | Research agent over-gathering, never deciding |
| Pipeline Break | Agent N completes but Agent N+1 never starts | Research done, orchestrator never called |
| Empty Result Loop | Tool returns null/empty → agent retries with variation | News API empty → 5 different query attempts |
| Eval Fail Cascade | research.completion eval fails, orchestrator still decides | Decision made on bad upstream data |
| Cost Anomaly | Session cost > mean + 2 sigma | Any session burning abnormally vs. baseline |
| Silent Exit | Session ends, no trade, no no-trade reason logged | 0 output, no explanation |

---

## RCA Output Format

What the dashboard shows for each incident. Example using the known yfinance hang:

```
INCIDENT — Tool Timeout Loop
Session: 2026-05-28 06:34 | Severity: CRITICAL
Cost wasted: $1.98 | Duration: 1027s | Outcome: 0 trades

ROOT CAUSE
  research_agent.analyze()
    └── tool_call: get_stock_data(symbol="NVDA")    [TIMEOUT 30s]
          └── retry 1: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── retry 2: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── retry 3: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── retry 4: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── retry 5: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── retry 6: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── retry 7: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── retry 8: get_stock_data(symbol="NVDA") [TIMEOUT 30s]
          └── session_timeout()                      [1027s elapsed]

EVALS FAILED
  research.completion        0.0  (no recommendation produced)
  research.token_efficiency  0.1  (45K tokens, zero useful output)
  session.pipeline_completion 0.0 (orchestrator never reached)
  session.outcome_linkage    0.0  (0 trades, no reason logged)

FIX
  Add max_retries=2 and timeout=10s to yfinance tool configuration in
  trading-agent-c/agents/research/tools.py
```

---

## Dashboard — New Screens

### Incidents Feed (new tab)
- One row per incident, sorted by cost_wasted descending
- Columns: timestamp, pattern name, severity badge, cost wasted, agents involved, evals failed
- Click any row → opens Interactive RCA View

### Interactive RCA View (drill-down)
- Top: incident summary card (pattern, cost, duration, outcome)
- Middle: call stack tree (collapsible, color-coded by step type and pass/fail)
- Right panel: eval scores for this session, per-agent + holistic
- Bottom: fix suggestion + link to relevant source file

---

## Build Plan

### Phase 1 — Data foundation (no UI)
- Create c_evals and c_incidents tables in Supabase
- Write eval_engine.py: compute all evals for existing 22 sessions, backfill c_evals
- Write pattern_detector.py: run pattern library against all sessions, backfill c_incidents

### Phase 2 — RCA generation
- For each incident, walk c_traces and build call_stack JSON
- Write human-readable root_cause and fix_suggestion
- Verify against known incidents: yfinance hang, research agent no-output sessions

### Phase 3 — Dashboard integration
- Add Incidents Feed tab to dashboard.py
- Add Interactive RCA View as drill-down from incidents
- Wire eval scores into existing Session Deep Dive tab

### Phase 4 — Real-time
- Supabase realtime subscription on c_sessions
- Trigger eval + pattern detection on session close
- Push incident to dashboard without refresh

---

## Known Incidents to Validate Against

These are real failures from Strategy C history. The prototype must surface all of them:

1. **2026-05-28 06:34** — yfinance timeout, research agent loop, $1.98, 0 trades
2. **Research agent hang** — sector context commits (d38950d, 487e4df) caused 10+ min hangs
3. **Unrealized P&L bug** — positions recorded with $0 P&L (fixed 2026-05-29)
4. **Upsert conflict bug** — duplicate session writes (fixed)
5. **Sector name schema bug** — wrong field name caused data loss (fixed)

If the pattern detector fires correctly on incidents 1-2 (the behavioral failures), the
prototype is working. Incidents 3-5 are schema bugs, not agent behavioral failures — they
validate a different detection class.

---

## Status

- Designed: 2026-05-29
- Not built yet
- Prerequisite: Strategy C must be running and writing to Supabase (confirmed, 22 sessions)
- Build estimate: Phase 1+2 = ~1 day, Phase 3 = ~1 day, Phase 4 = ~half day
