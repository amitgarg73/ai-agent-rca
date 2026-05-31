# AI Agent RCA — Product Design

*Single source of truth. Absorbs prototype-design.md, rca-product-idea.md, semantic_quality_design.md.*
*Last updated: 2026-05-31*

---

## 1. Core Insight

Current observability tools show you **what happened** — traces, costs, token counts.
None of them tell you **why** — what failure in the agent's reasoning or tool chain caused
the wasted spend or silent failure.

That gap is AI Agent Root Cause Analysis.

The concrete example: Strategy C research agent, 2026-05-28 01:26 UTC.
- Cost: $1.98 | Duration: 1027s | Output: 0 trades
- HTTP 200 on every call. No exception. Session record: "completed."
- Root cause: yfinance API timed out. Agent entered a retry loop, no exit condition.
- What the product would surface: "Tool Timeout Loop — yfinance, 8 retries, $1.98 wasted. Fix: add max_retries=2 and timeout=10s."

No human reads traces. System names the failure, quantifies the waste, suggests the fix.

The deeper problem this points to: **quality degradation precedes structural failure by 3-10 sessions.**
An agent starts producing less grounded recommendations, then less coherent reasoning, then eventually
causes an incident. If you only watch operational metrics, you see the incident. If you watch quality
metrics, you see it coming.

---

## 2. Problem Statement — Three Gaps

**Gap 1: Traces without diagnosis**
Observability tools show what happened. None tell you why. Reading raw traces to find a root cause
is manual, slow, and requires the person who built the system. It does not scale.

**Gap 2: Evals disconnected from execution**
Evals validate output quality. But they sit in a separate tool, disconnected from the trace that
produced the output. You know the answer was wrong. You do not know which agent step caused it
or what it cost.

**Gap 3: No named failure taxonomy**
Every incident is bespoke. Teams debug from scratch. There is no shared vocabulary for
"Tool Timeout Loop" or "Context Spiral" — no pattern library that lets you recognize a failure
the moment you see its signature, and no accumulated learning across sessions.

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              STRATEGY C (Live / Sim)                           │
│  Market → Research → Risk → Orchestrator → Trade Execution     │
└──────────────────────┬─────────────────────────────────────────┘
                       │ writes traces, sessions, positions
                       ▼
┌────────────────────────────────────────────────────────────────┐
│                        SUPABASE                                │
│  c_sessions | c_traces | c_positions (existing)                │
│  c_evals | c_incidents (added by this product)                 │
└──────────┬─────────────────────────────────────────────────────┘
           │ post-session (async, never in agent critical path)
     ┌─────┴──────────────────────────────────────┐
     │                                            │
     ▼                                            ▼
┌──────────────────┐                   ┌──────────────────────┐
│   Eval Engine    │                   │   Quality Judge      │
│  (rule-based,    │                   │  (LLM-as-judge,      │
│   <1ms, 17 evals)│                   │   Haiku, async,      │
│                  │                   │   ~$0.002/session)   │
│  Layer 1: ops    │                   │                      │
│  Layer 2: biz    │                   │  Layer 3: semantic   │
│  outcomes        │─────evals────────▶│  quality per agent   │
└──────────────────┘                   └──────────┬───────────┘
                                                  │
                                       ┌──────────▼───────────┐
                                       │   Pattern Detector   │
                                       │                      │
                                       │  Rule-based (6 ops + │
                                       │  4 quality patterns) │
                                       │                      │
                                       │  Isolation Forest    │
                                       │  (unknown patterns,  │
                                       │  20+ sessions)       │
                                       └──────────┬───────────┘
                                                  │ incidents
                                       ┌──────────▼───────────┐
                                       │     RCA Engine       │
                                       │  call stack builder  │
                                       │  fix suggestions     │
                                       │  recommendations     │
                                       └──────────┬───────────┘
                                                  │
                                       ┌──────────▼───────────┐
                                       │      DASHBOARD       │
                                       │  Ledger              │
                                       │  Quality Drift       │
                                       │  Incidents Feed      │
                                       │  RCA View            │
                                       │  Failure Simulator   │
                                       │  Trace Inspector     │
                                       └──────────────────────┘
```

---

## 4. Data Tables

### c_evals (written by eval engine + quality judge)
```sql
CREATE TABLE c_evals (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  uuid REFERENCES c_sessions(id),
    agent       text,    -- 'research' | 'risk' | 'orchestrator' | 'session' | 'business'
                         -- quality evals: 'research_quality' | 'risk_quality' | etc.
    eval_name   text,    -- e.g. 'tool_success_rate' | 'thesis_coherence' | 'cost_per_trade'
    score       float,   -- 0.0 to 1.0
    passed      boolean,
    threshold   float,
    detail      jsonb,   -- raw evidence + notes from LLM judge
    created_at  timestamptz DEFAULT now()
);
```

### c_incidents (written by pattern detector)
```sql
CREATE TABLE c_incidents (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     uuid REFERENCES c_sessions(id),
    pattern_name   text,
    severity       text,   -- 'critical' | 'warning' | 'info'
    root_cause     text,
    call_stack     jsonb,
    failed_evals   jsonb,
    cost_wasted    float,
    tokens_wasted  int,
    fix_suggestion text,
    is_simulated   boolean DEFAULT false,
    created_at     timestamptz DEFAULT now()
);
```

---

## 5. Eval Taxonomy — Three Layers

```
Layer 1: Operational Evals   — Did the pipeline run correctly?
         17 evals, rule-based, <1ms, zero LLM cost
         Stored in c_evals, backfilled via scripts/backfill_evals.py

Layer 2: Business Outcome Evals — Did the pipeline produce value?
         3 evals, computed on-the-fly in dashboard (not stored in c_evals)
         Uses sessions + traces DataFrames directly

Layer 3: Semantic Quality Evals — Did each agent do its job well?
         Per-agent per-dimension, LLM-as-judge (Haiku), ~$0.002/session
         Stored in c_evals with agent='research_quality' etc.
         Runs async post-session, never in agent critical path
```

### Layer 1: Operational Evals (17 total)

**Market agent**
| Eval | Pass condition | Threshold |
|---|---|---|
| data_completeness | Market agent produced at least one successful trace | 0.70 |
| data_freshness | First market trace within 10 min of session start | 0.50 |

**Research agent**
| Eval | Pass condition | Threshold |
|---|---|---|
| completion | LLM ran + at least one tool call succeeded | 0.70 |
| token_efficiency | < 30K tokens total | 0.50 |
| tool_success_rate | >= 80% of tool calls succeeded | 0.80 |
| tool_diversity | >= 2 distinct tools called | 2 |

**Risk agent**
| Eval | Pass condition | Threshold |
|---|---|---|
| assessment_complete | At least one successful risk trace | 1.00 |
| within_parameters | No error traces | 0.90 |

**Orchestrator agent**
| Eval | Pass condition | Threshold |
|---|---|---|
| decision_made | Trades executed or named good exit | 0.70 |
| exit_quality | terminal_reason in good exit set | 0.70 |

**Session holistic**
| Eval | Pass condition | Threshold |
|---|---|---|
| pipeline_completion | All 4 agents present | 1.00 |
| cost_anomaly | Cost within 2 sigma of rolling mean | 0.50 |
| outcome_linkage | Trades or named exit reason | 0.70 |
| tokens_per_decision | < 40K tokens | 0.50 |

### Layer 2: Business Outcome Evals (3 total, on-the-fly)

| Eval | Formula | Threshold |
|---|---|---|
| cost_per_trade | total_cost_usd / max(1, trades_executed) | < $0.50 |
| research_conversion | trades_executed / successful_research_llm_calls | > 30% |
| proposal_acceptance | trades_executed / trades_proposed | > 40% |

---

## 6. Semantic Quality Rubrics (Layer 3)

### Why quality evals exist

Operational evals catch infrastructure failures. Quality evals catch semantic failures —
sessions where everything ran correctly but agents produced low-quality outputs.

Degradation sequence:
```
Sessions 1-5:  quality healthy, ops healthy
Sessions 6-8:  data_grounding declining (0.80→0.60→0.45) — ops still healthy
Sessions 9-11: risk research_consistency drops — ops still healthy
Session 12:    pipeline break or cost anomaly fires — ops catches it now
```

Quality signals the problem at session 6. Ops only at session 12.

### Market Agent Quality

| Dimension | Measures | Score 1.0 | Score 0.0 |
|---|---|---|---|
| coverage_completeness | Data fetched for tickers research will analyze | Tickers match research traces | Market fetched MSFT, research analyzed NVDA |
| signal_clarity | Characterized market conditions | Directional/volatility characterization | Raw data dump, no interpretation |
| anomaly_flagging | Flagged unusual conditions | Flags anomaly or confirms clean | Silent on conditions that should be flagged |

**Composite:** mean of 3 dimensions. CB threshold: < 0.50

### Research Agent Quality

Highest-value evaluation point. Research quality is the single biggest predictor of whether
downstream agents can do their jobs well.

| Dimension | Weight | Score 1.0 | Score 0.5 | Score 0.0 |
|---|---|---|---|---|
| **data_grounding** | 0.25 | Cites specific price levels, volume, named catalyst | "Volume elevated", "positive momentum" | No data references |
| **thesis_coherence** | 0.25 | Clear directional thesis with supporting evidence | Thesis present, weakly supported | No thesis or thesis contradicts evidence |
| **actionability** | 0.20 | Direction + entry zone + specific ticker | Direction and ticker, no entry | Vague, not actionable |
| **catalyst_specificity** | 0.15 | "Breakout above $875" or "Earnings 2026-06-12" | "Some upward momentum" | No catalyst |
| **risk_acknowledgment** | 0.15 | "If fails to hold $860, thesis invalidated" | Generic "there is risk" | No downside mentioned |

**CB threshold:** composite < 0.60 OR data_grounding < 0.40

### Risk Agent Quality

| Dimension | Weight | Score 1.0 | Score 0.5 | Score 0.0 |
|---|---|---|---|---|
| **research_consistency** | 0.25 | Directly assesses research's specific ticker + direction | Same ticker, generic depth | Mismatch with research output |
| **parameter_completeness** | 0.25 | Direction + position size + stop loss all present | 2 of 3 | Missing critical parameters |
| **volatility_accounting** | 0.20 | "ATR $4.20 → sizing 150sh so 1 ATR = 0.84% portfolio" | "Market is volatile, sizing conservatively" | No volatility reference |
| **stop_loss_quality** | 0.20 | "Stop at $857 — below $860 support" | "Stop at 2% below entry" | No stop or no basis |
| **position_sizing_rationale** | 0.10 | Derived from portfolio %, stop distance, ATR | Partial rationale | Bare number |

**CB threshold:** composite < 0.60 OR research_consistency < 0.50

### Orchestrator Agent Quality

| Dimension | Weight | Score 1.0 | Score 0.5 | Score 0.0 |
|---|---|---|---|---|
| **decision_consistency** | 0.35 | Consistent with both research AND risk; any override explicitly reasoned | Consistent with one, minor gap with other | Contradicts upstream without explanation |
| **resolution_completeness** | 0.30 | "Execute: Long NVDA 150sh at market" or "No trade: risk not met" | "Likely execute if conditions hold" | Ambiguous or deferred |
| **reasoning_transparency** | 0.20 | Explicit chain: research said X → risk confirmed Y → decision Z | Inferable reasoning | Decision without reasoning |
| **upstream_integration** | 0.15 | Cites specific data from both research and risk | Cites one upstream agent | Makes decision without citing upstream |

**CB threshold:** composite < 0.60 OR decision_consistency < 0.50

### Session Coherence (cross-agent)

| Dimension | Score 1.0 | Score 0.0 |
|---|---|---|
| pipeline_coherence | Same ticker and direction throughout all agents | Any handoff breaks ticker/direction continuity |
| reasoning_chain | Each agent explicitly builds on previous output | Any agent ignores upstream output |

---

## 7. LLM-as-Judge Architecture

### Constraints
- Async, post-session only — never in agent critical path
- One Haiku call per session, all agents batched
- System prompt cached — ~$0.0003/call
- Total target: < $0.003/session
- Latency: 600-900ms — irrelevant post-session

### Input construction
Extract agent LLM output from traces: `step_type IN ('decision', 'llm_call')` AND `outcome='success'`,
chronological order. If actual response text not stored, judge falls back to structural trace
evidence (tool sequence, outcome pattern, token counts) as proxy.

### Storage
Results stored in `c_evals` with `agent='{agent}_quality'`:
```
agent='research_quality',  eval_name='thesis_coherence',  score=0.9
agent='research_quality',  eval_name='data_grounding',    score=0.4
agent='research_quality',  eval_name='composite_score',   score=0.72
agent='risk_quality',      eval_name='volatility_accounting', score=0.8
agent='session_quality',   eval_name='pipeline_coherence', score=1.0
```

No DDL changes — reuses existing `c_evals` schema. `_quality` suffix distinguishes from
operational evals in all queries.

### Response schema (JSON from Haiku)
```json
{
  "research": {
    "thesis_coherence":     {"score": 0.9, "note": "Clear bullish thesis with volume breakout evidence"},
    "data_grounding":       {"score": 0.4, "note": "No specific price levels cited"},
    "catalyst_specificity": {"score": 1.0, "note": "Breakout above $875 resistance identified"},
    "risk_acknowledgment":  {"score": 0.3, "note": "No downside risks mentioned"},
    "actionability":        {"score": 0.9, "note": "Direction and ticker clear"},
    "composite_score": 0.68,
    "recommendation": "Add downside risk: specify level at which thesis is invalidated"
  },
  "risk": { ... },
  "orchestrator": { ... },
  "session_coherence": {
    "pipeline_coherence": {"score": 1.0, "note": "NVDA consistent throughout"},
    "reasoning_chain":    {"score": 0.8, "note": "Risk references research output explicitly"}
  }
}
```

---

## 8. Pattern Library

### Operational Patterns (6, rule-based, implemented)

| Pattern | Severity | Detection rule | Fix template |
|---|---|---|---|
| **Tool Timeout Loop** | critical | Same tool_name has 3+ error traces | Add max_retries=2 + timeout=10s to tool config |
| **Context Spiral** | warning | Research agent > 40K tokens + 0 trades | Add hard token budget; force decision step if reached |
| **Pipeline Break** | critical | Research ran, orchestrator never started | Catch exceptions in research→orchestrator handoff |
| **Empty Result Loop** | warning | Same tool called 3+ times successfully + 0 trades | Add result quality validation; fail fast on empty returns |
| **Silent Exit** | info | 0 trades + no terminal_reason + no tool errors | Add terminal_reason to all orchestrator exit paths |
| **Cost Anomaly** | warning | Session cost > mean + 2 sigma | Check which agent consumed excess; review token usage |

### Quality Patterns (4, rule-based, to be implemented in Phase Q3)

| Pattern | Severity | Detection rule | Fix template |
|---|---|---|---|
| **Grounding Failure** | warning | research.data_grounding < 0.40 for 3 consecutive sessions | Add explicit output requirement: "cite at least one price level and one quantitative indicator" |
| **Coherence Break** | critical | orchestrator.decision_consistency < 0.50 in any session | Add consistency check in orchestrator prompt; structured handoff format between agents |
| **Quality Cascade** | warning→critical | 3+ quality dimensions declining >0.20 over 5 sessions | Review agent prompts for each degrading dimension; check market data quality |
| **Silent Degradation** | info→warning | Composite quality declining while all operational metrics stable | Most dangerous — manual review of recent outputs; check for prompt drift or model changes |

### Dynamic Patterns (Isolation Forest, Phase D1)

Catches unknown failure modes not covered by any named rule. Each session scored on anomaly
distance from pipeline's historical baseline.

**Feature vector:**
```
total_cost_usd, total_tokens, total_latency_ms, tool_error_rate,
pipeline_completion_score, op_score, biz_score, cost_per_trade,
trades_executed, research_composite_quality, risk_composite_quality,
orchestrator_composite_quality, research_data_grounding,
orchestrator_decision_consistency
```

Anomaly score > 0.65 → "Unknown Anomaly" incident with: features that deviated most,
z-score per feature, manual review prompt.

Data requirement: 20+ sessions for baseline; 50+ for reliable signal.
Auto-retrain every 10 new sessions, sliding 50-session window.

---

## 9. Circuit Breaker Design

### Principle

Circuit breakers fire after an agent completes, using in-memory traces — no DB reads.
Eval functions already take `(traces: list[dict], session: dict)` — zero round trips.
Compute time: ~0.5ms per agent. Smaller than LLM response time variance.

### Shadow mode (current state — Strategy C still testing)

CBs run in-process after each agent. Record `would_trigger_cb: True` in eval detail.
Never abort the pipeline. Dashboard shows where CBs would have fired and cost saved.
Goal: validate false positive rate < 10% before enabling real mode.

### Real mode (post-validation)

Same code path. Raise `CircuitBreakerError(eval_result)`. Orchestrator catches it,
writes `terminal_reason='circuit_breaker'`, logs which eval triggered, stops pipeline.

### CB configuration

```python
# Operational CBs — fire on structural failures
CIRCUIT_BREAKERS = {
    "research": [
        ("tool_success_rate", 0.80),  # abort risk+orchestrator if research tools failed
        ("completion",        0.70),
    ],
    "risk": [
        ("assessment_complete", 1.0), # abort orchestrator if risk never ran
    ],
}

# Quality CBs — fire on semantic failures (shadow mode, post-session)
QUALITY_CIRCUIT_BREAKERS = {
    "research": {
        "composite_score": 0.60,
        "data_grounding":  0.40,  # vague recommendation poisons all downstream
    },
    "risk": {
        "composite_score":        0.60,
        "research_consistency":   0.50,  # generic assessment is useless to orchestrator
    },
    "orchestrator": {
        "decision_consistency":   0.50,  # contradicting upstream breaks pipeline purpose
    },
}
```

### Interaction

Operational and quality CBs fire independently. Both can fire for the same session.
Incident records both. Quality CBs always post-session; operational CBs in-process.

### 1:26 AM scenario with circuit breaker

Research called `get_stock_data` 8 times, all timed out.
- `tool_success_rate = 0.0`, threshold 0.80 → CB fires after research agent
- Risk and orchestrator never run → ~$0.80 downstream cost saved
- `terminal_reason = 'circuit_breaker'` logged with eval details

---

## 10. Recommendation Engine

Every quality incident produces a structured recommendation. This is the product's
differentiation beyond standard observability: not just "what failed" but "what to do."

### Structure

```
DIMENSION FAILING:  research.data_grounding = 0.3 (threshold 0.40)
SESSIONS AFFECTED:  3 of last 5

What happened:
  Research recommendations are not citing specific data. Suggestions are
  generic rather than evidence-based.

Why it matters:
  Low grounding forces risk to assess a vague thesis. Orchestrator decisions
  lack evidence backing. Precedes increase in false-positive trades.

What to check:
  1. Review research system prompt — may have drifted or been modified
  2. Check market agent output — if data is thin, research cannot ground
  3. Open Session Deep Dive for last 3 sessions, review research trace outputs

How to fix:
  Add to research system prompt: "Your recommendation must include:
  (1) specific ticker, (2) at least one specific price level,
  (3) at least one quantitative indicator value (volume, ATR, RSI)."
  Consider adding structured output schema to enforce fields.
```

Stored in `c_incidents.fix_suggestion`. Generated based on which dimension failed,
session context, and trend direction.

---

## 11. Dashboard Design

### Current pages (implemented)

1. **Ledger** — KPIs + Cost by Agent donut + AgGrid session grid + inline detail + LLM analyst summary
2. **Quality Drift** — 3-tab redesign:
   - Option A: Eval heatmap (sessions × evals, pass/fail color) + business outcome bars
   - Option B: Sub-tabbed Operational | Business Outcomes with trend lines
   - Option C: Health timeline (green/amber/red dots) + dual op-vs-biz bars per session
3. **Incidents Feed** — filtered incident list, Run Analysis on All Sessions
4. **RCA View** — execution trace + inline eval scores + fix suggestion
5. **Failure Simulator** — inject any of 6 patterns, run analysis
6. **Trace Inspector** — raw trace browser

### Quality layer additions (Phase Q2)

**Quality Drift — Option B, third sub-tab "Quality":**
- Per-agent composite quality score trend (rolling 5-session line chart)
- Per-dimension breakdown for selected agent (scores per session, red = below threshold)
- Active recommendations panel: any dimension below threshold for 2+ sessions surfaces
  a structured recommendation with dismiss button

**Quality Drift — Option C Timeline:**
- Health dot color: 50% operational + 20% business + 30% quality
- Hover: Op: X% | Biz: Y% | Quality: Z%
- Quality incidents → triangle marker vs circle

**Session Deep Dive (Ledger + RCA View):**
- "Quality Scores" section alongside existing operational eval scores
- Per-dimension scores with LLM judge notes inline
- Session-level recommendation if any dimension below threshold

---

## 12. Implementation Phases

### Completed

| Phase | What | Done |
|---|---|---|
| 1 | Data: c_evals, c_incidents, eval engine (17 evals), pattern detector (6 patterns), backfill | 2026-05-30 |
| 2 | RCA: call stack builder, fix suggestions, rca_engine.py | 2026-05-30 |
| 3 | Dashboard: 6 pages, Quality Drift 3-tab, business evals, auth gate, LLM summaries | 2026-05-31 |

### Quality layer

| Phase | What | Effort | Gate |
|---|---|---|---|
| **Q1** | `engine/quality_judge.py` — Haiku batch eval, prompt construction, JSON parsing, c_evals storage. `scripts/backfill_quality.py`. Tests. | 2-3 days | Manual review of 10 sessions to validate scores |
| **Q2** | Quality tab in Quality Drift. Per-agent trend charts. Recommendations panel. Quality scores in session detail. | 2-3 days | Q1 done |
| **Q3** | 4 new quality patterns in pattern_detector.py. Drift detection rules (3-session decline, threshold breach). Tests. | 1-2 days | Q2 done |
| **Q4** | Shadow quality CBs. `would_trigger_cb` flag. CB config. Dashboard CB markers + cost-saved estimates. | 1-2 days | Q3 done |

### Real-time

| Phase | What | Effort | Gate |
|---|---|---|---|
| **4a** | In-process operational CBs in trading-agent-c (shadow mode). Per-agent eval after handoff. `CircuitBreakerError` skeleton. | 1-2 days | Q4 done; Strategy C still in testing |
| **4b** | Supabase realtime subscription on c_sessions. Auto-run evals + quality judge post-session. | 1 day | 4a done |
| **4c** | Flip CBs to real mode. Operational CBs first; quality CBs after separate validation. | 0.5 days | Strategy C testing complete; false positive rate < 10% confirmed |

### ML / dynamic detection

| Phase | What | Data needed | Notes |
|---|---|---|---|
| **D1** | `engine/anomaly_detector.py` — IsolationForestDetector, fit/score/explain/persist. Feature vector includes quality scores. Auto-retrain every 10 sessions. "Unknown Anomaly" incident type. | 20+ sessions with quality scores | scikit-learn IsolationForest |
| **D2** | `engine/sequence_miner.py` — SequenceTrie, mine_patterns, match. Offline batch. Human review gate on discovered patterns. | 50+ sessions | Regime split at 100+ |
| **P1** | Predictive early warning: declining eval score trend → incident probability. Simple logistic regression. Weekly health report (Haiku auto-generated). | 100+ sessions | — |

### Productization

| Phase | What | Gate |
|---|---|---|
| **9** | Multi-tenant schema (tenant_id), per-customer Isolation Forest models, hosted auth | LinkedIn demand test: 3+ engineers DM asking how to get it |

---

## 13. Product Positioning

### Category

AI Agent AIOPS. Traditional AIOPS: infrastructure anomaly → root cause → remediation.
AI Agent AIOPS: agent behavior anomaly → root cause in agent reasoning/tool chain → fix.

AIOPS buyers already understand this value. This extends a known category into a new domain.

### The failure pattern library as moat

Naming failure modes creates a shared vocabulary — the same way "circuit breaker" and
"blue/green deployment" became standard terms. If you name the patterns, you own the taxonomy.
The library grows with more customers: more labeled failures → better detection → more customers.

### Competitive position

- **Braintrust, Arize**: evals, no trace RCA — they know *if* output was wrong, not *why*
- **LangSmith, Langfuse**: traces, no eval linkage — they see *what happened*, not *whether it mattered*
- **Datadog, Dynatrace**: approaching from infrastructure upward — good at "API was slow," weak at "agent reasoning loop caused it"
- **This product**: owns the connection layer between traces and outcomes, with named behavioral taxonomy

**The window**: go deep on agent-native RCA before infrastructure vendors climb up from below.
Agent-first understanding of failure modes is the moat while it lasts.

### Domain generalization

The failure patterns appear across agent types:
- Trading agents: wasted cost = zero trades
- Support agents: wasted cost = unresolved tickets
- Coding agents: wasted cost = unmerged PRs
- Sales agents: wasted cost = no booked meetings

Outcome definition changes per domain. The detection layer does not.

### Where real moat comes from (honest assessment as of 2026-05-31)

The reasoning failure layer — detecting that an agent made a bad plan, hallucinated a fact, or
chose the wrong tool for the goal — requires reading unstructured agent output and understanding
intent. Nobody has solved this. It requires AI expertise AND deep observability domain knowledge
simultaneously. The semantic quality evals (Phase Q1-Q4) are the entry point to this layer.

Founder advantage: AIOPS/observability background means understanding what good RCA looks like.
Most people building AI observability tools come from ML eval or DevOps, not AIOPS.

---

## 14. Known Incidents to Validate Against

Real Strategy C failures. The prototype should surface all behavioral failures:

1. **2026-05-28 01:26** — yfinance timeout, 8 retries, $1.98, 0 trades — Tool Timeout Loop
2. **Research agent hang** — sector context commits (d38950d, 487e4df), 10+ min hangs — Pipeline Break
3. **Unrealized P&L bug** — $0 P&L recorded (fixed 2026-05-29) — schema bug, not behavioral
4. **Upsert conflict bug** — duplicate session writes (fixed) — schema bug
5. **Sector name schema bug** — wrong field name, data loss (fixed) — schema bug

Items 1-2 are behavioral failures that evals and pattern detector should catch.
Items 3-5 are schema bugs — validate a different detection class.

---

## 15. What This Does NOT Do

- Does NOT modify trading-agent-c pipeline logic
- Does NOT run during agent execution — quality evals are post-session async
- Does NOT make trading decisions — evaluates existing decisions
- Does NOT require schema changes to c_sessions, c_traces, c_positions
- Phases Q1-Q3 are observation only — no circuit breakers
- CBs remain shadow-mode while Strategy C is in testing

---

## 16. Repository

**GitHub:** `github.com/amitgarg73/ai-agent-rca`
**Working dir:** `/Users/amitgarg/Claude Projects/ai-agent-rca`
**Dashboard:** Streamlit Cloud (password protected, APP_PASSWORD in secrets)

```bash
streamlit run dashboard/dashboard.py         # local dev
python3 scripts/backfill_evals.py            # re-run operational evals
python3 scripts/backfill_evals.py --dry-run  # preview only
python3 -m pytest tests/ -v                  # run all tests (84 passing)
```
