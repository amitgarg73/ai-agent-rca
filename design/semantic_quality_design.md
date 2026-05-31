# AI Agent RCA — Semantic Quality Design

## Why This Matters

Operational evals catch infrastructure failures: tool timeouts, pipeline breaks, cost spikes.
They fire when something is structurally broken. They tell you nothing about whether the
agents are actually doing their jobs well.

The $1.98 session had a tool timeout — that's operational. But what about the session where
research produces a vague, ungrounded recommendation, risk rubber-stamps it, and orchestrator
makes a trade that loses? Every operational metric passes. The pipeline "completed." The agents
just did their jobs poorly.

Quality degradation typically leads structural failure by 3-10 sessions. An agent starts
producing less grounded recommendations, then less coherent reasoning, then eventually makes
decisions that create incidents. If you only watch operational metrics, you see the incident.
If you watch quality metrics, you see it coming.

---

## Eval Taxonomy — Three Layers

```
Layer 1: Operational Evals       — Did the pipeline run correctly?
         (17 evals, rule-based, <1ms, zero LLM cost)

Layer 2: Business Outcome Evals  — Did the pipeline produce value?
         (3 evals, computed on-the-fly from session+trace data)

Layer 3: Semantic Quality Evals  — Did each agent do its job well?
         (per-agent, per-dimension, LLM-as-judge, ~$0.002/session, post-session async)
```

Layer 3 is additive. It never blocks the trading pipeline. It runs async after session close.
Its scores feed circuit breakers in shadow mode first, real mode after validation.

---

## Agent Quality Rubrics

### Market Agent

Market agent is primarily a data fetch. Quality here is about coverage and signal clarity —
whether it fetched the right data and whether it surfaced a usable market signal for research.

| Dimension | What it measures | Pass condition | Fail signal |
|---|---|---|---|
| **coverage_completeness** | Did it fetch data for tickers research will analyze? | Tickers in market output match tickers in research traces | Market fetched MSFT, research analyzed NVDA — mismatch |
| **signal_clarity** | Did it characterize market conditions (trending, ranging, volatile)? | Summary includes directional or volatility characterization | Raw data dump with no interpretation |
| **anomaly_flagging** | Did it flag unusual conditions (halts, extreme vol, circuit breakers)? | Either flags anomaly or confirms clean conditions | Silent on conditions that should have been flagged |

**Composite market quality score:** mean of 3 dimensions.
**Threshold for CB consideration:** composite < 0.50

---

### Research Agent

This is the highest-value quality evaluation point. Research quality is the single biggest
predictor of whether downstream agents (risk, orchestrator) can do their jobs well.

| Dimension | What it measures | Score 1.0 | Score 0.5 | Score 0.0 |
|---|---|---|---|---|
| **thesis_coherence** | Is there a clear directional thesis that flows from the evidence? | "NVDA bullish — volume breakout above $875 resistance with above-average buy flow" | "NVDA looks interesting given recent activity" | No thesis, or thesis contradicts cited evidence |
| **data_grounding** | Does the recommendation cite specific data points? | References price level, volume figure, specific catalyst | General references ("volume elevated", "positive momentum") | Generic statements with no data ("market conditions favorable") |
| **catalyst_specificity** | Is there a specific near-term catalyst or technical level? | "Breakout above $875 resistance / Earnings beat on 2026-06-12" | "Some upward momentum building" | No catalyst identified |
| **risk_acknowledgment** | Did research identify the key risk to the thesis? | "If NVDA fails to hold $860 on close, thesis invalidated" | "As with all trades, there is risk involved" | No downside scenario mentioned |
| **actionability** | Is the output specific enough for risk/orchestrator to act on? | Direction + entry zone + specific ticker stated | Direction and ticker without entry clarity | Vague conclusion without actionable content |

**Composite research quality score:** weighted mean — data_grounding × 0.25, thesis_coherence × 0.25, actionability × 0.20, catalyst_specificity × 0.15, risk_acknowledgment × 0.15.

**Why weighting:** data_grounding and thesis_coherence are load-bearing for downstream agents. A recommendation with no data grounding forces risk and orchestrator to assess a guess.

**Threshold for CB consideration:** composite < 0.60 or data_grounding < 0.40

---

### Risk Agent

Risk quality is about whether the assessment actually engaged with the specific situation
research presented — not just whether it returned all required fields.

| Dimension | What it measures | Score 1.0 | Score 0.5 | Score 0.0 |
|---|---|---|---|---|
| **volatility_accounting** | Did it reference current volatility in sizing the position? | "ATR $4.20 → sizing at 150 shares so 1 ATR = 0.84% of $75K portfolio" | "Market is currently volatile, sizing conservatively" | Position sized with no volatility reference |
| **research_consistency** | Does the risk assessment address the same ticker and direction? | Directly assesses NVDA long with the exact thesis from research | Assesses NVDA but uses generic parameters | Assesses a different ticker or contradicts direction |
| **position_sizing_rationale** | Is position size mathematically derived or at least explained? | Size derived from portfolio %, stop distance, and ATR | Size given with partial rationale | Number with no explanation |
| **stop_loss_quality** | Is the stop technically or fundamentally justified? | "Stop at $857 — below the $860 support level" | "Stop at 2% below entry" | No stop loss, or stop without any basis |
| **parameter_completeness** | Are all three required parameters present: direction, size, stop? | All three present and specific | 2 of 3 present | Missing direction or stop |

**Composite risk quality score:** weighted mean — research_consistency × 0.25, parameter_completeness × 0.25, volatility_accounting × 0.20, stop_loss_quality × 0.20, position_sizing_rationale × 0.10.

**Why research_consistency is highest weight:** A risk assessment that doesn't engage with research's specific thesis is not a risk assessment — it's a generic template. The orchestrator cannot make a coherent decision from disconnected outputs.

**Threshold for CB consideration:** composite < 0.60 or research_consistency < 0.50

---

### Orchestrator Agent

Orchestrator quality is about decision integrity — whether the final decision can be
justified by what came before it, and whether it is fully resolved.

| Dimension | What it measures | Score 1.0 | Score 0.5 | Score 0.0 |
|---|---|---|---|---|
| **decision_consistency** | Does the decision align with research direction AND satisfy risk parameters? | Decision consistent with both; any override explicitly stated and reasoned | Consistent with one upstream agent, minor gap with the other | Contradicts research direction or violates risk parameters without explanation |
| **resolution_completeness** | Is the decision fully resolved — no hedging or deferral? | "Execute: Long NVDA 150 shares at market" or "No trade: risk parameters not met" | "Likely execute if conditions hold through market open" | Ambiguous, deferred, or no decision reached |
| **reasoning_transparency** | Can you understand WHY from the output alone? | Explicit chain: "Research identified X → risk confirmed parameters → decision is Y" | Implicit reasoning (inferable from context) | Decision with no reasoning |
| **upstream_integration** | Does the decision reference specific inputs from research and risk? | References specific data from both (ticker, direction, size, stop from risk) | References one upstream agent specifically | Makes decision without citing upstream inputs |

**Composite orchestrator quality score:** weighted mean — decision_consistency × 0.35, resolution_completeness × 0.30, reasoning_transparency × 0.20, upstream_integration × 0.15.

**Why decision_consistency is highest weight:** An orchestrator that contradicts research or violates risk without explanation breaks the entire pipeline's purpose. The output of a multi-agent system is only as trustworthy as the orchestrator's integration of upstream reasoning.

**Threshold for CB consideration:** composite < 0.60 or decision_consistency < 0.50

---

### Session Coherence (cross-agent quality)

Two cross-agent quality checks that no single agent's eval can catch:

| Dimension | What it measures | Score 1.0 | Score 0.0 |
|---|---|---|---|
| **pipeline_coherence** | Do all agents address the same ticker and direction throughout? | research ticker = risk ticker = orchestrator ticker | Any handoff breaks ticker/direction continuity |
| **reasoning_chain** | Does each agent build on the previous one's output? | Risk cites research, orchestrator cites risk | Any agent ignores upstream output entirely |

---

## LLM-as-Judge Architecture

### Design constraints
- Runs async, post-session only — never in agent critical path
- One Haiku call per session, all 4 agents batched
- System prompt cached — ~$0.0003 per call
- Total cost target: < $0.003/session
- Latency: 600-900ms — irrelevant since it runs after session close

### Input construction

Extract the LLM output text from each agent's traces. Look for `step_type IN ('decision', 'llm_call')` with `outcome='success'` — the agent's actual reasoning output. If multiple LLM steps, concatenate in chronological order.

```python
def _extract_agent_output(traces: list[dict], agent: str) -> str:
    agent_traces = sorted(
        [t for t in traces
         if t.get("agent", "").lower() == agent
         and t.get("step_type") in ("decision", "llm_call")
         and t.get("outcome") == "success"],
        key=lambda x: x.get("created_at", "")
    )
    # Combine text fields — actual output may be in a separate column
    # or derivable from the trace structure
    return "\n---\n".join(
        t.get("output", t.get("tool_name", "")) for t in agent_traces
    ) or "(no output captured)"
```

Note: if the actual LLM response text is not stored in c_traces, the judge falls back to
evaluating the structural trace evidence (tool calls made, outcome sequence, token counts)
as a proxy. Structural proxy scoring is less accurate but still directionally useful.

### System prompt (cached)

```
You are a quality evaluator for a multi-agent AI trading pipeline. You score each agent's
output on specific quality dimensions. Each score is 0.0 to 1.0. Be specific in notes —
vague notes have no debugging value.

RESEARCH AGENT RUBRIC:
- thesis_coherence: Is there a clear directional thesis supported by evidence? (1.0=clear+supported, 0.5=thesis without evidence, 0.0=no thesis)
- data_grounding: Does it cite specific price levels, volume figures, or named catalysts? (1.0=specific data cited, 0.5=general references, 0.0=no data)
- catalyst_specificity: Is there a specific near-term catalyst or technical level? (1.0=specific, 0.5=vague, 0.0=none)
- risk_acknowledgment: Is at least one downside risk identified? (1.0=specific risk+level, 0.5=generic risk mention, 0.0=none)
- actionability: Is the output specific enough to act on? (1.0=direction+entry+ticker, 0.5=direction+ticker, 0.0=vague)

RISK AGENT RUBRIC:
- volatility_accounting: Does it reference a volatility measure in sizing? (1.0=ATR/beta cited, 0.5=general vol reference, 0.0=none)
- research_consistency: Does it address the same ticker and direction as research? (1.0=direct reference, 0.5=same ticker different depth, 0.0=mismatch)
- position_sizing_rationale: Is size derived or at least explained? (1.0=derived with math, 0.5=with partial reasoning, 0.0=bare number)
- stop_loss_quality: Is the stop technically or fundamentally justified? (1.0=specific level with basis, 0.5=percentage-based, 0.0=none)
- parameter_completeness: Are direction, size, and stop all present? (1.0=all three, 0.5=two, 0.0=one or zero)

ORCHESTRATOR RUBRIC:
- decision_consistency: Does the decision align with research AND risk? (1.0=consistent with both, 0.5=consistent with one, 0.0=contradicts upstream)
- resolution_completeness: Is the decision fully resolved, no hedging? (1.0=clear execute or clear reject, 0.5=qualified, 0.0=ambiguous)
- reasoning_transparency: Is the WHY explicit in the output? (1.0=explicit chain, 0.5=inferable, 0.0=none)
- upstream_integration: Does it cite specific data from research and risk? (1.0=both cited, 0.5=one cited, 0.0=neither)

SESSION COHERENCE:
- pipeline_coherence: Do all agents address the same ticker throughout? (1.0=consistent, 0.0=mismatch)
- reasoning_chain: Does each agent build on the previous? (1.0=explicit, 0.5=implicit, 0.0=disconnected)

Respond ONLY with valid JSON matching this exact schema:
{
  "research": {
    "thesis_coherence": {"score": float, "note": "string"},
    "data_grounding": {"score": float, "note": "string"},
    "catalyst_specificity": {"score": float, "note": "string"},
    "risk_acknowledgment": {"score": float, "note": "string"},
    "actionability": {"score": float, "note": "string"},
    "composite_score": float,
    "recommendation": "string — one specific actionable fix, or null if composite >= 0.80"
  },
  "risk": {
    "volatility_accounting": {"score": float, "note": "string"},
    "research_consistency": {"score": float, "note": "string"},
    "position_sizing_rationale": {"score": float, "note": "string"},
    "stop_loss_quality": {"score": float, "note": "string"},
    "parameter_completeness": {"score": float, "note": "string"},
    "composite_score": float,
    "recommendation": "string or null"
  },
  "orchestrator": {
    "decision_consistency": {"score": float, "note": "string"},
    "resolution_completeness": {"score": float, "note": "string"},
    "reasoning_transparency": {"score": float, "note": "string"},
    "upstream_integration": {"score": float, "note": "string"},
    "composite_score": float,
    "recommendation": "string or null"
  },
  "session_coherence": {
    "pipeline_coherence": {"score": float, "note": "string"},
    "reasoning_chain": {"score": float, "note": "string"}
  }
}
```

### Storage

Quality eval results stored in `c_evals` as new rows with `agent="{agent}_quality"`:

```
agent="research_quality",  eval_name="thesis_coherence",   score=0.9, ...
agent="research_quality",  eval_name="data_grounding",     score=0.4, ...
agent="research_quality",  eval_name="composite_score",    score=0.72, ...
agent="risk_quality",      eval_name="volatility_accounting", score=0.8, ...
agent="session_quality",   eval_name="pipeline_coherence", score=1.0, ...
```

This reuses the existing `c_evals` schema with no DDL changes. The `_quality` suffix on agent
name distinguishes semantic evals from operational evals in all queries.

---

## Quality Drift Detection

### What early warning looks like

The Quality Drift page currently shows operational metrics (eval pass rates, tool success rates,
pipeline completion). Semantic quality adds a new signal layer that moves *before* these.

Degradation sequence typically looks like:
```
Sessions 1-5:  quality scores healthy (>0.80), ops healthy
Sessions 6-8:  data_grounding starts declining (0.75 → 0.60 → 0.45) — ops still healthy
Sessions 9-11: research_consistency in risk starts dropping — ops still healthy
Sessions 12:   pipeline breaks or cost anomaly fires — ops catches it
```

If you watch quality, you see the problem at session 6. If you watch ops only, you see it at session 12.

### Drift detection rules

A **quality drift alert** fires when any of these conditions hold across recent sessions:

1. **Dimension decline**: any single quality dimension drops by >0.20 over 3 consecutive sessions
2. **Composite decline**: any agent's composite quality score drops by >0.15 over 5 sessions
3. **Quality cascade**: 3+ dimensions declining simultaneously (any mix of agents)
4. **Threshold breach**: any dimension falls below its CB threshold for 2+ consecutive sessions

These fire as new incident types in `c_incidents` alongside the existing operational patterns.

---

## Quality-Based Pattern Library (additions to existing 6 patterns)

### Quality Cascade (warning → critical)
**Detection:** 3+ quality dimensions declining >0.20 over 5 sessions across any agents.
**Significance:** Systematic quality degradation — not one bad session, a trend. Often precedes structural failures by 5-10 sessions.
**Fix suggestion:** Review agent prompts for each degrading dimension. Check if market data quality has declined (market data quality affects all downstream agents). Review whether any system prompt changes were made recently.

### Grounding Failure (warning)
**Detection:** research.data_grounding < 0.40 for 3 consecutive sessions.
**Significance:** Research agent making recommendations without referencing specific data. Downstream risk assessment becomes a guess. Trade decisions have no evidence basis.
**Fix suggestion:** Add explicit output requirements to research system prompt: "Your recommendation must cite at minimum one specific price level and one quantitative indicator (volume, ATR, RSI, etc.)." Add output validation in research agent before handoff.

### Coherence Break (critical)
**Detection:** orchestrator.decision_consistency < 0.50 in any single session.
**Significance:** Orchestrator made a decision that contradicts what research or risk said. This means the multi-agent pipeline is not actually integrating its outputs. The orchestrator is acting independently.
**Fix suggestion:** Add explicit consistency check in orchestrator prompt: "Before deciding, confirm your decision aligns with research direction ({direction}) and satisfies risk parameters (size: {size}, stop: {stop}). If you deviate, state the specific reason." Consider adding a structured handoff format between agents.

### Silent Degradation (info → warning)
**Detection:** composite quality declining while all operational metrics stable (pass rates >0.85, no tool errors, no cost anomaly).
**Significance:** This is the most dangerous pattern because no existing alarm catches it. The pipeline looks healthy. Quality is quietly declining. Eventually leads to bad trades, not incidents.
**Fix suggestion:** Review recent session outputs manually. Compare agent reasoning from 10 sessions ago vs now. Look for prompt drift, model version changes, or data quality changes upstream.

---

## Circuit Breaker Integration — Quality Thresholds

These are shadow-mode only until Strategy C testing completes.

```python
QUALITY_CIRCUIT_BREAKERS = {
    "research": {
        "composite_score":    0.60,
        "data_grounding":     0.40,   # if not grounding in data, abort risk
    },
    "risk": {
        "composite_score":    0.60,
        "research_consistency": 0.50, # if not assessing research's ticker, abort orchestrator
    },
    "orchestrator": {
        "decision_consistency": 0.50, # if contradicting upstream, flag for review
    },
}
```

Quality CBs fire AFTER the agent completes (post-LLM-judge). They do not affect the current
session — they produce a `would_trigger_cb=True` flag in the eval record and a shadow CB
incident in `c_incidents`. In real mode (post-validation), a low quality score on research
would abort the pipeline before risk and orchestrator run.

**Interaction with operational CBs:** An agent can fail a quality CB without failing any
operational CB. Both fire independently. If both fire, the incident records both.

---

## Recommendation Engine

Every quality incident produces a structured recommendation. This is the product's
differentiation beyond standard observability.

### Recommendation structure

```
DIMENSION FAILING:  research.data_grounding = 0.3 (threshold 0.40)
SESSIONS AFFECTED:  last 3 of 5 sessions

What happened:
  Research recommendations no longer cite specific price levels or volume data.
  Suggestions are generic rather than evidence-based.

Why it matters:
  Low grounding forces risk agent to assess a vague thesis. Orchestrator decisions
  lack evidence backing. This precedes an increase in false-positive trades.

What to check:
  1. Review research agent system prompt — may have drifted or been modified
  2. Check market agent output quality — if market data is thin, research cannot ground
  3. Open Session Deep Dive for the last 3 sessions and review research trace outputs directly

How to fix:
  Add to research system prompt: "Your recommendation must include: (1) the specific
  ticker, (2) at least one specific price level (support, resistance, entry), (3) at
  least one quantitative indicator value."
  Consider adding a structured output schema to the research agent to enforce fields.
```

Recommendations are stored in `c_incidents.fix_suggestion` (existing field). The recommendation
engine generates them based on which dimension failed, the session context, and the trend.

---

## Dynamic Pattern Detection — Isolation Forest

### Purpose

Rule-based patterns catch known failures. Isolation Forest catches unknown ones.

Every session produces a feature vector. The model learns what "normal" looks like for
this specific pipeline. Anomalous sessions get flagged even if no named pattern matches.

### Feature vector (per session)

```python
FEATURES = [
    # Operational
    "total_cost_usd",
    "total_tokens",
    "total_latency_ms",
    "tool_error_rate",            # errors / total tool calls
    "pipeline_completion_score",  # from eval
    "op_score",                   # mean operational eval pass rate
    # Business
    "biz_score",                  # mean business eval pass rate
    "cost_per_trade",
    "trades_executed",
    # Quality (once LLM judge is running)
    "research_composite_quality",
    "risk_composite_quality",
    "orchestrator_composite_quality",
    "research_data_grounding",    # most predictive individual dimension
    "orchestrator_decision_consistency",
]
```

### Implementation plan

```
engine/anomaly_detector.py
  - IsolationForestDetector class
  - fit(sessions_df) — train on historical data
  - score(session) → float (0-1, higher = more anomalous)
  - explain(session) → list[str] (which features drove the anomaly score)
  - persist/load (pickle) so model survives restarts
```

Model retrains automatically when session count increases by 10 (sliding window of last 50).

When anomaly score > 0.65, fires as "Unknown Anomaly" incident with:
- The features that deviated most from baseline
- The z-score for each deviating feature
- "This session looks unusual in {N} dimensions. Manual review recommended."

At 20+ sessions: enough for a baseline. At 50+ sessions: reliable signal, regime-split possible.

### Integration with named patterns

Isolation Forest and named patterns are complementary:
- Named pattern fires → structured incident with specific fix
- Isolation Forest fires without named pattern → "Unknown Anomaly" incident prompts investigation
- Both fire → incident has both structured diagnosis and anomaly context

---

## Dashboard — Quality Layer Design

### Quality Drift page additions

The existing 3-tab layout gains quality data in each tab:

**Option A (Scorecard + Heatmap):**
- Heatmap now includes quality eval rows (research_quality.thesis_coherence, etc.)
- Quality rows highlighted with different background color to distinguish from operational
- "Quality composite" row shows the single most important signal

**Option B (Tabbed Operational | Business):**
- Add a third sub-tab: "Quality"
- Quality tab: per-agent composite quality score trend (line chart, rolling 5 sessions)
- Per-dimension breakdown for the selected agent (dimension scores per session, color = pass/fail)
- Recommendation panel: shows active recommendations from quality incidents

**Option C (Timeline):**
- Health dot color now incorporates quality: 50% operational, 20% business, 30% quality
- Hover shows: Op: X% | Biz: Y% | Quality: Z%
- Sessions with quality incidents get a distinct marker (triangle vs circle)

### New dashboard section: Quality Recommendations

Appears on the Quality Drift page when any quality dimension is below threshold for 2+ sessions:

```
ACTIVE QUALITY ALERTS

Research Agent — data_grounding declining (0.80 → 0.55 → 0.30, last 3 sessions)
"Your recommendation must include at least one specific price level..."
[View affected sessions] [Dismiss]

Orchestrator — reasoning_transparency low (0.40, last session)
"Add explicit reasoning chain: Research said X → Risk said Y → I decided Z because..."
[View session] [Dismiss]
```

### Session Deep Dive additions

When a session is opened in the RCA View or Ledger detail:
- New "Quality Scores" section alongside existing eval scores
- Per-dimension scores with inline notes from the LLM judge
- Session-level recommendation if any quality dimension is below threshold

---

## Implementation Phases (semantic quality + dynamic detection)

### Phase Q1: LLM Quality Judge Foundation
**Effort:** 2-3 days
**Deliverables:**
- `engine/quality_judge.py` — Haiku batch evaluation, prompt construction, JSON parsing
- `c_evals` storage with `_quality` agent suffix
- `scripts/backfill_quality.py` — run quality evals on all historical sessions
- Tests covering prompt construction, response parsing, score storage
- No dashboard changes yet — validate data quality first

**Gate:** Run backfill, review 10 sessions manually, confirm scores make sense.

### Phase Q2: Quality Drift Dashboard Integration
**Effort:** 2-3 days
**Deliverables:**
- Quality track in Quality Drift Option B (third sub-tab) and Option C (timeline overlay)
- Per-agent composite quality trend charts
- Per-dimension breakdown with session-level view
- Quality recommendations panel on Quality Drift page
- Quality scores section in Session Deep Dive (RCA View + Ledger detail)

### Phase Q3: Quality Patterns + Incidents
**Effort:** 1-2 days
**Deliverables:**
- 4 new named quality patterns in `pattern_detector.py`: Quality Cascade, Grounding Failure, Coherence Break, Silent Degradation
- Drift detection rules (3-session decline, threshold breach)
- Quality-based recommendations in `Incident.fix_suggestion`
- Tests for all 4 new patterns

### Phase Q4: Shadow Quality Circuit Breakers
**Effort:** 1-2 days
**Deliverables:**
- `would_trigger_cb` flag in quality eval records
- `QUALITY_CIRCUIT_BREAKERS` config in `eval_engine.py`
- Dashboard markers: "Quality CB would have fired" annotations on timeline
- Cost-saved estimate per session where quality CB would have triggered

### Phase D1: Isolation Forest Anomaly Detector
**Effort:** 2-3 days
**Data requirement:** 20+ sessions with quality scores
**Deliverables:**
- `engine/anomaly_detector.py` — IsolationForestDetector with fit/score/explain/persist
- Integration with `run_all_detectors` — fires "Unknown Anomaly" incident
- Feature vector extraction from sessions + evals DataFrames
- Auto-retrain trigger (every 10 new sessions)
- Dashboard: anomaly score column in Ledger grid, anomaly score overlay in Quality Drift timeline

### Phase D2: Sequence Pattern Mining
**Data requirement:** 50+ sessions
**Deliverables:**
- `engine/sequence_miner.py` — SequenceTrie, mine_patterns, match
- Offline batch job (not in real-time path)
- Discovered patterns added to named pattern library with human review gate

---

## Data Requirements Summary

| Phase | Sessions needed | Notes |
|---|---|---|
| Q1-Q4 (quality judge) | Any count | Works from session 1 |
| D1 (Isolation Forest) | 20+ | Training window; more = better |
| D2 (sequence mining) | 50+ | Split by regime at 100+ |
| Predictive early warning | 100+ | Need enough failure history |

Strategy C current count: ~27 sessions (as of 2026-05-30), growing each trading day.

---

## What This Does NOT Do

To be explicit about scope:

- Does NOT modify trading-agent-c's pipeline logic
- Does NOT run during agent execution — all quality evals are post-session async
- Does NOT make trading decisions — it evaluates existing decisions
- Does NOT require schema changes to c_sessions, c_traces, c_positions
- Phase Q1-Q3 do NOT require circuit breakers — observation only
- Circuit breakers (Phase Q4) remain shadow-mode until explicitly activated

The trading pipeline is unchanged. This layer reads what it produced and evaluates it.
