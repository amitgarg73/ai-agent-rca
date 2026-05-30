# AI Agent RCA — Product Idea (Brainstorm 2026-05-29)

## The Core Insight

Current observability tools (LangSmith, Langfuse, Datadog LLM Observability) show you
**what happened** — traces, costs, token counts.

None of them tell you **why** — what failure in the agent's reasoning or tool chain caused
the wasted spend or silent failure.

That gap is **AI Agent Root Cause Analysis (RCA)**.

---

## The Concrete Example

**Incident**: Strategy C research agent, 2026-05-28 06:34
- Cost: $1.98
- Duration: 1027 seconds
- Output: 0 trades

**Root cause (identified manually by reading traces)**:
yfinance API timed out. Research agent entered a retry loop — same tool call, repeated,
no exit condition. 45K tokens burned. Session ended with no useful output.

**What the product would have surfaced automatically**:
> "Tool Timeout Loop — yfinance API, 8 retries, 1027s, $1.98 wasted. Fix: add timeout
> handling and max retry limit to research agent tool configuration."

No human reads traces. System names the failure, quantifies the waste, suggests the fix.

---

## The Category: AI Agent AIOPS

Traditional AIOPS: infrastructure anomaly → automated root cause → remediation path.

AI Agent AIOPS: agent behavior anomaly → root cause in agent reasoning/tool chain → fix.

AIOPS buyers already understand this value. You are extending a known category into a new
domain, not inventing a new one from scratch.

---

## The Product: A Failure Pattern Library

There are probably 6-8 failure patterns that cover 80% of agent waste. Detect and name
them automatically from trace data.

| Pattern | Signal | Example |
|---|---|---|
| Tool Timeout Loop | Tool error → same tool retried N+ times | yfinance timeout → 8 retries |
| Context Spiral | Token count grows, no decision output | Research agent over-gathering |
| Empty Result Loop | Tool returns null → agent retries differently | News API returns empty |
| Schema Mismatch | Tool call error on parameters → agent retries same approach | Wrong API field name |
| Cascading Failure | Upstream agent bad output → downstream agents fail | Market agent wrong data format |
| Token Budget Exhaustion | Context window hit before decision | Long research session no output |
| Agent Deadlock | Two agents waiting on each other | Orchestrator + research stalled |

This library grows with more customers. More data = better pattern detection = better product.
That is a data network effect — defensible over time.

---

## Why This Framing Is Stronger Than "Dashboard"

- Dashboard = feature. Any team can build one in a day (we built ours in a day).
- RCA = product category. Requires pattern recognition, named taxonomy, cross-session learning.
- The pattern library creates a shared vocabulary. If you name the failure modes, you own
  the taxonomy — the same way "blue/green deployment" or "circuit breaker" became standard terms.
- AIOPS buyers recognize this immediately. No education required on why RCA matters.

---

## Domain-Agnostic — P&L Is Just One Example

The RCA problem is the same across agent types:
- Trading agents: wasted cost = zero trades (P&L outcome)
- Support agents: wasted cost = unresolved tickets (CSAT outcome)
- Coding agents: wasted cost = unmerged PRs (velocity outcome)
- Sales agents: wasted cost = no booked meetings (pipeline outcome)

The failure patterns (Tool Timeout Loop, Context Spiral, etc.) appear in all of them.
Outcome definition changes per domain; the detection layer does not.

---

## Competitive Position

**Who will get here eventually**: Dynatrace, Datadog — approaching from infrastructure upward.
They will be good at "the API was slow" and weak at "the agent's reasoning loop caused it."

**The window**: Go deep on agent-native RCA before infrastructure vendors climb up.
Agent-first understanding of failure modes is the moat while it lasts.

**Founder-market fit**: AIOPS/observability background means you know what good RCA looks
like. Most people building AI observability tools are coming from ML eval or DevOps, not
AIOPS. That is the unfair advantage.

---

## The Demo Build (2 hours)

Take existing Strategy C trace data. Write a detector for "Tool Timeout Loop":
- Look for: tool_call step type → error response → same tool called again within N seconds → repeat
- Fire alert: name the pattern, count retries, sum tokens, sum cost, suggest fix

Run it against the 2026-05-28 $1.98 incident. Record the output. That is the demo video.

---

## Open Questions for Further Debate

1. **Horizontal vs. vertical first** — build for all agent types immediately, or go deep in
   one domain (e.g., financial agents) to get the first paying customer?

2. **Detection approach** — rule-based pattern matching on traces (deterministic, fast to
   build) vs. LLM-based reasoning over trace summaries (flexible, expensive)?

3. **Where does the agent reasoning fit?** — tool call patterns are detectable from structured
   data; prompt/reasoning failures (hallucination, bad plan) require reading unstructured
   agent output. Different problem, much harder. Scope it in or out?

4. **Distribution wedge** — open source the pattern library to drive adoption, monetize
   on the platform? Or keep it proprietary?

5. **Relationship to existing schema work** — the OTel GenAI WG structured schema idea
   (tool_input/tool_output/agent_reasoning fields) is a prerequisite for detecting reasoning
   failures. Does RCA depend on that, or can it work from existing unstructured traces?

6. **First customer profile** — who feels this pain acutely enough to pay before the product
   is fully built? Teams running agents in production with real cost consequences and no
   internal platform team to build their own tooling.

---

## Evals as the Universal Outcome Signal (2026-05-29)

**The insight**: Instead of domain-specific business outcomes (P&L, tickets resolved,
conversion rate), use eval scores as the standardized outcome proxy. Every AI team already
runs evals. Pass them in. That becomes the "did this session succeed?" signal.

**Why this closes the generalization problem:**
Domain-specific outcomes require knowing the customer's business. Eval scores are
domain-agnostic in structure — you can eval any agent output with the same infrastructure.
The product works for trading agents, support agents, coding agents, sales agents without
needing to understand each domain.

**The missing link nobody has built:**
Evals and observability are two completely separate worlds today:
- Eval tools (Braintrust, Arize): "did the output quality meet the bar?"
- Observability tools (LangSmith, Langfuse): "what happened during execution?"

Nobody connects them. The product owns that connection:

> "This session scored 0.2 on your accuracy eval. Here's the exact tool call that caused
> it. It cost $0.43. The pattern is Schema Mismatch. Here's the fix."

**How it works:**
1. Customer instruments their agent (one decorator, OTel or custom JSON)
2. Customer runs evals they already have, or uses starter evals we provide
3. Eval score links back to session trace automatically
4. RCA runs: which step caused the eval failure, what did it cost, which pattern fired

**Why this solves the secret sauce problem:**
Eval-to-trace linkage creates the labeled failure dataset organically without domain
knowledge. Every customer who passes eval scores trains the pattern detector. That is the
data flywheel — and it does not require understanding each customer's domain at all.

**Competitive position:**
- Braintrust: evals, no trace RCA
- LangSmith: traces, no eval linkage
- This product: owns the connection layer between them

**Open questions this raises:**
- Evals can be wrong — if the customer's eval is bad, the outcome signal is noise. Do we
  provide reference evals, or trust whatever the customer brings?
- Evals are typically batch/offline. Does real-time RCA require real-time evals, or is
  post-session linkage sufficient?
- Braintrust could add trace linkage. How fast could they close this gap?

---

## Secret Sauce — Where Is the Moat? (Unresolved)

This is the right question to pressure-test. Honest assessment as of 2026-05-29.

**What feels like secret sauce but isn't:**
- Detecting "Tool Timeout Loop" from traces — anyone writes that rule in a weekend once they see the idea
- A dashboard with better charts — feature, not moat
- Being first — Datadog can replicate in a quarter with 100 engineers

**Three places real moat could come from:**

**A. Outcome-labeled failure data (flywheel)**
Knowing what "failure" looks like requires outcome ground truth. Most tools have traces but
no outcomes. Linking traces to business results lets you label which sessions were actually
failures vs. just slow. That labeled dataset trains better detectors than anyone else has.
More customers → more labeled failures → better detection → more customers.
Problem: circular early. Need customers to build the data.

**B. The reasoning failure layer (genuinely hard)**
Tool call pattern detection is easy — anyone can do it. Detecting that an agent made a bad
plan, hallucinated a fact, or chose the wrong tool for the goal requires reading unstructured
agent output and understanding intent. Nobody has cracked this. Requires AI expertise AND
deep observability domain knowledge simultaneously. If solved, defensible.

**C. OTel schema standard (political moat)**
Define the structured schema (tool_input, tool_output, agent_reasoning as typed fields) and
get the OTel GenAI working group to adopt it. Everyone who instruments their agents sends you
richer data than competitors receive. Not a technical moat but a standards moat. Slower to
build, much harder to displace once adopted.

**The uncomfortable truth:**
Right now the moat is founder-market fit: AIOPS background + a real running multi-agent
system + knowing what RCA looks like from the infrastructure side. Real advantage early,
not durable alone.

The actual secret sauce gets built after 10 customer conversations surface the one failure
pattern that keeps teams up at night and that nobody else has named yet.

**Status: unresolved. Revisit after first customer conversations.**

---

## Status

- Brainstormed: 2026-05-29
- Not validated — no customer conversations yet
- Demo build not started
- Secret sauce: unresolved (see section above)
- Relationship to existing PHASED_PLAN.md: this sharpens Phase 2 from "schema SDK" to
  "failure pattern detector"; Phase 1 (LinkedIn demand test) unchanged
