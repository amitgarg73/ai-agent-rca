"""
Generate AI_Agent_RCA_Architecture.docx
Run: python3 design/generate_architecture_doc.py
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

OUT = os.path.join(os.path.dirname(__file__), "AI_Agent_RCA_Architecture.docx")

NAVY  = RGBColor(0x0f, 0x17, 0x2a)
BLUE  = RGBColor(0x1e, 0x40, 0xaf)
SLATE = RGBColor(0x47, 0x55, 0x69)
RED   = RGBColor(0xdc, 0x26, 0x26)
GREEN = RGBColor(0x05, 0x96, 0x69)
AMBER = RGBColor(0xd9, 0x77, 0x06)
WHITE = RGBColor(0xff, 0xff, 0xff)
THEAD = RGBColor(0x1e, 0x3a, 0x5f)
LIGHT = RGBColor(0xf1, 0xf5, 0xf9)

doc = Document()

for sec in doc.sections:
    sec.top_margin    = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin   = Cm(2.5)
    sec.right_margin  = Cm(2.5)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _set_font(run, bold=False, italic=False, size=None, color=None, mono=False):
    run.bold   = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if mono:
        run.font.name = "Courier New"
        run._element.rPr.rFonts.set(qn("w:cs"), "Courier New")


def _shade_cell(cell, hex_color: str):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = NAVY if level == 1 else BLUE
        run.font.size = Pt(16 if level == 1 else 13)
    p.paragraph_format.space_before = Pt(18 if level == 1 else 12)
    p.paragraph_format.space_after  = Pt(6)
    return p


def body(text, bold=False, italic=False, color=None, size=10):
    p = doc.add_paragraph()
    r = p.add_run(text)
    _set_font(r, bold=bold, italic=italic, size=size, color=color)
    p.paragraph_format.space_after = Pt(4)
    return p


def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    _set_font(r, size=10, color=SLATE)
    p.paragraph_format.left_indent  = Inches(0.25 * (level + 1))
    p.paragraph_format.space_after  = Pt(2)
    return p


def code_block(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    _set_font(r, mono=True, size=8.5, color=NAVY)
    return p


def divider():
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"),   "single")
    bottom.set(qn("w:sz"),    "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "CBD5E1")
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p


def table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    # Header row
    hdr = t.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        _shade_cell(cell, "1E3A5F")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        _set_font(r, bold=True, size=9, color=WHITE)
    # Data rows
    for ri, row in enumerate(rows):
        tr = t.rows[ri + 1]
        fill = "F8FAFC" if ri % 2 == 0 else "FFFFFF"
        for ci, cell_text in enumerate(row):
            cell = tr.cells[ci]
            _shade_cell(cell, fill)
            p = cell.paragraphs[0]
            r = p.add_run(str(cell_text))
            _set_font(r, size=9, color=NAVY)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


# ── Cover ──────────────────────────────────────────────────────────────────────

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("AI Agent RCA")
_set_font(r, bold=True, size=28, color=NAVY)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Architecture and Technical Design")
_set_font(r, bold=False, size=16, color=SLATE)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Real-time failure detection, semantic quality evaluation, and root cause analysis\nfor multi-agent AI systems")
_set_font(r, italic=True, size=11, color=SLATE)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Amit Garg   |   amitgar@hotmail.com   |   June 2026")
_set_font(r, size=10, color=SLATE)

doc.add_page_break()


# ── 1. Core Problem ────────────────────────────────────────────────────────────

heading("1. Core Problem")
body(
    "AI agents fail silently. HTTP 200, no exception — but money burned and no output produced. "
    "Current observability tools show traces (what happened). None tell you why, which step caused it, "
    "or what it cost. Quality degrades quietly before operational metrics move."
)
doc.add_paragraph()
body("Concrete example — Strategy C, 2026-05-28 01:26 UTC:", bold=True)
bullet("Cost: $1.98  |  Duration: 1,027s  |  Output: 0 trades")
bullet("HTTP 200 on every call. No exception. Session record: completed.")
bullet("Root cause: yfinance API timed out. Agent entered a retry loop with no exit condition.")
bullet("What the product surfaces: \"Tool Timeout Loop — yfinance, 8 retries, $1.98 wasted. Fix: add max_retries=2 and timeout=10s.\"")
doc.add_paragraph()
body(
    "The deeper problem: quality degradation precedes structural failure by 3 to 10 sessions. "
    "An agent produces less grounded recommendations, then less coherent reasoning, then causes an incident. "
    "Operational metrics catch it at session 12. Quality metrics catch it at session 6.",
    italic=True, color=BLUE
)

divider()


# ── 2. System Architecture ─────────────────────────────────────────────────────

heading("2. System Architecture")

body("Four layers work together. Each layer feeds the next. Nothing runs in the agent critical path.", color=SLATE)
doc.add_paragraph()

for line in [
    "┌──────────────────────────────────────────────────────────────────┐",
    "│         STRATEGY C  (6-agent live trading pipeline)              │",
    "│  Market → Research → Risk → Orchestrator → Trade Execution       │",
    "└────────────────────────────┬─────────────────────────────────────┘",
    "                             │ writes c_sessions, c_traces, c_positions",
    "                             ▼",
    "┌──────────────────────────────────────────────────────────────────┐",
    "│                         SUPABASE                                 │",
    "│  c_sessions | c_traces | c_positions  (existing, read-only)      │",
    "│  c_evals | c_incidents | c_tool_spans (added by this system)     │",
    "└────────────┬─────────────────────────────────────────────────────┘",
    "             │ post-session async  (never blocks agent pipeline)",
    "     ┌───────┴────────────────────────┐",
    "     ▼                                ▼",
    "┌──────────────────┐      ┌───────────────────────┐",
    "│   Eval Engine    │      │    Quality Judge       │",
    "│  17 rule-based   │      │  Haiku, ~$0.002/sess   │",
    "│  evals, <1ms     │      │  20 dims, 4 composites │",
    "│  Layer 1: ops    │      │  Layer 3: semantic     │",
    "│  Layer 2: biz    │      │  quality per agent     │",
    "└────────┬─────────┘      └───────────┬────────────┘",
    "         └──────────┬─────────────────┘",
    "                    ▼",
    "┌──────────────────────────────────────────────────────────────────┐",
    "│                    Pattern Detector                              │",
    "│  10 named patterns (rule-based) + Isolation Forest (ML)         │",
    "│  Shadow circuit breakers: compute_shadow_cb_fires()             │",
    "└────────────────────────────┬─────────────────────────────────────┘",
    "                             ▼",
    "┌──────────────────────────────────────────────────────────────────┐",
    "│                      RCA Engine                                  │",
    "│  call stack builder  |  fix suggestions  |  cost attribution    │",
    "└────────────────────────────┬─────────────────────────────────────┘",
    "                             ▼",
    "┌──────────────────────────────────────────────────────────────────┐",
    "│                      Dashboard (Streamlit)                       │",
    "│  Ledger | Quality Drift | Incidents Feed | RCA View              │",
    "│  Failure Simulator | Trace Inspector                             │",
    "└──────────────────────────────────────────────────────────────────┘",
]:
    code_block(line)

divider()


# ── 3. Three-Layer Eval Taxonomy ───────────────────────────────────────────────

heading("3. Three-Layer Eval Taxonomy")

body(
    "Evals run post-session, async, never in the agent critical path. "
    "Each layer answers a different question.", color=SLATE
)
doc.add_paragraph()

table(
    ["Layer", "Question", "Method", "Count", "Cost", "Storage"],
    [
        ["1 — Operational", "Did the pipeline run correctly?", "Rule-based, pure Python", "17 evals", "<1ms, $0", "c_evals"],
        ["2 — Business", "Did the pipeline produce value?", "Computed on-the-fly in dashboard", "3 evals", "<1ms, $0", "Not stored"],
        ["3 — Semantic", "Did each agent reason well?", "LLM-as-judge (Haiku), batched", "20 dims, 4 composites", "~$0.002/session", "c_evals"],
    ],
    col_widths=[1.2, 1.8, 1.6, 0.9, 1.0, 0.9],
)

heading("Layer 1: Operational Evals (17)", level=2)
table(
    ["Agent", "Eval", "Pass Condition", "Threshold"],
    [
        ["market",       "data_completeness",  "Market agent produced at least one successful trace", "0.70"],
        ["market",       "data_freshness",     "First market trace within 10 min of session start",  "0.50"],
        ["research",     "completion",         "LLM ran + at least one tool call succeeded",          "0.70"],
        ["research",     "token_efficiency",   "< 30K tokens total",                                  "0.50"],
        ["research",     "tool_success_rate",  ">= 80% of tool calls succeeded",                      "0.80"],
        ["research",     "tool_diversity",     ">= 2 distinct tools called",                          "2"],
        ["risk",         "assessment_complete","At least one successful risk trace",                  "1.00"],
        ["risk",         "within_parameters",  "No error traces",                                     "0.90"],
        ["orchestrator", "decision_made",      "Trades executed or named good exit",                  "0.70"],
        ["orchestrator", "exit_quality",       "terminal_reason in good exit set",                    "0.70"],
        ["session",      "pipeline_completion","All 4 agents present",                                "1.00"],
        ["session",      "cost_anomaly",       "Cost within 2 sigma of rolling mean",                 "0.50"],
        ["session",      "outcome_linkage",    "Trades or named exit reason",                         "0.70"],
        ["session",      "tokens_per_decision","< 40K tokens",                                        "0.50"],
    ],
    col_widths=[1.1, 1.6, 2.8, 0.8],
)

heading("Layer 2: Business Outcome Evals (3, on-the-fly)", level=2)
table(
    ["Eval", "Formula", "Threshold"],
    [
        ["cost_per_trade",      "total_cost_usd / max(1, trades_executed)",          "< $0.50"],
        ["research_conversion", "trades_executed / successful_research_llm_calls",   "> 30%"],
        ["proposal_acceptance", "trades_executed / trades_proposed",                 "> 40%"],
    ],
    col_widths=[1.6, 3.4, 1.2],
)

divider()


# ── 4. Semantic Quality Scoring ────────────────────────────────────────────────

heading("4. Semantic Quality Scoring")

body(
    "Quality evals catch what operational evals miss: sessions where everything ran correctly "
    "but agents produced low-quality reasoning. Scoring is structural proxy — inferred from trace "
    "patterns (tool sequences, token counts, decision traces) without storing raw LLM output text.",
    color=SLATE
)
doc.add_paragraph()
body("Degradation sequence:", bold=True)
bullet("Sessions 1–5:   quality healthy, ops healthy")
bullet("Sessions 6–8:   data_grounding declining (0.80 → 0.60 → 0.45) — ops still passing")
bullet("Sessions 9–11:  risk research_consistency drops — ops still passing")
bullet("Session 12:     pipeline break or cost anomaly fires — ops catches it now")
doc.add_paragraph()
body("Quality signals the problem at session 6. Ops catches it at session 12.", italic=True, color=BLUE)
doc.add_paragraph()

heading("Research Agent (5 dimensions)", level=2)
table(
    ["Dimension", "Weight", "What it measures"],
    [
        ["data_grounding",        "25%", "Did research cite specific price levels and quantitative indicators?"],
        ["thesis_coherence",      "25%", "Did research produce a clear directional thesis with evidence?"],
        ["actionability",         "20%", "Was the recommendation specific enough for risk to evaluate?"],
        ["catalyst_specificity",  "15%", "Did research identify a current reason to act (news, breakout, earnings)?"],
        ["volatility_accounting", "15%", "Did research check ATR or volatility before sizing the recommendation?"],
    ],
    col_widths=[1.8, 0.7, 3.8],
)
body("CB threshold: composite < 0.60 or data_grounding < 0.40", color=SLATE, size=9)
doc.add_paragraph()

heading("Risk Agent (4 dimensions)", level=2)
table(
    ["Dimension", "Weight", "What it measures"],
    [
        ["research_consistency",     "30%", "Did risk directly assess the research recommendation, not a generic scenario?"],
        ["parameter_completeness",   "30%", "Did risk output direction + position size + stop loss?"],
        ["stop_loss_quality",        "25%", "Was the stop loss anchored to a technical level, not an arbitrary percentage?"],
        ["position_sizing_rationale","15%", "Was position size derived from portfolio %, stop distance, and ATR?"],
    ],
    col_widths=[1.8, 0.7, 3.8],
)
body("CB threshold: composite < 0.60 or research_consistency < 0.50", color=SLATE, size=9)
doc.add_paragraph()

heading("Orchestrator Agent (4 dimensions)", level=2)
table(
    ["Dimension", "Weight", "What it measures"],
    [
        ["decision_consistency",    "35%", "Did the final decision align with both research AND risk outputs?"],
        ["resolution_completeness", "30%", "Did the orchestrator reach a definitive conclusion (execute or no-trade)?"],
        ["reasoning_transparency",  "20%", "Did the orchestrator log the chain: research said X → risk confirmed Y → decision Z?"],
        ["upstream_integration",    "15%", "Did the orchestrator cite specific data from both upstream agents?"],
    ],
    col_widths=[1.8, 0.7, 3.8],
)
body("CB threshold: composite < 0.60 or decision_consistency < 0.50", color=SLATE, size=9)
doc.add_paragraph()

heading("Session Coherence (cross-agent, 2 dimensions)", level=2)
table(
    ["Dimension", "What it measures"],
    [
        ["pipeline_coherence", "Same ticker and direction consistent throughout all agents"],
        ["reasoning_chain",    "Each agent explicitly builds on the previous agent's output"],
    ],
    col_widths=[1.8, 4.5],
)

divider()


# ── 5. LLM Quality Drift Signals ──────────────────────────────────────────────

heading("5. LLM Quality Drift Detection")

body(
    "Structural proxy scoring catches most quality degradation. These additional signals extend "
    "detection range — some are available now from existing trace data, others require output text storage.",
    color=SLATE
)
doc.add_paragraph()

heading("Available now (no schema change required)", level=2)
table(
    ["Signal", "Source", "How to detect", "What it catches"],
    [
        ["Output token trend",    "c_traces output_tokens",    "Rolling avg per agent; flag >20% shift over 5 sessions",           "LLMs producing shorter outputs after model updates; 'lazy' degradation"],
        ["Tool retry rate",       "c_traces",                  "Count repeated calls to same tool per agent per session",          "Rising retries = agent not satisfied with first result; precedes tool_success_rate drop"],
        ["Decision variance",     "c_sessions",                "Same market condition → different decisions across sessions",      "Coherence drift — inconsistency under similar inputs"],
        ["Model version tracking","c_sessions metadata",       "Add model_name field; flag sessions where version changes",        "Silent model updates (Anthropic, OpenAI) — most common cause of systemic drift"],
    ],
    col_widths=[1.5, 1.4, 2.0, 2.3],
)

heading("Requires output text storage", level=2)
table(
    ["Signal", "Method", "What it catches"],
    [
        ["Embedding drift",               "Cosine similarity of output embeddings to rolling baseline",          "Generic/repetitive outputs that pass all structural checks"],
        ["Hedge word frequency",          "Count 'unable to determine', 'insufficient data', 'unclear' per agent","Models expressing uncertainty through hedging — early grounding failure signal"],
        ["Structured output completeness","Did agent populate all expected output schema fields?",               "Partial outputs that pass structural checks but are semantically incomplete"],
    ],
    col_widths=[1.8, 2.8, 2.6],
)
body("Implementation priority: output token trend + model version tracking first (zero schema change, 10-line additions).", italic=True, color=BLUE, size=9)

divider()


# ── 6. Pattern Library ────────────────────────────────────────────────────────

heading("6. Pattern Library")

body("Named failure patterns create a shared vocabulary. Each pattern has a detection rule, severity, and fix template.", color=SLATE)
doc.add_paragraph()

heading("Operational Patterns (10, rule-based, implemented)", level=2)
table(
    ["Pattern", "Severity", "Detection rule", "Fix"],
    [
        ["Tool Timeout Loop",        "critical", "Same tool has 3+ error traces",                        "max_retries=2 + timeout=10s"],
        ["Context Spiral",           "warning",  "Research > 40K tokens + 0 trades",                     "Hard token budget; force decision step"],
        ["Pipeline Break",           "critical", "Research ran, orchestrator never started",              "Catch exceptions at research→orchestrator handoff"],
        ["Empty Result Loop",        "warning",  "Same tool called 3+ times successfully + 0 trades",    "Result quality validation; fail fast on empty returns"],
        ["Silent Exit",              "info",     "0 trades + no terminal_reason + no tool errors",       "Add terminal_reason to all exit paths"],
        ["Cost Anomaly",             "warning",  "Session cost > mean + 2 sigma",                        "Check which agent consumed excess"],
        ["Hyperactive Polling Loop", "warning",  "Same tool called 5+ times in <60s, same args",         "Add result caching and backoff logic"],
        ["Tool Call Fabrication",    "critical", "Tool call references non-existent tool or invalid args","Add tool schema validation before execution"],
        ["Handoff Schema Break",     "critical", "Agent output does not match next agent's input schema", "Schema validation at each handoff boundary"],
        ["Error Misinterpretation",  "warning",  "Agent proceeds normally after tool error response",     "Explicit error detection; treat error responses as failed steps"],
    ],
    col_widths=[1.6, 0.75, 2.1, 1.8],
)

heading("Quality Patterns (4, planned — Phase Q3)", level=2)
table(
    ["Pattern", "Severity", "Detection rule", "Fix"],
    [
        ["Grounding Failure",  "warning",          "research.data_grounding < 0.40 for 3 consecutive sessions",       "Add output requirement: cite at least one price level and one indicator"],
        ["Coherence Break",    "critical",          "orchestrator.decision_consistency < 0.50 in any session",         "Structured handoff format; consistency check in orchestrator prompt"],
        ["Quality Cascade",    "warning → critical","3+ quality dimensions declining >0.20 over 5 sessions",           "Review agent prompts for each degrading dimension; check market data quality"],
        ["Silent Degradation", "info → warning",    "Composite quality declining while all operational metrics stable", "Manual review of recent outputs; check for prompt drift or model version change"],
    ],
    col_widths=[1.6, 0.75, 2.2, 1.7],
)

heading("Dynamic Patterns (Isolation Forest, implemented)", level=2)
body(
    "Catches unknown failure modes not covered by any named rule. Each session scored on anomaly "
    "distance from the pipeline's historical baseline. Score > 0.65 generates an 'Unknown Anomaly' "
    "incident with the top deviating features and a manual review prompt."
)
body("Feature vector: total_cost_usd, total_tokens, latency_ms, tool_error_rate, pipeline_completion, op_score, biz_score, trades_executed, quality composites per agent", color=SLATE, size=9)

divider()


# ── 7. Circuit Breaker Design ──────────────────────────────────────────────────

heading("7. Circuit Breaker Design")

body("CBs run in-process after each agent using in-memory traces. Zero DB round trips. Compute time: ~0.5ms per agent.", color=SLATE)
doc.add_paragraph()

table(
    ["Mode", "Behavior", "Status"],
    [
        ["Shadow (current)", "CB fires, records would_trigger_cb=True, never aborts pipeline. Dashboard shows where CB would have fired + cost saved.", "Active — Strategy C still in testing"],
        ["Real (post-validation)", "Same code path. Raises CircuitBreakerError. Orchestrator catches it, writes terminal_reason='circuit_breaker', stops pipeline.", "Blocked until false positive rate < 10%"],
    ],
    col_widths=[1.3, 4.2, 1.7],
)
doc.add_paragraph()

heading("CB Configuration", level=2)
for line in [
    "# Operational CBs",
    "CIRCUIT_BREAKERS = {",
    "    'research': [('tool_success_rate', 0.80), ('completion', 0.70)],",
    "    'risk':     [('assessment_complete', 1.0)],",
    "}",
    "",
    "# Quality CBs (shadow post-session, Phase Q4)",
    "QUALITY_CIRCUIT_BREAKERS = {",
    "    'research':     {'composite_score': 0.60, 'data_grounding': 0.40},",
    "    'risk':         {'composite_score': 0.60, 'research_consistency': 0.50},",
    "    'orchestrator': {'decision_consistency': 0.50},",
    "}",
]:
    code_block(line)

divider()


# ── 8. Dashboard ───────────────────────────────────────────────────────────────

heading("8. Dashboard Structure")

body("Six-page Streamlit app. Password protected. Live on Streamlit Cloud.", color=SLATE)
doc.add_paragraph()

table(
    ["Page", "Content", "Key interactions"],
    [
        ["Ledger",            "KPIs, Cost by Agent donut, AgGrid session grid, inline session detail, LLM analyst summary",             "Click row for detail; agent pills for cost breakdown"],
        ["Quality Drift",     "Two tabs: Operational/Business Health (session health timeline, eval trends, business outcomes) and Semantic Health (drift cards, agent composites, dimension breakdown)", "Tab switching; session picker for dimension breakdown"],
        ["Incidents Feed",    "Filtered incident list with severity, pattern, cost wasted; bulk analysis trigger",                       "Filter by severity/pattern; RCA link per incident"],
        ["RCA View",          "Full execution trace, inline eval scores, pattern detail expander, fix suggestion",                       "Session selector; expandable pattern detail"],
        ["Failure Simulator", "Inject synthetic traces for any of 10 named patterns, run analysis, view results",                       "Pattern selector; dry-run vs persist"],
        ["Trace Inspector",   "Raw trace browser with session + agent filter",                                                           "Session + agent filter"],
    ],
    col_widths=[1.3, 3.4, 1.6],
)

divider()


# ── 9. Data Model ──────────────────────────────────────────────────────────────

heading("9. Key Data Tables")

heading("c_evals (operational + quality evals)", level=2)
for line in [
    "session_id  uuid    -- links to c_sessions",
    "agent       text    -- 'research' | 'risk_quality' | 'session' | ...",
    "eval_name   text    -- 'tool_success_rate' | 'thesis_coherence' | ...",
    "score       float   -- 0.0 to 1.0",
    "passed      boolean",
    "threshold   float",
    "detail      jsonb   -- raw evidence from judge or rule",
]:
    code_block(line)
doc.add_paragraph()

heading("c_incidents (pattern detector output)", level=2)
for line in [
    "session_id     uuid",
    "pattern_name   text    -- 'Tool Timeout Loop' | 'Unknown Anomaly' | ...",
    "severity       text    -- 'critical' | 'warning' | 'info'",
    "root_cause     text",
    "call_stack     jsonb",
    "failed_evals   jsonb   -- list of {agent, eval_name, score, threshold}",
    "cost_wasted    float",
    "fix_suggestion text",
    "is_simulated   boolean",
]:
    code_block(line)
doc.add_paragraph()

heading("c_tool_spans (SDK tool tracer)", level=2)
for line in [
    "session_id   uuid",
    "agent        text",
    "tool_name    text",
    "latency_ms   float",
    "llm_cost_usd float   -- nested LLM cost inside tool, if any",
    "error        text",
]:
    code_block(line)

divider()


# ── 10. Roadmap ────────────────────────────────────────────────────────────────

heading("10. Roadmap")

table(
    ["Phase", "What", "Status", "Gate"],
    [
        ["Q3",  "Quality patterns: Grounding Failure, Coherence Break, Quality Cascade, Silent Degradation",        "Not started",    "Q2 done"],
        ["Q4",  "Shadow quality CBs: would_trigger_cb flag, CB config, dashboard CB markers with cost-saved",       "Not started",    "Q3 done"],
        ["4a",  "In-process operational CBs in trading-agent-c (shadow, in-memory, ~0.5ms)",                        "Not started",    "Q4 done"],
        ["4c",  "Flip CBs to real mode — ops CBs first, quality CBs after validation",                              "Blocked",        "Strategy C testing complete; <10% false positives"],
        ["D2",  "Sequence pattern mining — discover unnamed failure patterns from trace subsequences",               "Not started",    "50+ sessions"],
        ["P1",  "Predictive early warning — declining score trend → incident probability; weekly health report",     "Not started",    "100+ sessions"],
        ["LDQ", "LLM drift signals: output token trend, model version tracking, tool retry rate",                   "Planned",        "Next sprint"],
        ["9",   "Multi-tenant: per-customer Isolation Forest, tenant_id schema, hosted auth",                       "Not started",    "LinkedIn demand test: 3+ engineer DMs"],
    ],
    col_widths=[0.55, 3.2, 1.05, 1.9],
)

divider()


# ── 11. Tech Stack ─────────────────────────────────────────────────────────────

heading("11. Tech Stack")

table(
    ["Component", "Technology", "Notes"],
    [
        ["Data store",       "Supabase (PostgreSQL)",         "Shared with trading-agent-c; schema extended, not modified"],
        ["Eval engine",      "Pure Python",                   "17 evals, <1ms, zero LLM cost"],
        ["Quality judge",    "Anthropic Haiku (claude-haiku-4-5)", "Batched per session, prompt-cached system prompt, ~$0.002/session"],
        ["Anomaly detector", "scikit-learn IsolationForest",  "Persisted to engine/if_model.pkl; auto-retrain every 10 sessions"],
        ["Realtime monitor", "Supabase Realtime websocket",   "Polling fallback (30s); runs evals + quality judge post-session"],
        ["Dashboard",        "Streamlit + Plotly + AgGrid",   "Streamlit Cloud, password-protected"],
        ["Tool tracer SDK",  "Python decorator (@trace_tool)","Thread-local span stack; writes c_tool_spans"],
        ["Tests",            "pytest",                        "211 passing across evals, patterns, quality judge, realtime monitor"],
    ],
    col_widths=[1.4, 2.2, 2.7],
)

doc.add_paragraph()
p = doc.add_paragraph()
r = p.add_run("GitHub: github.com/amitgarg73/ai-agent-rca  |  Dashboard: Streamlit Cloud  |  Contact: amitgar@hotmail.com")
_set_font(r, size=9, color=SLATE)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.save(OUT)
print(f"Written: {OUT}")
