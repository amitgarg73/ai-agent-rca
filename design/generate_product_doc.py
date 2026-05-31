"""
Generate AI_Agent_RCA_Product_Design.docx from product-design.md content.
Run: python3 design/generate_product_doc.py
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy, os

OUT = os.path.join(os.path.dirname(__file__), "AI_Agent_RCA_Product_Design.docx")

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x0f, 0x17, 0x2a)
BLUE   = RGBColor(0x1e, 0x40, 0xaf)
SLATE  = RGBColor(0x47, 0x55, 0x69)
LIGHT  = RGBColor(0xf1, 0xf5, 0xf9)
RED    = RGBColor(0xdc, 0x26, 0x26)
GREEN  = RGBColor(0x05, 0x96, 0x69)
AMBER  = RGBColor(0xd9, 0x77, 0x06)
WHITE  = RGBColor(0xff, 0xff, 0xff)
THEAD  = RGBColor(0x1e, 0x3a, 0x5f)

doc = Document()

# ── Page margins ───────────────────────────────────────────────────────────────
for sec in doc.sections:
    sec.top_margin    = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin   = Cm(2.5)
    sec.right_margin  = Cm(2.5)

# ── Style helpers ──────────────────────────────────────────────────────────────
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

def _set_cell_border(cell, **kwargs):
    """Set cell borders. kwargs: top, bottom, left, right = (size, color_hex)"""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side, (sz, col) in kwargs.items():
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"),   "single")
        el.set(qn("w:sz"),    str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), col)
        tcBorders.append(el)
    tcPr.append(tcBorders)

def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after  = Pt(6)
    run = p.add_run(text)
    _set_font(run, bold=True, size=18, color=NAVY)
    # Bottom border
    pPr  = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot  = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), "1e40af")
    pBdr.append(bot)
    pPr.append(pBdr)
    return p

def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    _set_font(run, bold=True, size=13, color=BLUE)
    return p

def h3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(3)
    run = p.add_run(text)
    _set_font(run, bold=True, size=11, color=SLATE)
    return p

def body(text, indent=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    if indent:
        p.paragraph_format.left_indent = Cm(0.6)
    run = p.add_run(text)
    _set_font(run, size=10, color=NAVY)
    return p

def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent  = Cm(0.4 + level * 0.4)
    p.paragraph_format.space_after  = Pt(2)
    # parse inline bold (**...**)
    parts = text.split("**")
    for i, part in enumerate(parts):
        run = p.add_run(part)
        _set_font(run, bold=(i % 2 == 1), size=10, color=NAVY)
    return p

def code_block(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.5)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    # Light grey shading via XML
    pPr  = p._p.get_or_add_pPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  "F1F5F9")
    pPr.append(shd)
    run = p.add_run(text)
    _set_font(run, size=8.5, color=SLATE, mono=True)
    return p

def note(text):
    """Italic caption / note paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    _set_font(run, italic=True, size=9, color=SLATE)
    return p

def table(headers, rows, col_widths=None):
    n_cols = len(headers)
    t = doc.add_table(rows=1 + len(rows), cols=n_cols)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hrow = t.rows[0]
    for i, hdr in enumerate(headers):
        cell = hrow.cells[i]
        _shade_cell(cell, "1e3a5f")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(hdr)
        _set_font(run, bold=True, size=9, color=WHITE)

    # Data rows
    for ri, row_data in enumerate(rows):
        row = t.rows[ri + 1]
        fill = "F8FAFC" if ri % 2 == 0 else "FFFFFF"
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            _shade_cell(cell, fill)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            # parse bold markdown inline
            parts = str(cell_text).split("**")
            for pi, part in enumerate(parts):
                run = p.add_run(part)
                _set_font(run, bold=(pi % 2 == 1), size=9, color=NAVY)

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def divider():
    p = doc.add_paragraph()
    pPr  = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot  = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "4")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), "CBD5E1")
    pBdr.append(bot)
    pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(6)

def callout(text, color_hex="EFF6FF", border_hex="3B82F6"):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.4)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(6)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  color_hex)
    pPr.append(shd)
    pBdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"),   "single")
    left.set(qn("w:sz"),    "18")
    left.set(qn("w:space"), "4")
    left.set(qn("w:color"), border_hex)
    pBdr.append(left)
    pPr.append(pBdr)
    run = p.add_run(text)
    _set_font(run, size=9.5, color=NAVY)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ══════════════════════════════════════════════════════════════════════════════

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(40)
p.paragraph_format.space_after  = Pt(6)
run = p.add_run("AI Agent RCA")
_set_font(run, bold=True, size=32, color=NAVY)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

p2 = doc.add_paragraph()
p2.paragraph_format.space_after = Pt(4)
run2 = p2.add_run("Product Design")
_set_font(run2, size=18, color=BLUE)

note("Single source of truth  ·  Last updated: 2026-05-31")
divider()

callout(
    "Current observability tools show you WHAT happened — traces, costs, token counts. "
    "None tell you WHY — what failure in the agent's reasoning or tool chain caused the wasted "
    "spend or silent failure. That gap is AI Agent Root Cause Analysis.",
    "EFF6FF", "1E40AF"
)

doc.add_page_break()


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORE INSIGHT
# ══════════════════════════════════════════════════════════════════════════════
h1("1. Core Insight")

body(
    "Current observability tools show you what happened — traces, costs, token counts. "
    "None tell you why — what failure in the agent's reasoning or tool chain caused the "
    "wasted spend or silent failure."
)

h3("The Concrete Example")
callout(
    "Strategy C research agent  ·  2026-05-28 01:26 UTC\n"
    "Cost: $1.98  |  Duration: 1027s  |  Output: 0 trades\n\n"
    "HTTP 200 on every call. No exception. Session record: \"completed.\"\n\n"
    "Root cause: yfinance API timed out. Agent entered a retry loop — no exit condition. "
    "8 retries, 45K tokens burned.\n\n"
    "What the product surfaces automatically:\n"
    "\"Tool Timeout Loop — yfinance, 8 retries, $1.98 wasted. "
    "Fix: add max_retries=2 and timeout=10s to the research agent tool configuration.\"",
    "FFF7ED", "D97706"
)

body(
    "No human reads traces. The system names the failure, quantifies the waste, and suggests the fix."
)

h3("The Deeper Problem: Quality Leads Operational by 3-10 Sessions")
body(
    "An agent starts producing less grounded recommendations, then less coherent reasoning, "
    "then eventually causes an incident. If you only watch operational metrics, you see the incident. "
    "If you watch quality metrics, you see it coming."
)
code_block(
    "Sessions 1-5:   quality healthy, operational healthy\n"
    "Sessions 6-8:   data_grounding declining (0.80 → 0.60 → 0.45) — ops still green\n"
    "Sessions 9-11:  risk research_consistency drops — ops still green\n"
    "Session 12:     pipeline break fires — ops catches it now\n\n"
    "Quality signals the problem at session 6. Ops catches it at session 12."
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════════════════════
h1("2. Problem Statement — Three Gaps")

h3("Gap 1: Traces Without Diagnosis")
body(
    "Observability tools show what happened. None tell you why. Reading raw traces to find a "
    "root cause is manual, slow, and requires the person who built the system. It does not scale."
)

h3("Gap 2: Evals Disconnected from Execution")
body(
    "Evals validate output quality but sit in a separate tool, disconnected from the trace that "
    "produced the output. You know the answer was wrong. You do not know which agent step caused "
    "it or what it cost."
)

h3("Gap 3: No Named Failure Taxonomy")
body(
    "Every incident is bespoke. Teams debug from scratch. There is no shared vocabulary for "
    "\"Tool Timeout Loop\" or \"Context Spiral\" — no pattern library that lets you recognize "
    "a failure the moment you see its signature."
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 3. EVAL TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════
h1("3. Eval Taxonomy — Three Layers")

table(
    ["Layer", "Description", "Count", "Cost", "When"],
    [
        ["1 — Operational", "Did the pipeline run correctly?", "17 evals", "< 1ms, zero LLM cost", "On-demand / backfill"],
        ["2 — Business Outcomes", "Did the pipeline produce value?", "3 evals", "Zero (computed from data)", "On-the-fly in dashboard"],
        ["3 — Semantic Quality", "Did each agent do its job well?", "Per-agent per-dimension", "~$0.002/session (Haiku)", "Async post-session"],
    ],
    col_widths=[1.2, 2.2, 1.1, 1.5, 1.4]
)

h2("Layer 1: Operational Evals (17 total)")

h3("Market Agent")
table(
    ["Eval", "Pass Condition", "Threshold"],
    [
        ["data_completeness", "Market agent produced at least one successful trace", "0.70"],
        ["data_freshness", "First market trace within 10 min of session start", "0.50"],
    ],
    col_widths=[1.5, 3.8, 1.0]
)

h3("Research Agent")
table(
    ["Eval", "Pass Condition", "Threshold"],
    [
        ["completion", "LLM ran + at least one tool call succeeded", "0.70"],
        ["token_efficiency", "< 30K tokens total", "0.50"],
        ["tool_success_rate", ">= 80% of tool calls succeeded", "0.80"],
        ["tool_diversity", ">= 2 distinct tools called", "2"],
    ],
    col_widths=[1.5, 3.8, 1.0]
)

h3("Risk Agent")
table(
    ["Eval", "Pass Condition", "Threshold"],
    [
        ["assessment_complete", "At least one successful risk trace", "1.00"],
        ["within_parameters", "No error traces", "0.90"],
    ],
    col_widths=[1.5, 3.8, 1.0]
)

h3("Orchestrator Agent")
table(
    ["Eval", "Pass Condition", "Threshold"],
    [
        ["decision_made", "Trades executed or named good exit", "0.70"],
        ["exit_quality", "terminal_reason in good exit set", "0.70"],
    ],
    col_widths=[1.5, 3.8, 1.0]
)

h3("Session Holistic")
table(
    ["Eval", "Pass Condition", "Threshold"],
    [
        ["pipeline_completion", "All 4 agents present", "1.00"],
        ["cost_anomaly", "Cost within 2 sigma of rolling mean", "0.50"],
        ["outcome_linkage", "Trades or named exit reason", "0.70"],
        ["tokens_per_decision", "< 40K tokens", "0.50"],
    ],
    col_widths=[1.5, 3.8, 1.0]
)

h2("Layer 2: Business Outcome Evals (3 total, on-the-fly)")
table(
    ["Eval", "Formula", "Pass Threshold"],
    [
        ["cost_per_trade", "total_cost_usd / max(1, trades_executed)", "< $0.50"],
        ["research_conversion", "trades_executed / successful research LLM calls", "> 30%"],
        ["proposal_acceptance", "trades_executed / trades_proposed", "> 40%"],
    ],
    col_widths=[1.5, 3.2, 1.5]
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEMANTIC QUALITY RUBRICS
# ══════════════════════════════════════════════════════════════════════════════
h1("4. Semantic Quality Rubrics (Layer 3)")

body(
    "Operational evals catch infrastructure failures. Quality evals catch semantic failures — "
    "sessions where everything ran correctly but agents produced low-quality outputs. "
    "Quality degradation signals problems 3-10 sessions before operational metrics move."
)

h2("Research Agent Quality")
note("Highest-value evaluation point. Research quality is the strongest predictor of downstream agent performance.")

table(
    ["Dimension", "Weight", "Score 1.0", "Score 0.5", "Score 0.0"],
    [
        ["data_grounding", "0.25", "Cites specific price levels, volume, named catalyst", "\"Volume elevated\", \"positive momentum\"", "No data references"],
        ["thesis_coherence", "0.25", "Clear directional thesis with supporting evidence", "Thesis present, weakly supported", "No thesis or contradicts evidence"],
        ["actionability", "0.20", "Direction + entry zone + specific ticker", "Direction and ticker, no entry", "Vague, not actionable"],
        ["catalyst_specificity", "0.15", "\"Breakout above $875\" or specific date", "\"Some upward momentum\"", "No catalyst"],
        ["risk_acknowledgment", "0.15", "\"If fails to hold $860, thesis invalidated\"", "Generic \"there is risk\"", "No downside mentioned"],
    ],
    col_widths=[1.4, 0.6, 1.7, 1.6, 1.5]
)
callout("CB threshold: composite < 0.60 OR data_grounding < 0.40", "FEF2F2", "DC2626")

h2("Risk Agent Quality")
table(
    ["Dimension", "Weight", "Score 1.0", "Score 0.5", "Score 0.0"],
    [
        ["research_consistency", "0.25", "Directly assesses research's specific ticker + direction", "Same ticker, generic depth", "Mismatch with research"],
        ["parameter_completeness", "0.25", "Direction + position size + stop loss all present", "2 of 3", "Missing critical parameters"],
        ["volatility_accounting", "0.20", "\"ATR $4.20 → sizing 150sh so 1 ATR = 0.84% portfolio\"", "\"Market is volatile, sizing conservatively\"", "No volatility reference"],
        ["stop_loss_quality", "0.20", "\"Stop at $857 — below $860 support\"", "\"Stop at 2% below entry\"", "No stop or no basis"],
        ["position_sizing_rationale", "0.10", "Derived from portfolio %, stop distance, ATR", "Partial rationale", "Bare number"],
    ],
    col_widths=[1.4, 0.6, 1.7, 1.6, 1.5]
)
callout("CB threshold: composite < 0.60 OR research_consistency < 0.50", "FEF2F2", "DC2626")

h2("Orchestrator Agent Quality")
table(
    ["Dimension", "Weight", "Score 1.0", "Score 0.5", "Score 0.0"],
    [
        ["decision_consistency", "0.35", "Consistent with both research AND risk; any override explicitly reasoned", "Consistent with one, minor gap", "Contradicts upstream without explanation"],
        ["resolution_completeness", "0.30", "\"Execute: Long NVDA 150sh\" or \"No trade: risk not met\"", "\"Likely execute if conditions hold\"", "Ambiguous or deferred"],
        ["reasoning_transparency", "0.20", "Explicit chain: research X → risk Y → decision Z", "Inferable reasoning", "Decision without reasoning"],
        ["upstream_integration", "0.15", "Cites specific data from both research and risk", "Cites one upstream agent", "Makes decision without citing upstream"],
    ],
    col_widths=[1.4, 0.6, 1.7, 1.6, 1.5]
)
callout("CB threshold: composite < 0.60 OR decision_consistency < 0.50", "FEF2F2", "DC2626")

h2("Market Agent Quality")
table(
    ["Dimension", "Score 1.0", "Score 0.0"],
    [
        ["coverage_completeness", "Tickers fetched match what research will analyze", "Market fetched MSFT, research analyzed NVDA"],
        ["signal_clarity", "Directional or volatility characterization provided", "Raw data dump, no interpretation"],
        ["anomaly_flagging", "Flags anomaly or confirms clean conditions", "Silent on unusual conditions"],
    ],
    col_widths=[1.5, 2.8, 2.8]
)

h2("Session Coherence (Cross-Agent)")
table(
    ["Dimension", "Score 1.0", "Score 0.0"],
    [
        ["pipeline_coherence", "Same ticker and direction throughout all agents", "Any handoff breaks ticker/direction continuity"],
        ["reasoning_chain", "Each agent explicitly builds on previous output", "Any agent ignores upstream output"],
    ],
    col_widths=[1.5, 2.8, 2.8]
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 5. LLM-AS-JUDGE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
h1("5. LLM-as-Judge Architecture")

table(
    ["Constraint", "Value"],
    [
        ["Model", "Claude Haiku — one call per session, all agents batched"],
        ["Timing", "Async, post-session only — never in agent critical path"],
        ["System prompt", "Cached (~$0.0003/call)"],
        ["Total cost target", "< $0.003 per session"],
        ["Latency", "600-900ms — irrelevant post-session"],
        ["Storage", "c_evals rows with agent='research_quality', 'risk_quality', etc."],
        ["Schema change", "None — reuses existing c_evals, _quality suffix distinguishes from ops evals"],
    ],
    col_widths=[1.8, 5.0]
)

h3("Storage Example")
code_block(
    "agent='research_quality',  eval_name='thesis_coherence',  score=0.9, passed=True\n"
    "agent='research_quality',  eval_name='data_grounding',    score=0.4, passed=False\n"
    "agent='research_quality',  eval_name='composite_score',   score=0.72, passed=True\n"
    "agent='risk_quality',      eval_name='volatility_accounting', score=0.8, passed=True\n"
    "agent='session_quality',   eval_name='pipeline_coherence', score=1.0, passed=True"
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 6. PATTERN LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
h1("6. Pattern Library")

h2("Operational Patterns — 6 (Implemented)")
table(
    ["Pattern", "Severity", "Detection Rule", "Fix Template"],
    [
        ["Tool Timeout Loop", "CRITICAL", "Same tool_name has 3+ error traces", "Add max_retries=2 + timeout=10s to tool config"],
        ["Context Spiral", "WARNING", "Research agent > 40K tokens + 0 trades", "Add hard token budget; force decision step if reached"],
        ["Pipeline Break", "CRITICAL", "Research ran, orchestrator never started", "Catch exceptions in research→orchestrator handoff"],
        ["Empty Result Loop", "WARNING", "Same tool called 3+ times successfully + 0 trades", "Add result quality validation; fail fast on empty returns"],
        ["Silent Exit", "INFO", "0 trades + no terminal_reason + no tool errors", "Add terminal_reason to all orchestrator exit paths"],
        ["Cost Anomaly", "WARNING", "Session cost > mean + 2 sigma", "Check which agent consumed excess; review token usage"],
    ],
    col_widths=[1.3, 0.8, 2.1, 2.6]
)

h2("Quality Patterns — 4 (Phase Q3)")
table(
    ["Pattern", "Severity", "Detection Rule", "Fix Template"],
    [
        ["Grounding Failure", "WARNING", "research.data_grounding < 0.40 for 3 consecutive sessions", "Add output requirement: cite ≥1 price level and ≥1 quantitative indicator"],
        ["Coherence Break", "CRITICAL", "orchestrator.decision_consistency < 0.50 in any session", "Add consistency check in orchestrator prompt; structured handoff format"],
        ["Quality Cascade", "WARNING → CRITICAL", "3+ quality dimensions declining > 0.20 over 5 sessions", "Review agent prompts; check market data quality; look for prompt drift"],
        ["Silent Degradation", "INFO → WARNING", "Composite quality declining while all operational metrics stable", "Most dangerous — manual review of recent outputs; check for model changes"],
    ],
    col_widths=[1.3, 0.9, 2.0, 2.6]
)

h2("Dynamic Patterns — Isolation Forest (Phase D1)")
body(
    "Catches unknown failure modes not covered by any named rule. Each session is scored on "
    "anomaly distance from the pipeline's historical baseline. Requires 20+ sessions to train; "
    "50+ for reliable signal. Auto-retrains every 10 new sessions on a sliding 50-session window."
)
note(
    "Feature vector: total_cost_usd, total_tokens, total_latency_ms, tool_error_rate, "
    "pipeline_completion_score, op_score, biz_score, cost_per_trade, trades_executed, "
    "research_composite_quality, risk_composite_quality, orchestrator_composite_quality, "
    "research_data_grounding, orchestrator_decision_consistency"
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 7. CIRCUIT BREAKER DESIGN
# ══════════════════════════════════════════════════════════════════════════════
h1("7. Circuit Breaker Design")

body(
    "Circuit breakers fire after an agent completes, using in-memory traces — no DB reads. "
    "Eval functions take (traces, session) with zero round trips. Compute time: ~0.5ms per agent."
)

table(
    ["Mode", "Behaviour", "Current State"],
    [
        ["Shadow mode", "Run evals in-process, record would_trigger_cb=True, never abort. Dashboard shows where CBs would have fired and cost saved.", "Active — Strategy C still in testing"],
        ["Real mode", "Same code path. Raise CircuitBreakerError. Orchestrator catches it, writes terminal_reason='circuit_breaker', stops pipeline.", "Blocked until Strategy C testing complete + false positive rate < 10%"],
    ],
    col_widths=[1.2, 4.1, 1.9]
)

h3("Operational CB Config")
code_block(
    "CIRCUIT_BREAKERS = {\n"
    "    \"research\": [\n"
    "        (\"tool_success_rate\", 0.80),  # abort risk+orchestrator if tools failed\n"
    "        (\"completion\",        0.70),\n"
    "    ],\n"
    "    \"risk\": [\n"
    "        (\"assessment_complete\", 1.0), # abort orchestrator if risk never ran\n"
    "    ],\n"
    "}"
)

h3("Quality CB Config (Shadow Mode)")
code_block(
    "QUALITY_CIRCUIT_BREAKERS = {\n"
    "    \"research\": {\n"
    "        \"composite_score\": 0.60,\n"
    "        \"data_grounding\":  0.40,   # vague recommendation poisons all downstream\n"
    "    },\n"
    "    \"risk\": {\n"
    "        \"composite_score\":      0.60,\n"
    "        \"research_consistency\": 0.50, # generic assessment is useless to orchestrator\n"
    "    },\n"
    "    \"orchestrator\": {\n"
    "        \"decision_consistency\": 0.50, # contradicting upstream breaks pipeline purpose\n"
    "    },\n"
    "}"
)

h3("The 1:26 AM Scenario With Circuit Breaker")
callout(
    "Research called get_stock_data 8 times, all timed out.\n"
    "tool_success_rate = 0.0 (threshold 0.80) → CB fires after research agent.\n"
    "Risk and orchestrator never run → ~$0.80 downstream cost saved.\n"
    "terminal_reason = 'circuit_breaker' logged with eval details.",
    "F0FDF4", "059669"
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 8. RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
h1("8. Recommendation Engine")

body(
    "Every quality incident produces a structured recommendation. Not just what failed — "
    "what to do. Stored in c_incidents.fix_suggestion, generated based on which dimension "
    "failed, the session context, and the trend direction."
)

h3("Example Output")
code_block(
    "DIMENSION FAILING:  research.data_grounding = 0.3 (threshold 0.40)\n"
    "SESSIONS AFFECTED:  3 of last 5\n\n"
    "What happened:\n"
    "  Research recommendations are not citing specific data. Suggestions are generic\n"
    "  rather than evidence-based.\n\n"
    "Why it matters:\n"
    "  Low grounding forces risk to assess a vague thesis. Orchestrator decisions lack\n"
    "  evidence backing. Precedes increase in false-positive trades.\n\n"
    "What to check:\n"
    "  1. Review research system prompt — may have drifted or been modified\n"
    "  2. Check market agent output — if data is thin, research cannot ground\n"
    "  3. Open Session Deep Dive for last 3 sessions, review research trace outputs\n\n"
    "How to fix:\n"
    "  Add to research system prompt: 'Your recommendation must include:\n"
    "  (1) specific ticker, (2) at least one specific price level,\n"
    "  (3) at least one quantitative indicator value (volume, ATR, RSI).'"
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 9. IMPLEMENTATION PHASES
# ══════════════════════════════════════════════════════════════════════════════
h1("9. Implementation Phases")

h2("Completed")
table(
    ["Phase", "What", "Done"],
    [
        ["1", "Data: c_evals, c_incidents, eval engine (17 evals), pattern detector (6 patterns), backfill", "2026-05-30"],
        ["2", "RCA: call stack builder, fix suggestions, rca_engine.py", "2026-05-30"],
        ["3", "Dashboard: 6 pages, Quality Drift 3-tab, business evals, auth gate, LLM summaries", "2026-05-31"],
    ],
    col_widths=[0.6, 5.4, 1.2]
)

h2("Quality Layer")
table(
    ["Phase", "What", "Effort", "Gate"],
    [
        ["Q1", "engine/quality_judge.py — Haiku batch eval, prompt construction, JSON parsing, c_evals storage. scripts/backfill_quality.py. Tests.", "2-3 days", "Manual review of 10 sessions"],
        ["Q2", "Quality tab in Quality Drift. Per-agent trend charts. Recommendations panel. Quality scores in session detail.", "2-3 days", "Q1 done"],
        ["Q3", "4 new quality patterns in pattern_detector.py. Drift detection rules. Tests.", "1-2 days", "Q2 done"],
        ["Q4", "Shadow quality CBs. would_trigger_cb flag. CB config. Dashboard CB markers + cost-saved estimates.", "1-2 days", "Q3 done"],
    ],
    col_widths=[0.5, 3.8, 0.9, 1.9]
)

h2("Real-time")
table(
    ["Phase", "What", "Effort", "Gate"],
    [
        ["4a", "In-process operational CBs in trading-agent-c (shadow mode). Per-agent eval after handoff. CircuitBreakerError skeleton.", "1-2 days", "Q4 done; Strategy C in testing"],
        ["4b", "Supabase realtime subscription on c_sessions. Auto-run evals + quality judge post-session.", "1 day", "4a done"],
        ["4c", "Flip CBs to real mode. Operational first; quality after separate validation.", "0.5 days", "Strategy C testing complete; FP < 10%"],
    ],
    col_widths=[0.5, 3.8, 0.9, 1.9]
)

h2("ML / Dynamic Detection")
table(
    ["Phase", "What", "Data Needed"],
    [
        ["D1", "engine/anomaly_detector.py — IsolationForestDetector, fit/score/explain/persist. Feature vector includes quality scores. Auto-retrain every 10 sessions.", "20+ sessions with quality scores"],
        ["D2", "engine/sequence_miner.py — SequenceTrie, mine_patterns, match. Offline batch. Human review gate on discovered patterns.", "50+ sessions"],
        ["P1", "Predictive early warning: declining eval score trend → incident probability. Weekly health report (Haiku auto-generated).", "100+ sessions"],
    ],
    col_widths=[0.5, 4.5, 2.2]
)

h2("Productization")
table(
    ["Phase", "What", "Gate"],
    [
        ["9", "Multi-tenant schema (tenant_id), per-customer Isolation Forest models, hosted auth.", "LinkedIn demand test: 3+ engineers DM asking how to get it"],
    ],
    col_widths=[0.5, 4.5, 2.2]
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 10. PRODUCT POSITIONING
# ══════════════════════════════════════════════════════════════════════════════
h1("10. Product Positioning")

h2("Category: AI Agent AIOPS")
body(
    "Traditional AIOPS: infrastructure anomaly → root cause → remediation. "
    "AI Agent AIOPS: agent behavior anomaly → root cause in agent reasoning/tool chain → fix. "
    "AIOPS buyers already understand this value. This extends a known category into a new domain."
)

h2("Competitive Position")
table(
    ["Competitor", "What they have", "Gap"],
    [
        ["Braintrust, Arize", "Evals, no trace RCA", "Know if output was wrong, not why"],
        ["LangSmith, Langfuse", "Traces, no eval linkage", "See what happened, not whether it mattered"],
        ["Datadog, Dynatrace", "Infrastructure observability", "Good at \"API was slow\", weak at \"agent reasoning loop caused it\""],
        ["This product", "Trace + eval linkage + named behavioral taxonomy + semantic quality", "Owns the connection layer nobody else has built"],
    ],
    col_widths=[1.5, 2.0, 3.7]
)

h2("Domain Generalization")
body("The failure patterns appear across all agent types. Outcome definition changes; the detection layer does not.")
table(
    ["Agent Type", "Wasted Cost ="],
    [
        ["Trading agents", "Zero trades"],
        ["Support agents", "Unresolved tickets"],
        ["Coding agents", "Unmerged PRs"],
        ["Sales agents", "No booked meetings"],
    ],
    col_widths=[2.0, 5.0]
)

h2("Where Real Moat Comes From")
callout(
    "The reasoning failure layer — detecting that an agent made a bad plan, hallucinated a fact, "
    "or chose the wrong tool for the goal — requires reading unstructured agent output and "
    "understanding intent. Nobody has solved this. It requires AI expertise AND deep observability "
    "domain knowledge simultaneously. The semantic quality evals (Phase Q1-Q4) are the entry point "
    "to this layer.\n\n"
    "Founder advantage: AIOPS/observability background means understanding what good RCA looks like. "
    "Most people building AI observability tools come from ML eval or DevOps, not AIOPS.",
    "EFF6FF", "1E40AF"
)
divider()


# ══════════════════════════════════════════════════════════════════════════════
# 11. CONSTRAINTS & KNOWN INCIDENTS
# ══════════════════════════════════════════════════════════════════════════════
h1("11. Constraints and Known Incidents")

h2("What This Does NOT Do")
for line in [
    "Does NOT modify trading-agent-c pipeline logic",
    "Does NOT run during agent execution — quality evals are post-session async",
    "Does NOT make trading decisions — evaluates existing decisions",
    "Does NOT require schema changes to c_sessions, c_traces, c_positions",
    "Phases Q1-Q3 are observation only — no circuit breakers",
    "CBs remain shadow-mode while Strategy C is in testing",
]:
    bullet(line)

h2("Known Incidents to Validate Against")
table(
    ["#", "Date / Description", "Pattern", "Type"],
    [
        ["1", "2026-05-28 01:26 — yfinance timeout, 8 retries, $1.98, 0 trades", "Tool Timeout Loop", "Behavioral"],
        ["2", "Research agent hang — sector context commits (d38950d, 487e4df), 10+ min hangs", "Pipeline Break", "Behavioral"],
        ["3", "Unrealized P&L bug — $0 P&L recorded (fixed 2026-05-29)", "Schema bug", "Infrastructure"],
        ["4", "Upsert conflict bug — duplicate session writes (fixed)", "Schema bug", "Infrastructure"],
        ["5", "Sector name schema bug — wrong field name, data loss (fixed)", "Schema bug", "Infrastructure"],
    ],
    col_widths=[0.3, 3.2, 1.5, 1.2]
)
note("Items 1-2 are behavioral failures the eval engine and pattern detector must surface. Items 3-5 are schema bugs that validate a different detection class.")

doc.save(OUT)
print(f"Saved: {OUT}")
