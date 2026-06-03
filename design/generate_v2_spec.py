"""Generate dashboard_v2_spec.docx — full specification for the v2 dashboard redesign."""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

OUT = os.path.join(os.path.dirname(__file__), "dashboard_v2_spec.docx")

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN  = RGBColor(0x16, 0x65, 0x34)
AMBER  = RGBColor(0x92, 0x40, 0x0e)
RED    = RGBColor(0x99, 0x1b, 0x1b)
SLATE  = RGBColor(0x47, 0x55, 0x69)
BLUE   = RGBColor(0x1d, 0x4e, 0xd8)
HEADER = RGBColor(0x0f, 0x17, 0x2a)
SUBHDR = RGBColor(0x1e, 0x29, 0x3b)
MUTED  = RGBColor(0x64, 0x74, 0x8b)
CELL_H = RGBColor(0xf1, 0xf5, 0xf9)
CELL_B = RGBColor(0xe2, 0xe8, 0xf0)

def set_cell_bg(cell, hex_color: str):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def set_cell_border(cell, sides=("top","bottom","left","right"), color="E2E8F0", sz="4"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in sides:
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), sz)
        el.set(qn("w:color"), color)
        tcBorders.append(el)
    tcPr.append(tcBorders)

def h1(doc, text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = HEADER
    p.runs[0].font.size = Pt(18)
    return p

def h2(doc, text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = SUBHDR
    p.runs[0].font.size = Pt(14)
    return p

def h3(doc, text):
    p = doc.add_heading(text, level=3)
    p.runs[0].font.color.rgb = SLATE
    p.runs[0].font.size = Pt(12)
    return p

def body(doc, text, bold=False, italic=False, color=None, size=10):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color
    return p

def bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    for run in p.runs:
        run.font.size = Pt(10)
    return p

def note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    run = p.add_run(f"Note: {text}")
    run.font.size = Pt(9)
    run.italic = True
    run.font.color.rgb = MUTED
    return p

def divider(doc):
    doc.add_paragraph("─" * 80)

def table_2col(doc, rows, header=None, col_widths=(2.5, 4.0)):
    t = doc.add_table(rows=len(rows) + (1 if header else 0), cols=2)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    if header:
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = h
            cell.paragraphs[0].runs[0].bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(9)
            cell.paragraphs[0].runs[0].font.color.rgb = SUBHDR
            set_cell_bg(cell, "F1F5F9")
    for ri, (c0, c1) in enumerate(rows):
        row = t.rows[ri + (1 if header else 0)]
        row.cells[0].text = c0
        row.cells[1].text = c1
        for c in row.cells:
            for para in c.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(10)
    for row in t.rows:
        row.cells[0].width = Inches(col_widths[0])
        row.cells[1].width = Inches(col_widths[1])
    return t

def table_multicol(doc, headers, rows, col_widths=None):
    n = len(headers)
    t = doc.add_table(rows=len(rows) + 1, cols=n)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = SUBHDR
        set_cell_bg(cell, "F1F5F9")
    for ri, row_data in enumerate(rows):
        row = t.rows[ri + 1]
        for ci, val in enumerate(row_data):
            row.cells[ci].text = val
            for para in row.cells[ci].paragraphs:
                for run in para.runs:
                    run.font.size = Pt(10)
    if col_widths:
        for row in t.rows:
            for ci, w in enumerate(col_widths):
                row.cells[ci].width = Inches(w)
    return t

# ── Build document ───────────────────────────────────────────────────────────
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# Default style
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE PAGE
# ─────────────────────────────────────────────────────────────────────────────
p = doc.add_heading("AI Agent RCA Dashboard", level=0)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.runs[0].font.color.rgb = HEADER
p.runs[0].font.size = Pt(24)

p2 = doc.add_paragraph("Version 2 — Design Specification")
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p2.runs:
    run.font.size = Pt(14)
    run.font.color.rgb = SLATE

p3 = doc.add_paragraph("June 2026  ·  Amit Garg")
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p3.runs:
    run.font.size = Pt(10)
    run.font.color.rgb = MUTED

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 1. PURPOSE AND APPROACH
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "1. Purpose and Approach")

body(doc, (
    "This document specifies the v2 redesign of the AI Agent RCA dashboard. "
    "Three pages are rebuilt as new routes alongside the existing v1 pages. "
    "V1 pages remain untouched for side-by-side comparison until v2 is validated "
    "and v1 is formally deprecated."
))

body(doc, "Each page has one job:", bold=True)
bullet(doc, "Overview v2 — Is my pipeline healthy right now?")
bullet(doc, "Ledger v2 — What am I paying for and is it delivering value?")
bullet(doc, "Quality v2 — Why is the pipeline behaving this way, and is it getting worse?")

body(doc, "\nImplementation platform:", bold=True)
bullet(doc, "Phase 1: Streamlit (new page routes, shared Python components)")
bullet(doc, "Phase 2: React (Vercel) + Supabase JS client, once design is validated")

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL THRESHOLDS AND COLOR SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "2. Global Thresholds and Color System")

body(doc, (
    "All health signals use a consistent three-level color system. "
    "Thresholds apply to KPI cards, chart annotations, heatmap cells, "
    "bubble colors, and radar chart fills."
))

doc.add_paragraph()
table_multicol(doc,
    headers=["Signal Type", "Green (healthy)", "Amber (watch)", "Red (act now)"],
    rows=[
        ["Pass rate (operational evals)", "≥ 80%", "60 – 80%", "< 60%"],
        ["Session reliability", "≥ 85%", "60 – 85%", "< 60%"],
        ["Cost efficiency (productive spend %)", "≥ 70%", "40 – 70%", "< 40%"],
        ["Semantic quality score (0–1)", "≥ 0.60", "0.40 – 0.60", "< 0.40"],
        ["Pipeline efficiency (% sessions with trades)", "≥ 70%", "40 – 70%", "< 40%"],
    ],
    col_widths=[2.8, 1.5, 1.5, 1.5]
)

doc.add_paragraph()
body(doc, "Hex color values:", bold=True)
table_2col(doc, [
    ("Green (healthy)", "#10b981 background  ·  #166534 text"),
    ("Amber (watch)",   "#f59e0b background  ·  #92400e text"),
    ("Red (act now)",   "#ef4444 background  ·  #991b1b text"),
    ("Slate (neutral)", "#475569"),
    ("Hollow / no data","#e2e8f0 background  ·  #94a3b8 text"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "3. Data Sources")

body(doc, "All data is read from Supabase (Strategy C project). No writes from the dashboard.")

doc.add_paragraph()
table_multicol(doc,
    headers=["Table", "Key columns used", "Notes"],
    rows=[
        ["c_sessions",
         "id, started_at, total_cost_usd, total_latency_ms, total_tokens_input, "
         "total_tokens_output, trades_executed, agents_invoked, terminal_reason, "
         "cost_breakdown (JSON), is_simulated",
         "cost_breakdown is a per-agent dict: {agent: {cost_usd, input, output, cache_read, cache_write, model}}"],
        ["c_evals",
         "session_id, agent, eval_name, score (0–1), passed (bool)",
         "Operational evals: agent in {market, research, risk, orchestrator, session}. "
         "Quality evals: agent ends with _quality (e.g. research_quality). "
         "composite_score is a synthetic eval_name averaging all dimensions."],
        ["c_traces",
         "session_id, agent, step_type, tool_name, tokens_input, tokens_output, "
         "latency_ms, outcome, error, created_at",
         "step_type values: llm_call, tool_call, decision, error"],
        ["c_incidents",
         "id, session_id, pattern_name, severity, root_cause, failed_evals (JSON), "
         "created_at",
         "severity values: critical, warning, info. "
         "failed_evals is a list of {agent, eval_name, score} dicts."],
    ],
    col_widths=[1.5, 2.8, 2.4]
)

doc.add_paragraph()
body(doc, "Period filtering:", bold=True)
body(doc, (
    "All three pages expose a period selector: 7d / 14d / 30d. "
    "The selected window filters sessions by started_at. "
    "The previous equivalent window (same duration, immediately preceding) "
    "is used to compute delta values on KPI cards."
))

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 4. SHARED COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "4. Shared Components")

body(doc, "These functions are built once and reused across all three pages.")

doc.add_paragraph()
h2(doc, "4.1  signal_card(label, value, delta_html, color)")
body(doc, "Renders one KPI card with a traffic-light color, large value, and delta vs previous period.")
table_2col(doc, [
    ("label",      "Card title text"),
    ("value",      "Primary value string, e.g. '73%' or '$0.038'"),
    ("delta_html", "Pre-formatted HTML: '▲ 8% vs prev' (green) or '▼ 12% vs prev' (red)"),
    ("color",      "One of: 'green', 'amber', 'red' — determines background and text color"),
], header=["Parameter", "Description"], col_widths=(1.5, 5.0))

doc.add_paragraph()
h2(doc, "4.2  fleet_strip(agent_pass_rates: dict)")
body(doc, (
    "Renders the aggregated pipeline health strip on Overview v2. "
    "Shows ORC → MKT → NEWS → RES → RSK → ORC with each node colored by "
    "aggregate pass rate across all sessions in the period. "
    "Each node is clickable and navigates to Quality v2 filtered to that agent."
))
table_2col(doc, [
    ("agent_pass_rates", "Dict mapping agent name to pass rate float, e.g. {'research': 0.67, 'risk': 0.83, ...}"),
    ("node color logic", "Apply global pass rate thresholds: ≥80% green, 60–80% amber, <60% red"),
    ("node label",       "market→MKT, news_analyst→NEWS, research→RES, risk→RSK, orchestrator→ORC"),
    ("click behavior",   "st.query_params update to navigate to Quality v2 with agent filter pre-set"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "4.3  compute_savings(sessions_df) → dict")
body(doc, "Computes realized and projected savings for the Ledger v2 savings strip.")
table_2col(doc, [
    ("avg_full_cost",
     "Mean of total_cost_usd where terminal_reason in ('converged', 'eod_complete'). "
     "Represents the baseline cost of a full successful pipeline run."),
    ("early_exit_sessions",
     "Sessions where terminal_reason = 'no_viable_proposals'. "
     "These exited before Research and Risk ran, saving cost vs a full run."),
    ("realized_savings",
     "Sum of max(0, avg_full_cost - session.total_cost_usd) "
     "for each early exit session in the window. "
     "Represents actual money not spent due to smart pipeline early exits."),
    ("shadow_cb_savings",
     "From c_incidents: count of incidents where pattern fired in shadow mode "
     "and session completed (did not exit early). "
     "Estimated savings = count × avg_full_cost × 0.4 (heuristic: CB would have "
     "saved ~40% of session cost by halting mid-pipeline). "
     "Displayed as projected, not realized."),
    ("return value",
     "{'realized': float, 'realized_count': int, 'projected': float, 'projected_count': int}"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "4.4  compute_impact_bridge(evals_df, sessions_df, mode) → list")
body(doc, (
    "Computes eval-to-outcome correlation for the Impact Bridge sections in Quality v2. "
    "Used in both Operational (mode='operational') and Semantic (mode='semantic') tabs."
))
table_2col(doc, [
    ("mode = 'operational'",
     "For each unique (agent, eval_name) pair in evals_df where agent is not a _quality agent: "
     "split sessions into passed group and failed group. "
     "Compute 0-trade rate and incident rate for each group. "
     "Rank by abs(0-trade rate delta) descending. Return top N."),
    ("mode = 'semantic'",
     "For each quality agent (agent ends with _quality, eval_name = composite_score): "
     "split sessions into above-threshold (score ≥ 0.60) and below-threshold groups. "
     "Compute 0-trade rate, avg cost, and incident rate for each group. "
     "Rank by abs(0-trade rate delta) descending. Return top N."),
    ("return value",
     "List of dicts: {agent, eval_name, pass_0trade_rate, fail_0trade_rate, "
     "pass_incident_rate, fail_incident_rate, delta, pass_count, fail_count}"),
    ("minimum sample",
     "Exclude any eval where either group has fewer than 2 sessions. "
     "Show 'insufficient data' note if fewer than 3 evals qualify."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PAGE: OVERVIEW V2
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "5. Page: Overview v2")
body(doc, "Route: page == 'Overview v2'")
body(doc, "Job: Answer 'is my pipeline healthy right now?' in under 30 seconds.", bold=True)
doc.add_paragraph()

h2(doc, "5.1  Page Header")
table_2col(doc, [
    ("Title",          "'System Overview'"),
    ("Period selector","Radio: 7d / 14d / 30d. Default 7d. Filters all sections on the page."),
    ("Timestamp",      "'As of <UTC datetime>' — pulled at page load, not from data"),
    ("Refresh button", "Clears load_sessions, load_traces, load_all_evals cache and reruns"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "5.2  Section 1 — 4 Health Signal Cards")
body(doc, "Four columns across full width. Each uses signal_card() shared component.")
doc.add_paragraph()
table_multicol(doc,
    headers=["Card", "Formula", "Green", "Amber", "Red"],
    rows=[
        ["Reliability",
         "sessions_clean / sessions_total × 100\n"
         "sessions_clean = sessions with no incident in window",
         "≥ 85%", "60–85%", "< 60%"],
        ["Cost Efficiency",
         "cost_productive / cost_total × 100\n"
         "cost_productive = sum cost where trades_executed > 0",
         "≥ 70%", "40–70%", "< 40%"],
        ["Operational Quality",
         "evals_passed / evals_total × 100\n"
         "exclude _quality agent evals",
         "≥ 80%", "60–80%", "< 60%"],
        ["Semantic Quality",
         "mean(composite_score) across sessions in window\n"
         "agent = *_quality, eval_name = composite_score",
         "≥ 0.60", "0.40–0.60", "< 0.40"],
    ],
    col_widths=[1.8, 2.8, 0.8, 0.8, 0.8]
)
doc.add_paragraph()
body(doc, "Delta computation for each card:", bold=True)
body(doc, (
    "prev_value = same metric computed over the previous equivalent window "
    "(same duration, ending at window start). "
    "delta_pct = (current - prev) / prev × 100. "
    "Display as '▲ N%' (green if direction is good) or '▼ N%' (red if direction is bad). "
    "For Reliability, Cost Efficiency, Op Quality: higher is better. "
    "For Semantic Quality: higher is better."
))

doc.add_paragraph()
h2(doc, "5.3  Section 2 — Pipeline Fleet Health Strip")
body(doc, (
    "Full-width horizontal strip showing aggregate health of each pipeline agent "
    "across all sessions in the selected window. Uses fleet_strip() shared component."
))
table_2col(doc, [
    ("Node order",    "ORC → MKT → NEWS → RES → RSK → ORC (left ORC = coordination, right ORC = synthesis)"),
    ("Node color",    "Aggregate eval pass rate for that agent in window. Apply global pass rate thresholds."),
    ("Node label",    "Agent abbreviation (ORC / MKT / NEWS / RES / RSK) + aggregate pass rate % below node"),
    ("Node click",    "Navigates to Quality v2 with agent filter pre-set to the clicked agent"),
    ("Connector",     "Arrow lines between nodes (→). Color = gray if both nodes healthy, amber/red if either is degraded"),
    ("Pass rate calc","evals_passed / evals_total for that agent across all sessions in window. "
                      "Market agent has no quality evals — use operational evals only."),
    ("No data state", "Node renders as hollow circle with 'No data' label. Does not navigate on click."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "5.4  Section 3 — Session Outcomes Chart")
body(doc, "Stacked bar chart by calendar day across the selected window.")
table_2col(doc, [
    ("X-axis",        "Date (one bar per calendar day in window)"),
    ("Y-axis",        "Session count"),
    ("Bar segments",  "Green = clean (no incident, trades > 0)  ·  Amber = 0-trade (no incident, trades = 0)  ·  Red = incident (incident fired, any trade count)"),
    ("Hover tooltip", "Date · N clean · N 0-trade · N incident · Total cost $X.XX for that day"),
    ("Cost in hover", "Sum of total_cost_usd for sessions on that date. Not shown as axis — hover only."),
    ("Height",        "240px"),
    ("No data state", "st.info('No sessions in this range.')"),
], col_widths=(2.0, 4.5))
note(doc, "Session is classified as 'incident' if it appears in c_incidents for the window, regardless of trade count. A session can have incidents AND trades.")

doc.add_paragraph()
h2(doc, "5.5  Section 4 — Active Incidents Panel")
body(doc, (
    "Right panel, 40% width, shares row with Session Outcomes Chart. "
    "Shows severity distribution and the highest-priority incidents in the window."
))
table_2col(doc, [
    ("Donut chart",     "3 slices: Critical (red #ef4444) / Warning (amber #f59e0b) / Info (blue #3b82f6). "
                        "Total incident count in center. Height 160px."),
    ("Severity legend", "Below donut: '■ Critical N  ■ Warning N  ■ Info N'"),
    ("Incident list",   "Top 2 incidents. Default sort: severity (critical first), then recency within same severity. "
                        "Sort toggle: 'Severity' (default) | 'Date'. Toggle is a small radio or pill selector."),
    ("Incident row",    "[SEVERITY badge] pattern_name · agent · timestamp · [→ RCA] button"),
    ("SEVERITY badge",  "Red pill for critical, amber for warning, blue for info"),
    ("→ RCA button",    "Navigates to RCA View with that incident pre-selected"),
    ("Footer link",     "'View all N incidents →' navigates to Incidents Feed page"),
    ("Empty state",     "If no incidents in window: 'No incidents in this period' with green checkmark"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "5.6  Layout")
body(doc, "Vertical stack, top to bottom:", bold=True)
bullet(doc, "[Period selector row]")
bullet(doc, "[4 Health Signal Cards — equal width columns]")
bullet(doc, "[Pipeline Fleet Health Strip — full width]")
bullet(doc, "[Session Outcomes Chart 60%]  [Active Incidents Panel 40%]  ← same row")

doc.add_paragraph()
h2(doc, "5.7  What is NOT on Overview v2 (removed from v1)")
table_2col(doc, [
    ("Recent Sessions AgGrid",  "Moved to Ledger v2 — session detail is a ledger concern"),
    ("Agent Eval Health table", "Replaced by Fleet Health strip (aggregate) + signal cards"),
    ("Avg Cost/Session KPI",    "Replaced by Cost Efficiency card (more meaningful signal)"),
    ("Top Pattern KPI",         "Subsumed into Active Incidents panel"),
], col_widths=(2.5, 4.0))

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 6. PAGE: LEDGER V2
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "6. Page: Ledger v2")
body(doc, "Route: page == 'Ledger v2'")
body(doc, "Job: Financial and audit record — every dollar traceable to a decision and outcome.", bold=True)
body(doc, (
    "This page serves two audiences: the operator who wants to understand cost efficiency, "
    "and the auditor who needs a verifiable record. Every number is period-filtered, "
    "timestamped, and exportable."
))
doc.add_paragraph()

h2(doc, "6.1  Page Header")
table_2col(doc, [
    ("Title",           "'Cost Ledger'"),
    ("As-of timestamp", "'As of <UTC datetime>' — displayed prominently in header"),
    ("Period selector", "7d / 14d / 30d. Filters all sections except the session ledger table (which shows all-time by default, filterable)."),
    ("Generate Summary","Button — on click, calls LLM to produce CFO-style audit brief. Hidden until clicked. See Section 6.3."),
    ("Refresh",         "Clears session and eval cache, reruns page"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.2  Section 1 — 4 Financial KPI Cards")
body(doc, "Period-aware. All use signal_card() component.")
doc.add_paragraph()
table_multicol(doc,
    headers=["Card", "Formula", "Color logic"],
    rows=[
        ["Total Spend",
         "sum(total_cost_usd) for sessions in window",
         "No threshold — informational. Display neutral slate color."],
        ["Cost per Trade",
         "Total Spend / sum(trades_executed). Show '—' if no trades.",
         "No fixed threshold — show delta vs prev period. Red if rising, green if falling."],
        ["Wasted Spend",
         "sum(total_cost_usd) for sessions where trades_executed = 0. "
         "Display as '$X.XX  (N%)' where N% = wasted / total × 100.",
         "Green < 15%, Amber 15–40%, Red > 40% of total spend"],
        ["Pipeline Efficiency",
         "sessions with trades_executed > 0 / total sessions × 100",
         "Apply pipeline efficiency thresholds: ≥70% green, 40–70% amber, <40% red"],
    ],
    col_widths=[1.8, 3.0, 1.9]
)

doc.add_paragraph()
h2(doc, "6.3  Section 2 — On-Demand CFO Summary")
body(doc, (
    "Hidden by default. Shown when user clicks 'Generate Summary'. "
    "Calls Anthropic API (Haiku model for cost). Cached by data snapshot hash."
))
table_2col(doc, [
    ("Tone",        "CFO audit brief — formal, factual, no filler. 3–4 sentences maximum."),
    ("Content",     "1. Total spend and efficiency ratio for the period. "
                    "2. Largest waste driver (exit reason or agent). "
                    "3. Trend direction vs previous period. "
                    "4. One actionable observation (e.g. 'timeout sessions account for 61% of wasted spend — reducing agent timeouts is the highest-leverage cost action')."),
    ("Prompt inputs","n_sessions, total_cost, efficiency_pct, wasted_cost, wasted_pct, "
                     "top_waste_exit_reason, top_agent_by_cost, top_agent_pct, "
                     "realized_savings, prev_period_efficiency_pct"),
    ("Error state", "If API unavailable: 'Summary unavailable — API key not configured.'"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.4  Section 3 — Pipeline Savings Strip")
body(doc, (
    "Full-width band with green accent. Two side-by-side callout boxes. "
    "Uses compute_savings() shared component."
))
table_2col(doc, [
    ("Left box — Realized savings",
     "Solid green border. Shows: '$X.XX saved  ·  N early exits before Research/Risk ran'. "
     "Tooltip: 'Early exits occur when the Market agent determines no viable opportunities exist. "
     "Research and Risk agents never run, saving their cost vs a full pipeline run. "
     "Estimated vs average full-run cost of $X.XX.'"),
    ("Right box — Projected savings",
     "Dashed green border (visually distinct — not yet real). "
     "Shows: '$X.XX projected  ·  Shadow circuit breakers fired N times'. "
     "Tooltip: 'Shadow circuit breakers logged N firing events but did not stop the pipeline. "
     "If enabled, these would have halted execution mid-run and saved an estimated $X.XX.'"),
    ("[Enable circuit breakers →]",
     "Link in right box footer. Navigates to circuit breaker configuration (future feature). "
     "For now, renders as disabled text with note: (Coming soon)"),
    ("If no savings",
     "Left box: '$0.00 saved  ·  No early exits this period'. "
     "Right box: '$0.00 projected  ·  No shadow CB fires this period'. "
     "Both boxes still render — they frame the potential value, even when zero."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.5  Section 4 — Cost Waterfall Chart")
body(doc, "Full-width Plotly horizontal waterfall. Primary metric = dollars. This is the centerpiece of the page.")
doc.add_paragraph()
table_2col(doc, [
    ("Row 1 — Total Spend",
     "Full bar, gray. Width = total cost in window."),
    ("Rows 2–6 — By Agent",
     "One row per agent (research, risk, orchestrator, market, other). "
     "Red bars showing cost consumed. Sorted descending by cost. "
     "Label shows agent name + dollar amount. "
     "research_TICKER keys in cost_breakdown are normalized to 'research'."),
    ("Row — Productive",
     "Green bar. sum(total_cost_usd) for sessions where trades_executed > 0."),
    ("Row — Wasted",
     "Red bar. sum(total_cost_usd) for sessions where trades_executed = 0."),
    ("Row — Saved (exits)",
     "Lighter green bar (opacity 0.6). realized_savings from compute_savings(). "
     "Represents cost not spent due to early exits. "
     "Label: 'Saved via early exits'."),
    ("Hover tooltip",
     "Each bar: '$X.XXXX  ·  N sessions'. Session count on hover only, not on chart."),
    ("Click behavior",
     "Clicking any agent bar in rows 2–6 expands the Agent Detail Panel below (Section 6.6). "
     "Active agent bar gets a highlight border."),
    ("X-axis",  "Cost in USD, formatted as $X.XXXX"),
    ("Height",  "320px"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.6  Section 5 — Agent Detail Panel")
body(doc, (
    "Expands below the waterfall when a user clicks an agent bar. "
    "Hidden by default — shows placeholder 'Click an agent bar above to drill down.'"
))
table_2col(doc, [
    ("Header",         "Agent name (title case) + total cost for period"),
    ("4 metrics",      "Cost $X.XXXX  ·  Input tokens N  ·  Output tokens N  ·  Cache reads N"),
    ("Model caption",  "Model used (from cost_breakdown) and cache write token count"),
    ("Wasted %",       "Share of this agent's cost that came from 0-trade sessions. "
                       "Shown as colored pill: '$X.XX (N%) in 0-trade sessions'."),
    ("Per-ticker table","Research agent only. Bar chart: ticker on Y, cost on X. "
                        "Sorted descending. Shows which stocks consumed the most research budget."),
    ("Tool call table", "From c_traces: group by tool_name. Columns: Tool / Calls / Avg Latency (ms) / Errors / Error %. "
                        "Sorted by Calls descending."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.7  Section 6 — Spend Over Time")
body(doc, "Full-width Plotly line chart. Shows daily spend trend.")
table_2col(doc, [
    ("Primary line",   "Daily sum(total_cost_usd). Color #3b82f6 (blue). Width 2px."),
    ("Rolling average","7-day rolling average of daily spend. Color #94a3b8 (gray). Dashed. Width 1.5px."),
    ("X-axis",         "Date"),
    ("Y-axis",         "Cost USD, formatted as $X.XXXX"),
    ("Hover",          "Date · Daily spend $X.XXXX · 7-day avg $X.XXXX"),
    ("Height",         "200px"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.8  Section 7 — Waste and Savings Analysis")
body(doc, "Two panels sharing a row: 55% left, 45% right.")
doc.add_paragraph()
body(doc, "Left panel — Exit reason bifurcated chart:", bold=True)
table_2col(doc, [
    ("Chart type",  "Horizontal diverging bar. Center = zero line. Left = green (savings). Right = red (waste)."),
    ("Y-axis",      "Terminal reason labels"),
    ("Savings side (left)",
     "Exits classified as efficient: terminal_reason = 'no_viable_proposals'. "
     "Value = realized savings for those sessions (avg_full_cost - actual_cost per session, summed)."),
    ("Waste side (right)",
     "Exits classified as wasteful: all_rejected, timeout, watchdog_timeout, unknown. "
     "Value = total_cost_usd for those sessions."),
    ("eod_complete / superseded",
     "Classified as neutral — not shown on this chart (they are expected outcomes, not waste or savings)."),
    ("Hover",       "Exit reason · $X.XXXX · N sessions"),
    ("Labels",      "Dollar amounts on each bar end. Session count in hover only."),
    ("Height",      "280px"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
body(doc, "Right panel — Wasted % per agent:", bold=True)
table_2col(doc, [
    ("Chart type",  "Horizontal bar. One bar per agent."),
    ("Value",       "Wasted cost for agent / total cost for agent × 100. Colored by threshold: <25% green, 25–50% amber, >50% red."),
    ("Hover",       "Agent · $X.XXXX wasted of $X.XXXX total · N 0-trade sessions"),
    ("Height",      "280px"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.9  Section 8 — Session Ledger")
body(doc, "Full-width AgGrid. Primary record — every session, every dollar.")
table_2col(doc, [
    ("Column order", "Session (MM-DD HH:MM) / Cost ($) / Trades / Exit Reason / Duration (s) / Incidents / Waste flag"),
    ("Waste flag",   "Icon column: ⚠ (amber) if cost > $0.01 and trades_executed = 0. Blank otherwise."),
    ("Exit Reason",  "Color-coded: converged = green, eod_complete/superseded = slate, "
                     "no_viable_proposals/all_rejected = amber, timeout/watchdog_timeout = red"),
    ("Row style",    "Red background if session has incidents. Amber background if 0-trade. Normal if clean."),
    ("Filters",      "Checkbox: 'Incidents only' and '0-trade sessions only'. Can combine."),
    ("Pagination",   "10 rows per page. Pagination bar shown when > 10 rows."),
    ("Pipeline strip","Selecting a row shows pipeline detail panel below grid (same as v1 — session cost / trades / exit badge / pipeline node strip / incident banners)."),
    ("CSV export",   "Small spreadsheet icon (📊) button top-right of grid. Downloads current filtered view as CSV. "
                     "Filename: 'cost_ledger_<period>_<date>.csv'. "
                     "Columns: session_id, started_at, cost_usd, trades_executed, terminal_reason, duration_s, incident_count."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "6.10  Layout")
bullet(doc, "[Header: title · as-of · period selector · Generate Summary · Refresh]")
bullet(doc, "[4 Financial KPI Cards]")
bullet(doc, "[Pipeline Savings Strip: Realized | Projected]")
bullet(doc, "[On-demand CFO Summary — hidden until requested]")
bullet(doc, "[Cost Waterfall — full width]")
bullet(doc, "[Agent Detail Panel — expands on bar click]")
bullet(doc, "[Spend Over Time — full width]")
bullet(doc, "[Exit breakdown 55%]  [Wasted % by agent 45%]  ← same row")
bullet(doc, "[Session Ledger — full width, with 📊 CSV export]")

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 7. PAGE: QUALITY V2
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "7. Page: Quality v2")
body(doc, "Route: page == 'Quality v2'")
body(doc, "Job: Why is the pipeline behaving this way, and is it getting worse?", bold=True)
body(doc, (
    "Two tabs cover two distinct signal types: rule-based operational evals (reactive, "
    "deterministic) and LLM-judge semantic quality scores (leading indicator, probabilistic). "
    "Both tabs include an Impact Bridge that connects quality signals to business outcomes. "
    "Business outcome charts (P&L, cost efficiency trends) are removed from this page — "
    "those belong in Ledger v2."
))
doc.add_paragraph()

h2(doc, "7.1  Tab 1: Operational Health")
body(doc, "Rule-based evals. Something broke — find it.")
doc.add_paragraph()

h3(doc, "Section 1 — 4 Eval Health KPI Cards")
table_multicol(doc,
    headers=["Card", "Formula", "Color logic"],
    rows=[
        ["Overall Pass Rate",
         "sum(passed) / count(*) × 100 for non-quality evals in window",
         "Apply pass rate thresholds"],
        ["Failing Agents",
         "Count of distinct agents where per-agent pass rate < 80% in window",
         "0 = green, 1 = amber, ≥2 = red"],
        ["Top Failing Eval",
         "eval_name with lowest pass rate across all agents in window. Show as text.",
         "No color — informational"],
        ["Sessions with Failures",
         "Sessions with at least one eval failed / total sessions × 100",
         "Apply reliability thresholds: <15% green, 15–40% amber, >40% red"],
    ],
    col_widths=[2.0, 2.8, 1.9]
)

doc.add_paragraph()
h3(doc, "Section 2 — Agent × Eval Bubble Matrix")
body(doc, (
    "Plotly bubble chart (scatter). Replaces the flat heatmap from v1. "
    "Two visual dimensions — color for health signal, size for data confidence."
))
table_2col(doc, [
    ("X-axis",         "Eval name (one position per unique eval in window)"),
    ("Y-axis",         "Agent name (market, research, risk, orchestrator, session)"),
    ("Bubble color",   "Pass rate for (agent, eval) pair in window. Apply pass rate color thresholds. "
                       "Continuous color scale: #ef4444 (0%) → #f59e0b (60%) → #10b981 (100%)."),
    ("Bubble size",    "Number of sessions evaluated for this (agent, eval) pair. "
                       "Min size = 8px (1 session). Max size = 24px (≥20 sessions). "
                       "Small bubble = low confidence. Large bubble = reliable signal."),
    ("Bubble label",   "Pass rate % shown inside each bubble (white text, bold, 9pt). "
                       "Hidden if bubble too small (< 12px)."),
    ("Hover tooltip",  "Agent · eval_name · Pass rate N% · N/M sessions passed · "
                       "Confidence: Low/Medium/High (based on session count)"),
    ("Click row",      "Clicking a Y-axis agent label expands a sparkline panel below the chart "
                       "showing per-eval pass rate over last 10 sessions for that agent."),
    ("Sparkline panel","One small line chart per eval for the selected agent. "
                       "X = session date. Y = 0 (failed) or 1 (passed), plotted as step line. "
                       "Pass rate % shown as title. 0.80 threshold as dotted horizontal line."),
    ("Height",         "360px for bubble matrix. Sparkline panel: 180px per row, max 3 cols."),
    ("No data",        "If fewer than 2 sessions in window: 'Not enough data for bubble matrix — need at least 2 sessions.'"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h3(doc, "Section 3 — Impact Bridge (Operational)")
body(doc, (
    "Auto-surfaces the top 3 eval failures most correlated with bad outcomes. "
    "Uses compute_impact_bridge(mode='operational')."
))
table_2col(doc, [
    ("Display",
     "3 side-by-side cards (or stacked if fewer than 3 qualify). Each card shows one (agent, eval) pair."),
    ("Card content",
     "Eval name + agent as header. "
     "Two rows of outcome stats:\n"
     "  When FAILS:  0-trade N% ·  Incident N%\n"
     "  When PASSES: 0-trade N% ·  Incident N%\n"
     "Delta highlighted: '3.2× more likely to produce 0 trades when this eval fails'"),
    ("Ranking",
     "Ranked by abs(0-trade rate when failed - 0-trade rate when passed) descending. "
     "Ties broken by incident rate delta."),
    ("Minimum data",
     "Require at least 3 sessions in each group (passed and failed). "
     "If a pair has fewer, skip it. If fewer than 3 pairs qualify, show available ones with note."),
    ("'Explore all' expander",
     "Expander below the top-3 cards: 'Explore all eval correlations ▼'. "
     "Contains a selectbox listing all qualifying (agent, eval) pairs. "
     "Selecting one renders the same two-row outcome stat card for that pair."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "7.2  Tab 2: Semantic Quality")
body(doc, "LLM-judge scores. Something is drifting — catch it early.")
doc.add_paragraph()

h3(doc, "Section 1 — 4 Trend Direction Cards")
body(doc, "Research / Risk / Orchestrator / Session composite. Keep from v1 — design is proven.")
table_2col(doc, [
    ("Score",     "Most recent composite_score for that agent"),
    ("Trend",     "Linear slope across last 5 sessions: > +0.01 = ▲ improving, < -0.01 = ▼ declining, else → stable"),
    ("Delta",     "current_score - oldest_of_last_5"),
    ("Colors",    "Card background and trend arrow use quality score thresholds"),
    ("Session",   "'Session' card = mean of all 4 agent composite scores for the most recent session"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h3(doc, "Section 2 — Quality Drift Charts")
body(doc, "3-column layout: Research / Risk / Orchestrator. Score over time.")
table_2col(doc, [
    ("Score line",     "Composite quality score per session. Color = agent color (research #f59e0b, risk #10b981, orchestrator #3b82f6)."),
    ("Trend line",     "Linear regression across all sessions shown. Dashed, darker shade of agent color."),
    ("Threshold",      "Dotted gray horizontal line at 0.60. Annotation 'threshold'."),
    ("Incident dots",  "Red filled circles on sessions that have incidents. Hover shows pattern names."),
    ("Height",         "200px per chart"),
    ("Session quality","Fourth chart below the 3-column row: mean composite across all agents per session. Blue line."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h3(doc, "Section 3 — Per-Agent Radar Charts")
body(doc, (
    "4 radar charts side by side (Research / Risk / Orchestrator / Session). "
    "Replaces the Pipeline Quality Snapshot heatmap from v1. "
    "Shows the 'shape' of each agent's quality profile and how it has shifted."
))
table_2col(doc, [
    ("Spokes",           "One spoke per quality eval dimension for that agent (excluding composite_score). "
                         "research_quality evals: data_grounding, thesis_coherence, actionability, catalyst_specificity, "
                         "volatility_accounting, risk_acknowledgment. "
                         "risk_quality evals: research_consistency, parameter_completeness, position_sizing_rationale, "
                         "stop_loss_quality. "
                         "orchestrator_quality evals: decision_consistency, resolution_completeness, reasoning_transparency, "
                         "upstream_integration, pipeline_coherence, reasoning_chain. "
                         "session_quality: uses session-level holistic evals."),
    ("Current period",   "Solid line, agent color, filled with transparent agent color. "
                         "Values = mean score per dimension across sessions in window."),
    ("Previous period",  "Dashed gray line, no fill. Values = mean score per dimension over previous equivalent window. "
                         "Labeled 'prev period' in legend."),
    ("Scale",            "0 to 1 on all spokes. 0.60 threshold ring as light gray dashed circle."),
    ("Interpretation",   "Shape shrinking toward center = quality degrading. One spoke collapsing = specific dimension failing. "
                         "Shape shifting asymmetrically = uneven degradation."),
    ("Hover",            "Dimension name · Current: X.XX · Prev: X.XX · Delta: ±X.XX"),
    ("Height",           "280px per radar. Layout: 4 equal columns."),
    ("No prev data",     "If window has no previous equivalent period data, render current only. "
                         "Note 'Previous period data unavailable' below chart."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h3(doc, "Section 4 — Impact Bridge (Semantic)")
body(doc, (
    "Auto-surfaces top 3 quality-to-outcome correlations. "
    "Uses compute_impact_bridge(mode='semantic'). "
    "Threshold split: score < 0.60 vs score ≥ 0.60."
))
table_2col(doc, [
    ("Display",
     "3 side-by-side cards. Each card covers one quality agent (research_quality, risk_quality, orchestrator_quality)."),
    ("Card content",
     "Agent name as header (Research / Risk / Orchestrator). "
     "Two rows of outcome stats:\n"
     "  Quality < 0.60:  0-trade N% · avg cost $X.XX · incident N%\n"
     "  Quality ≥ 0.60:  0-trade N% · avg cost $X.XX · incident N%\n"
     "Delta highlighted: 'Quality below threshold → Nx more likely to produce 0 trades'"),
    ("Ranking",
     "Ranked by abs(0-trade rate below threshold - 0-trade rate above threshold) descending."),
    ("'Explore all' expander",
     "Below the 3 cards: 'Explore all quality correlations ▼'. "
     "Selectbox: agent (research / risk / orchestrator / session). "
     "Renders the full outcome stats for the selected agent at two thresholds: "
     "< 0.40 (red) / 0.40–0.60 (amber) / ≥ 0.60 (green). "
     "Shows how outcomes degrade as quality worsens."),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h3(doc, "Section 5 — Session Detail (AgGrid)")
body(doc, "Kept from v1. Session-level quality scores with pipeline strip on selection.")
table_2col(doc, [
    ("Columns",          "Session / Research / Risk / Orchestrator / Session Q / Exit / Incidents"),
    ("Score coloring",   "Green ≥ 0.60 · Amber 0.40–0.60 · Red < 0.40"),
    ("Row style",        "Red background if session has incidents"),
    ("Pagination",       "10 rows per page"),
    ("Pipeline strip",   "Row selection shows pipeline detail panel with quality score badges per agent"),
    ("Explore expander", "'Explore all quality correlations ▼' from Section 4 appears directly below the AgGrid"),
], col_widths=(2.0, 4.5))

doc.add_paragraph()
h2(doc, "7.3  Layout")
body(doc, "Tab 1 — Operational Health:", bold=True)
bullet(doc, "[4 Eval Health KPI Cards]")
bullet(doc, "[Agent × Eval Bubble Matrix — full width]")
bullet(doc, "[Sparkline panel — expands on agent row click]")
bullet(doc, "[Impact Bridge: top 3 correlations — 3 columns]")
bullet(doc, "[Explore all correlations ▼ expander]")
doc.add_paragraph()
body(doc, "Tab 2 — Semantic Quality:", bold=True)
bullet(doc, "[4 Trend Direction Cards]")
bullet(doc, "[Quality Drift Charts — 3 columns + system composite below]")
bullet(doc, "[Radar Charts — 4 columns: Research / Risk / Orchestrator / Session]")
bullet(doc, "[Impact Bridge: top 3 quality correlations — 3 columns]")
bullet(doc, "[Explore all quality correlations ▼ expander]")
bullet(doc, "[Session Detail AgGrid + pipeline strip on row select]")

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 8. BUILD PLAN
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "8. Build Plan")

body(doc, (
    "V1 pages remain untouched throughout. V2 pages are added as new routes. "
    "Nav shows both during transition. Deprecate v1 after side-by-side validation."
))

doc.add_paragraph()
table_multicol(doc,
    headers=["Phase", "Task", "Depends on"],
    rows=[
        ["1", "Shared components: signal_card(), fleet_strip(), compute_savings(), compute_impact_bridge()", "—"],
        ["2", "Overview v2: all 4 sections + fleet strip + incidents panel", "Phase 1"],
        ["3", "Ledger v2: waterfall chart + savings strip + CFO summary + session ledger + CSV export", "Phase 1"],
        ["4", "Quality v2 Tab 1: bubble matrix + sparkline expand + impact bridge", "Phase 1"],
        ["5", "Quality v2 Tab 2: radar charts + drift charts + impact bridge + session detail", "Phase 1"],
        ["6", "Nav wiring: add v2 routes to sidebar nav alongside v1", "Phases 2–5"],
        ["7", "Side-by-side comparison and validation", "Phase 6"],
        ["8", "Deprecate v1 pages", "Phase 7 sign-off"],
    ],
    col_widths=[0.6, 3.8, 2.3]
)

doc.add_paragraph()
h2(doc, "8.1  Notes on Streamlit constraints")
bullet(doc, "Radar charts: use Plotly go.Scatterpolar. 4 subplots via make_subplots(rows=1, cols=4, specs=[[{'type':'polar'}×4]]).")
bullet(doc, "Bubble matrix: use Plotly go.Scatter with mode='markers', marker_size from session count, marker_color from pass rate continuous scale.")
bullet(doc, "CSV export: encode DataFrame as base64, render as st.download_button — native Streamlit, no JS needed.")
bullet(doc, "Fleet strip node click navigation: use st.query_params to pass agent filter, read on Quality v2 page load.")
bullet(doc, "CFO summary: use @st.cache_data keyed by data snapshot hash (hash of sessions df) to avoid re-calling LLM on every rerun.")

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 9. OPEN QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
h1(doc, "9. Open Questions (deferred)")
bullet(doc, "React migration: Vercel (frontend) + Supabase JS client. Trigger: when design is validated and external demo is needed.")
bullet(doc, "Circuit breaker enable flow: the '[Enable circuit breakers →]' link in the Savings Strip has no destination yet. Defer to Phase 4c implementation.")
bullet(doc, "Session quality composite on radar: session_quality agent evals vary — confirm which eval_names are available before building spokes.")
bullet(doc, "Bubble matrix minimum session threshold: currently set at 2. Revisit after seeing real data density — may need to raise to 3.")

doc.add_paragraph()
divider(doc)
p = doc.add_paragraph("End of specification")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p.runs:
    run.font.color.rgb = MUTED
    run.font.size = Pt(9)

doc.save(OUT)
print(f"Written: {OUT}")
