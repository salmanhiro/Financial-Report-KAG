"""
Generate a cash-flow analysis report (Markdown + PDF).

Each "period result" passed in is a dict:
    {
        "company": str,
        "year": int,
        "metrics": dict[str, float|None],
        "rules": list[str],      # retrieved rule texts, e.g. "[R2] ..."
        "prompt": str,           # full augmented prompt (optional, for appendix)
        "llm_output": str | None # may be None in dry-run
    }
"""

from __future__ import annotations

import io
import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


VERDICT_RE = re.compile(r"VERDICT\s*:\s*(HEALTHY|WEAK|MIXED)", re.IGNORECASE)


def _verdict_from(text: str | None) -> str:
    if not text:
        return "N/A"
    m = VERDICT_RE.search(text)
    return m.group(1).upper() if m else "UNCLASSIFIED"


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1_000:
            return f"{v:,.0f}"
        return f"{v:.3f}"
    return str(v)


# ---------- Markdown ---------------------------------------------------------

def build_markdown(results: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out: list[str] = []
    out.append("# Cash-Flow Analysis Report")
    out.append(f"*Generated {now} · KAG pipeline with local Ollama*\n")

    # Executive summary
    out.append("## Executive summary\n")
    out.append("| Company | Year | Verdict | OCF | FCF | OCF/NI | FCF margin |")
    out.append("|---|---|---|---|---|---|---|")
    for r in results:
        m = r["metrics"]
        out.append(
            f"| {r['company']} | {r['year']} | **{_verdict_from(r.get('llm_output'))}** "
            f"| {_fmt(m.get('operating_cash_flow'))} "
            f"| {_fmt(m.get('free_cash_flow'))} "
            f"| {_fmt(m.get('ocf_to_ni'))} "
            f"| {_fmt(m.get('fcf_margin'))} |"
        )
    out.append("")

    # Per-period detail
    for r in results:
        out.append(f"## {r['company']} — {r['year']}\n")
        out.append(f"**Verdict:** {_verdict_from(r.get('llm_output'))}\n")

        out.append("### Metrics")
        for k, v in r["metrics"].items():
            out.append(f"- `{k}`: {_fmt(v)}")
        out.append("")

        out.append("### Rules applied (from knowledge graph)")
        for rule in r["rules"]:
            out.append(f"- {rule}")
        out.append("")

        out.append("### LLM analysis")
        out.append(r.get("llm_output") or "_(dry-run — LLM not called)_")
        out.append("")

    return "\n".join(out)


# ---------- PDF --------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, spaceAfter=6),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=16),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13),
        "mono": ParagraphStyle("mono", parent=base["BodyText"], fontName="Courier", fontSize=8.5, leading=11),
        "verdict_h": ParagraphStyle("vh", parent=base["BodyText"], fontSize=10, fontName="Helvetica-Bold"),
    }
    return styles


def _verdict_color(v: str):
    return {
        "HEALTHY": colors.HexColor("#16a34a"),
        "WEAK":    colors.HexColor("#dc2626"),
        "MIXED":   colors.HexColor("#ca8a04"),
    }.get(v, colors.grey)


def build_pdf(results: list[dict], output_path: str) -> str:
    st = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )
    story = []

    story.append(Paragraph("Cash-Flow Analysis Report", st["title"]))
    story.append(Paragraph(
        f"Generated {datetime.now():%Y-%m-%d %H:%M} · KAG pipeline with local Ollama",
        st["caption"],
    ))

    # --- executive summary table ---
    story.append(Paragraph("Executive summary", st["h2"]))
    header = ["Company", "Year", "Verdict", "OCF", "FCF", "OCF/NI", "FCF margin"]
    data = [header]
    for r in results:
        m = r["metrics"]
        data.append([
            r["company"],
            str(r["year"]),
            _verdict_from(r.get("llm_output")),
            _fmt(m.get("operating_cash_flow")),
            _fmt(m.get("free_cash_flow")),
            _fmt(m.get("ocf_to_ni")),
            _fmt(m.get("fcf_margin")),
        ])
    t = Table(data, repeatRows=1, hAlign="LEFT")
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
    for i, r in enumerate(results, start=1):
        ts.add("TEXTCOLOR", (2, i), (2, i),
               _verdict_color(_verdict_from(r.get("llm_output"))))
        ts.add("FONTNAME", (2, i), (2, i), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)

    # --- per-period detail ---
    for r in results:
        story.append(PageBreak())
        v = _verdict_from(r.get("llm_output"))
        story.append(Paragraph(f"{r['company']} — {r['year']}", st["h2"]))
        story.append(Paragraph(
            f'Verdict: <font color="{_verdict_color(v).hexval()}"><b>{v}</b></font>',
            st["verdict_h"],
        ))
        story.append(Spacer(1, 6))

        # metrics table
        story.append(Paragraph("Metrics", st["h3"]))
        mdata = [["Metric", "Value"]] + [[k, _fmt(v)] for k, v in r["metrics"].items()]
        mt = Table(mdata, colWidths=[6 * cm, 4 * cm], hAlign="LEFT")
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ]))
        story.append(mt)

        # rules
        story.append(Paragraph("Rules applied (from knowledge graph)", st["h3"]))
        for rule in r["rules"]:
            story.append(Paragraph("• " + _escape(rule), st["body"]))

        # LLM analysis
        story.append(Paragraph("LLM analysis", st["h3"]))
        text = r.get("llm_output") or "(dry-run — LLM not called)"
        for para in text.split("\n\n"):
            story.append(Paragraph(_escape(para).replace("\n", "<br/>"), st["body"]))
            story.append(Spacer(1, 4))

    doc.build(story)
    return output_path


def build_pdf_bytes(results: list[dict]) -> bytes:
    buf = io.BytesIO()
    # SimpleDocTemplate accepts a file-like object too
    build_pdf(results, buf)  # type: ignore[arg-type]
    return buf.getvalue()


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))
