"""
KAG (Knowledge Augmented Generation) demo for cash flow analysis.

Accepts either:
  - a CSV with one row per company-year, OR
  - a PDF cash-flow statement (e.g. IDX-style Indonesian disclosures).

Pipeline:
  1. Parse a financial document (CSV or PDF) into facts.
  2. Compute key metrics (FCF, OCF/NI, FCF margin, Debt/OCF...).
  3. Retrieve relevant definitions + rules from the domain knowledge graph.
  4. Build an augmented prompt and ask a LOCAL Ollama model to produce a verdict.

Examples:
  # CSV
  python analyze.py sample_cashflow.csv --company AcmeCorp --model llama3.2

  # PDF (IDX-style, numbers reported in millions of IDR)
  python analyze.py filing.pdf --company "PT ABC Tbk" --year 2024 --scale 1000000

  # Inspect the augmented prompt without calling the LLM
  python analyze.py filing.pdf --company "PT ABC Tbk" --year 2024 --dry-run
"""

import argparse
import csv
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

from knowledge_graph import build_cashflow_kg, retrieve_rules_for, describe_concepts


OLLAMA_URL = "http://localhost:11434/api/generate"


# --- loaders -----------------------------------------------------------------

def load_csv(path: str, company: str | None) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k != "company":
                r[k] = float(v)
    if company:
        rows = [r for r in rows if r["company"].lower() == company.lower()]
    rows.sort(key=lambda r: r["year"])
    return rows


def load_pdf(path: str, company: str | None, year: int | None, scale: float | None,
             model: str = "granite3.2-vision") -> list[dict]:
    # Lazy import so CSV users don't need pdfplumber installed.
    from pdf_extract import extract_cashflow_fields
    record = extract_cashflow_fields(path, company=company, year=year, scale=scale, model=model)
    record.pop("_extraction", None)
    # Zero-fill any fields the parser couldn't find so downstream math works;
    # warnings have already been printed by the extractor.
    for k in ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
              "net_income", "capex", "total_debt", "revenue"):
        if record.get(k) is None:
            record[k] = 0.0
    return [record]


def load_document(path: str, company: str | None, year: int | None, scale: float | None,
                  model: str = "granite3.2-vision") -> list[dict]:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv(path, company)
    if suffix == ".pdf":
        return load_pdf(path, company, year, scale, model=model)
    raise ValueError(f"Unsupported file type: {suffix}")


# --- metrics + KAG -----------------------------------------------------------

def compute_metrics(row: dict, prev: dict | None) -> dict:
    ocf = row["operating_cash_flow"]
    capex = row["capex"]
    ni = row["net_income"]
    rev = row["revenue"]
    debt = row["total_debt"]
    fcf = ocf - capex
    metrics = {
        "operating_cash_flow": ocf,
        "free_cash_flow": fcf,
        "ocf_to_ni": ocf / ni if ni else None,
        "fcf_margin": fcf / rev if rev else None,
        "debt_to_ocf": debt / ocf if ocf > 0 else None,
        "financing_cash_flow": row["financing_cash_flow"],
        "investing_cash_flow": row["investing_cash_flow"],
    }
    if prev:
        prev_ocf = prev["operating_cash_flow"]
        if prev_ocf:
            metrics["ocf_yoy_change"] = (ocf - prev_ocf) / prev_ocf
    return metrics


def relevant_concepts(metrics: dict) -> list[str]:
    concepts = ["OperatingCashFlow", "FreeCashFlow", "OCFtoNI", "FCFMargin", "DebtToOCF"]
    if metrics.get("financing_cash_flow", 0) > 0 and metrics["operating_cash_flow"] < 0:
        concepts.append("FinancingCashFlow")
    return concepts


def build_prompt(company: str, year: int, metrics: dict, defs: list[str], rules: list[str]) -> str:
    facts = "\n".join(
        f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
        for k, v in metrics.items() if v is not None
    )
    return f"""You are a financial analyst. Determine whether {company}'s cash flow in {year} is HEALTHY, WEAK, or MIXED, and explain why.

# Knowledge Graph — Concepts
{chr(10).join(defs)}

# Knowledge Graph — Rules (must ground your reasoning in these)
{chr(10).join(rules)}

# Computed Facts for {company} ({year})
{facts}

# Instructions
1. Cite the rule IDs (e.g. R2, R4) that justify each observation.
2. Conclude with one line: `VERDICT: <HEALTHY|WEAK|MIXED>`.
"""


def call_ollama(model: str, prompt: str) -> str:
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["response"]
    except urllib.error.URLError as e:
        return (f"[Ollama unreachable at {OLLAMA_URL}: {e}]\n"
                f"Start it with `ollama serve` and `ollama pull {model}`.\n"
                "Showing prompt instead so you can inspect the KAG output:\n\n" + prompt)


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to cash flow CSV or PDF")
    ap.add_argument("--company", default=None,
                    help="CSV: filter to this company. PDF: override the LLM-extracted name.")
    ap.add_argument("--year", type=int, default=None,
                    help="PDF: override the LLM-extracted reporting year.")
    ap.add_argument("--scale", type=float, default=None,
                    help="PDF: override the LLM-inferred unit scale (e.g. 1000000 for millions).")
    ap.add_argument("--model", default="granite3.2-vision")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the augmented prompt without calling the LLM")
    ap.add_argument("--report", default=None,
                    help="Write a PDF report to this path (e.g. report.pdf)")
    ap.add_argument("--report-md", default=None,
                    help="Write a Markdown report to this path")
    args = ap.parse_args()

    rows = load_document(args.path, args.company, args.year, args.scale, model=args.model)
    if not rows:
        print("No matching rows.", file=sys.stderr)
        sys.exit(1)

    g = build_cashflow_kg()
    prev = None
    results = []
    for row in rows:
        metrics = compute_metrics(row, prev)
        concepts = relevant_concepts(metrics)
        defs = describe_concepts(g, concepts)
        rules = retrieve_rules_for(g, concepts)
        prompt = build_prompt(row["company"], int(row["year"]), metrics, defs, rules)

        print("=" * 70)
        print(f"{row['company']} — {int(row['year'])}")
        print("=" * 70)
        llm_output = None
        if args.dry_run:
            print(prompt)
        else:
            llm_output = call_ollama(args.model, prompt)
            print(llm_output)

        results.append({
            "company": row["company"],
            "year": int(row["year"]),
            "metrics": metrics,
            "rules": rules,
            "prompt": prompt,
            "llm_output": llm_output,
        })
        prev = row

    if args.report or args.report_md:
        from report import build_markdown, build_pdf
        if args.report:
            build_pdf(results, args.report)
            print(f"\n[report] PDF written to {args.report}")
        if args.report_md:
            with open(args.report_md, "w") as f:
                f.write(build_markdown(results))
            print(f"[report] Markdown written to {args.report_md}")


if __name__ == "__main__":
    main()
