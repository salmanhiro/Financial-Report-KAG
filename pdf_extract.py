"""
PDF extraction for cash-flow statements using a local Ollama LLM.

Pure LLM approach — no regex, no keyword matching, no tabular assumptions.
Optimized for speed: small context windows, tight page digests, model
kept warm across calls.

Pipeline:
  1. PyMuPDF (pymupdf) extracts text page-by-page — 10-20x faster than pdfplumber.
  2. Small doc → single extraction call.
     Large doc → locate pass (tiny per-page digest) picks 1–3 pages,
                 then extraction runs on only those pages.
  3. Ollama `format: "json"` + low temperature + num_predict cap.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path

import pymupdf


OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Speed tuning knobs -----------------------------------------------------
# Characters, not tokens. 1 token ≈ 3-4 chars for English/Indonesian.
SMALL_DOC_LIMIT      = 20_000   # send whole doc in one call if it fits this
PAGE_SNIPPET_CHARS   = 400      # per-page digest for the locate pass
LOCATE_DIGEST_CAP    = 30_000   # total chars sent to locate pass
EXTRACT_TEXT_CAP     = 20_000   # max chars sent to extract pass
EXTRACT_NUM_CTX      = 8_192    # smaller context = much faster
LOCATE_NUM_CTX       = 8_192
EXTRACT_NUM_PREDICT  = 512      # cap output tokens — JSON needs very little
LOCATE_NUM_PREDICT   = 128
KEEP_ALIVE           = "30m"    # keep model in RAM between calls

HTTP_TIMEOUT         = 600

EXTRACT_SCHEMA = {
    "company":              "Company name.",
    "year":                 "Reporting year (4-digit integer).",
    "currency":             "Currency code, e.g. 'IDR', 'USD'.",
    "unit_scale":           "Unit multiplier: 1, 1000, 1000000, or 1000000000. "
                            "'dalam jutaan' / 'in millions' = 1000000.",
    "operating_cash_flow":  "Net cash from operating activities (TOTAL line).",
    "investing_cash_flow":  "Net cash from investing activities (TOTAL line).",
    "financing_cash_flow":  "Net cash from financing activities (TOTAL line).",
    "net_income":           "Net income / profit.",
    "capex":                "Capital expenditure. Always positive.",
    "total_debt":           "Total interest-bearing debt / borrowings.",
    "revenue":              "Total revenue / net sales.",
}

EXTRACT_PROMPT = """\
Find the cash-flow statement in the document and return a single JSON object with these fields (use null if not present):
{schema}

- Parentheses like (1.234) are negative. Capex is always positive.
- Return numbers as printed; do not multiply by unit_scale.
- If the document has no cash-flow statement, set every numeric field to null.
- JSON only, no prose.

--- DOCUMENT ---
{text}
--- END ---
"""

LOCATE_PROMPT = """\
Text below is from a financial PDF with [[PAGE N]] markers. Return JSON {{"pages": [page numbers]}} listing only pages that contain the cash-flow statement (operating / investing / financing activities table). Empty list if none.

--- DOCUMENT ---
{text}
--- END ---
"""


class NotACashFlowStatement(RuntimeError):
    """Raised when the document does not contain a cash-flow statement."""


NUMERIC_FIELDS = [
    "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
    "net_income", "capex", "total_debt", "revenue",
]


def _build_schema_block() -> str:
    return "\n".join(f'  - {k}: {desc}' for k, desc in EXTRACT_SCHEMA.items())


def extract_pdf_pages(pdf_path: str | Path) -> list[str]:
    """Return a list of page texts (index 0 = page 1).

    Uses PyMuPDF (`pymupdf`) — 10-20x faster than pdfplumber on large PDFs.
    The "text" mode preserves reading order and keeps table rows intact
    enough for the LLM to interpret.
    """
    pages: list[str] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return pages


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _parse_json(text: str) -> dict:
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = None
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ValueError(f"No JSON object in response:\n{text[:500]}")


def _call_ollama(model: str, prompt: str,
                 num_ctx: int = EXTRACT_NUM_CTX,
                 num_predict: int = EXTRACT_NUM_PREDICT) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
        },
    }
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read())["response"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama unreachable at {OLLAMA_URL}: {e}\n"
            "Make sure Ollama is running."
        ) from e


def _locate_cashflow_pages(pages: list[str], model: str) -> list[int]:
    chunks: list[str] = []
    budget = LOCATE_DIGEST_CAP
    for i, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        head = page_text[:PAGE_SNIPPET_CHARS]
        chunk = f"\n[[PAGE {i}]]\n{head}"
        if len(chunk) > budget:
            break
        chunks.append(chunk)
        budget -= len(chunk)

    prompt = LOCATE_PROMPT.format(text="".join(chunks))
    raw = _call_ollama(model, prompt,
                       num_ctx=LOCATE_NUM_CTX,
                       num_predict=LOCATE_NUM_PREDICT)
    try:
        out = _parse_json(raw)
        found = out.get("pages") or []
        return [int(p) for p in found if str(p).strip().lstrip("-").isdigit()]
    except Exception:
        return []


def _pack_pages(pages: list[str], wanted: list[int], cap: int) -> str:
    """Concatenate wanted page texts up to `cap` chars."""
    out: list[str] = []
    used = 0
    for p in wanted:
        if not (1 <= p <= len(pages)):
            continue
        body = pages[p - 1]
        if not body.strip():
            continue
        block = f"[[PAGE {p}]]\n{body}\n\n"
        if used + len(block) > cap:
            block = block[: cap - used]
            out.append(block)
            break
        out.append(block)
        used += len(block)
    return "".join(out)


def extract_cashflow_fields(
    pdf_path: str | Path,
    company: str | None = None,
    year: int | None = None,
    scale: float | None = None,
    model: str = "granite3.2-vision",
) -> dict:
    """Extract one cash-flow record from a PDF using a local LLM."""
    print(f"[pdf_extract] reading {pdf_path} …")
    pages = extract_pdf_pages(pdf_path)
    full_text = "\n\n".join(pages)
    total = len(full_text)

    # --- Choose what to send the model --------------------------------------
    if total <= SMALL_DOC_LIMIT:
        document_text = full_text
        print(f"[pdf_extract] small doc ({total:,} chars) — single extract call")
    else:
        print(f"[pdf_extract] large doc ({total:,} chars) — locating pages …")
        target = _locate_cashflow_pages(pages, model)
        if target:
            neighbors = sorted({p + d for p in target for d in (-1, 0, 1)
                               if 1 <= p + d <= len(pages)})
            print(f"[pdf_extract] located pages {target} → sending {neighbors}")
            document_text = _pack_pages(pages, neighbors, EXTRACT_TEXT_CAP)
        else:
            print("[pdf_extract] no page located — using document head")
            document_text = full_text[:EXTRACT_TEXT_CAP]

    # --- Extract ------------------------------------------------------------
    prompt = EXTRACT_PROMPT.format(schema=_build_schema_block(), text=document_text)
    print(f"[pdf_extract] extracting with {model} ({len(document_text):,} chars) …")
    response = _call_ollama(model, prompt)

    try:
        extracted = _parse_json(response)
    except ValueError as e:
        raise RuntimeError(
            f"LLM did not return parseable JSON. Try a stronger model.\n\n"
            f"Raw:\n{response[:500]}"
        ) from e

    if all(extracted.get(k) is None for k in NUMERIC_FIELDS):
        raise NotACashFlowStatement(
            "This PDF does not contain a cash-flow statement."
        )

    resolved_company = company or extracted.get("company") or "Unknown"
    resolved_year = year or _coerce_int(extracted.get("year")) or 0
    resolved_scale = float(scale) if scale is not None else (
        _coerce_float(extracted.get("unit_scale")) or 1.0
    )
    print(f"[pdf_extract] company={resolved_company!r}, year={resolved_year}, "
          f"currency={extracted.get('currency')!r}, scale={resolved_scale}")

    result: dict = {"company": resolved_company, "year": resolved_year}
    for key in NUMERIC_FIELDS:
        val = extracted.get(key)
        if val is None:
            result[key] = None
            continue
        try:
            result[key] = float(val) * resolved_scale
        except (TypeError, ValueError):
            result[key] = None

    if result.get("capex") is not None and result["capex"] < 0:
        result["capex"] = -result["capex"]

    result["_extraction"] = {
        "company": extracted.get("company"),
        "year": extracted.get("year"),
        "currency": extracted.get("currency"),
        "unit_scale": extracted.get("unit_scale"),
        "resolved_scale": resolved_scale,
    }
    return result


def _coerce_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LLM-based cash-flow extractor.")
    ap.add_argument("pdf")
    ap.add_argument("--company", default=None)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--model", default="granite3.2-vision")
    args = ap.parse_args()

    result = extract_cashflow_fields(
        args.pdf, company=args.company, year=args.year,
        scale=args.scale, model=args.model,
    )
    print(json.dumps(result, indent=2))