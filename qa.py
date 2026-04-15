"""
Follow-up Q&A over KAG analysis results.

Given a user question and the already-computed analysis results (metrics,
rules retrieved from the KG, LLM verdict), construct an augmented prompt
that asks the LLM to answer using ONLY those facts and rules — and to
cite them explicitly as SOURCES.

Citation contract:
  - Knowledge-graph rules are cited as "Rule R2", "Rule R4", etc.
  - Computed facts are cited as "Fact: operating_cash_flow = 820 150 000".
  - Prior LLM verdict is cited as "Prior analysis (YYYY)".
The LLM is instructed to end every answer with a `## Sources` section
listing the citations actually used.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"


QA_SYSTEM = """\
You are a financial analyst assistant answering follow-up questions about a
company's cash flow. You MUST ground every claim in one of three sources:

  1. RULES — the numbered rules (R1, R2, …) from the knowledge graph.
  2. FACTS — the computed numeric facts shown below.
  3. PRIOR — the prior LLM analysis/verdict for the same period.

Rules of engagement:
  - If the question cannot be answered from the provided context, say so
    plainly. DO NOT invent data or cite rules that don't exist.
  - Keep the answer concise (a few short paragraphs).
  - Inline-cite as you go, e.g. "leverage is moderate (Rule R5)" or
    "with OCF of 820.15 M (Fact: operating_cash_flow)".
  - End with a `## Sources` section that lists every citation you actually
    used, one per line, in this exact format:
        - Rule R2 — <one-line paraphrase of the rule>
        - Fact: operating_cash_flow = <value>
        - Prior analysis (<year>) — <one-line gist>
"""


def _format_results_context(results: list[dict]) -> str:
    """Serialize the analysis results into a compact, LLM-readable context block."""
    blocks: list[str] = []
    seen_rules: dict[str, str] = {}  # rule_id -> text

    for r in results:
        lines = [f"### {r['company']} — {r['year']}"]
        if r.get("llm_output"):
            lines.append("PRIOR ANALYSIS:")
            lines.append(r["llm_output"].strip())
        lines.append("FACTS:")
        for k, v in r["metrics"].items():
            if v is None:
                continue
            if isinstance(v, float):
                val = f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.0f}"
            else:
                val = str(v)
            lines.append(f"  - {k} = {val}")
        lines.append("RULES APPLIED:")
        for rule in r["rules"]:
            lines.append(f"  - {rule}")
            # Parse "[R2] text..." into id + paraphrase for dedup
            if rule.startswith("[") and "]" in rule:
                rid = rule[1:rule.index("]")]
                seen_rules[rid] = rule[rule.index("]") + 1:].strip()
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_qa_prompt(question: str, results: list[dict]) -> str:
    ctx = _format_results_context(results)
    return f"""{QA_SYSTEM}

# Context (all the information you are allowed to use)

{ctx}

# Question
{question}

# Answer (remember: inline-cite, then end with `## Sources`)
"""


def answer_question(question: str, results: list[dict], model: str = "granite3.2-vision") -> str:
    prompt = build_qa_prompt(question, results)
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["response"]
    except urllib.error.URLError as e:
        return (f"[Ollama unreachable at {OLLAMA_URL}: {e}]\n"
                "Start Ollama with `ollama serve`.")
