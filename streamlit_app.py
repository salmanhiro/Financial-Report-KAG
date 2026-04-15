"""
Streamlit UI for the KAG cash-flow analyzer.

Run:
    pip install -r requirements.txt
    ollama serve &
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from knowledge_graph import build_cashflow_kg, describe_concepts, retrieve_rules_for
from analyze import (
    build_prompt,
    call_ollama,
    compute_metrics,
    load_csv,
    relevant_concepts,
)
from pdf_extract import extract_cashflow_fields, NotACashFlowStatement
from report import build_markdown, build_pdf_bytes
from qa import answer_question


st.set_page_config(page_title="KAG Cash-Flow Analyzer", page_icon="💵", layout="wide")
st.title("💵 KAG Cash-Flow Analyzer")
st.caption(
    "Knowledge Augmented Generation over a financial knowledge graph — "
    "local Ollama, no API key required."
)

# --- sidebar: config ---------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    model = st.text_input("Ollama model", value="granite3.2-vision",
                          help="e.g.llama3.2, mistral, phi3")
    dry_run = st.checkbox(
        "Dry-run (skip LLM, show augmented prompt only)", value=False
    )
    st.divider()
    st.header("Knowledge Graph")
    g = build_cashflow_kg()
    st.metric("Concepts", sum(1 for _, d in g.nodes(data=True) if d.get("type") == "concept"))
    st.metric("Rules", sum(1 for _, d in g.nodes(data=True) if d.get("type") == "rule"))
    with st.expander("Browse rules"):
        for n, d in g.nodes(data=True):
            if d.get("type") == "rule":
                st.markdown(f"**{n}** — {d['text']}")

# --- main: input -------------------------------------------------------------
st.subheader("1. Upload a cash-flow document")
st.caption(
    "Upload a **PDF** (any language/layout — the LLM extracts everything, "
    "including company name, reporting year, and unit scale) or a pre-structured **CSV**. "
    "Or tick the sample checkbox to try the demo data."
)
col_a, col_b = st.columns([2, 1])

with col_a:
    uploaded = st.file_uploader(
        "PDF or CSV",
        type=["csv", "pdf"],
    )
with col_b:
    use_sample = st.checkbox("Use bundled sample_cashflow.csv", value=not bool(uploaded))

# For CSVs we still need a company name to filter rows. For PDFs the LLM reads
# it from the document, but the user can override it below if needed.
csv_company_filter = ""
if uploaded is not None and uploaded.name.lower().endswith(".csv"):
    csv_company_filter = st.text_input(
        "Company name (filter the CSV rows)", value="AcmeCorp"
    )
elif use_sample:
    csv_company_filter = st.text_input(
        "Company name (filter the CSV rows)", value="AcmeCorp"
    )

with st.expander("⚙️  Override LLM extraction (optional, PDF only)"):
    st.caption(
        "Leave these empty to let the LLM extract everything from the document. "
        "Fill them in only to correct a mis-read."
    )
    override_company = st.text_input("Company name override", value="")
    oc1, oc2 = st.columns(2)
    with oc1:
        override_year = st.number_input(
            "Reporting year override", min_value=0, max_value=2100, value=0, step=1,
            help="0 means use the LLM's reading.",
        )
    with oc2:
        override_scale = st.number_input(
            "Unit-scale override", min_value=0.0, value=0.0, step=1.0, format="%.0f",
            help="0 means use the LLM's inference (e.g. 'dalam jutaan Rupiah' → 1 000 000).",
        )

# --- load rows ---------------------------------------------------------------
rows: list[dict] = []
source_label = ""

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    source_label = uploaded.name
    try:
        if suffix == ".csv":
            rows = load_csv(tmp_path, csv_company_filter or None)
            if not rows:
                st.warning(f"No rows for company '{csv_company_filter}' in CSV — showing all rows.")
                rows = load_csv(tmp_path, None)
        else:
            with st.spinner(f"Extracting fields from PDF using {model}…"):
                rec = extract_cashflow_fields(
                    tmp_path,
                    company=override_company or None,
                    year=int(override_year) or None,
                    scale=float(override_scale) if override_scale else None,
                    model=model,
                )
            # Show what the LLM read from the document
            meta = rec.pop("_extraction", {})
            with st.container(border=True):
                st.markdown("**🤖 LLM auto-extracted from the document:**")
                mcols = st.columns(4)
                mcols[0].metric("Company", meta.get("company") or "—")
                mcols[1].metric("Year", meta.get("year") or "—")
                mcols[2].metric("Currency", meta.get("currency") or "—")
                mcols[3].metric("Unit scale", f"×{meta.get('resolved_scale', 1):,.0f}")

            for k in ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
                      "net_income", "capex", "total_debt", "revenue"):
                if rec.get(k) is None:
                    st.warning(f"Could not extract **{k}** from PDF — defaulted to 0.")
                    rec[k] = 0.0
            rows = [rec]
    except NotACashFlowStatement as e:
        st.error(f"📄 **{uploaded.name}** doesn't look like a cash-flow statement")
        st.info(str(e))
        st.caption(
            "Tip: this tool expects a *laporan arus kas* / cash-flow statement page "
            "from an annual report or quarterly filing — not bond prospectuses, "
            "legal announcements, or general press releases."
        )
    except Exception as e:
        st.error(f"Failed to parse {uploaded.name}")
        with st.expander("Error details"):
            st.code(str(e))
        st.caption(
            "If the PDF is a real cash-flow statement, try switching to a more "
            "capable Ollama model in the sidebar (e.g. `granite3.2-vision:7b`, `mistral`, "
            "or `llama3.1:8b`)."
        )

elif use_sample:
    sample_path = Path(__file__).parent / "sample_cashflow.csv"
    if sample_path.exists():
        rows = load_csv(str(sample_path), csv_company_filter or None)
        if not rows:
            rows = load_csv(str(sample_path), None)
        source_label = sample_path.name

# --- display + run -----------------------------------------------------------
if rows:
    st.subheader(f"2. Extracted facts — {source_label}")
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.subheader("3. Run KAG analysis")
    if st.button("Analyze", type="primary"):
        prev = None
        results = []
        for row in rows:
            metrics = compute_metrics(row, prev)
            concepts = relevant_concepts(metrics)
            defs = describe_concepts(g, concepts)
            rules = retrieve_rules_for(g, concepts)
            prompt = build_prompt(row["company"], int(row["year"]), metrics, defs, rules)

            with st.container(border=True):
                st.markdown(f"### {row['company']} — {int(row['year'])}")

                mcol, rcol = st.columns(2)
                with mcol:
                    st.markdown("**Computed metrics**")
                    display = {k: (f"{v:.3f}" if isinstance(v, float) else v)
                               for k, v in metrics.items() if v is not None}
                    st.json(display)
                with rcol:
                    st.markdown("**Retrieved rules (from KG)**")
                    for r in rules:
                        st.markdown(f"- {r}")

                with st.expander("Augmented prompt sent to the LLM"):
                    st.code(prompt, language="markdown")

                llm_out = None
                if dry_run:
                    st.info("Dry-run enabled — LLM call skipped.")
                else:
                    with st.spinner(f"Asking Ollama ({model})…"):
                        llm_out = call_ollama(model, prompt)
                    st.markdown("**LLM analysis**")
                    st.markdown(llm_out)

            results.append({
                "company": row["company"],
                "year": int(row["year"]),
                "metrics": metrics,
                "rules": rules,
                "prompt": prompt,
                "llm_output": llm_out,
            })
            prev = row

        st.session_state["results"] = results
        st.session_state["chat"] = []  # reset Q&A history for new analysis

    # Report downloads persist across reruns via session_state.
    if "results" in st.session_state and st.session_state["results"]:
        st.subheader("4. Download report")
        results = st.session_state["results"]
        md = build_markdown(results)
        with st.expander("Preview report (Markdown)"):
            st.markdown(md)
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "⬇️ Download PDF report",
                data=build_pdf_bytes(results),
                file_name="cashflow_report.pdf",
                mime="application/pdf",
                type="primary",
            )
        with dcol2:
            st.download_button(
                "⬇️ Download Markdown report",
                data=md,
                file_name="cashflow_report.md",
                mime="text/markdown",
            )

        # --- 5. Follow-up Q&A (grounded in KG rules + extracted facts) ------
        st.subheader("5. Ask a follow-up question")
        st.caption(
            "Ask anything about the analyzed cash flow. Answers are grounded in the "
            "retrieved knowledge-graph rules and the extracted facts — every claim is "
            "cited, and every response ends with a **Sources** section."
        )

        if "chat" not in st.session_state:
            st.session_state["chat"] = []

        # Replay history
        for turn in st.session_state["chat"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        # Suggested questions
        suggestions = [
            "Why is the verdict what it is?",
            "Which year had the weakest cash flow and why?",
            "Is the company over-leveraged?",
            "Is the company's earnings quality good?",
        ]
        sug_cols = st.columns(len(suggestions))
        picked: str | None = None
        for col, s in zip(sug_cols, suggestions):
            with col:
                if st.button(s, use_container_width=True, key=f"sug_{s}"):
                    picked = s

        user_msg = picked or st.chat_input("Ask a follow-up…")
        if user_msg:
            st.session_state["chat"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                with st.spinner(f"Answering with {model}…"):
                    answer = answer_question(user_msg, results, model=model)
                st.markdown(answer)
            st.session_state["chat"].append({"role": "assistant", "content": answer})

        if st.session_state["chat"] and st.button("Clear chat history"):
            st.session_state["chat"] = []
            st.rerun()
else:
    st.info("Upload a file or tick 'Use bundled sample' to get started.")
