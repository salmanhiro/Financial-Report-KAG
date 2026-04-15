"""
Domain knowledge graph for cash flow analysis.

This is the "K" in KAG (Knowledge Augmented Generation). We encode financial
domain knowledge — concepts, relationships, and heuristic rules — as a graph
using networkx. The graph is later queried to retrieve relevant rules and
definitions that ground the LLM's reasoning.
"""

import networkx as nx


def build_cashflow_kg() -> nx.DiGraph:
    g = nx.DiGraph()

    # --- Concepts (nodes) ---
    concepts = {
        "OperatingCashFlow": "Cash generated from core business operations. The most important indicator of sustainable cash generation.",
        "InvestingCashFlow": "Cash used for or generated from investments (capex, acquisitions, asset sales). Usually negative for growing companies.",
        "FinancingCashFlow": "Cash from debt/equity issuance or repayment, dividends. Signals how a company funds itself.",
        "FreeCashFlow": "Operating Cash Flow minus Capital Expenditures. Cash truly available to investors.",
        "NetIncome": "Accounting profit; may diverge from cash due to accruals.",
        "CapEx": "Capital expenditures; investment in long-term assets.",
        "TotalDebt": "Sum of short and long-term debt obligations.",
        "Revenue": "Top-line sales.",
        "OCFtoNI": "Ratio of Operating Cash Flow to Net Income; quality-of-earnings indicator.",
        "FCFMargin": "Free Cash Flow / Revenue; efficiency of cash generation.",
        "DebtToOCF": "Total Debt / Operating Cash Flow; how many years of OCF to repay debt.",
        "CashFlowHealth": "Overall assessment of a company's cash flow situation.",
    }
    for name, desc in concepts.items():
        g.add_node(name, type="concept", description=desc)

    # --- Relationships (edges) ---
    g.add_edge("FreeCashFlow", "OperatingCashFlow", rel="derived_from")
    g.add_edge("FreeCashFlow", "CapEx", rel="derived_from")
    g.add_edge("OCFtoNI", "OperatingCashFlow", rel="derived_from")
    g.add_edge("OCFtoNI", "NetIncome", rel="derived_from")
    g.add_edge("FCFMargin", "FreeCashFlow", rel="derived_from")
    g.add_edge("FCFMargin", "Revenue", rel="derived_from")
    g.add_edge("DebtToOCF", "TotalDebt", rel="derived_from")
    g.add_edge("DebtToOCF", "OperatingCashFlow", rel="derived_from")
    g.add_edge("CashFlowHealth", "OperatingCashFlow", rel="assessed_by")
    g.add_edge("CashFlowHealth", "FreeCashFlow", rel="assessed_by")
    g.add_edge("CashFlowHealth", "OCFtoNI", rel="assessed_by")
    g.add_edge("CashFlowHealth", "FCFMargin", rel="assessed_by")
    g.add_edge("CashFlowHealth", "DebtToOCF", rel="assessed_by")

    # --- Heuristic rules (as nodes linked to concepts) ---
    rules = [
        ("R1", "Positive and growing Operating Cash Flow is a strong sign of healthy operations.", "OperatingCashFlow"),
        ("R2", "Operating Cash Flow should generally exceed Net Income (OCF/NI > 1). A ratio below 1 may indicate aggressive accounting or working-capital issues.", "OCFtoNI"),
        ("R3", "Free Cash Flow > 0 means the company funds its growth internally. FCF < 0 is acceptable only for early-stage or heavy-investment phases.", "FreeCashFlow"),
        ("R4", "FCF Margin above 10% is considered strong; below 5% is weak for mature companies.", "FCFMargin"),
        ("R5", "Debt/OCF above 4 signals elevated leverage risk; above 6 is concerning.", "DebtToOCF"),
        ("R6", "Financing cash inflows combined with negative operating cash flow can indicate the company is burning cash and relying on external funding.", "FinancingCashFlow"),
        ("R7", "Sharp year-over-year drops in Operating Cash Flow warrant investigation, especially if revenue is stable or rising.", "OperatingCashFlow"),
    ]
    for rid, text, concept in rules:
        g.add_node(rid, type="rule", text=text)
        g.add_edge(rid, concept, rel="applies_to")

    return g


def retrieve_rules_for(g: nx.DiGraph, concepts: list[str]) -> list[str]:
    """Retrieve all rule texts that apply to any of the given concepts."""
    out = []
    for node, data in g.nodes(data=True):
        if data.get("type") != "rule":
            continue
        targets = [t for _, t, d in g.out_edges(node, data=True) if d.get("rel") == "applies_to"]
        if any(c in targets for c in concepts):
            out.append(f"[{node}] {data['text']}")
    return out


def describe_concepts(g: nx.DiGraph, concepts: list[str]) -> list[str]:
    out = []
    for c in concepts:
        if c in g.nodes and g.nodes[c].get("type") == "concept":
            out.append(f"- {c}: {g.nodes[c]['description']}")
    return out


if __name__ == "__main__":
    g = build_cashflow_kg()
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    print("\nSample rules for [OperatingCashFlow, FreeCashFlow]:")
    for r in retrieve_rules_for(g, ["OperatingCashFlow", "FreeCashFlow"]):
        print(" ", r)
