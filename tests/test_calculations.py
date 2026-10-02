"""
RuleBeaconAI — derived-metric / calculation regression tests.

These are pure-logic checks (no Qdrant, embedding model or LLM required):

    1. Fiscal-year range expansion ("FY2021 to FY2024").
    2. Derived-metric intent detection (growth / margin / share).
    3. Reranker synonym handling for
       "Net Earnings Attributable to Controlling Interests".
    4. Out-of-corpus year guard still fires for an unsupported year range.
    5. Figure-aware metric coverage (a numeric table row vs a prose mention).

Run from the project root:

    python -m tests.test_calculations
"""

from src.rag.calculations import (
    detect_calculation,
    detect_metric_concepts,
    metric_value_present,
    primary_metric,
)
from src.rag.reranker import DocumentReranker
from src.rag.service import (
    RAGService,
    detect_requested_years,
    expand_year_ranges,
)


class FailingVectorStore:

    def __getattr__(self, name):
        raise AssertionError(
            "Vector store should not be accessed "
            "for an unsupported fiscal year."
        )


class FailingLLM:

    def invoke(self, *args, **kwargs):
        raise AssertionError(
            "LLM should not be called "
            "for an unsupported fiscal year."
        )


def check(label, condition, results):
    status = "PASS" if condition else "FAIL"
    results.append((label, condition))
    print(f"[{status}] {label}")
    return condition


def test_year_ranges(results):
    query = (
        "How did Microsoft's revenue grow each year "
        "from FY2021 to FY2024?"
    )

    expanded = expand_year_ranges(query)
    check(
        "range expansion covers FY2021-FY2024 interior years",
        expanded == [2021, 2022, 2023, 2024],
        results,
    )

    requested = detect_requested_years(query)
    check(
        "detected years include 2022 and 2023 for FY2021-FY2024",
        {2021, 2022, 2023, 2024}.issubset(set(requested)),
        results,
    )

    check(
        "comparison phrasing (vs) is not treated as a range",
        expand_year_ranges("Compare NVIDIA revenue FY2024 vs FY2022") == [],
        results,
    )

    check(
        "single unsupported year is still detected",
        detect_requested_years(
            "What was Apple's total revenue in fiscal year 2019?"
        ) == [2019],
        results,
    )


def test_calculation_intents(results):
    growth = detect_calculation(
        "Compare NVIDIA's revenue growth rate in FY2022 vs FY2023 vs FY2024."
    )
    check(
        "growth intent detected for revenue growth question",
        growth.kind == "growth"
        and "revenue" in growth.required_metrics
        and growth.needs_prior_year,
        results,
    )

    margin = detect_calculation(
        "What was Tesla's operating income in FY2022, "
        "and what was its operating margin?"
    )
    check(
        "margin intent requires operating income and revenue",
        margin.kind == "margin"
        and "operating_income" in margin.required_metrics
        and "revenue" in margin.required_metrics,
        results,
    )

    share = detect_calculation(
        "What was NVIDIA's data center revenue in FY2024, "
        "and what share of total revenue did it represent?"
    )
    check(
        "share intent requires revenue",
        share.kind == "share" and "revenue" in share.required_metrics,
        results,
    )

    plain = detect_calculation(
        "What was NVIDIA's fiscal year 2024 end date?"
    )
    check(
        "non-derived question yields no calculation spec",
        plain.kind is None and not plain.required_metrics,
        results,
    )


def test_attributable_synonym(results):
    concepts = detect_metric_concepts(
        "What was Archer Daniels Midland's net earnings "
        "attributable to ADM in FY2023?"
    )
    check(
        "attributable phrasing maps to net_income",
        "net_income" in concepts,
        results,
    )

    attributable_doc = (
        "| Net Earnings Attributable to Controlling Interests "
        "| | | $ | 3,483 | 4,340 | 3,226 |"
    )
    score = DocumentReranker._exact_metric_score(
        "What was Archer Daniels Midland's net earnings "
        "attributable to ADM in FY2023?",
        attributable_doc,
    )
    check(
        "reranker boosts the attributable net earnings label",
        score == 1.0,
        results,
    )

    qualified_doc = (
        "Net earnings including noncontrolling interests 4,368"
    )
    qualified_score = DocumentReranker._exact_metric_score(
        "net earnings",
        qualified_doc,
    )
    check(
        "reranker still ignores qualified 'including' variant",
        qualified_score == 0.0,
        results,
    )

    revenue_doc = "| Revenue | | | $ | 60,922 |"
    revenue_score = DocumentReranker._exact_metric_score(
        "What was NVIDIA's revenue in fiscal year 2024?",
        revenue_doc,
    )
    check(
        "reranker still boosts a plain revenue table label",
        revenue_score == 1.0,
        results,
    )


def test_figure_aware_coverage(results):
    adm_attributable = (
        "| Net Earnings Attributable to Controlling Interests "
        "| | | $ | 3,483 | 4,340 | 3,226 |"
    )
    check(
        "figure check accepts the attributable income-statement row",
        metric_value_present(adm_attributable, "net_income"),
        results,
    )

    adm_including = (
        "| Net earnings including noncontrolling interests "
        "| | | $ | 3,466 |"
    )
    check(
        "figure check rejects the including-noncontrolling row",
        not metric_value_present(adm_including, "net_income"),
        results,
    )

    prose = (
        "Net earnings attributable to controlling interests "
        "decreased 20% versus the prior year."
    )
    check(
        "figure check ignores prose that only mentions the metric",
        not metric_value_present(prose, "net_income"),
        results,
    )

    tsla_revenue = "| Total revenues | 81,462 | 53,823 | 31,536 |"
    check(
        "figure check accepts the total revenues row",
        metric_value_present(tsla_revenue, "revenue"),
        results,
    )

    tsla_operating = "| Income from operations | 13,656 | 6,523 | 1,994 |"
    check(
        "figure check accepts the income from operations row",
        metric_value_present(tsla_operating, "operating_income"),
        results,
    )

    # A numeric row whose *label* is not the metric term must not count as a
    # figure just because a matching number sits in the table. Segment or
    # geography subtotal rows (label "Total") are the common corpus case.
    subtotal_row = "| Total | $ | 245,122 | $ | 211,915 | $ | 198,270 |"
    check(
        "figure check rejects a numeric subtotal row not labelled with the metric",
        not metric_value_present(subtotal_row, "revenue"),
        results,
    )

    # Prose emitted inside table pipes (how the corpus renders MD&A bullets)
    # must not be mistaken for a financial-statement line item.
    bullet_row = "| | \u2022 | Revenue increased 32% from the prior year. |"
    check(
        "figure check rejects a bulleted prose row that mentions the metric",
        not metric_value_present(bullet_row, "revenue"),
        results,
    )

    prose_row = "| Revenue increased 32% from the prior year. |"
    check(
        "figure check rejects a bare prose row that mentions the metric",
        not metric_value_present(prose_row, "revenue"),
        results,
    )

    percent_row = "| As a percent of revenue | 4% | 3% | 1ppt |"
    check(
        "figure check rejects a percentage-only change row",
        not metric_value_present(percent_row, "revenue"),
        results,
    )

    # A genuine label may carry a trailing footnote marker; still a figure.
    footnoted_row = "| Total revenue (a) | 245,122 | 211,915 |"
    check(
        "figure check accepts a metric label with a footnote marker",
        metric_value_present(footnoted_row, "revenue"),
        results,
    )

    headline = primary_metric(
        detect_metric_concepts(
            "What was Archer Daniels Midland's net earnings "
            "attributable to ADM in FY2023?"
        )
    )
    check(
        "headline metric for the attributable question is net_income",
        headline == "net_income",
        results,
    )


def test_unsupported_range_guard(results):
    service = RAGService(
        vector_store=FailingVectorStore(),
        llm=FailingLLM(),
        reranker=DocumentReranker.__new__(DocumentReranker),
    )

    result = service.ask(
        "What was Apple's total revenue from FY2019 to FY2021?"
    )

    answer = result["answer"]
    sources = result["sources"]

    check(
        "unsupported year inside a range is refused",
        "No filing for fiscal year" in answer
        and "current document corpus" in answer,
        results,
    )
    check(
        "no retrieval occurs for an unsupported year range",
        sources == [],
        results,
    )


def main():
    print("=" * 80)
    print("RuleBeaconAI CALCULATION / DERIVED-METRIC REGRESSION")
    print("=" * 80)

    results = []

    test_year_ranges(results)
    test_calculation_intents(results)
    test_attributable_synonym(results)
    test_figure_aware_coverage(results)
    test_unsupported_range_guard(results)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Overall: {passed}/{total} ({passed / total * 100:.1f}%)")
    print("=" * 80)

    if passed == total:
        print("✓ ALL CALCULATION REGRESSION TESTS PASSED")
    else:
        print("✗ CALCULATION REGRESSION FAILED")
        for label, ok in results:
            if not ok:
                print(f"  - FAILED: {label}")

    return passed == total


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
