"""
General derived-metric / calculation support for RuleBeaconAI.

This module centralizes:

    1. Canonical financial-metric concept terminology, shared with the
       reranker so synonym handling does not drift between components.
    2. Detection of "calculation intents" (growth rate, margin, share of
       total, ...) and the input metrics each intent requires.

Nothing here is specific to an individual question. The intent is to let
the retrieval + generation layers recognize *when* a question needs a
derived metric and *which* inputs must be present before answering.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Canonical metric concept terminology
# ---------------------------------------------------------------------------
#
# Ordering matters: longer / more specific variants are listed first so that
# exact table labels such as
#     "Net Earnings Attributable to Controlling Interests"
# are matched before the bare "net earnings" phrasing.

METRIC_CONCEPT_TERMS = {
    "net_income": [
        "net earnings attributable to controlling interests",
        "net income attributable to controlling interests",
        "net earnings attributable to controlling interest",
        "net income attributable to controlling interest",
        "net earnings attributable to",
        "net income attributable to",
        "net earnings",
        "net income",
        "net profit",
    ],
    "revenue": [
        "total revenues",
        "total revenue",
        "revenues",
        "revenue",
        "net sales",
        "net revenues",
        "net revenue",
    ],
    "operating_income": [
        "total income from operations",
        "income from operations",
        "operating income",
        "operating profit",
    ],
    "gross_profit": [
        "gross profit",
    ],
    "research_development": [
        "research and development",
        "r&d",
    ],
    "eps": [
        "earnings per share",
        "diluted earnings per share",
        "basic earnings per share",
    ],
    "cash": [
        "cash and cash equivalents",
    ],
    "long_term_debt": [
        "long-term debt",
    ],
    "short_term_debt": [
        "short-term debt",
    ],
    "operating_cash_flow": [
        "operating cash flow",
    ],
    "assets": [
        "total assets",
    ],
    "liabilities": [
        "total liabilities",
    ],
}


def normalize_text(text: str) -> str:
    """Lower-case and collapse whitespace for terminology matching."""

    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_metric_concepts(text: str) -> list[str]:
    """
    Return the normalized financial concepts mentioned in a piece of text.

    Terminology variants belonging to the same concept are treated as
    equivalent (e.g. "net sales" -> revenue).
    """

    normalized = normalize_text(text)

    concepts = []

    for concept, terms in METRIC_CONCEPT_TERMS.items():
        if any(
            re.search(
                rf"\b{re.escape(term)}\b",
                normalized,
            )
            for term in terms
        ):
            concepts.append(concept)

    return concepts


def metric_present(text: str, concept: str) -> bool:
    """Return True when a chunk/query text mentions a metric concept."""

    normalized = normalize_text(text)

    return any(
        re.search(
            rf"\b{re.escape(term)}\b",
            normalized,
        )
        for term in METRIC_CONCEPT_TERMS.get(concept, [])
    )


# Qualifiers that change *which figure* a table row reports. Rows carrying a
# noncontrolling-interest qualifier (e.g. "net earnings including noncontrolling
# interests") must not be treated as coverage for the attributable metric.
_FIGURE_DISQUALIFIERS = (
    "including noncontrolling",
    "attributable to noncontrolling",
    "losses attributable to noncontrolling",
)


# A row made up only of percentages / percentage-point deltas (for example the
# MD&A "As a percent of revenue | 4% | 3% | 1ppt" change table) is not an
# absolute figure for the underlying metric, so it must not satisfy coverage.
_PERCENTAGE_RE = re.compile(r"\d[\d,\.]*\s*%")
_PERCENTAGE_POINT_RE = re.compile(r"\d[\d,\.]*\s*ppt\b", re.IGNORECASE)
_FISCAL_YEAR_TOKEN_RE = re.compile(r"\b20\d{2}\b")
_ABSOLUTE_NUMBER_RE = re.compile(r"\d[\d,\.]*")

# Wording that betrays an MD&A prose sentence rather than a line-item label.
_PROSE_MARKERS = (
    " increased ",
    " decreased ",
    " grew ",
    " declined ",
    " driven by ",
    " due to ",
    " offset by ",
    " primarily ",
    " was ",
    " were ",
)

_MAX_LABEL_LENGTH = 70


def _looks_like_line_item(label: str) -> bool:
    """True when a table row's first cell reads like a line-item label."""

    if not label or len(label) > _MAX_LABEL_LENGTH:
        return False

    if label.endswith("."):
        return False

    padded = f" {label} "

    return not any(marker in padded for marker in _PROSE_MARKERS)


def _has_absolute_figure(line: str) -> bool:
    """
    True when a table row still carries an absolute number once percentages,
    percentage-point deltas and bare fiscal-year tokens are removed.
    """

    stripped = _PERCENTAGE_RE.sub(" ", line)
    stripped = _PERCENTAGE_POINT_RE.sub(" ", stripped)
    stripped = _FISCAL_YEAR_TOKEN_RE.sub(" ", stripped)

    return bool(_ABSOLUTE_NUMBER_RE.search(stripped))


def metric_value_present(text: str, concept: str) -> bool:
    """
    Return True when a chunk contains an actual *figure* for a metric concept.

    Unlike ``metric_present`` (which fires on any prose mention), this requires
    the metric term to appear on a numeric table row -- the shape a
    financial-statement line item takes, for example::

        | Total revenues | 81,462 |

    Rows carrying a meaning-changing qualifier (e.g. "net earnings including
    noncontrolling interests") do not count as coverage, so the attributable
    metric is not mistaken for the including-noncontrolling variant.
    """

    terms = METRIC_CONCEPT_TERMS.get(concept, [])
    if not terms:
        return False

    number_pattern = re.compile(r"\d")

    for raw_line in text.splitlines():
        line = normalize_text(raw_line)

        if "|" not in line or not number_pattern.search(line):
            continue

        if any(qualifier in line for qualifier in _FIGURE_DISQUALIFIERS):
            continue

        # The metric must be the row *label* -- the first non-empty cell of a
        # markdown table row. This rejects numeric rows that merely happen to
        # end with the metric word (e.g. "... | Net earnings |") and prose
        # bullets like "| | - | Revenue increased 32% ... |".
        label = next(
            (cell.strip() for cell in line.split("|") if cell.strip()),
            "",
        )

        if not _looks_like_line_item(label):
            continue

        if not _has_absolute_figure(line):
            continue

        if any(
            re.search(
                rf"\b{re.escape(term)}\b",
                label,
            )
            for term in terms
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Calculation intent detection
# ---------------------------------------------------------------------------


GROWTH_TRIGGERS = [
    r"\bgrowth\b",
    r"\bgrew\b",
    r"\bgrow\b",
    r"\bgrown\b",
    r"\bgrowing\b",
    r"\byoy\b",
    r"\byear[- ]over[- ]year\b",
    r"\bpercentage change\b",
    r"\bpercent change\b",
]

MARGIN_TRIGGERS = [
    r"\bmargin\b",
]

SHARE_TRIGGERS = [
    r"\bshare of\b",
    r"\bshare\b",
    r"\bpercentage of total\b",
    r"\bpercent of total\b",
    r"\bproportion of\b",
]

# Preferred numerator/denominator metrics per intent.
_MARGIN_NUMERATORS = ("operating_income", "gross_profit", "net_income")
_SHARE_DENOMINATOR = "revenue"
_DEFAULT_GROWTH_METRIC = "revenue"


@dataclass
class CalculationSpec:
    """
    Describes a derived-metric request.

    kind:             "growth" | "margin" | "share" | None
    required_metrics: metric concepts that must be present in the context
    needs_prior_year: for growth, the prior fiscal year is also required
    ratio_terms:      (numerator concept, denominator concept)
    description:      human/LLM-readable formula guidance
    """

    kind: str | None = None
    required_metrics: list[str] = field(default_factory=list)
    needs_prior_year: bool = False
    ratio_terms: tuple[str | None, str | None] = (None, None)
    description: str = ""


def _primary_metric(concepts: list[str]) -> str | None:
    """Pick the base metric a growth question most likely refers to."""

    for preferred in (
        "revenue",
        "operating_income",
        "net_income",
        "gross_profit",
    ):
        if preferred in concepts:
            return preferred

    return concepts[0] if concepts else None


def primary_metric(concepts: list[str]) -> str | None:
    """Public helper: the headline metric a question most likely asks about."""

    return _primary_metric(concepts)


def detect_calculation(query: str) -> CalculationSpec:
    """
    Detect whether a question asks for a derived metric and, if so,
    which input metrics the derivation requires.
    """

    normalized = normalize_text(query)
    concepts = detect_metric_concepts(query)

    is_margin = any(
        re.search(pattern, normalized) for pattern in MARGIN_TRIGGERS
    )
    is_share = any(
        re.search(pattern, normalized) for pattern in SHARE_TRIGGERS
    )
    is_growth = any(
        re.search(pattern, normalized) for pattern in GROWTH_TRIGGERS
    )

    if is_margin:
        numerator = next(
            (concept for concept in _MARGIN_NUMERATORS if concept in concepts),
            "operating_income",
        )
        denominator = "revenue"

        required = [numerator]
        if denominator not in required:
            required.append(denominator)

        return CalculationSpec(
            kind="margin",
            required_metrics=required,
            needs_prior_year=False,
            ratio_terms=(numerator, denominator),
            description=(
                f"margin (%) = {numerator} / {denominator} * 100. "
                "The numerator and denominator must come from the same "
                "fiscal year."
            ),
        )

    if is_share:
        numerator = next(
            (
                concept
                for concept in concepts
                if concept != _SHARE_DENOMINATOR
            ),
            None,
        )

        required = [_SHARE_DENOMINATOR]
        if numerator and numerator not in required:
            required.append(numerator)

        return CalculationSpec(
            kind="share",
            required_metrics=required,
            needs_prior_year=False,
            ratio_terms=(numerator, _SHARE_DENOMINATOR),
            description=(
                f"share (%) = {numerator or 'the requested line item'} / "
                f"{_SHARE_DENOMINATOR} * 100."
            ),
        )

    if is_growth:
        base_metric = _primary_metric(concepts) or _DEFAULT_GROWTH_METRIC

        return CalculationSpec(
            kind="growth",
            required_metrics=[base_metric],
            needs_prior_year=True,
            ratio_terms=(base_metric, base_metric),
            description=(
                f"growth rate of {base_metric} for a fiscal year = "
                "(current fiscal year value - prior fiscal year value) / "
                "prior fiscal year value * 100. Use consecutive "
                "fiscal-year figures; do not reuse one year's value for "
                "two years."
            ),
        )

    return CalculationSpec()
