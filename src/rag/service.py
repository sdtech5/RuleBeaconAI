import re

from qdrant_client import models

from src.rag.retriever import (
    hybrid_retrieve_documents,
    create_source_reference,
)
from src.rag.reranker import DocumentReranker
from src.rag.generator import generate_answer
from src.rag.calculations import (
    METRIC_CONCEPT_TERMS,
    detect_calculation,
    detect_metric_concepts,
    metric_present,
    metric_value_present,
    primary_metric,
)


SUPPORTED_COMPANIES = {
    "AAPL",
    "ADM",
    "AMZN",
    "GOOGL",
    "JPM",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "TSLA",
    "WMT",
}


SUPPORTED_YEARS = {
    2020,
    2021,
    2022,
    2023,
    2024,
}


COMPANY_ALIASES = {
    "APPLE": "AAPL",
    "ADM": "ADM",
    "ARCHER DANIELS MIDLAND": "ADM",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "JPMORGAN": "JPM",
    "JPMORGAN CHASE": "JPM",
    "META": "META",
    "FACEBOOK": "META",
    "MICROSOFT": "MSFT",
    "NETFLIX": "NFLX",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "WALMART": "WMT",
}


def detect_company(query: str) -> str | None:
    normalized_query = query.upper()

    aliases = sorted(
        COMPANY_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, ticker in aliases:
        pattern = rf"\b{re.escape(alias)}\b"

        if re.search(pattern, normalized_query):
            return ticker

    return None


# Connectors that express an inclusive fiscal-year span, e.g.
# "FY2021 to FY2024", "2021-2024", "2021 through 2024".
_YEAR_RANGE_PATTERN = re.compile(
    r"(?:FY\s*)?(20\d{2})\s*"
    r"(?:-|–|—|to|through|thru|until)\s*"
    r"(?:FY\s*)?(20\d{2})",
    re.IGNORECASE,
)


def expand_year_ranges(query: str) -> list[int]:
    """
    Expand explicit fiscal-year *ranges* into every year they cover.

    "FY2021 to FY2024" -> [2021, 2022, 2023, 2024]

    Only span-like connectors are treated as ranges, so "2021 and 2023"
    is not expanded. A sanity bound guards against accidental matches.
    """

    expanded = []

    for match in _YEAR_RANGE_PATTERN.finditer(query):
        start = int(match.group(1))
        end = int(match.group(2))

        if start > end:
            start, end = end, start

        if 1 <= end - start <= 10:
            expanded.extend(range(start, end + 1))

    return expanded


def detect_requested_years(query: str) -> list[int]:
    matches = re.findall(
        r"\b(?:FY\s*)?(20\d{2})\b",
        query.upper(),
    )

    seen = set()
    years = []

    for match in matches:
        year_val = int(match)
        if year_val not in seen:
            seen.add(year_val)
            years.append(year_val)

    # Add any interior years implied by an explicit range (e.g. the 2022 and
    # 2023 filings for "FY2021 to FY2024").
    for year_val in expand_year_ranges(query):
        if year_val not in seen:
            seen.add(year_val)
            years.append(year_val)

    return years


def detect_requested_year(query: str) -> int | None:
    years = detect_requested_years(query)
    return years[0] if years else None


def detect_year(query: str) -> int | None:
    requested_year = detect_requested_year(query)

    if requested_year in SUPPORTED_YEARS:
        return requested_year

    return None


def build_company_filter(
    ticker: str | None,
    year: int | list[int] | None = None,
):
    if ticker is None:
        return None

    if year is not None:
        if isinstance(year, int):
            years_to_use = [year]
        else:
            years_to_use = list(year)

        sources = [
            f"data\\raw\\{ticker}\\10-K_{filing_year}.md"
            for filing_year in sorted(set(years_to_use))
        ]
    else:
        sources = [
            f"data\\raw\\{ticker}\\10-K_{filing_year}.md"
            for filing_year in sorted(SUPPORTED_YEARS)
        ]

    return models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchAny(any=sources),
            )
        ]
    )


_RERANKER_CAP = 12

# How many metric-terminology variants a focused refill query may try before
# giving up (most specific phrasing first).
_FOCUSED_QUERY_TERM_LIMIT = 4

# How many figure-bearing chunks a single missing (metric, year) input may
# contribute. One is enough because the candidates are ranked with the
# audited statement preferred (see ``_is_financial_statement``); adding more
# merely duplicates the same table and inflates the prompt, which must stay
# under the provider's 8k tokens-per-minute request ceiling.
_REFILL_CANDIDATES_PER_INPUT = 1

_FS_KEYWORDS = {"financial statements", "supplementary data", "item 8"}


def _is_financial_statement(result) -> bool:
    """True when a chunk belongs to the financial statements / Item 8 section."""

    section = str(result.document.metadata.get("section", "")).lower()

    return any(keyword in section for keyword in _FS_KEYWORDS)


def chunk_fiscal_year(metadata: dict) -> int | None:
    """Derive the filing fiscal year from chunk metadata."""

    for key in ("source", "file_name"):
        value = str(metadata.get(key, ""))
        match = re.search(r"10-K_(\d{4})", value)
        if match:
            return int(match.group(1))

    return None


def chunk_matches_metric(result, concepts) -> bool:
    """Return True when a chunk mentions any of the given metric concepts."""

    if not concepts:
        return False

    text = result.document.page_content

    return any(metric_present(text, concept) for concept in concepts)


def chunk_has_metric_value(result, concepts) -> bool:
    """Return True when a chunk contains an actual figure for any concept."""

    if not concepts:
        return False

    text = result.document.page_content

    return any(
        metric_value_present(text, concept) for concept in concepts
    )


def _focused_terms(concept: str) -> list[str]:
    """
    Metric terminology for a focused refill query, most specific first.

    A filing's statement label is not always the most specific phrasing (for
    example a filing that labels its top line "Revenue" rather than "Total
    revenues"), so the variants are tried in turn until one surfaces a figure.
    """

    terms = METRIC_CONCEPT_TERMS.get(concept, [concept])

    ordered = [term for term in dict.fromkeys(terms) if term]

    return (ordered or [concept])[:_FOCUSED_QUERY_TERM_LIMIT]


def select_reranker_input(results, years, cap=_RERANKER_CAP):
    """
    Choose the candidate pool handed to the cross-encoder.

    For multi-year questions a flat ``results[:cap]`` slice can omit an
    entire fiscal year (the hybrid pool is ordered globally). Instead we
    reserve a per-year quota so every requested year is represented before
    filling the remaining budget by global score.
    """

    if len(years) <= 1:
        return results[:cap]

    per_year = max(2, cap // len(years))

    selected = []
    selected_ids = set()

    for year in years:
        count = 0
        for result in results:
            if count >= per_year:
                break

            chunk_id = result.document.metadata.get("chunk_id")
            if chunk_id in selected_ids:
                continue

            if chunk_fiscal_year(result.document.metadata) == year:
                selected.append(result)
                selected_ids.add(chunk_id)
                count += 1

    for result in results:
        if len(selected) >= cap:
            break

        chunk_id = result.document.metadata.get("chunk_id")
        if chunk_id not in selected_ids:
            selected.append(result)
            selected_ids.add(chunk_id)

    return selected[:cap]


def select_with_year_coverage(
    reranked_results,
    years,
    target_final,
    required_metrics=None,
):
    """
    Select the final context, guaranteeing representation for every year.

    Pass 1 reserves one chunk per requested fiscal year (preferring
    financial-statement / metric-matching chunks). Pass 2 spends any
    remaining budget by score. The total never exceeds ``target_final`` and
    no requested year is dropped.
    """

    required_metrics = required_metrics or []

    selected = []
    selected_chunk_ids = set()

    # --- Pass 1: one representative chunk per year ---
    for year in years:
        best_result = None
        best_key = None

        for result in reranked_results:
            if chunk_fiscal_year(result.document.metadata) != year:
                continue

            chunk_id = result.document.metadata.get("chunk_id")
            if chunk_id in selected_chunk_ids:
                continue

            section = str(
                result.document.metadata.get("section", "")
            ).lower()
            is_fs = any(keyword in section for keyword in _FS_KEYWORDS)
            matches_metric = chunk_matches_metric(result, required_metrics)
            has_value = chunk_has_metric_value(result, required_metrics)

            if is_fs and has_value:
                priority = 3
            elif has_value:
                priority = 2
            elif is_fs or matches_metric:
                priority = 1
            else:
                priority = 0

            key = (priority, result.score)

            if best_result is None or key > best_key:
                best_result = result
                best_key = key

        if best_result is not None:
            selected.append(best_result)
            selected_chunk_ids.add(
                best_result.document.metadata.get("chunk_id")
            )

    # --- Pass 2: fill remaining budget by score ---
    for result in reranked_results:
        if len(selected) >= target_final:
            break

        chunk_id = result.document.metadata.get("chunk_id")
        if chunk_id not in selected_chunk_ids:
            selected.append(result)
            selected_chunk_ids.add(chunk_id)

    selected.sort(key=lambda result: result.score, reverse=True)

    for index, result in enumerate(selected, start=1):
        result.rank = index

    return selected


class RAGService:

    def __init__(
        self,
        vector_store,
        llm,
        reranker,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

    def _refill_missing_inputs(
        self,
        question,
        company,
        years,
        required_metrics,
        results,
    ):
        """
        Ensure every required (metric concept, fiscal year) input is present.

        Missing inputs are fetched with a focused supplementary retrieval
        (metric terminology + fiscal year + company) and appended to the
        context. This keeps derived-metric answers grounded instead of
        computed from partial data.
        """

        if not required_metrics or not years:
            return results

        # One refill is allowed per missing (metric, year) input; the cap
        # tracks the number of requested years so a five-year question (for
        # example FY2021-FY2024 plus the growth prior year) cannot silently
        # drop an input.
        max_refills = min(6, max(4, len(years)))

        covered = set()
        selected_chunk_ids = set()

        for result in results:
            selected_chunk_ids.add(
                result.document.metadata.get("chunk_id")
            )
            year = chunk_fiscal_year(result.document.metadata)
            if year is None:
                continue
            for concept in required_metrics:
                if metric_value_present(result.document.page_content, concept):
                    covered.add((concept, year))

        missing = [
            (concept, year)
            for concept in required_metrics
            for year in years
            if (concept, year) not in covered
        ]

        if not missing:
            return results

        attempted = 0

        for concept, year in missing:
            if attempted >= max_refills:
                break

            attempted += 1

            candidate_filter = build_company_filter(
                ticker=company,
                year=year,
            )

            # The fiscal year is already enforced by ``candidate_filter``, so
            # the query text must NOT repeat "fiscal year YYYY": that phrase
            # matches MD&A prose bullets and drowns out the statement rows.
            # Try the metric terminology from most specific to least.
            #
            # Candidates stay in hybrid-retrieval order: the cross-encoder is
            # by far the most expensive step in the pipeline (~16s per call on
            # CPU), which would otherwise push multi-year questions past the
            # API request timeout. The refill only needs a chunk that really
            # carries the figure, and the ``metric_value_present`` gate below
            # already guarantees that.
            figure_matches = []
            fallback = None

            for term in _focused_terms(concept):
                focused_query = f"{company or ''} {term}".strip()

                candidates = hybrid_retrieve_documents(
                    vectorstore=self.vector_store,
                    query=focused_query,
                    k=12,
                    query_filter=candidate_filter,
                )

                if not candidates:
                    continue

                for candidate in candidates:
                    chunk_id = candidate.document.metadata.get("chunk_id")
                    if chunk_id in selected_chunk_ids:
                        continue

                    content = candidate.document.page_content

                    if metric_value_present(content, concept):
                        figure_matches.append(candidate)
                    elif fallback is None and metric_present(content, concept):
                        fallback = candidate

                if len(figure_matches) >= _REFILL_CANDIDATES_PER_INPUT:
                    break

            # Audited-statement rows are the better source than a segment or
            # subtotal table carrying the same metric (stable sort keeps the
            # hybrid-retrieval order within each group).
            figure_matches.sort(key=_is_financial_statement, reverse=True)

            chosen_list = figure_matches[:_REFILL_CANDIDATES_PER_INPUT]

            if not chosen_list and fallback is not None:
                chosen_list = [fallback]

            for chosen in chosen_list:
                results.append(chosen)
                selected_chunk_ids.add(
                    chosen.document.metadata.get("chunk_id")
                )
                if metric_value_present(chosen.document.page_content, concept):
                    covered.add((concept, year))

        results.sort(key=lambda result: result.score, reverse=True)
        for index, result in enumerate(results, start=1):
            result.rank = index

        return results

    def ask(
        self,
        question,
        k=5,
        retrieval_k=15,
    ):
        company = detect_company(question)
        requested_years = detect_requested_years(question)

        # Check for unsupported explicit years
        for yr in requested_years:
            if yr not in SUPPORTED_YEARS:
                return {
                    "question": question,
                    "answer": (
                        f"No filing for fiscal year "
                        f"{yr} is available "
                        f"in the current document corpus."
                    ),
                    "sources": [],
                }

        valid_years = [y for y in requested_years if y in SUPPORTED_YEARS]

        calculation = detect_calculation(question)

        # Report-style metric questions (no derived metric) still need the
        # actual figure present. Track the headline metric so the figure-aware
        # coverage/refill path guards them too.
        required_metrics = list(calculation.required_metrics)
        if not required_metrics:
            headline = primary_metric(detect_metric_concepts(question))
            if headline:
                required_metrics = [headline]

        # Derived metrics require every input metric for every year they
        # cover. Growth additionally needs the prior year as the denominator.
        retrieval_years = list(valid_years)
        if calculation.needs_prior_year and valid_years:
            prior_year = min(valid_years) - 1
            if (
                prior_year in SUPPORTED_YEARS
                and prior_year not in retrieval_years
            ):
                retrieval_years.append(prior_year)

        if len(retrieval_years) > 1:
            filter_year = sorted(retrieval_years)
            # Retrieve 8 candidates per year (was 15×N — too slow for
            # cross-encoder on CPU).
            effective_retrieval_k = max(
                retrieval_k,
                8 * len(retrieval_years),
            )
        elif len(retrieval_years) == 1:
            filter_year = retrieval_years[0]
            effective_retrieval_k = retrieval_k
        else:
            filter_year = None
            effective_retrieval_k = retrieval_k

        query_filter = build_company_filter(
            ticker=company,
            year=filter_year,
        )

        retrieval_results = hybrid_retrieve_documents(
            vectorstore=self.vector_store,
            query=question,
            k=effective_retrieval_k,
            query_filter=query_filter,
        )

        # Cap reranker input: the cross-encoder runs on CPU and is the main
        # latency bottleneck. For multi-year questions the cap reserves a
        # per-year quota so no requested year is dropped before reranking.
        reranker_input = select_reranker_input(
            retrieval_results,
            retrieval_years,
        )

        reranked_results = self.reranker.rerank(
            query=question,
            results=reranker_input,
            top_k=len(reranker_input),
        )

        # Ensure every requested (and prior) fiscal year is represented, and
        # allow a slightly larger context when a derived metric needs several
        # input figures.
        target_final = k
        if len(retrieval_years) > 1:
            target_final = min(8, max(k, len(retrieval_years) + 1))

        final_results = select_with_year_coverage(
            reranked_results,
            retrieval_years,
            target_final,
            required_metrics=required_metrics,
        )

        # If any required (metric concept, fiscal year) input is still
        # missing, fetch it with a focused supplementary query rather than
        # answering from partial data.
        final_results = self._refill_missing_inputs(
            question=question,
            company=company,
            years=retrieval_years,
            required_metrics=required_metrics,
            results=final_results,
        )

        documents_for_generation = [
            result.document
            for result in final_results
        ]

        answer = generate_answer(
            self.llm,
            question,
            documents_for_generation,
            calculation_hint=calculation.description or None,
        )

        sources = []

        for result in final_results:
            source = create_source_reference(
                result.document,
                result.rank,
            )

            sources.append(
                {
                    "rank": source.rank,
                    "score": result.score,
                    "document_id": source.document_id,
                    "chunk_id": source.chunk_id,
                    "file_name": source.file_name,
                    "source": source.source,
                    "section": source.section,
                    "page": source.page,
                }
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }