import re

from qdrant_client import models

from src.rag.retriever import (
    hybrid_retrieve_documents,
    create_source_reference,
)
from src.rag.reranker import DocumentReranker
from src.rag.generator import generate_answer


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

        if len(valid_years) > 1:
            filter_year = valid_years
            # Retrieve 8 candidates per year (was 15×N — too slow for cross-encoder on CPU)
            effective_retrieval_k = max(retrieval_k, 8 * len(valid_years))
        elif len(valid_years) == 1:
            filter_year = valid_years[0]
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
        # latency bottleneck. 12 candidates give excellent quality while
        # keeping scoring time well under 10s.
        RERANKER_CAP = 12
        reranker_input = retrieval_results[:RERANKER_CAP]

        reranked_results = self.reranker.rerank(
            query=question,
            results=reranker_input,
            top_k=RERANKER_CAP,
        )

        # For multi-year queries, ensure each requested year is well-represented.
        # Priority: financial-statement / table chunks first (2 slots per year),
        # then any remaining high-scoring chunks up to k total.
        SLOTS_PER_YEAR = 2
        FS_KEYWORDS = {"financial statements", "supplementary data", "item 8"}
        selected_results = []
        selected_chunk_ids = set()

        # --- Pass 1: financial-statement / table chunks per year ---
        for yr in valid_years:
            target_filename = f"10-K_{yr}.md"
            slots = 0
            for r in reranked_results:
                if slots >= SLOTS_PER_YEAR:
                    break
                chunk_source = str(r.document.metadata.get("source", ""))
                chunk_fn = str(r.document.metadata.get("file_name", ""))
                cid = r.document.metadata.get("chunk_id")
                section = str(r.document.metadata.get("section", "")).lower()
                is_table = bool(r.document.metadata.get("is_table", False))
                from_right_year = (
                    target_filename in chunk_source or target_filename in chunk_fn
                )
                is_fs = is_table or any(kw in section for kw in FS_KEYWORDS)
                if from_right_year and is_fs and cid not in selected_chunk_ids:
                    selected_results.append(r)
                    selected_chunk_ids.add(cid)
                    slots += 1

        # --- Pass 2: any remaining chunks per year up to SLOTS_PER_YEAR ---
        for yr in valid_years:
            target_filename = f"10-K_{yr}.md"
            slots = sum(
                1 for r in selected_results
                if target_filename in str(r.document.metadata.get("source", ""))
                or target_filename in str(r.document.metadata.get("file_name", ""))
            )
            for r in reranked_results:
                if slots >= SLOTS_PER_YEAR:
                    break
                chunk_source = str(r.document.metadata.get("source", ""))
                chunk_fn = str(r.document.metadata.get("file_name", ""))
                cid = r.document.metadata.get("chunk_id")
                from_right_year = (
                    target_filename in chunk_source or target_filename in chunk_fn
                )
                if from_right_year and cid not in selected_chunk_ids:
                    selected_results.append(r)
                    selected_chunk_ids.add(cid)
                    slots += 1

        # --- Pass 3: fill remaining k slots from the full reranked pool ---
        for r in reranked_results:
            cid = r.document.metadata.get("chunk_id")
            if cid not in selected_chunk_ids:
                selected_results.append(r)
                selected_chunk_ids.add(cid)
                if len(selected_results) >= k:
                    break

        selected_results.sort(key=lambda r: r.score, reverse=True)
        for idx, r in enumerate(selected_results, 1):
            r.rank = idx
        final_results = selected_results[:k]

        documents_for_generation = [
            result.document
            for result in final_results
        ]

        answer = generate_answer(
            self.llm,
            question,
            documents_for_generation,
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