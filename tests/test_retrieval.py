from src.rag.embeddings import create_embeddings
from src.rag.vectorstore import load_qdrant_store
from src.rag.reranker import DocumentReranker
from src.rag.retriever import hybrid_retrieve_documents
from src.rag.service import (
    detect_company,
    detect_year,
    build_company_filter,
)

from tests.retrieval_questions import RETRIEVAL_QUESTIONS


RETRIEVAL_K = 30
TOP_K = 5


def evaluate_question(
    question_case,
    vector_store,
    reranker,
):
    question = question_case["question"]
    expected_company = question_case["company"]
    expected_year = question_case["year"]
    expected_terms = question_case["expected_terms"]

    detected_company = detect_company(question)
    detected_year = detect_year(question)

    company_ok = detected_company == expected_company
    year_ok = detected_year == expected_year

    query_filter = build_company_filter(
        ticker=detected_company,
        year=detected_year,
    )

    retrieval_results = hybrid_retrieve_documents(
        vectorstore=vector_store,
        query=question,
        k=RETRIEVAL_K,
        query_filter=query_filter,
    )

    reranked_results = reranker.rerank(
        query=question,
        results=retrieval_results,
        top_k=TOP_K,
    )

    correct_filing_results = []

    for result in reranked_results:
        source = result.document.metadata.get(
            "source",
            "",
        )

        expected_source = (
            f"data\\raw\\{expected_company}"
            f"\\10-K_{expected_year}.md"
        )

        if source == expected_source:
            correct_filing_results.append(result)

    filing_hit = len(correct_filing_results) > 0

    metric_hit = False

    for result in correct_filing_results:
        content = result.document.page_content.lower()

        if any(
            term.lower() in content
            for term in expected_terms
        ):
            metric_hit = True
            break

    success = (
        company_ok
        and year_ok
        and filing_hit
        and metric_hit
    )

    return {
        "question": question,
        "company": expected_company,
        "year": expected_year,
        "company_detection": company_ok,
        "year_detection": year_ok,
        "filing_hit_at_5": filing_hit,
        "metric_hit_at_5": metric_hit,
        "success": success,
        "results": reranked_results,
    }


def main():
    print("=" * 80)
    print("RuleBeaconAI RETRIEVAL REGRESSION")
    print("=" * 80)

    print(f"\nQuestions: {len(RETRIEVAL_QUESTIONS)}")
    print(f"Hybrid candidates: {RETRIEVAL_K}")
    print(f"Final results: {TOP_K}")

    print("\nLoading embedding model...")
    embeddings = create_embeddings()

    print("Connecting to Qdrant...")
    vector_store = load_qdrant_store(embeddings)

    print("Loading reranker...")
    reranker = DocumentReranker()

    results = []

    try:
        for index, question_case in enumerate(
            RETRIEVAL_QUESTIONS,
            start=1,
        ):
            print("\n" + "-" * 80)
            print(
                f"[{index}/{len(RETRIEVAL_QUESTIONS)}] "
                f"{question_case['question']}"
            )

            result = evaluate_question(
                question_case,
                vector_store,
                reranker,
            )

            results.append(result)

            status = "PASS" if result["success"] else "FAIL"

            print(f"Result: {status}")
            print(
                f"Company detection: "
                f"{'PASS' if result['company_detection'] else 'FAIL'}"
            )
            print(
                f"Year detection: "
                f"{'PASS' if result['year_detection'] else 'FAIL'}"
            )
            print(
                f"Filing Hit@5: "
                f"{'PASS' if result['filing_hit_at_5'] else 'FAIL'}"
            )
            print(
                f"Metric Hit@5: "
                f"{'PASS' if result['metric_hit_at_5'] else 'FAIL'}"
            )

            print("\nTop sources:")

            for retrieved in result["results"]:
                metadata = retrieved.document.metadata

                print(
                    f"  #{retrieved.rank} "
                    f"{metadata.get('file_name')} "
                    f"| {retrieved.score:.4f}"
                )

    finally:
        vector_store.close()

    total = len(results)
    passed = sum(
        1
        for result in results
        if result["success"]
    )

    filing_hits = sum(
        1
        for result in results
        if result["filing_hit_at_5"]
    )

    metric_hits = sum(
        1
        for result in results
        if result["metric_hit_at_5"]
    )

    print("\n" + "=" * 80)
    print("REGRESSION SUMMARY")
    print("=" * 80)

    print(
        f"\nOverall: {passed}/{total} "
        f"({passed / total * 100:.1f}%)"
    )

    print(
        f"Filing Hit@5: {filing_hits}/{total} "
        f"({filing_hits / total * 100:.1f}%)"
    )

    print(
        f"Metric Hit@5: {metric_hits}/{total} "
        f"({metric_hits / total * 100:.1f}%)"
    )

    print("\nBy company:")

    for result in results:
        status = "PASS" if result["success"] else "FAIL"

        print(
            f"  {result['company']:5} "
            f"{result['year']} "
            f"{status}"
        )

    print("\n" + "=" * 80)

    if passed == total:
        print("✓ ALL RETRIEVAL TESTS PASSED")
    else:
        print(
            f"⚠ {total - passed} "
            f"retrieval test(s) failed"
        )


if __name__ == "__main__":
    main()