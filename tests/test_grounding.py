# tests/test_grounding.py

from src.rag.embeddings import create_embeddings
from src.rag.vectorstore import load_qdrant_store
from src.rag.generator import create_llm
from src.rag.service import RAGService
from src.rag.reranker import DocumentReranker


TEST_CASES = [
    {
        "company": "AAPL",
        "year": 2023,
        "question": "What were Apple's net sales in fiscal year 2023?",
    },
    {
        "company": "ADM",
        "year": 2024,
        "question": "What was ADM's net earnings in fiscal year 2024?",
    },
    {
        "company": "NVDA",
        "year": 2024,
        "question": "What was NVIDIA's revenue in fiscal year 2024?",
    },
    {
        "company": "TSLA",
        "year": 2022,
        "question": (
            "What was Tesla's operating income in FY2022, "
            "and what was its operating margin?"
        ),
    },
    {
        "company": "NVDA",
        "years": [2021, 2022, 2023, 2024],
        "question": (
            "Compare NVIDIA's revenue growth rate in "
            "FY2022 vs FY2023 vs FY2024."
        ),
    },
]


def build_service():

    print("Loading embedding model...")
    embeddings = create_embeddings()

    print("Connecting to Qdrant...")
    vector_store = load_qdrant_store(embeddings)

    print("Loading LLM...")
    llm = create_llm()

    print("Loading reranker...")
    reranker = DocumentReranker()

    return RAGService(
        vector_store=vector_store,
        llm=llm,
        reranker=reranker,
    )


def source_matches(source, company, years):
    """
    A source is grounded when it belongs to the expected company and to
    one of the requested fiscal years. Multi-year questions may legally
    cite more than one filing year.
    """

    source_path = source["source"].replace("/", "\\").upper()

    expected_company = f"\\{company}\\"

    if expected_company not in source_path:
        return False

    return any(
        f"10-K_{year}.MD" in source_path
        for year in years
    )


def main():

    print("=" * 80)
    print("RuleBeaconAI GROUNDING VALIDATION")
    print("=" * 80)

    service = build_service()

    passed = 0

    for index, test in enumerate(TEST_CASES, start=1):

        print("-" * 80)
        print(
            f"[{index}/{len(TEST_CASES)}] "
            f"{test['question']}"
        )

        result = service.ask(test["question"])

        answer = result["answer"]
        sources = result["sources"]

        print()
        print("ANSWER:")
        print(answer)

        print()
        print("SOURCES:")

        for source in sources:
            print(
                f"  #{source['rank']} "
                f"{source['source']} | "
                f"{source['score']:.4f}"
            )

        expected_years = test.get("years") or [test["year"]]

        answer_pass = bool(answer.strip())

        source_pass = (
            len(sources) > 0
            and all(
                source_matches(
                    source,
                    test["company"],
                    expected_years,
                )
                for source in sources
            )
        )

        print()

        if answer_pass:
            print("Answer present: PASS")
        else:
            print("Answer present: FAIL")

        if source_pass:
            print("Source grounding: PASS")
        else:
            print("Source grounding: FAIL")

        if answer_pass and source_pass:
            print("Result: PASS")
            passed += 1
        else:
            print("Result: FAIL")

    print()
    print("=" * 80)
    print("GROUNDING SUMMARY")
    print("=" * 80)

    print(
        f"Overall: {passed}/{len(TEST_CASES)} "
        f"({passed / len(TEST_CASES) * 100:.1f}%)"
    )

    print("=" * 80)

    if passed == len(TEST_CASES):
        print("✓ ALL GROUNDING TESTS PASSED")
    else:
        print("✗ GROUNDING VALIDATION FAILED")


if __name__ == "__main__":
    main()