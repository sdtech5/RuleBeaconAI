# tests/test_generation.py

import re

from src.rag.embeddings import create_embeddings
from src.rag.vectorstore import load_qdrant_store
from src.rag.generator import create_llm
from src.rag.service import RAGService
from src.rag.reranker import DocumentReranker


QUESTIONS = [
    {
        "company": "AAPL",
        "year": 2023,
        "question": "What were Apple's net sales in fiscal year 2023?",
        "expected_values": [383.3, 383285],
    },
    {
        "company": "ADM",
        "year": 2024,
        "question": "What was ADM's net earnings in fiscal year 2024?",
        "expected_values": [2036, 2.036],
    },
    {
        "company": "NVDA",
        "year": 2024,
        "question": "What was NVIDIA's revenue in fiscal year 2024?",
        "expected_values": [60.9, 60922, 60.922],
    },
    {
        "company": "TSLA",
        "year": 2022,
        "question": (
            "What was Tesla's operating income in FY2022, "
            "and what was its operating margin?"
        ),
        "expected_values": [
            13656,
            13.656,
            13.7,
            81462,
            81.462,
            81.46,
            81.5,
            16.8,
            16.76,
            16.7,
        ],
    },
    {
        "company": "NVDA",
        "year": 2024,
        "question": (
            "Compare NVIDIA's revenue growth rate in "
            "FY2022 vs FY2023 vs FY2024."
        ),
        "expected_values": [
            26974,
            26.974,
            26.97,
            60922,
            60.922,
            60.9,
            125.8,
            125.9,
            126,
            61.4,
            61,
        ],
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


def normalize_text(text):
    return re.sub(r"[^a-z0-9.]", "", text.lower())


def answer_contains_expected_value(answer, expected_values):

    normalized = normalize_text(answer)

    for value in expected_values:

        value_str = str(value)

        # Direct textual match
        if normalize_text(value_str) in normalized:
            return True

        # Handle billion/million formatting
        if value >= 1000:
            billion_value = value / 1000

            if f"{billion_value:.1f}" in normalized:
                return True

    return False


def source_matches(source, company, year):

    source_path = source["source"].replace("/", "\\").upper()

    expected_company = f"\\{company}\\"
    expected_file = f"10-K_{year}.MD"

    return (
        expected_company in source_path
        and expected_file in source_path
    )


def main():

    print("=" * 80)
    print("RuleBeaconAI GENERATION VALIDATION")
    print("=" * 80)

    service = build_service()

    passed = 0

    for index, test in enumerate(QUESTIONS, start=1):

        print("-" * 80)
        print(
            f"[{index}/{len(QUESTIONS)}] "
            f"{test['question']}"
        )

        result = service.ask(test["question"])

        answer = result["answer"]

        print()
        print("ANSWER:")
        print(answer)

        print()
        print("SOURCES:")

        for source in result["sources"]:
            print(
                f"  #{source['rank']} "
                f"{source['source']} | "
                f"{source['score']:.4f}"
            )

        answer_pass = answer_contains_expected_value(
            answer,
            test["expected_values"],
        )

        source_pass = any(
            source_matches(
                source,
                test["company"],
                test["year"],
            )
            for source in result["sources"]
        )

        print()

        if answer_pass:
            print("Answer validation: PASS")
        else:
            print("Answer validation: FAIL")

        if source_pass:
            print("Source validation: PASS")
        else:
            print("Source validation: FAIL")

        if answer_pass and source_pass:
            print("Result: PASS")
            passed += 1
        else:
            print("Result: FAIL")

    print()
    print("=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)

    print(
        f"Overall: {passed}/{len(QUESTIONS)} "
        f"({passed / len(QUESTIONS) * 100:.1f}%)"
    )

    print("=" * 80)

    if passed == len(QUESTIONS):
        print("✓ ALL GENERATION TESTS PASSED")
    else:
        print("✗ GENERATION VALIDATION FAILED")


if __name__ == "__main__":
    main()