# tests/test_year_guard.py

from src.rag.service import RAGService


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


class FailingReranker:

    def rerank(self, *args, **kwargs):
        raise AssertionError(
            "Reranker should not be called "
            "for an unsupported fiscal year."
        )


QUESTIONS = [
    (
        "What was Apple's total revenue in fiscal year 2019 "
        "according to the provided documents?"
    ),
    (
        "What was Microsoft's revenue in fiscal year 2025 "
        "according to the provided documents?"
    ),
    (
        "What was NVIDIA's revenue in FY2026 "
        "according to the provided documents?"
    ),
]


def main():

    print("=" * 80)
    print("RuleBeaconAI OUT-OF-CORPUS YEAR GUARD")
    print("=" * 80)

    service = RAGService(
        vector_store=FailingVectorStore(),
        llm=FailingLLM(),
        reranker=FailingReranker(),
    )

    passed = 0

    for index, question in enumerate(QUESTIONS, start=1):

        print("-" * 80)
        print(f"[{index}/{len(QUESTIONS)}] {question}")

        result = service.ask(question)

        answer = result["answer"]
        sources = result["sources"]

        print()
        print("ANSWER:")
        print(answer)

        print()
        print(f"SOURCES: {len(sources)}")

        answer_pass = (
            "No filing for fiscal year" in answer
            and "available" in answer
            and "current document corpus" in answer
        )

        sources_pass = len(sources) == 0

        if answer_pass:
            print("Deterministic response: PASS")
        else:
            print("Deterministic response: FAIL")

        if sources_pass:
            print("No retrieval: PASS")
        else:
            print("No retrieval: FAIL")

        if answer_pass and sources_pass:
            print("Result: PASS")
            passed += 1
        else:
            print("Result: FAIL")

    print()
    print("=" * 80)
    print("YEAR GUARD SUMMARY")
    print("=" * 80)

    print(
        f"Overall: {passed}/{len(QUESTIONS)} "
        f"({passed / len(QUESTIONS) * 100:.1f}%)"
    )

    print("=" * 80)

    if passed == len(QUESTIONS):
        print("✓ ALL OUT-OF-CORPUS YEAR GUARD TESTS PASSED")
    else:
        print("✗ OUT-OF-CORPUS YEAR GUARD FAILED")


if __name__ == "__main__":
    main()