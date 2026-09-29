"""
RuleBeaconAI — 10-question end-to-end backend test

Run from the RuleBeaconAI project root after starting FastAPI:

    uvicorn src.api.main:app --reload

Then, in another terminal:

    python -m tests.test_10_questions

The script sends all 10 questions sequentially to the existing /ask endpoint,
prints concise Question/Answer results, and saves the same output to
test_results_10_questions.txt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000/ask"
OUTPUT_FILE = Path("test_results_10_questions.txt")

QUESTIONS = [
    "What was NVIDIA's total revenue in fiscal year 2024?",
    "What was Meta Platforms' total revenue in fiscal year 2020?",
    "Compare NVIDIA's revenue growth rate in FY2022 vs FY2023 vs FY2024.",
    "How did Microsoft's revenue grow each year from FY2021 to FY2024?",
    "What were Apple's iPhone, Mac, iPad, Wearables, and Services revenues in FY2023?",
    "What was NVIDIA's data center revenue in FY2024, and what share of total revenue did it represent?",
    "What was Apple's operating margin in FY2023? (Operating income / Total net sales)",
    "What was Tesla's operating income in FY2022, and what was its operating margin?",
    "What was NVIDIA's fiscal year 2024 end date?",
    "What was Archer Daniels Midland's net earnings attributable to ADM in FY2023?",
]


def call_api(question: str) -> tuple[int, dict]:
    payload = json.dumps({"question": question}).encode("utf-8")

    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=180) as response:
        return response.status, json.loads(
            response.read().decode("utf-8")
        )


def format_result(
    number: int,
    question: str,
    status: str,
    answer: str,
) -> str:
    return (
        f"{number}. QUESTION\n"
        f"{question}\n\n"
        f"ANSWER\n"
        f"{answer}\n\n"
        f"STATUS: {status}\n"
    )


def main() -> None:
    print("\n" + "=" * 78)
    print("RULEBEACONAI — 10 QUESTION BACKEND TEST")
    print("=" * 78)
    print(f"API: {API_URL}")
    print("Running questions sequentially...\n")

    results = []
    total_start = time.perf_counter()

    for number, question in enumerate(QUESTIONS, start=1):
        print(f"[{number}/10] Running...", end="", flush=True)
        start = time.perf_counter()

        try:
            status_code, data = call_api(question)
            elapsed = time.perf_counter() - start

            if status_code != 200:
                answer = (
                    f"API returned HTTP {status_code}: "
                    f"{data.get('detail', data)}"
                )
                status = f"ERROR — HTTP {status_code}"
            else:
                answer = str(data.get("answer", "")).strip()

                if not answer:
                    answer = "[No answer returned by API]"
                    status = "ERROR — EMPTY ANSWER"
                else:
                    status = "OK"

            print(f" done ({elapsed:.1f}s)")

        except HTTPError as exc:
            elapsed = time.perf_counter() - start

            try:
                error_body = exc.read().decode("utf-8")
                parsed_error = json.loads(error_body)
                detail = parsed_error.get("detail", parsed_error)
            except Exception:
                detail = str(exc)

            answer = f"HTTP {exc.code}: {detail}"
            status = f"ERROR — HTTP {exc.code}"
            print(f" failed ({elapsed:.1f}s)")

        except URLError:
            answer = (
                "Could not connect to FastAPI. "
                "Start it with: uvicorn src.api.main:app --reload"
            )
            status = "ERROR — API UNAVAILABLE"
            print(" failed — API unavailable")

        except Exception as exc:
            answer = f"Unexpected test error: {type(exc).__name__}: {exc}"
            status = "ERROR — TEST FAILURE"
            print(" failed")

        results.append(
            format_result(
                number,
                question,
                status,
                answer,
            )
        )

    total_elapsed = time.perf_counter() - total_start

    output = (
        "RULEBEACONAI — 10 QUESTION BACKEND TEST RESULTS\n"
        + "=" * 78
        + "\n\n"
        + "\n".join(results)
        + "\n"
        + "=" * 78
        + "\n"
        + f"Total questions: {len(QUESTIONS)}\n"
        + f"Total runtime: {total_elapsed:.1f}s\n"
    )

    OUTPUT_FILE.write_text(output, encoding="utf-8")

    print("\n" + output)
    print(f"Saved copy to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
