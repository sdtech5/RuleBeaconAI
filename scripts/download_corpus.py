from pathlib import Path
import re

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "astr010/sec-10k-markdown-uncompressed"

CORPUS_ROOT = Path("data/raw")

# Final launch corpus:
# 11 companies × 5 years (2020–2024) = 55 filings.
TARGET_COMPANIES = [
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
]

TARGET_YEARS = range(2020, 2025)

FILING_PATTERN = re.compile(
    r"^(?P<ticker>[^/]+)/10-K_(?P<year>\d{4})\.md$"
)


def get_available_filings():
    print("\nInspecting available filings...")

    api = HfApi()

    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    available = {}

    for path in files:
        match = FILING_PATTERN.match(path)

        if not match:
            continue

        ticker = match.group("ticker").upper()
        year = int(match.group("year"))

        available.setdefault(ticker, set()).add(year)

    return available


def validate_target_filings(available):
    print("\n=== Validating Target Corpus ===")

    missing = []

    for ticker in TARGET_COMPANIES:
        available_years = available.get(ticker, set())

        missing_years = [
            year
            for year in TARGET_YEARS
            if year not in available_years
        ]

        if missing_years:
            missing.append((ticker, missing_years))
            print(
                f"  {ticker}: MISSING {missing_years}"
            )
        else:
            print(
                f"  {ticker}: OK (2020–2024)"
            )

    if missing:
        print(
            "\nERROR: Target corpus validation failed."
        )

        for ticker, years in missing:
            print(
                f"  {ticker}: missing {years}"
            )

        raise RuntimeError(
            "Target corpus validation failed."
        )

    print(
        "\n✓ Corpus validation passed."
    )


def download_filing(ticker: str, year: int):
    repo_path = f"{ticker}/10-K_{year}.md"

    destination_dir = CORPUS_ROOT / ticker
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_path = (
        destination_dir / f"10-K_{year}.md"
    )

    if destination_path.exists():
        print(
            f"SKIP  {repo_path} "
            "(already exists)"
        )
        return False

    print(f"Downloading {repo_path}...")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type="dataset",
    )

    source = Path(downloaded_path)

    destination_path.write_bytes(
        source.read_bytes()
    )

    print(f"OK    {destination_path}")

    return True


def remove_obsolete_filings():
    """
    Remove filings belonging to target companies
    that fall outside the final 2020–2024 window.
    """

    print(
        "\n=== Removing Obsolete Local Filings ==="
    )

    removed = []

    for ticker in TARGET_COMPANIES:
        company_dir = CORPUS_ROOT / ticker

        if not company_dir.exists():
            continue

        for file_path in company_dir.glob(
            "10-K_*.md"
        ):
            match = re.match(
                r"^10-K_(\d{4})\.md$",
                file_path.name,
            )

            if not match:
                continue

            year = int(match.group(1))

            if year not in TARGET_YEARS:
                print(f"REMOVE {file_path}")

                file_path.unlink()
                removed.append(file_path)

    if not removed:
        print("No obsolete filings found.")
    else:
        print(
            f"\nRemoved {len(removed)} "
            "obsolete local filings."
        )

    return removed


def print_final_local_corpus():
    print(
        "\n=== Final Local Corpus Check ==="
    )

    total = 0

    for ticker in TARGET_COMPANIES:
        company_dir = CORPUS_ROOT / ticker
        years = []

        if company_dir.exists():
            for file_path in company_dir.glob(
                "10-K_*.md"
            ):
                match = re.match(
                    r"^10-K_(\d{4})\.md$",
                    file_path.name,
                )

                if match:
                    years.append(
                        int(match.group(1))
                    )

        years.sort()

        print(
            f"  {ticker}: {years}"
        )

        total += len(years)

    expected = (
        len(TARGET_COMPANIES)
        * len(TARGET_YEARS)
    )

    print(
        f"\nTotal local filings: {total}"
    )

    print(
        f"Expected filings: {expected}"
    )

    if total != expected:
        raise RuntimeError(
            "Final local corpus count does not "
            "match the expected target."
        )

    print(
        f"\n✓ Final corpus contains exactly "
        f"{expected} filings."
    )


def main():
    print(
        "\n=== RuleBeaconAI Corpus Update ==="
    )

    available = get_available_filings()

    # Validate the complete target before
    # making any local changes.
    validate_target_filings(available)

    print("\nTarget corpus:")

    for ticker in TARGET_COMPANIES:
        print(
            f"  {ticker}: 2020–2024"
        )

    expected = (
        len(TARGET_COMPANIES)
        * len(TARGET_YEARS)
    )

    print(
        f"\nTarget total: "
        f"{len(TARGET_COMPANIES)} companies × "
        f"{len(TARGET_YEARS)} years = "
        f"{expected} filings"
    )

    # Remove only obsolete filings belonging
    # to target companies.
    remove_obsolete_filings()

    print(
        "\n=== Downloading Missing Filings ==="
    )

    downloaded = 0
    skipped = 0

    for ticker in TARGET_COMPANIES:
        print(f"\n--- {ticker} ---")

        for year in TARGET_YEARS:
            was_downloaded = download_filing(
                ticker=ticker,
                year=year,
            )

            if was_downloaded:
                downloaded += 1
            else:
                skipped += 1

    print(
        "\n=== Corpus Update Complete ==="
    )

    print(
        f"New files downloaded: {downloaded}"
    )

    print(
        f"Existing files kept: {skipped}"
    )

    print_final_local_corpus()


if __name__ == "__main__":
    main()