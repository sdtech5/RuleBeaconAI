from src.rag.ingestion import ingest_documents


def main():

    print(
        "\n=== RuleBeaconAI Incremental Ingestion ==="
    )

    result = ingest_documents(
        directory_path="data/raw",
    )

    print(
        "\nIngestion Summary:"
    )

    print(
        f"  New: {result['new']}"
    )

    print(
        f"  Changed: {result['changed']}"
    )

    print(
        f"  Unchanged: {result['unchanged']}"
    )

    print(
        f"  Deleted: {result['deleted']}"
    )

    print(
        f"  Chunks upserted: "
        f"{result['chunks_upserted']}"
    )


if __name__ == "__main__":
    main()