import json
from pathlib import Path


MANIFEST_PATH = Path(
    "data/manifest.json"
)


def load_manifest() -> dict:
    """
    Load the ingestion manifest.

    Returns an empty manifest when no previous
    ingestion has been recorded.
    """

    if not MANIFEST_PATH.exists():
        return {}

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_manifest(
    manifest: dict,
) -> None:
    """
    Persist the ingestion manifest to disk.
    """

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = MANIFEST_PATH.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    temporary_path.replace(
        MANIFEST_PATH
    )


def get_document_record(
    manifest: dict,
    file_name: str,
) -> dict | None:
    """
    Return the manifest record for a document.
    """

    return manifest.get(file_name)


def update_document_record(
    manifest: dict,
    file_name: str,
    document_id: str,
    source: str,
    chunk_ids: list[str],
) -> None:
    """
    Add or update a document's ingestion record.
    """

    manifest[file_name] = {
        "document_id": document_id,
        "source": source,
        "chunk_ids": chunk_ids,
    }


def remove_document_record(
    manifest: dict,
    file_name: str,
) -> None:
    """
    Remove a document from the manifest.
    """

    manifest.pop(
        file_name,
        None,
    )