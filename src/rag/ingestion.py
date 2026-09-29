from pathlib import Path

from src.rag.loader import load_document
from src.rag.parser import parse_document
from src.rag.chunker import chunk_document

from src.rag.manifest import (
    load_manifest,
    save_manifest,
    update_document_record,
    remove_document_record,
)

from src.rag.bulk_uploader import BulkUploader


def discover_documents(
    directory_path: str,
) -> list[Path]:
    """
    Discover all Markdown documents recursively.
    """

    directory = Path(
        directory_path
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    return sorted(
        directory.rglob("*.md")
    )


def _document_key(
    file_path: Path,
    directory_path: str,
) -> str:
    """
    Create a stable manifest key relative
    to the corpus root.
    """

    directory = Path(
        directory_path
    )

    return file_path.relative_to(
        directory
    ).as_posix()


def ingest_documents(
    directory_path: str,
) -> dict:
    """
    Perform incremental document ingestion.

    Lifecycle:

        NEW
        CHANGED
        UNCHANGED
        DELETED

    New and changed documents are:

        loaded
        parsed
        chunked
        embedded
        uploaded to Qdrant

    The manifest is updated only after
    successful Qdrant upload.
    """

    manifest = load_manifest()

    files = discover_documents(
        directory_path
    )

    current_files = {
        _document_key(
            file_path,
            directory_path,
        )
        for file_path in files
    }

    new_files = []
    changed_files = []
    unchanged_files = []
    deleted_files = []

    # --------------------------------------------------
    # Detect new / changed / unchanged documents
    # --------------------------------------------------

    for file_path in files:

        document = load_document(
            str(file_path)
        )

        file_key = _document_key(
            file_path,
            directory_path,
        )

        previous_record = manifest.get(
            file_key
        )

        if previous_record is None:

            new_files.append(
                (
                    file_path,
                    document,
                    file_key,
                )
            )

        elif (
            previous_record["document_id"]
            != document.metadata["document_id"]
        ):

            changed_files.append(
                (
                    file_path,
                    document,
                    file_key,
                    previous_record,
                )
            )

        else:

            unchanged_files.append(
                file_key
            )

    # --------------------------------------------------
    # Detect deleted documents
    # --------------------------------------------------

    for file_key in list(
        manifest.keys()
    ):

        if file_key not in current_files:

            deleted_files.append(
                (
                    file_key,
                    manifest[file_key],
                )
            )

    print(
        "\n=== Incremental Ingestion ==="
    )

    print(
        f"New: {len(new_files)}"
    )

    print(
        f"Changed: {len(changed_files)}"
    )

    print(
        f"Unchanged: {len(unchanged_files)}"
    )

    print(
        f"Deleted: {len(deleted_files)}"
    )

    # --------------------------------------------------
    # Initialize bulk uploader
    # --------------------------------------------------

    uploader = BulkUploader()

    total_chunks_upserted = 0

    try:

        # ----------------------------------------------
        # Handle deleted documents
        # ----------------------------------------------

        for (
            file_key,
            record,
        ) in deleted_files:

            print(
                f"\nDeleting removed document: "
                f"{file_key}"
            )

            uploader.delete_chunks(
                record["chunk_ids"]
            )

            remove_document_record(
                manifest,
                file_key,
            )

            save_manifest(
                manifest
            )

        # ----------------------------------------------
        # Handle changed documents
        # ----------------------------------------------

        for (
            file_path,
            document,
            file_key,
            previous_record,
        ) in changed_files:

            print(
                f"\nReplacing changed document: "
                f"{file_key}"
            )

            # Remove old chunks first.
            uploader.delete_chunks(
                previous_record[
                    "chunk_ids"
                ]
            )

            sections = parse_document(
                document
            )

            chunks = chunk_document(
                sections
            )

            print(
                f"  Sections: "
                f"{len(sections)}"
            )

            print(
                f"  Chunks: "
                f"{len(chunks)}"
            )

            uploaded = (
                uploader.upload_documents(
                    chunks
                )
            )

            total_chunks_upserted += (
                uploaded
            )

            chunk_ids = [
                chunk.metadata[
                    "chunk_id"
                ]
                for chunk in chunks
            ]

            update_document_record(
                manifest=manifest,
                file_name=file_key,
                document_id=document.metadata[
                    "document_id"
                ],
                source=document.metadata[
                    "source"
                ],
                chunk_ids=chunk_ids,
            )

            save_manifest(
                manifest
            )

            print(
                f"  ✓ Completed {file_key}"
            )

        # ----------------------------------------------
        # Handle new documents
        # ----------------------------------------------

        for (
            file_path,
            document,
            file_key,
        ) in new_files:

            print(
                f"\nProcessing new document: "
                f"{file_key}"
            )

            sections = parse_document(
                document
            )

            chunks = chunk_document(
                sections
            )

            print(
                f"  Sections: "
                f"{len(sections)}"
            )

            print(
                f"  Chunks: "
                f"{len(chunks)}"
            )

            uploaded = (
                uploader.upload_documents(
                    chunks
                )
            )

            total_chunks_upserted += (
                uploaded
            )

            chunk_ids = [
                chunk.metadata[
                    "chunk_id"
                ]
                for chunk in chunks
            ]

            # IMPORTANT:
            # Only update the manifest after
            # Qdrant upload succeeds.

            update_document_record(
                manifest=manifest,
                file_name=file_key,
                document_id=document.metadata[
                    "document_id"
                ],
                source=document.metadata[
                    "source"
                ],
                chunk_ids=chunk_ids,
            )

            save_manifest(
                manifest
            )

            print(
                f"  ✓ Completed {file_key}"
            )

    finally:

        uploader.close()

    return {
        "new": len(new_files),
        "changed": len(changed_files),
        "unchanged": len(unchanged_files),
        "deleted": len(deleted_files),
        "chunks_upserted":
            total_chunks_upserted,
    }