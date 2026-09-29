from hashlib import sha256

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(
    documents: list[Document],
) -> list[Document]:
    """
    Split parsed document sections into smaller overlapping chunks.

    Chunk IDs are deterministic so that unchanged documents produce
    stable chunk identifiers across ingestion runs.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    section_chunk_counters = {}

    for chunk in chunks:
        document_id = chunk.metadata["document_id"]

        section_key = (
            document_id,
            chunk.metadata.get("part"),
            chunk.metadata.get("item"),
            chunk.metadata.get("section"),
        )

        chunk_index = section_chunk_counters.get(
            section_key,
            0,
        )

        section_chunk_counters[section_key] = (
            chunk_index + 1
        )

        raw_chunk_id = (
            f"{document_id}:"
            f"{chunk.metadata.get('part')}:"
            f"{chunk.metadata.get('item')}:"
            f"{chunk.metadata.get('section')}:"
            f"{chunk_index}"
        )

        chunk.metadata["chunk_id"] = sha256(
            raw_chunk_id.encode("utf-8")
        ).hexdigest()

    return chunks