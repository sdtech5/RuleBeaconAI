from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document


def load_document(file_path: str) -> Document:
    """
    Load a single document and assign stable document metadata.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    text = path.read_text(encoding="utf-8")

    document_id = sha256(text.encode("utf-8")).hexdigest()

    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "source": str(path),
            "file_name": path.name,
        },
    )


def load_documents(directory_path: str) -> list[Document]:
    """
    Load all Markdown documents from a directory.
    """

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    documents = []

    for file_path in sorted(directory.glob("*.md")):

        document = load_document(
            str(file_path)
        )

        documents.append(document)

    return documents
