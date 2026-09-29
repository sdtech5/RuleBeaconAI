from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class SourceReference:
    rank: int
    document_id: str
    chunk_id: str
    file_name: str
    source: str
    section: str | None = None
    page: int | None = None


@dataclass
class RetrievalResult:
    """
    Represents one retrieved chunk together with
    its ranking information.
    """

    document: Document
    score: float
    rank: int