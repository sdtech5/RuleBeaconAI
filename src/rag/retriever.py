from qdrant_client import models

from src.rag.schemas import RetrievalResult, SourceReference


def hybrid_retrieve_documents(
    vectorstore,
    query,
    k=5,
    query_filter=None,
):
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        query_filter=query_filter,
    )

    retrieval_results = []

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):
        retrieval_results.append(
            RetrievalResult(
                document=document,
                score=float(score),
                rank=rank,
            )
        )

    return retrieval_results


def create_source_reference(document, rank):
    metadata = dict(document.metadata or {})

    # Support both the current flat Qdrant metadata format
    # and older nested metadata returned by legacy AAPL payloads.
    nested_metadata = metadata.get("metadata")

    if isinstance(nested_metadata, dict):
        merged_metadata = dict(nested_metadata)
        merged_metadata.update(
            {
                key: value
                for key, value in metadata.items()
                if key != "metadata"
            }
        )
        metadata = merged_metadata

    return SourceReference(
        rank=rank,
        document_id=metadata.get("document_id"),
        chunk_id=metadata.get("chunk_id"),
        file_name=metadata.get("file_name"),
        source=metadata.get("source"),
        section=metadata.get("section"),
        page=metadata.get("page"),
    )