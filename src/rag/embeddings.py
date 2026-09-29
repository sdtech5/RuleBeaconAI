from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Create the embedding model used by the RAG pipeline.
    """

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )