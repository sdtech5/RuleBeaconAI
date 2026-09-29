import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding


load_dotenv()

COLLECTION_NAME = "sec_filings"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

SPARSE_MODEL_NAME = "Qdrant/bm25"

QDRANT_TIMEOUT = 300


class NativeQdrantHybridStore:
    """
    Query adapter for the RuleBeaconAI Qdrant collection.

    The collection is populated using the native Qdrant client, so
    queries are also performed using the native Qdrant client.

    Retrieval strategy:
        Dense search
        +
        Sparse BM25 search
        ↓
        Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
    ):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError("QDRANT_URL is not configured.")

        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY is not configured.")

        self.embeddings = embeddings

        print("Loading sparse BM25 model...")
        self.sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME
        )

        print("Connecting to Qdrant...")
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=QDRANT_TIMEOUT,
            prefer_grpc=True,
        )

    @staticmethod
    def _to_sparse_vector(sparse_vector) -> models.SparseVector:
        return models.SparseVector(
            indices=[
                int(index)
                for index in sparse_vector.indices
            ],
            values=[
                float(value)
                for value in sparse_vector.values
            ],
        )

    @staticmethod
    def _payload_to_document(payload: dict) -> Document:
        """
        Convert the native Qdrant payload into a LangChain Document.

        Qdrant stores document text under "text" and all document
        metadata as top-level payload fields.
        """

        metadata = {
            key: value
            for key, value in payload.items()
            if key != "text"
        }

        return Document(
            page_content=payload.get("text", ""),
            metadata=metadata,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        query_filter=None,
    ):
        """
        Perform hybrid dense + sparse retrieval using Qdrant RRF.

        Optional query_filter allows structured metadata filtering
        before hybrid retrieval.
        """

        # Dense query embedding
        dense_vector = self.embeddings.embed_query(query)

        # Sparse BM25 query embedding
        sparse_vector = next(
            self.sparse_model.embed([query])
        )

        sparse_query = self._to_sparse_vector(
            sparse_vector
        )

        # Retrieve candidates independently from both
        # dense and sparse indexes, then fuse them using RRF.
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=k,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR_NAME,
                    limit=k,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            document = self._payload_to_document(
                payload
            )

            results.append(
                (
                    document,
                    float(point.score),
                )
            )

        return results

    def close(self):
        self.client.close()


def load_qdrant_store(
    embeddings: HuggingFaceEmbeddings,
) -> NativeQdrantHybridStore:

    return NativeQdrantHybridStore(
        embeddings=embeddings
    )