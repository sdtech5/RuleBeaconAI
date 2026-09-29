import os
import uuid
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer


load_dotenv()


COLLECTION_NAME = "sec_filings"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"

EMBEDDING_BATCH_SIZE = 256

QDRANT_UPLOAD_BATCH_SIZE = 256
QDRANT_UPLOAD_PARALLELISM = 2
QDRANT_MAX_RETRIES = 3
QDRANT_TIMEOUT = 300


class BulkUploader:
    """
    Production-oriented bulk ingestion pipeline.

    Responsibilities:

        Document chunks
            ↓
        Dense embeddings
            ↓
        Sparse BM25 embeddings
            ↓
        Qdrant PointStruct objects
            ↓
        Native Qdrant bulk upload

    The uploader deliberately bypasses LangChain's
    VectorStore.add_documents() for bulk ingestion.

    Qdrant's native upload_points() API provides:

        - batching
        - retries
        - controlled parallelism
        - lazy iteration
        - gRPC transport
    """

    def __init__(self):

        qdrant_url = os.getenv(
            "QDRANT_URL"
        )

        qdrant_api_key = os.getenv(
            "QDRANT_API_KEY"
        )

        if not qdrant_url:
            raise ValueError(
                "QDRANT_URL is not configured."
            )

        if not qdrant_api_key:
            raise ValueError(
                "QDRANT_API_KEY is not configured."
            )

        print(
            "Loading dense embedding model..."
        )

        self.dense_model = (
            SentenceTransformer(
                DENSE_MODEL_NAME
            )
        )

        print(
            "Loading sparse BM25 model..."
        )

        self.sparse_model = (
            SparseTextEmbedding(
                model_name=SPARSE_MODEL_NAME
            )
        )

        print(
            "Connecting to Qdrant..."
        )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=QDRANT_TIMEOUT,
            prefer_grpc=True,
        )

    @staticmethod
    def _qdrant_point_id(
        chunk_id: str,
    ) -> str:
        """
        Convert the deterministic SHA-256 chunk ID
        into a deterministic UUID accepted by Qdrant.
        """

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_id,
            )
        )

    @staticmethod
    def _sparse_vector_to_qdrant(
        sparse_vector,
    ) -> models.SparseVector:
        """
        Convert FastEmbed's sparse embedding into
        Qdrant's native SparseVector representation.
        """

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

    def _create_points(
        self,
        documents: list[Document],
    ) -> list[models.PointStruct]:
        """
        Create Qdrant points for one embedding batch.

        Dense and sparse embeddings are generated
        in batches to avoid excessive memory usage.
        """

        if not documents:
            return []

        texts = [
            document.page_content
            for document in documents
        ]

        dense_vectors = (
            self.dense_model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        sparse_vectors = list(
            self.sparse_model.embed(
                texts
            )
        )

        points = []

        for (
            document,
            dense_vector,
            sparse_vector,
        ) in zip(
            documents,
            dense_vectors,
            sparse_vectors,
        ):

            chunk_id = document.metadata[
                "chunk_id"
            ]

            point_id = (
                self._qdrant_point_id(
                    chunk_id
                )
            )

            payload = dict(
                document.metadata
            )

            payload[
                "text"
            ] = document.page_content

            point = models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME:
                        dense_vector.tolist(),

                    SPARSE_VECTOR_NAME:
                        self._sparse_vector_to_qdrant(
                            sparse_vector
                        ),
                },
                payload=payload,
            )

            points.append(point)

        return points

    def upload_documents(
        self,
        documents: Iterable[Document],
    ) -> int:
        """
        Embed and upload documents to Qdrant.

        Documents are consumed lazily in bounded
        batches rather than materializing the entire
        corpus in memory.

        Returns:
            Number of uploaded chunks.
        """

        batch = []
        total_uploaded = 0

        for document in documents:

            batch.append(document)

            if len(batch) >= EMBEDDING_BATCH_SIZE:

                total_uploaded += (
                    self._upload_batch(
                        batch
                    )
                )

                batch = []

        if batch:

            total_uploaded += (
                self._upload_batch(
                    batch
                )
            )

        return total_uploaded

    def _upload_batch(
        self,
        documents: list[Document],
    ) -> int:
        """
        Embed and upload one bounded batch.
        """

        if not documents:
            return 0

        print(
            f"Embedding {len(documents)} chunks..."
        )

        points = self._create_points(
            documents
        )

        print(
            f"Uploading {len(points)} chunks "
            f"to Qdrant..."
        )

        self.client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
            batch_size=QDRANT_UPLOAD_BATCH_SIZE,
            parallel=QDRANT_UPLOAD_PARALLELISM,
            max_retries=QDRANT_MAX_RETRIES,
            wait=True,
        )

        print(
            f"Uploaded {len(points)} chunks."
        )

        return len(points)

    def delete_chunks(
        self,
        chunk_ids: list[str],
            ) -> None:
        """
        Delete chunks using their deterministic
        Qdrant point IDs.
        """

        if not chunk_ids:
            return

        qdrant_ids = [
            self._qdrant_point_id(
                chunk_id
            )
            for chunk_id in chunk_ids
        ]

        print(
            f"Deleting {len(qdrant_ids)} "
            f"Qdrant points..."
        )

        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=qdrant_ids
            ),
            wait=True,
        )

        print(
            "Delete completed."
        )
    
    def close(self):
        """
        Close the Qdrant client.
        """

        self.client.close()