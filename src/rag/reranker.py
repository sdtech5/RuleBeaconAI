import re

from sentence_transformers import CrossEncoder

from src.rag.calculations import (
    METRIC_CONCEPT_TERMS,
    detect_metric_concepts,
)
from src.rag.schemas import RetrievalResult


MODEL_NAME = "BAAI/bge-reranker-base"


class DocumentReranker:
    def __init__(self, model_name=MODEL_NAME):
        self.model = CrossEncoder(model_name)

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_metric_terms(query: str) -> list[str]:
        """
        Extract financial metric concepts from the question.

        Synonyms are grouped conceptually so that different common
        phrasings can match the same underlying financial metric.
        """

        # Concept terminology is centralized in src.rag.calculations so the
        # reranker and the calculation/retrieval layers stay in sync.
        return detect_metric_concepts(query)

    @staticmethod
    def _concept_terms(concept: str) -> list[str]:
        """
        Return terminology variants for a normalized financial concept.
        """

        return METRIC_CONCEPT_TERMS.get(concept, [])

    @staticmethod
    def _exact_metric_score(
        query: str,
        document: str,
    ) -> float:
        """
        Give a strong boost when the requested financial metric appears
        as an exact standalone table/statement label.

        Terminology variants belonging to the same concept are treated
        as equivalent.

        Example:

            Query: "net profit"

            Document:
                Net income | $4,368

        receives a metric boost because both belong to the net_income
        concept.
        """

        concepts = DocumentReranker._extract_metric_terms(query)

        if not concepts:
            return 0.0

        normalized_document = DocumentReranker._normalize(document)

        score = 0.0

        for concept in concepts:

            terms = DocumentReranker._concept_terms(concept)

            for metric in terms:

                # Strongest case:
                # markdown table label:
                # | Net income | ...
                table_pattern = (
                    rf"(?:\|\s*){re.escape(metric)}"
                    rf"(?:\s*\||\s*$)"
                )

                if re.search(
                    table_pattern,
                    normalized_document,
                ):
                    score += 1.0
                    break

                # Standalone metric followed by punctuation/sentence
                # boundary. Avoid qualified variants such as:
                #
                # "net earnings including..."
                #
                standalone_pattern = (
                    rf"\b{re.escape(metric)}\b"
                    rf"(?!\s+(?:including|attributable|"
                    rf"available|before|after|from|to)\b)"
                )

                if re.search(
                    standalone_pattern,
                    normalized_document,
                ):
                    score += 0.35
                    break

        return score

    def rerank(
        self,
        query,
        results,
        top_k=5,
    ):

        if not results:
            return []

        pairs = [
            (
                query,
                result.document.page_content,
            )
            for result in results
        ]

        cross_encoder_scores = self.model.predict(pairs)

        reranked = []

        for result, cross_score in zip(
            results,
            cross_encoder_scores,
        ):

            metric_score = self._exact_metric_score(
                query,
                result.document.page_content,
            )

            final_score = (
                float(cross_score)
                + metric_score
            )

            reranked.append(
                RetrievalResult(
                    document=result.document,
                    score=final_score,
                    rank=0,
                )
            )

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        reranked = reranked[:top_k]

        for rank, result in enumerate(
            reranked,
            start=1,
        ):
            result.rank = rank

        return reranked