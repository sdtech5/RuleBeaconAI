import re

from sentence_transformers import CrossEncoder

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

        normalized = DocumentReranker._normalize(query)

        metric_groups = {
            "net_income": [
                "net earnings",
                "net income",
                "net profit",
            ],
            "revenue": [
                "total revenues",
                "total revenue",
                "revenues",
                "revenue",
                "net sales",
                "net revenues",
                "net revenue",
            ],
            "operating_income": [
                "operating income",
                "operating profit",
                "income from operations",
                "total income from operations",
            ],
            "gross_profit": [
                "gross profit",
            ],
            "research_development": [
                "research and development",
                "r&d",
            ],
            "eps": [
                "earnings per share",
                "diluted earnings per share",
                "basic earnings per share",
            ],
            "cash": [
                "cash and cash equivalents",
            ],
            "long_term_debt": [
                "long-term debt",
            ],
            "short_term_debt": [
                "short-term debt",
            ],
            "operating_cash_flow": [
                "operating cash flow",
            ],
            "assets": [
                "total assets",
            ],
            "liabilities": [
                "total liabilities",
            ],
        }

        matched_concepts = []

        for concept, terms in metric_groups.items():
            if any(
                re.search(
                    rf"\b{re.escape(term)}\b",
                    normalized,
                )
                for term in terms
            ):
                matched_concepts.append(concept)

        return matched_concepts

    @staticmethod
    def _concept_terms(concept: str) -> list[str]:
        """
        Return terminology variants for a normalized financial concept.
        """

        concept_terms = {
            "net_income": [
                "net earnings",
                "net income",
                "net profit",
            ],
            "revenue": [
                "total revenues",
                "total revenue",
                "revenues",
                "revenue",
                "net sales",
                "net revenues",
                "net revenue",
            ],
            "operating_income": [
                "operating income",
                "operating profit",
                "income from operations",
                "total income from operations",
            ],
            "gross_profit": [
                "gross profit",
            ],
            "research_development": [
                "research and development",
                "r&d",
            ],
            "eps": [
                "earnings per share",
                "diluted earnings per share",
                "basic earnings per share",
            ],
            "cash": [
                "cash and cash equivalents",
            ],
            "long_term_debt": [
                "long-term debt",
            ],
            "short_term_debt": [
                "short-term debt",
            ],
            "operating_cash_flow": [
                "operating cash flow",
            ],
            "assets": [
                "total assets",
            ],
            "liabilities": [
                "total liabilities",
            ],
        }

        return concept_terms.get(concept, [])

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