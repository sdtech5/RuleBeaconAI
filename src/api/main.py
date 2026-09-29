import concurrent.futures

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.embeddings import create_embeddings
from src.rag.vectorstore import load_qdrant_store
from src.rag.generator import create_llm
from src.rag.reranker import DocumentReranker
from src.rag.service import RAGService


app = FastAPI(
    title="RuleBeaconAI",
    description=(
        "SEC document research assistant "
        "powered by hybrid RAG and reranking."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Request / Response Schemas
# --------------------------------------------------


class QuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    rank: int
    score: float
    document_id: str
    chunk_id: str
    file_name: str
    source: str
    section: str | None = None
    page: int | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


# --------------------------------------------------
# Application Startup
# --------------------------------------------------


print("Loading RuleBeaconAI...")


print("Loading embedding model...")

embeddings = create_embeddings()


print("Connecting to Qdrant...")

vector_store = load_qdrant_store(
    embeddings
)


print("Loading LLM...")

llm = create_llm()


print("Loading reranker...")

reranker = DocumentReranker()


rag_service = RAGService(
    vector_store=vector_store,
    llm=llm,
    reranker=reranker,
)


print(
    "RuleBeaconAI ready — "
    "connected to existing Qdrant collection."
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------


@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "RuleBeaconAI",
    }


# --------------------------------------------------
# Question Endpoint
# --------------------------------------------------


@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(
    request: QuestionRequest,
):
    REQUEST_TIMEOUT = 80  # seconds — well within the 90s client timeout

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(rag_service.ask, request.question)
        try:
            result = future.result(timeout=REQUEST_TIMEOUT)
        except concurrent.futures.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=(
                    "The query took too long to process. "
                    "Try a more specific question or narrow the date range."
                ),
            )

    return result