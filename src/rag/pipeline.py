from src.rag.loader import load_documents
from src.rag.parser import parse_document
from src.rag.chunker import chunk_document
from src.rag.embeddings import create_embeddings
from src.rag.vectorstore import create_qdrant_store
from src.rag.generator import create_llm
from src.rag.service import RAGService


def main():

    print("\n=== RuleBeaconAI RAG Pipeline ===")

    # ---------------------------------------------------------
    # 1. Load documents
    # ---------------------------------------------------------

    print("\n[1/5] Loading documents...")

    documents = load_documents(
        "data/raw/AAPL"
    )

    print(
        f"Loaded {len(documents)} document(s)."
    )

    for document in documents:
        print(
            f"  - {document.metadata['file_name']}"
        )

    # ---------------------------------------------------------
    # 2. Parse and chunk documents
    # ---------------------------------------------------------

    print("\n[2/5] Parsing and chunking documents...")

    all_sections = []

    for document in documents:

        sections = parse_document(document)

        all_sections.extend(sections)

        print(
            f"  {document.metadata['file_name']}: "
            f"{len(sections)} sections"
        )

    chunks = chunk_document(
        all_sections
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # ---------------------------------------------------------
    # 3. Create embeddings
    # ---------------------------------------------------------

    print("\n[3/5] Creating embeddings...")

    embeddings = create_embeddings()

    print(
        "Dense embedding model loaded."
    )

    # ---------------------------------------------------------
    # 4. Build Qdrant vector store
    # ---------------------------------------------------------

    print("\n[4/5] Building Qdrant vector store...")

    vector_store = create_qdrant_store(
        chunks,
        embeddings,
    )

    print(
        "Qdrant vector store created successfully."
    )

    # ---------------------------------------------------------
    # Load LLM
    # ---------------------------------------------------------

    llm = create_llm()

    print(
        "LLM loaded successfully."
    )

    # ---------------------------------------------------------
    # Create RAG service
    # ---------------------------------------------------------

    rag_service = RAGService(
        vector_store=vector_store,
        llm=llm,
    )

    # ---------------------------------------------------------
    # 5. Test RAG service
    # ---------------------------------------------------------

    print("\n[5/5] Testing RAG service...")

    questions = [
        "What was Apple's total net sales in fiscal year 2023?",
        "How much did Apple spend on research and development in 2023?",
        "What were Apple's iPhone net sales in 2023?",
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print(
            f"Question: {question}"
        )
        print("=" * 60)

        result = rag_service.ask(
            question
        )

        print("\nAnswer:")
        print(
            result["answer"]
        )

        print("\nRetrieved Sources:")

        for source in result["sources"]:

            print(
                f"\nSOURCE {source['rank']}"
            )

            print(
                f"  File: {source['file_name']}"
            )

            print(
                f"  Chunk ID: {source['chunk_id']}"
            )

            print(
                f"  Document ID: {source['document_id']}"
            )

            print(
                f"  Section: {source['section']}"
            )

            print(
                f"  Page: {source['page']}"
            )

            print(
                f"  Hybrid Score: {source['score']}"
            )


if __name__ == "__main__":
    main()