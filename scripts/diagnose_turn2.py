import os
import streamlit as st
import app
from src.rag.embeddings import create_embeddings
from src.rag.vectorstore import load_qdrant_store
from src.rag.reranker import DocumentReranker
from src.rag.generator import create_llm, generate_answer
from src.rag.service import RAGService
from src.rag.retriever import hybrid_retrieve_documents

# Turn 1
q1 = "What were Alphabet's net sales in fiscal year 2023?"
c1 = app.build_contextual_question(q1)
print("Contextual Q1:")
print(c1)

# Turn 2
q2 = "How does it compare to 2022 net sales? Percentage wise"
c2 = app.build_contextual_question(q2)
print("\nContextual Q2:")
print(c2)

embeddings = create_embeddings()
vs = load_qdrant_store(embeddings)
reranker = DocumentReranker()
llm = create_llm()
service = RAGService(vs, llm, reranker)

chunks = service.retrieve_candidates(c2)
print(f"RETRIEVED {len(chunks)} CHUNKS:")
for i, c in enumerate(chunks):
    print(f"--- CHUNK {i+1} [{c.metadata.get('file_name')}] [{c.metadata.get('section')}] is_table={c.metadata.get('is_table')} ---")
    print(c.page_content[:400])
    print()

res = service.ask(c2)
print("\nANSWER:")
print(res["answer"])
