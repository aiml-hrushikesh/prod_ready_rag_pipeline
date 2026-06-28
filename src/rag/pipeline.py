from typing import Dict, List

from src.chunking.chunking_strategies import BaseChunker
from src.config import settings
from src.llm.base import BaseLLM
from src.vectorstore.milvus_store import MilvusStore


class RAGPipeline:
    def __init__(self, llm: BaseLLM, chunker: BaseChunker, vector_store: MilvusStore):
        self.llm = llm
        self.chunker = chunker
        self.vector_store = vector_store

        # Initialize vector store collection
        self.vector_store.create_collection()

    def add_documents(self, documents: List[str]) -> None:
        """
        Ingest documents by chunking them, generating embeddings,
        and storing them in the vector store.
        """
        all_chunks = []
        for doc in documents:
            if not doc:
                continue
            chunks = self.chunker.chunk_text(doc)
            all_chunks.extend(chunks)

        # Filter out empty or whitespace-only chunks
        all_chunks = [c.strip() for c in all_chunks if c and c.strip()]

        if not all_chunks:
            print("No valid chunks generated from the documents.")
            return

        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = []
        for i, chunk in enumerate(all_chunks):
            # Print a progress indicator every 10 chunks if needed
            if i > 0 and i % 50 == 0:
                print(f"Processed {i}/{len(all_chunks)} chunks...")
            emb = self.llm.get_embeddings(chunk)
            embeddings.append(emb)

        print("Storing chunks in vector store...")
        self.vector_store.insert_embeddings(embeddings=embeddings, texts=all_chunks)

    def query(
        self,
        question: str,
        top_k: int = settings.DEFAULT_TOP_K,
        temperature: float = settings.DEFAULT_TEMPERATURE,
    ) -> str:
        """Query the RAG pipeline with a question."""

        question_embedding = self.llm.get_embeddings(question)
        results: List[Dict[str, float]] = self.vector_store.retrieve(
            query_embedding=question_embedding, limit=top_k
        )
        context = "\n\n".join(str(hit["text"]) for hit in results)
        prompt = f"""You are a professional assistant representing Equal Experts. 
Use the following context to answer the question. If the context does not contain enough information to answer the question, say "I cannot answer this question based on the available information."

Context:
{context}

Question: {question}

Answer:"""
        answer = self.llm.generate(prompt=prompt, temperature=temperature)
        return answer

