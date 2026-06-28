import logging
from typing import Dict, Generator, Iterable, List

from src.chunking.chunking_strategies import BaseChunker
from src.config import settings
from src.llm.base import BaseLLM
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


def _batch_items(items: Iterable[str], batch_size: int) -> Generator[List[str], None, None]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class RAGPipeline:
    def __init__(self, llm: BaseLLM, chunker: BaseChunker, vector_store: BaseVectorStore):
        self.llm = llm
        self.chunker = chunker
        self.vector_store = vector_store
        self.vector_store.create_collection(dim=settings.EMBEDDING_DIMENSION)

    def add_documents(self, documents: List[str]) -> None:
        """Ingest documents by chunking, embedding, and storing them."""
        if not documents:
            raise ValueError("No documents were provided to add_documents.")

        all_chunks: List[str] = []
        for doc_index, doc in enumerate(documents, start=1):
            if not doc or not doc.strip():
                logger.debug("Skipping empty document at index %d", doc_index)
                continue
            chunks = self.chunker.chunk_text(doc)
            all_chunks.extend(chunk.strip() for chunk in chunks if chunk and chunk.strip())

        if not all_chunks:
            logger.warning("No valid chunks generated from the provided documents.")
            return

        logger.info("Generating embeddings for %d chunks.", len(all_chunks))
        embeddings: List[List[float]] = []
        for batch_index, batch in enumerate(_batch_items(all_chunks, settings.DEFAULT_EMBEDDING_BATCH_SIZE), start=1):
            logger.debug("Processing embedding batch %d/%d", batch_index, (len(all_chunks) + settings.DEFAULT_EMBEDDING_BATCH_SIZE - 1) // settings.DEFAULT_EMBEDDING_BATCH_SIZE)
            for chunk in batch:
                embeddings.append(self.llm.get_embeddings(chunk))

        logger.info("Storing chunks in vector store.")
        self.vector_store.insert_embeddings(embeddings=embeddings, texts=all_chunks)

    def _build_prompt(self, context: str, question: str) -> str:
        return (
            "You are a professional assistant for a customer-facing knowledge base. "
            "Use the context below to answer the question precisely and honestly. "
            "If the context does not contain enough information, respond with: "
            'I cannot answer this question based on the available information.'\n\n'
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    def query(
        self,
        question: str,
        top_k: int = settings.DEFAULT_TOP_K,
        temperature: float = settings.DEFAULT_TEMPERATURE,
    ) -> str:
        """Query the RAG pipeline and return the generated answer."""
        question_embedding = self.llm.get_embeddings(question)
        results: List[Dict[str, float]] = self.vector_store.retrieve(
            query_embedding=question_embedding,
            limit=top_k,
        )

        if not results:
            logger.warning("No retrieval results for question: %s", question)
            return "I cannot answer this question based on the available information."

        context = "\n\n".join(str(hit["text"]) for hit in results)
        prompt = self._build_prompt(context=context, question=question)
        answer = self.llm.generate(prompt=prompt, temperature=temperature)
        return answer

