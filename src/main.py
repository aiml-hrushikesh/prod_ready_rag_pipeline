import json
from typing import List

from src.chunking.chunking_strategies import AdvancedChunker, SimpleChunker
from src.config import settings
from src.llm.ollama import OllamaLLM
from src.rag.evaluation import run_evaluation
from src.rag.pipeline import RAGPipeline
from src.vectorstore.milvus_store import MilvusStore

# Sample questions used for both pipeline runs and evaluation
EVAL_QUESTIONS = [
    "What work did Equal Experts do for IG group?",
    "What was demonstrated in the Forrester study?",
    "How did Equal Experts help HMRC?",
]


def run_pipeline(chunker_name: str, chunker: object) -> None:  # type: ignore[type-arg]
    """Run the full pipeline with the given chunker and evaluate results."""
    print(f"\n{'='*60}")
    print(f"Running pipeline with: {chunker_name}")
    print("=" * 60)

    llm = OllamaLLM()
    vector_store = MilvusStore()
    pipeline = RAGPipeline(llm=llm, chunker=chunker, vector_store=vector_store)  # type: ignore[arg-type]

    try:
        with open(settings.CASE_STUDIES_PATH, "r", encoding="utf-8") as f:
            case_studies = json.load(f)
            documents: List[str] = [study["content"] for study in case_studies]

        print(f"Processing {len(documents)} documents...")
        pipeline.add_documents(documents)
        print("Documents stored.\n")

        # Collect answers and retrieved contexts for RAGAS evaluation
        answers: List[str] = []
        contexts: List[List[str]] = []

        for question in EVAL_QUESTIONS:
            print(f"Q: {question}")

            # Retrieve contexts explicitly so we can pass them to RAGAS
            q_embedding = llm.get_embeddings(question)
            hits = vector_store.retrieve(q_embedding, limit=settings.DEFAULT_TOP_K)
            retrieved_contexts = [str(hit["text"]) for hit in hits]

            answer = pipeline.query(question, top_k=settings.DEFAULT_TOP_K)
            print(f"A: {answer}\n")

            answers.append(answer)
            contexts.append(retrieved_contexts)

        # --- RAGAS Evaluation ---
        print(f"\n--- LLM-as-Judge Evaluation for {chunker_name} ---")
        scores = run_evaluation(
            questions=EVAL_QUESTIONS,
            answers=answers,
            contexts=contexts,
            llm=llm,
        )
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        llm.close()


def main() -> None:
    # Run with SimpleChunker (character-based sliding window)
    simple_chunker = SimpleChunker(
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        overlap=settings.DEFAULT_CHUNK_OVERLAP,
    )
    run_pipeline("SimpleChunker (character-based)", simple_chunker)

    # Run with AdvancedChunker (recursive paragraph/sentence-aware)
    advanced_chunker = AdvancedChunker(
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        overlap=settings.DEFAULT_CHUNK_OVERLAP,
    )
    run_pipeline("AdvancedChunker (sentence-aware)", advanced_chunker)


if __name__ == "__main__":
    main()
