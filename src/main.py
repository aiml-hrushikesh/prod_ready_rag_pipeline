import json
import logging
from pathlib import Path
from typing import List

import typer

from src.chunking.chunking_strategies import AdvancedChunker, SimpleChunker
from src.config import settings
from src.llm.ollama import OllamaLLM
from src.rag.evaluation import run_evaluation
from src.rag.pipeline import RAGPipeline
from src.vectorstore.milvus_store import MilvusStore

app = typer.Typer()

EVAL_QUESTIONS = [
    "What work did Equal Experts do for IG group?",
    "What was demonstrated in the Forrester study?",
    "How did Equal Experts help HMRC?",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_documents(source_path: Path) -> List[str]:
    with source_path.open("r", encoding="utf-8") as f:
        case_studies = json.load(f)
        return [study["content"] for study in case_studies if study.get("content")]


def run_pipeline(chunker_name: str, chunker: object, evaluate: bool = True) -> None:  # type: ignore[type-arg]
    logger = logging.getLogger(__name__)
    logger.info("Running pipeline with: %s", chunker_name)

    llm = OllamaLLM()
    vector_store = MilvusStore()
    pipeline = RAGPipeline(llm=llm, chunker=chunker, vector_store=vector_store)  # type: ignore[arg-type]

    try:
        documents = load_documents(Path(settings.CASE_STUDIES_PATH))
        logger.info("Processing %d documents", len(documents))
        pipeline.add_documents(documents)

        answers: List[str] = []
        contexts: List[List[str]] = []

        for question in EVAL_QUESTIONS:
            logger.info("Asking question: %s", question)
            q_embedding = llm.get_embeddings(question)
            hits = vector_store.retrieve(q_embedding, limit=settings.DEFAULT_TOP_K)
            retrieved_contexts = [str(hit["text"]) for hit in hits]

            answer = pipeline.query(question, top_k=settings.DEFAULT_TOP_K)
            logger.info("Answer: %s", answer)

            answers.append(answer)
            contexts.append(retrieved_contexts)

        if evaluate:
            logger.info("Evaluating pipeline results")
            scores = run_evaluation(
                questions=EVAL_QUESTIONS,
                answers=answers,
                contexts=contexts,
                llm=llm,
            )
            for metric, score in scores.items():
                logger.info("%s: %.4f", metric, score)

    except Exception:
        logger.exception("Pipeline execution failed")
        raise
    finally:
        llm.close()


@app.command()
def run(
    chunker: str = typer.Option(
        "advanced",
        "--chunker",
        "-c",
        help="Choose chunker implementation: simple or advanced.",
    ),
    skip_evaluation: bool = typer.Option(
        False,
        "--skip-evaluation",
        help="Skip the RAGAS evaluation step.",
    ),
) -> None:
    configure_logging()
    chunker_choice = chunker.lower()
    if chunker_choice == "simple":
        selected_chunker = SimpleChunker(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            overlap=settings.DEFAULT_CHUNK_OVERLAP,
        )
        chunker_name = "SimpleChunker (character-based)"
    else:
        selected_chunker = AdvancedChunker(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            overlap=settings.DEFAULT_CHUNK_OVERLAP,
        )
        chunker_name = "AdvancedChunker (sentence-aware)"

    run_pipeline(chunker_name=chunker_name, chunker=selected_chunker, evaluate=not skip_evaluation)


if __name__ == "__main__":
    app()
