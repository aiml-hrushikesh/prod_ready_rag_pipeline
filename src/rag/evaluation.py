"""
RAG Evaluation using LLM-as-a-judge pattern.

Uses the same local Ollama LLM to score:
  - context_relevance: Are the retrieved chunks relevant to the question? (0-1)
  - answer_faithfulness: Is the answer grounded in the context? (0-1)
  - answer_relevance: Does the answer actually address the question? (0-1)
"""

from typing import Any, Dict, List

from src.config import settings


CONTEXT_RELEVANCE_PROMPT = """You are an evaluation assistant. 
Score how relevant the following context chunks are to the question.

Question: {question}

Retrieved Context:
{context}

Score from 0 to 10 where:
0 = completely irrelevant
5 = somewhat relevant
10 = perfectly relevant

Reply with ONLY a single integer number between 0 and 10. No explanation."""


FAITHFULNESS_PROMPT = """You are an evaluation assistant.
Score whether the answer is faithful and grounded in the provided context.

Context:
{context}

Answer: {answer}

Score from 0 to 10 where:
0 = answer is completely made up / not in context
5 = answer is partially supported by context
10 = answer is fully supported by context

Reply with ONLY a single integer number between 0 and 10. No explanation."""


ANSWER_RELEVANCE_PROMPT = """You are an evaluation assistant.
Score how well the answer addresses the question.

Question: {question}

Answer: {answer}

Score from 0 to 10 where:
0 = answer is completely off-topic
5 = answer partially addresses the question
10 = answer perfectly addresses the question

Reply with ONLY a single integer number between 0 and 10. No explanation."""


def _parse_score(response: str) -> float:
    """Extract numeric score from LLM response."""
    response = response.strip()
    for token in response.split():
        try:
            score = float(token)
            if 0 <= score <= 10:
                return score / 10.0  # normalize to 0-1
        except ValueError:
            continue
    return 0.0


def run_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    llm: Any = None,
) -> Dict[str, Any]:
    """
    Evaluate RAG pipeline using LLM-as-a-judge.

    Args:
        questions: List of user questions.
        answers:   Corresponding generated answers from the RAG pipeline.
        contexts:  For each question, the list of retrieved context chunks.
        llm:       LLM instance with generate() method (OllamaLLM).

    Returns:
        A dict with metric names mapped to their average scores (0-1).
    """
    if llm is None:
        print("No LLM provided for evaluation, skipping.")
        return {
            "context_relevance": "N/A",
            "answer_faithfulness": "N/A",
            "answer_relevance": "N/A",
        }

    context_relevance_scores = []
    faithfulness_scores = []
    answer_relevance_scores = []

    for i, (question, answer, context_chunks) in enumerate(
        zip(questions, answers, contexts)
    ):
        print(f"  Evaluating question {i + 1}/{len(questions)}...")
        context_text = "\n\n".join(context_chunks)

        try:
            # Score 1: Context Relevance
            prompt = CONTEXT_RELEVANCE_PROMPT.format(
                question=question, context=context_text
            )
            response = llm.generate(prompt=prompt, temperature=0.0)
            context_relevance_scores.append(_parse_score(response))

            # Score 2: Faithfulness
            prompt = FAITHFULNESS_PROMPT.format(
                context=context_text, answer=answer
            )
            response = llm.generate(prompt=prompt, temperature=0.0)
            faithfulness_scores.append(_parse_score(response))

            # Score 3: Answer Relevance
            prompt = ANSWER_RELEVANCE_PROMPT.format(
                question=question, answer=answer
            )
            response = llm.generate(prompt=prompt, temperature=0.0)
            answer_relevance_scores.append(_parse_score(response))

        except Exception as e:
            print(f"  Error evaluating question '{question}': {e}")
            context_relevance_scores.append(0.0)
            faithfulness_scores.append(0.0)
            answer_relevance_scores.append(0.0)

    return {
        "context_relevance": sum(context_relevance_scores) / len(context_relevance_scores),
        "answer_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "answer_relevance": sum(answer_relevance_scores) / len(answer_relevance_scores),
    }
