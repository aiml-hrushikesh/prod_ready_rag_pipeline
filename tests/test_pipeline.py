from unittest.mock import Mock, call

import pytest

from src.rag.pipeline import RAGPipeline


@pytest.fixture
def mock_components():
    llm = Mock()
    chunker = Mock()
    vector_store = Mock()

    # Setup mock returns
    llm.get_embeddings.return_value = [0.1, 0.2, 0.3]
    llm.generate.return_value = "test answer"
    vector_store.retrieve.return_value = [
        {"text": "relevant text 1", "score": 0.9},
        {"text": "relevant text 2", "score": 0.8},
    ]
    chunker.chunk_text.return_value = ["chunk one", "chunk two"]

    return {"llm": llm, "chunker": chunker, "vector_store": vector_store}


def test_pipeline_query(mock_components):
    pipeline = RAGPipeline(
        llm=mock_components["llm"],
        chunker=mock_components["chunker"],
        vector_store=mock_components["vector_store"],
    )

    result = pipeline.query("test question")

    mock_components["llm"].get_embeddings.assert_called_once_with("test question")
    mock_components["vector_store"].retrieve.assert_called_once()
    mock_components["llm"].generate.assert_called_once()
    assert result == "test answer"


def test_add_documents(mock_components):
    """Test that add_documents chunks, embeds, and stores all chunks correctly."""
    pipeline = RAGPipeline(
        llm=mock_components["llm"],
        chunker=mock_components["chunker"],
        vector_store=mock_components["vector_store"],
    )

    documents = ["document one", "document two"]
    pipeline.add_documents(documents)

    # Chunker should be called once per document
    assert mock_components["chunker"].chunk_text.call_count == 2
    mock_components["chunker"].chunk_text.assert_any_call("document one")
    mock_components["chunker"].chunk_text.assert_any_call("document two")

    # Each of the 4 chunks (2 docs × 2 chunks) should get embeddings
    assert mock_components["llm"].get_embeddings.call_count == 4

    # Vector store should be called once with all chunks and embeddings
    mock_components["vector_store"].insert_embeddings.assert_called_once()
    call_kwargs = mock_components["vector_store"].insert_embeddings.call_args[1]
    assert len(call_kwargs["texts"]) == 4
    assert len(call_kwargs["embeddings"]) == 4


def test_add_documents_empty_input(mock_components):
    """Test that empty documents are handled gracefully."""
    pipeline = RAGPipeline(
        llm=mock_components["llm"],
        chunker=mock_components["chunker"],
        vector_store=mock_components["vector_store"],
    )

    mock_components["chunker"].chunk_text.return_value = []
    pipeline.add_documents([""])

    mock_components["vector_store"].insert_embeddings.assert_not_called()


def test_query_prompt_contains_context(mock_components):
    """Test that the query prompt passed to the LLM includes the retrieved context."""
    pipeline = RAGPipeline(
        llm=mock_components["llm"],
        chunker=mock_components["chunker"],
        vector_store=mock_components["vector_store"],
    )

    pipeline.query("What did Equal Experts do for HMRC?")

    # Extract the prompt passed to llm.generate
    generate_call = mock_components["llm"].generate.call_args
    prompt = generate_call[1]["prompt"]
    assert "relevant text 1" in prompt
    assert "relevant text 2" in prompt
    assert "What did Equal Experts do for HMRC?" in prompt
