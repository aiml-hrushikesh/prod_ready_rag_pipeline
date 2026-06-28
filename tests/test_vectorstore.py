from unittest.mock import MagicMock, patch

import pytest

from src.vectorstore.milvus_store import MilvusStore


@pytest.fixture
def mock_milvus():
    with patch("src.vectorstore.milvus_store.MilvusClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.has_collection.return_value = False
        mock_client.create_schema.return_value = MagicMock()
        mock_client.prepare_index_params.return_value = MagicMock()

        yield mock_client


def test_milvus_store_init(mock_milvus):
    store = MilvusStore()
    assert store.client is mock_milvus


def test_milvus_store_create_collection(mock_milvus):
    store = MilvusStore()
    store.create_collection()

    mock_milvus.has_collection.assert_called_once()
    mock_milvus.create_collection.assert_called_once()


def test_milvus_store_create_collection_existing(mock_milvus):
    """If collection already exists, skip creation."""
    mock_milvus.has_collection.return_value = True
    store = MilvusStore()
    store.create_collection()

    mock_milvus.create_collection.assert_not_called()


def test_milvus_store_insert_embeddings(mock_milvus):
    store = MilvusStore()

    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    texts = ["text1", "text2"]

    store.insert_embeddings(embeddings=embeddings, texts=texts)

    mock_milvus.insert.assert_called_once()
    call_kwargs = mock_milvus.insert.call_args[1]
    assert call_kwargs["collection_name"] == store.collection_name
    assert len(call_kwargs["data"]) == 2


def test_milvus_store_retrieve(mock_milvus):
    store = MilvusStore()

    # Mock search results
    mock_hit = {"entity": {"text": "test text"}, "distance": 0.95}
    mock_milvus.search.return_value = [[mock_hit]]

    results = store.retrieve([1.0, 2.0], limit=1)

    assert len(results) == 1
    assert results[0]["text"] == "test text"
    assert results[0]["score"] == 0.95
    mock_milvus.search.assert_called_once()


def test_milvus_store_insert_embeddings_validation(mock_milvus):
    store = MilvusStore()

    with pytest.raises(ValueError, match="Number of embeddings and texts must match"):
        store.insert_embeddings(embeddings=[[1.0, 2.0]], texts=["text1", "text2"])
