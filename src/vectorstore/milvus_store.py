from __future__ import annotations

import logging
from typing import Dict, List

from pymilvus import MilvusClient, DataType

from src.config import settings
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusStore(BaseVectorStore):
    def __init__(self) -> None:
        self.collection_name = settings.COLLECTION_NAME
        self.client = MilvusClient(uri=settings.MILVUS_DB_PATH)

    def create_collection(self, dim: int = settings.EMBEDDING_DIMENSION) -> None:
        """Create the collection if it does not already exist."""
        if self.client.has_collection(self.collection_name):
            logger.debug("Using existing Milvus collection: %s", self.collection_name)
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="IP",
            index_type="AUTOINDEX",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created Milvus collection: %s", self.collection_name)

    def insert_embeddings(
        self, embeddings: List[List[float]], texts: List[str]
    ) -> None:
        """Insert text and their embeddings into the collection."""
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings and texts must match")

        entities = [
            {"text": text, "embedding": embedding}
            for text, embedding in zip(texts, embeddings)
        ]

        self.client.insert(collection_name=self.collection_name, data=entities)
        logger.debug("Inserted %d embeddings into Milvus collection", len(entities))

    def retrieve(
        self, query_embedding: List[float], limit: int
    ) -> List[Dict[str, float]]:
        """Retrieve similar documents using the query embedding."""
        self.client.load_collection(self.collection_name)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params={"metric_type": "IP"},
            limit=limit,
            output_fields=["text"],
        )

        hits: List[Dict[str, float]] = []
        for hit in results[0]:
            hits.append({"text": hit["entity"]["text"], "score": float(hit["distance"])});

        logger.debug("Retrieved %d hits from Milvus", len(hits))
        return hits

    def close(self) -> None:
        """Close the Milvus client connection."""
        try:
            self.client.close()
            logger.debug("Closed Milvus client connection")
        except Exception as exc:
            logger.warning("Failed to close Milvus client cleanly: %s", exc)
