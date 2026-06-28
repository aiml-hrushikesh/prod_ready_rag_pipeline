from typing import Dict, List

from pymilvus import MilvusClient, DataType

from src.config import settings


class MilvusStore:
    def __init__(self) -> None:
        self.collection_name = settings.COLLECTION_NAME
        self.client = MilvusClient(uri=settings.MILVUS_DB_PATH)

    def create_collection(self, dim: int = settings.EMBEDDING_DIMENSION) -> None:
        """Create a new collection, reuse if it already exists."""
        if self.client.has_collection(self.collection_name):
            return  # reuse existing collection

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

        hits = []
        for hit in results[0]:
            hits.append({"text": hit["entity"]["text"], "score": hit["distance"]})

        return hits
