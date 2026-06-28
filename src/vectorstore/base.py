from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseVectorStore(ABC):
    """Abstract vector store interface for RAG storage implementations."""

    @abstractmethod
    def create_collection(self, dim: int) -> None:
        """Create or initialize the target collection for embeddings."""
        pass

    @abstractmethod
    def insert_embeddings(self, embeddings: List[List[float]], texts: List[str]) -> None:
        """Store embeddings alongside text chunks in the vector store."""
        pass

    @abstractmethod
    def retrieve(self, query_embedding: List[float], limit: int) -> List[Dict[str, float]]:
        """Retrieve nearest neighbor text chunks for the query embedding."""
        pass

    def close(self) -> None:
        """Close any underlying connections or cleanup resources."""
        return None
