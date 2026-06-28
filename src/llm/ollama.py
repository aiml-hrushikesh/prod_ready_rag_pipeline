from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from src.config import settings
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = settings.LLM_MODEL,
        embedding_model: str = settings.EMBEDDING_MODEL,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = httpx.Client(base_url=self.base_url, timeout=60.0)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                last_exception = exc
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError("Failed to call Ollama API") from last_exception

    def generate(
        self,
        prompt: str,
        temperature: float = settings.DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = settings.MAX_TOKENS,
        **kwargs: Any
    ) -> str:
        """Generate text using Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        result = self._post("/api/generate", payload)
        return str(result.get("response", "")).strip()

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Ollama API."""
        payload = {"model": self.embedding_model, "prompt": text}
        result = self._post("/api/embeddings", payload)
        embedding = result.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Invalid embedding response from Ollama")
        return [float(value) for value in embedding]

    def close(self) -> None:
        self.client.close()
