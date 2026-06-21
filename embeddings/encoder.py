import asyncio
import json
from typing import Any, List, Optional
import numpy as np
import httpx
from redis.asyncio import Redis
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from lexrag.config import settings
from lexrag.logger import get_logger
from lexrag.exceptions import EmbeddingError
from lexrag.utils.hashing import compute_sha256

logger = get_logger(__name__)

def _before_retry_log(retry_state: Any) -> None:
    logger.warning("Ollama connection error, retrying...", attempt=retry_state.attempt_number)

class BGEEncoder:
    """Production-grade embedding pipeline using Ollama."""
    
    def __init__(
        self, 
        ollama_url: str = settings.ollama_base_url, 
        model: str = settings.ollama_embedding_model, 
        cache: Optional[Redis] = None
    ) -> None:
        self.ollama_url = ollama_url
        self.model = model
        self.cache = cache
        self.client = httpx.AsyncClient(timeout=120.0)
        
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into a numpy array of dense embeddings."""
        if not texts:
            return np.empty((0, settings.qdrant_vector_size), dtype=np.float32)
            
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        cache_misses = []
        cache_miss_indices = []
        
        # 1. Check Redis cache
        if self.cache:
            for i, text in enumerate(texts):
                text_hash = compute_sha256(text)[:16]
                cache_key = f"embedding_cache:{text_hash}"
                cached_bytes = await self.cache.get(cache_key)
                
                if cached_bytes:
                    results[i] = np.array(json.loads(cached_bytes), dtype=np.float32)
                else:
                    cache_misses.append(text)
                    cache_miss_indices.append(i)
        else:
            cache_misses = texts
            cache_miss_indices = list(range(len(texts)))
            
        # 2. Batch encode cache misses
        batch_size = 32
        for i in range(0, len(cache_misses), batch_size):
            batch_texts = cache_misses[i:i+batch_size]
            batch_indices = cache_miss_indices[i:i+batch_size]
            
            embeddings = await self._call_ollama_batch(batch_texts)
            
            # 5. Store new embeddings in Redis and reconstruct
            for j, emb in enumerate(embeddings):
                # L2 Normalization
                norm = np.linalg.norm(emb)
                normalized_emb = emb / norm if norm > 0 else emb
                
                orig_idx = batch_indices[j]
                results[orig_idx] = normalized_emb
                
                # Assert dimension
                if len(normalized_emb) != settings.qdrant_vector_size:
                    raise EmbeddingError(
                        f"Embedding dimension mismatch. Expected {settings.qdrant_vector_size}, got {len(normalized_emb)}"
                    )
                
                if self.cache:
                    text_hash = compute_sha256(batch_texts[j])[:16]
                    cache_key = f"embedding_cache:{text_hash}"
                    # 7 days TTL (604800 seconds)
                    await self.cache.setex(cache_key, 604800, json.dumps(normalized_emb.tolist()))
                    
        # All items should be filled now
        final_results = [r for r in results if r is not None]
        return np.array(final_results, dtype=np.float32)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        before_sleep=_before_retry_log
    )
    async def _call_ollama_batch(self, batch: List[str]) -> List[np.ndarray]:
        logger.debug("Calling Ollama API for batch", batch_size=len(batch))
        response = await self.client.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": batch}
        )
        response.raise_for_status()
        data = response.json()
        return [np.array(e, dtype=np.float32) for e in data.get("embeddings", [])]

    async def close(self) -> None:
        """Close the internal HTTP client."""
        await self.client.aclose()
