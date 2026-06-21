import asyncio
import json
from typing import Any, List, Optional
import numpy as np
from redis.asyncio import Redis

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from lexrag.config import settings
from lexrag.logger import get_logger
from lexrag.exceptions import EmbeddingError
from lexrag.utils.hashing import compute_sha256

logger = get_logger(__name__)

class BGEEncoder:
    """Production-grade embedding pipeline using HuggingFace Cloud Inference API."""
    
    def __init__(
        self, 
        api_key: str = settings.huggingface_api_key, 
        model: str = settings.embedding_model, 
        cache: Optional[Redis] = None
    ) -> None:
        if not api_key:
            logger.warning("HUGGINGFACE_API_KEY is not set. Embedding calls will fail.")
            
        self.model = model
        self.cache = cache
        
        self.hf_embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=api_key,
            model=self.model,
        )
        
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
            
        # 2. Batch encode cache misses using LangChain wrapper
        if cache_misses:
            try:
                # Langchain currently doesn't have true async for HF API, so we run in threadpool
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    self.hf_embeddings.embed_documents, 
                    cache_misses
                )
            except Exception as e:
                logger.error("Failed to generate embeddings from HuggingFace", error=str(e))
                raise EmbeddingError(f"HuggingFace embedding failed: {str(e)}")
            
            # 3. Store new embeddings in Redis and reconstruct
            for j, emb in enumerate(embeddings):
                # L2 Normalization
                norm = np.linalg.norm(emb)
                normalized_emb = np.array(emb, dtype=np.float32)
                normalized_emb = normalized_emb / norm if norm > 0 else normalized_emb
                
                orig_idx = cache_miss_indices[j]
                results[orig_idx] = normalized_emb
                
                # Assert dimension
                if len(normalized_emb) != settings.qdrant_vector_size:
                    raise EmbeddingError(
                        f"Embedding dimension mismatch. Expected {settings.qdrant_vector_size}, got {len(normalized_emb)}"
                    )
                
                if self.cache:
                    text_hash = compute_sha256(cache_misses[j])[:16]
                    cache_key = f"embedding_cache:{text_hash}"
                    # 7 days TTL
                    await self.cache.setex(cache_key, 604800, json.dumps(normalized_emb.tolist()))
                    
        final_results = [r for r in results if r is not None]
        return np.array(final_results, dtype=np.float32)

    async def close(self) -> None:
        """Cleanup resources."""
        pass
