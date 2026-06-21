import math
import time
from typing import List, Optional

from sentence_transformers import CrossEncoder

from lexrag.config import settings
from lexrag.logger import get_logger
from lexrag.retrieval.hybrid import RetrievalResult

logger = get_logger(__name__)

class CrossEncoderReranker:
    """Cross-encoder re-ranker using BGE Reranker v2.
    Implemented as a Singleton to load the model only once.
    """
    
    _instance: Optional['CrossEncoderReranker'] = None
    
    def __new__(cls) -> 'CrossEncoderReranker':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self) -> None:
        if not self._initialized:
            logger.info("Initializing CrossEncoder", model=settings.reranker_model)
            # Load locally via sentence_transformers
            self.model = CrossEncoder(settings.reranker_model, max_length=512)
            self._initialized = True
            
    def _sigmoid(self, x: float) -> float:
        """Normalize raw logits to probability [0, 1]."""
        return 1 / (1 + math.exp(-x))
        
    def rerank(
        self, 
        query: str, 
        results: List[RetrievalResult], 
        top_n: int = 5, 
        min_relevance_threshold: float = 0.3
    ) -> List[RetrievalResult]:
        """Re-rank retrieval results using the cross-encoder model."""
        
        if not results:
            return []
            
        start_time = time.perf_counter()
        
        # Build pairs. Prefer parent_text for more context if available.
        pairs = [(query, r.parent_text or r.text) for r in results]
        
        # Predict scores
        logits = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        
        # Normalize to probabilities using sigmoid
        scores = [self._sigmoid(float(logit)) for logit in logits]
        
        # Attach new scores
        for r, score in zip(results, scores):
            r.score = score
            
        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Apply threshold filtering
        filtered_results = [r for r in results if r.score >= min_relevance_threshold]
        
        if not filtered_results and results:
            logger.warning("All results below relevance threshold, returning top-1 anyway", query=query)
            filtered_results = [results[0]]
            
        # Take top_n and assign ranks
        final_results = filtered_results[:top_n]
        for i, r in enumerate(final_results):
            r.reranker_rank = i + 1
            
        # Statistics for logging
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Compute average absolute rank change
        rank_changes = [abs(r.rank - r.reranker_rank) for r in final_results if r.reranker_rank is not None]
        avg_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0.0
        
        logger.info(
            "Reranking complete",
            query=query,
            input_count=len(results),
            output_count=len(final_results),
            top_score=final_results[0].score if final_results else 0.0,
            bottom_score=final_results[-1].score if final_results else 0.0,
            mean_score=sum(r.score for r in final_results) / len(final_results) if final_results else 0.0,
            latency_ms=latency_ms,
            avg_rank_change=avg_rank_change
        )
        
        return final_results
