from typing import Any, Dict, List, Optional

from lexrag.logger import get_logger
from .hybrid import HybridRetriever, RetrievalResult
from .reranker import CrossEncoderReranker

logger = get_logger(__name__)

class RetrievalPipeline:
    """Composes hybrid retrieval and cross-encoder re-ranking into a single pipeline."""
    
    def __init__(self, hybrid_retriever: HybridRetriever, reranker: CrossEncoderReranker) -> None:
        self.retriever = hybrid_retriever
        self.reranker = reranker
        
    async def retrieve_and_rerank(
        self, 
        query: str, 
        top_k: int = 20, 
        top_n: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None,
        min_relevance_threshold: float = 0.3
    ) -> List[RetrievalResult]:
        """Execute the full retrieval and reranking pipeline."""
        
        logger.info("Executing retrieval pipeline", query=query)
        
        # 1. Base Retrieval (Hybrid)
        initial_results = await self.retriever.retrieve(
            query=query, 
            top_k=top_k, 
            filter_dict=filter_dict
        )
        
        if not initial_results:
            logger.info("No results found in initial retrieval")
            return []
            
        # 2. Re-ranking (Cross-Encoder)
        final_results = self.reranker.rerank(
            query=query,
            results=initial_results,
            top_n=top_n,
            min_relevance_threshold=min_relevance_threshold
        )
        
        return final_results
