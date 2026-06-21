from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client.http import models

from lexrag.logger import get_logger
from lexrag.exceptions import RetrievalError
from lexrag.embeddings.dual_encoder import DualEncoder
from .vector_store import QdrantVectorStore

logger = get_logger(__name__)

@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    text: str
    parent_text: Optional[str]
    score: float
    rank: int
    metadata: Dict[str, Any]
    retrieval_method: str
    reranker_rank: Optional[int] = None

class HybridRetriever:
    """Retriever that combines dense and sparse search using Reciprocal Rank Fusion."""
    
    def __init__(self, vector_store: QdrantVectorStore, dual_encoder: DualEncoder) -> None:
        self.vector_store = vector_store
        self.encoder = dual_encoder
        
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 20, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval."""
        logger.info("Starting hybrid retrieval", query=query, top_k=top_k)
        
        # 1. Encode Query
        dense_vecs, sparse_vecs = await self.encoder.encode([query])
        dense_vec = dense_vecs[0].tolist()
        sparse_vec = sparse_vecs[0]
        
        # 2. Build Qdrant Filter
        qdrant_filter = None
        if filter_dict:
            must_conditions = []
            for key, val in filter_dict.items():
                must_conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=val)
                ))
            qdrant_filter = models.Filter(must=must_conditions)
            
        # 3. Query Qdrant with Prefetch and Fusion
        try:
            results = await self.vector_store.client.query_points(
                collection_name=self.vector_store.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=top_k * 2,
                        filter=qdrant_filter
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"]
                        ),
                        using="sparse",
                        limit=top_k * 2,
                        filter=qdrant_filter
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant query failed: {str(e)}")
            
        # 4. Map Results
        retrieval_results = []
        for i, point in enumerate(results.points):
            payload = point.payload or {}
            retrieval_results.append(RetrievalResult(
                chunk_id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                parent_text=payload.get("window_text"),
                score=point.score,
                rank=i + 1,
                metadata=payload,
                retrieval_method="hybrid_rrf"
            ))
            
        logger.info("Hybrid retrieval complete", count=len(retrieval_results))
        return retrieval_results
