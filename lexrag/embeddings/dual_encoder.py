import re
from typing import Any, Dict, List, Tuple
import numpy as np

from lexrag.logger import get_logger
from .encoder import BGEEncoder

logger = get_logger(__name__)

class DualEncoder:
    """Extracts both dense and sparse vectors for hybrid search.
    
    Note: Standard Ollama REST API /api/embed only returns dense vectors.
    To satisfy the sparse vector requirement natively, a local FlagEmbedding model
    could be used. Here we combine the Ollama dense embeddings with a local
    BM25/TF-IDF token hash approach to simulate the sparse vector format 
    expected by Qdrant's sparse indices.
    """
    
    def __init__(self, dense_encoder: BGEEncoder) -> None:
        self.dense_encoder = dense_encoder
        
    async def encode(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Encode texts into dense and sparse vectors."""
        logger.info("Starting dual encoding", text_count=len(texts))
        
        # 1. Get Dense Embeddings from Ollama
        dense_vecs = await self.dense_encoder.encode(texts)
        
        # 2. Get Sparse Vectors (Token frequencies)
        sparse_vecs: List[Dict[str, Any]] = []
        for text in texts:
            # Tokenize by word characters
            tokens = re.findall(r'\b\w+\b', text.lower())
            
            token_counts: Dict[int, float] = {}
            for t in tokens:
                # Use a simple string hash mod 100000 for the index space
                idx = hash(t) % 100000
                token_counts[idx] = token_counts.get(idx, 0.0) + 1.0
                
            indices = list(token_counts.keys())
            values = list(token_counts.values())
            
            # Optionally normalize sparse vectors with BM25-like weights here
            # For basic Qdrant sparse support, term frequency is a good baseline
            sparse_vecs.append({
                "indices": indices, 
                "values": values
            })
            
        return dense_vecs, sparse_vecs

    async def close(self) -> None:
        """Close the underlying dense encoder."""
        await self.dense_encoder.close()
