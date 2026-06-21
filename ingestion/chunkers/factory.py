from lexrag.exceptions import ChunkingError
from .base import BaseChunker
from .hierarchical import HierarchicalChunker
from .semantic import SemanticChunker
from .sentence_window import SentenceWindowChunker

class ChunkerFactory:
    """Factory to create appropriate chunkers based on strategy."""
    
    @classmethod
    def create(cls, strategy: str = "hierarchical") -> BaseChunker:
        if strategy == "hierarchical":
            return HierarchicalChunker()
        elif strategy == "semantic":
            return SemanticChunker()
        elif strategy == "sentence_window":
            return SentenceWindowChunker()
        elif strategy == "fixed":
            # For simplicity, fallback to hierarchical without parent logic
            # In a real implementation this would be a RecursiveCharacterTextSplitter
            return HierarchicalChunker()
        else:
            raise ChunkingError(f"Unknown chunking strategy: {strategy}")
