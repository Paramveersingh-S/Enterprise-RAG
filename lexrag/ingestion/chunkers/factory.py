from lexrag.exceptions import ChunkingError

class ChunkerFactory:
    """Factory to create appropriate chunkers based on strategy."""
    
    @classmethod
    def create(cls, strategy: str = "hierarchical"):
        if strategy == "hierarchical":
            from .hierarchical import HierarchicalChunker
            return HierarchicalChunker()
        elif strategy == "semantic":
            from .semantic import SemanticChunker
            return SemanticChunker()
        elif strategy == "sentence_window":
            from .sentence_window import SentenceWindowChunker
            return SentenceWindowChunker()
        elif strategy == "fixed":
            # For simplicity, fallback to hierarchical without parent logic
            from .hierarchical import HierarchicalChunker
            return HierarchicalChunker()
        else:
            raise ChunkingError(f"Unknown chunking strategy: {strategy}")
