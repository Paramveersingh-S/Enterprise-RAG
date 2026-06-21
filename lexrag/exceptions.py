from typing import Any, Dict, Optional

class LexRAGError(Exception):
    """Base exception class for all LexRAG errors."""
    def __init__(self, message: str, code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

class IngestionError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_INGEST_001", details)

class ParsingError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_PARSE_001", details)

class ChunkingError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_CHUNK_001", details)

class EmbeddingError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_EMBED_001", details)

class RetrievalError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_RETRIEVAL_001", details)

class GraphError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_GRAPH_001", details)

class GenerationError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_GEN_001", details)

class RateLimitError(LexRAGError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LEXRAG_RATELIMIT_001", details)
