import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from lexrag.ingestion.parsers.base import DocumentMetadata, ParsedDocument

@dataclass
class ChunkMetadata:
    chunk_index: int
    total_chunks: int
    page_numbers: List[int]
    section_header: Optional[str]
    token_count: int
    character_count: int
    strategy: str
    overlap_with_previous: bool
    document_metadata: DocumentMetadata
    window_text: Optional[str] = None

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    parent_chunk_id: Optional[str]
    text: str
    metadata: ChunkMetadata

class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """Split a parsed document into chunks."""
        pass
        
    def _detect_section_header(self, text: str) -> Optional[str]:
        """Detect a section header in the text."""
        lines = text.split("\n")
        
        # Check from bottom up to get the most recent header
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
                
            # Markdown heading
            if re.match(r"^#{1,6}\s+(.+)$", line):
                return line.strip("# ")
                
            # ALL CAPS heading (at least 5 characters)
            if re.match(r"^[A-Z][A-Z ]{5,}$", line):
                return line
                
            # Numbered heading e.g., 1., 1.1, 1.2.3
            if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
                return line
                
        return None
