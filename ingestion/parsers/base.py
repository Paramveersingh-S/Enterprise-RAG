from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class ImageRef:
    image_id: str
    path: str
    description: Optional[str] = None

@dataclass
class Table:
    markdown: str
    html: Optional[str] = None
    title: Optional[str] = None

@dataclass
class DocumentMetadata:
    source_path: str
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""
    language: str = "en"
    word_count: int = 0

@dataclass
class ParsedDocument:
    content: str
    metadata: DocumentMetadata
    tables: List[Table] = field(default_factory=list)
    images: List[ImageRef] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)

class BaseParser(ABC):
    """Abstract base class for all document parsers."""
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        pass
        
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse the file and return a ParsedDocument."""
        pass
