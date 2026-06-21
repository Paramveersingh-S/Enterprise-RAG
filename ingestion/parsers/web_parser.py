from pathlib import Path
from typing import List

from unstructured.partition.html import partition_html

from lexrag.exceptions import ParsingError
from lexrag.logger import get_logger
from lexrag.utils.hashing import compute_sha256
from .base import BaseParser, DocumentMetadata, ParsedDocument

logger = get_logger(__name__)

class WebParser(BaseParser):
    """Parser for HTML and Markdown files."""
    
    SUPPORTED_EXTENSIONS = {".html", ".htm", ".md"}
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        
    def parse(self, file_path: Path) -> ParsedDocument:
        logger.info("Parsing Web document", file_path=str(file_path))
        
        try:
            ext = file_path.suffix.lower()
            
            if ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                # HTML parsing using unstructured to strip boilerplate
                elements = partition_html(filename=str(file_path))
                content = "\n\n".join([str(el) for el in elements])
                
            file_stat = file_path.stat()
            
            metadata = DocumentMetadata(
                source_path=str(file_path),
                filename=file_path.name,
                file_type=ext[1:],
                file_size_bytes=file_stat.st_size,
                page_count=1,
                content_hash=compute_sha256(content),
                word_count=len(content.split())
            )
            
            return ParsedDocument(
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            raise ParsingError(f"Web parse failed: {str(e)}")
