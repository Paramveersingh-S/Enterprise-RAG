from pathlib import Path
from typing import List

from unstructured.partition.auto import partition
from unstructured.documents.elements import Title, Table as UnstructuredTable, ListItem

from lexrag.exceptions import ParsingError
from lexrag.logger import get_logger
from lexrag.utils.hashing import compute_sha256
from .base import BaseParser, DocumentMetadata, ParsedDocument, Table

logger = get_logger(__name__)

class OfficeParser(BaseParser):
    """Parser for DOCX, PPTX, and XLSX using Unstructured."""
    
    SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        
    def parse(self, file_path: Path) -> ParsedDocument:
        logger.info("Parsing Office document", file_path=str(file_path))
        
        try:
            elements = partition(filename=str(file_path))
            
            content_parts: List[str] = []
            tables: List[Table] = []
            
            for element in elements:
                if isinstance(element, Title):
                    content_parts.append(f"## {element.text}")
                elif isinstance(element, UnstructuredTable):
                    # Unstructured provides text_as_html for tables
                    html_table = getattr(element.metadata, "text_as_html", element.text)
                    tables.append(Table(markdown=element.text, html=html_table))
                    content_parts.append(element.text)
                elif isinstance(element, ListItem):
                    content_parts.append(f"- {element.text}")
                else:
                    content_parts.append(element.text)
                    
            full_content = "\n\n".join(content_parts)
            file_stat = file_path.stat()
            
            metadata = DocumentMetadata(
                source_path=str(file_path),
                filename=file_path.name,
                file_type=file_path.suffix.lower()[1:],
                file_size_bytes=file_stat.st_size,
                page_count=0, # Unstructured doesn't easily map to page count across formats
                content_hash=compute_sha256(full_content),
                word_count=len(full_content.split())
            )
            
            return ParsedDocument(
                content=full_content,
                metadata=metadata,
                tables=tables
            )
            
        except Exception as e:
            raise ParsingError(f"Office parse failed: {str(e)}")
