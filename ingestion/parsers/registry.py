from pathlib import Path
from typing import List

from lexrag.exceptions import ParsingError
from lexrag.logger import get_logger

from .base import BaseParser
from .pdf_parser import PDFParser
from .docling_parser import DoclingParser
from .office_parser import OfficeParser
from .web_parser import WebParser

logger = get_logger(__name__)

class ParserRegistry:
    """Registry to manage and select document parsers."""
    
    def __init__(self, use_docling_for_pdf: bool = True) -> None:
        self.parsers: List[BaseParser] = []
        
        # Docling is preferred for PDFs by default
        if use_docling_for_pdf:
            self.parsers.append(DoclingParser())
        else:
            self.parsers.append(PDFParser())
            
        self.parsers.append(OfficeParser())
        self.parsers.append(WebParser())
        
    def get_parser(self, file_path: Path) -> BaseParser:
        """Get the appropriate parser for the given file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
                
        raise ParsingError(f"No parser found for file type: {file_path.suffix}")
