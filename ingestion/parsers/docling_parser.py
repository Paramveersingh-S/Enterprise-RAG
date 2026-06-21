from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from lexrag.exceptions import ParsingError
from lexrag.logger import get_logger
from lexrag.utils.hashing import compute_sha256
from .base import BaseParser, DocumentMetadata, ParsedDocument, Table

logger = get_logger(__name__)

class DoclingParser(BaseParser):
    """Parser using Docling for complex PDFs with mixed layouts and tables."""
    
    def __init__(self) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"
        
    def parse(self, file_path: Path) -> ParsedDocument:
        logger.info("Parsing PDF with Docling", file_path=str(file_path))
        
        try:
            result = self.converter.convert(file_path)
            doc = result.document
            
            content = doc.export_to_markdown()
            
            tables = []
            if hasattr(doc, 'tables'):
                for idx, table in enumerate(doc.tables):
                    tables.append(Table(markdown=table.export_to_markdown()))
            
            file_stat = file_path.stat()
            metadata = DocumentMetadata(
                source_path=str(file_path),
                filename=file_path.name,
                file_type="pdf",
                file_size_bytes=file_stat.st_size,
                page_count=0, # Docling abstract representation doesn't easily map to page count
                content_hash=compute_sha256(content),
                word_count=len(content.split())
            )
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                tables=tables
            )
        except Exception as e:
            raise ParsingError(f"Docling parse failed: {str(e)}")
