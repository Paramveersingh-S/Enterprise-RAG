import io
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from lexrag.exceptions import ParsingError
from lexrag.logger import get_logger
from lexrag.utils.hashing import compute_sha256
from .base import BaseParser, DocumentMetadata, ParsedDocument

logger = get_logger(__name__)

class PDFParser(BaseParser):
    """Parser for PDF files using PyMuPDF and Tesseract OCR fallback."""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"
        
    def parse(self, file_path: Path) -> ParsedDocument:
        logger.info("Parsing PDF file", file_path=str(file_path))
        
        try:
            doc = fitz.open(file_path)
            if doc.is_encrypted:
                logger.warning("PDF is password protected, skipping.", file_path=str(file_path))
                raise ParsingError("PDF is password protected.")
                
            content_parts: List[str] = []
            parse_errors: List[str] = []
            ocr_pages = 0
            native_pages = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                
                if len(text.strip()) < 50:
                    # Fallback to OCR
                    try:
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        text = pytesseract.image_to_string(img)
                        ocr_pages += 1
                        logger.debug("Used OCR", page=page_num)
                    except Exception as e:
                        error_msg = f"OCR failed on page {page_num}: {str(e)}"
                        logger.error(error_msg)
                        parse_errors.append(error_msg)
                        text = ""
                else:
                    native_pages += 1
                
                if text.strip():
                    content_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                    
            full_content = "\n\n".join(content_parts)
            
            # Extract metadata
            doc_info = doc.metadata or {}
            file_stat = file_path.stat()
            
            metadata = DocumentMetadata(
                source_path=str(file_path),
                filename=file_path.name,
                file_type="pdf",
                file_size_bytes=file_stat.st_size,
                page_count=len(doc),
                title=doc_info.get("title"),
                author=doc_info.get("author"),
                content_hash=compute_sha256(full_content),
                word_count=len(full_content.split())
            )
            
            logger.info("PDF parsing complete", 
                       native_pages=native_pages, 
                       ocr_pages=ocr_pages)
                       
            return ParsedDocument(
                content=full_content,
                metadata=metadata,
                parse_errors=parse_errors
            )
            
        except Exception as e:
            if not isinstance(e, ParsingError):
                raise ParsingError(f"Failed to parse PDF: {str(e)}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()
