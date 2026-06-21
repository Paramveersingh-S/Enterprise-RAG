import asyncio
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiosqlite
from redis.asyncio import Redis

from lexrag.config import settings
from lexrag.logger import get_logger
from lexrag.utils.hashing import compute_sha256
from lexrag.exceptions import ParsingError
from .parsers.base import ParsedDocument
from .parsers.registry import ParserRegistry

logger = get_logger(__name__)

@dataclass
class IngestionResult:
    parsed_document: Optional[ParsedDocument]
    status: str
    file_path: str
    document_id: Optional[str] = None
    error: Optional[str] = None

class IngestionPipeline:
    """Pipeline for ingesting and parsing documents."""
    
    def __init__(self, use_docling: bool = True):
        self.registry = ParserRegistry(use_docling_for_pdf=use_docling)
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self.db_path = "lexrag_failures.db"
        
    async def initialize(self) -> None:
        """Initialize the failure database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS parse_failures (
                    id TEXT PRIMARY KEY,
                    file_path TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    stack_trace TEXT,
                    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await db.commit()

    async def log_failure(self, file_path: str, error_code: str, error_message: str, stack_trace: str) -> None:
        """Log a parsing failure to the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO parse_failures (id, file_path, error_code, error_message, stack_trace) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), file_path, error_code, error_message, stack_trace)
            )
            await db.commit()
            
    async def ingest_file(self, file_path: Path) -> IngestionResult:
        """Ingest a single file."""
        logger.info("Starting ingestion", file_path=str(file_path))
        
        try:
            # 1. Compute hash and check deduplication
            with open(file_path, "rb") as f:
                content_bytes = f.read()
                
            file_hash = compute_sha256(content_bytes)
            redis_key = f"doc:hash:{file_hash}"
            
            existing_doc_id = await self.redis.get(redis_key)
            if existing_doc_id:
                logger.info("Duplicate document skipped", file_path=str(file_path), doc_id=existing_doc_id)
                return IngestionResult(parsed_document=None, status="skipped_duplicate", file_path=str(file_path), document_id=existing_doc_id)
                
            # 2. Select parser
            parser = self.registry.get_parser(file_path)
            
            # 3. Parse document
            parsed_doc = parser.parse(file_path)
            
            # 4. Store hash
            document_id = str(uuid.uuid4())
            await self.redis.set(redis_key, document_id)
            
            # 5. Log metadata
            logger.info("Document parsed successfully", 
                        file_path=str(file_path),
                        document_id=document_id,
                        metadata=parsed_doc.metadata.__dict__)
                        
            return IngestionResult(parsed_document=parsed_doc, status="success", file_path=str(file_path), document_id=document_id)
            
        except Exception as e:
            error_msg = str(e)
            stack = traceback.format_exc()
            error_code = getattr(e, "code", "LEXRAG_PARSE_UNKNOWN")
            
            logger.error("Ingestion failed", file_path=str(file_path), error=error_msg)
            await self.log_failure(str(file_path), error_code, error_msg, stack)
            
            return IngestionResult(parsed_document=None, status="error", file_path=str(file_path), error=error_msg)

    async def ingest_directory(self, dir_path: Path, recursive: bool = True) -> List[IngestionResult]:
        """Ingest all supported files in a directory."""
        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a valid directory.")
            
        pattern = "**/*" if recursive else "*"
        files = [f for f in dir_path.glob(pattern) if f.is_file()]
        
        # Filter supported files based on registry parsers
        supported_files = []
        for f in files:
            try:
                if self.registry.get_parser(f):
                    supported_files.append(f)
            except ParsingError:
                pass
                
        logger.info(f"Found {len(supported_files)} supported files in {dir_path}")
        
        # Run concurrently, max 4 at a time
        semaphore = asyncio.Semaphore(4)
        
        async def bounded_ingest(file_path: Path) -> IngestionResult:
            async with semaphore:
                return await self.ingest_file(file_path)
                
        tasks = [bounded_ingest(f) for f in supported_files]
        results = await asyncio.gather(*tasks)
        
        return list(results)

if __name__ == "__main__":
    # Test script for CLI usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents.")
    parser.add_argument("--file", type=str, help="Single file to ingest")
    parser.add_argument("--directory", type=str, help="Directory to ingest")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    pipeline = IngestionPipeline()
    
    async def main() -> None:
        await pipeline.initialize()
        if args.file:
            result = await pipeline.ingest_file(Path(args.file))
            print(f"Result: {result.status} for {result.file_path}")
            if result.parsed_document and args.verbose:
                print(result.parsed_document.metadata)
        elif args.directory:
            results = await pipeline.ingest_directory(Path(args.directory))
            success = sum(1 for r in results if r.status == "success")
            print(f"Ingested {success}/{len(results)} files successfully.")
            
    asyncio.run(main())
