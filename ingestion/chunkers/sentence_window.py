import uuid
from typing import List

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import Document as LlamaDocument

from lexrag.ingestion.parsers.base import ParsedDocument
from lexrag.logger import get_logger
from .base import BaseChunker, Chunk, ChunkMetadata
from .token_counter import TokenCounter

logger = get_logger(__name__)

class SentenceWindowChunker(BaseChunker):
    """Chunking that extracts single sentences but stores context window."""
    
    def __init__(self, window_size: int = 3) -> None:
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        doc_id = document.metadata.content_hash or str(uuid.uuid4())
        logger.info("Sentence window chunking started", document_id=doc_id)
        
        llama_doc = LlamaDocument(text=document.content, id_=doc_id)
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])
        
        chunks = []
        for i, node in enumerate(nodes):
            section_header = self._detect_section_header(node.text)
            
            metadata = ChunkMetadata(
                chunk_index=i,
                total_chunks=len(nodes),
                page_numbers=[],
                section_header=section_header,
                token_count=TokenCounter.count(node.text),
                character_count=len(node.text),
                strategy="sentence_window",
                overlap_with_previous=True,
                document_metadata=document.metadata,
                window_text=node.metadata.get("window")
            )
            
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_chunk_id=None,
                text=node.text,
                metadata=metadata
            ))
            
        logger.info("Sentence window chunking complete", chunks_produced=len(chunks))
        return chunks
