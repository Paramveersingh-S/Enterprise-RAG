import uuid
from typing import List

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from lexrag.ingestion.parsers.base import ParsedDocument
from lexrag.logger import get_logger
from .base import BaseChunker, Chunk, ChunkMetadata
from .token_counter import TokenCounter

logger = get_logger(__name__)

class SemanticChunker(BaseChunker):
    """Semantic chunking based on embedding similarity drops."""
    
    def __init__(self, threshold: float = 0.75) -> None:
        # Using a fast local model for semantic splitting
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # The percentile threshold roughly correlates to cosine similarity drops
        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=90, 
            embed_model=embed_model
        )
        
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        doc_id = document.metadata.content_hash or str(uuid.uuid4())
        logger.info("Semantic chunking started", document_id=doc_id)
        
        llama_doc = LlamaDocument(text=document.content, id_=doc_id)
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])
        
        chunks = []
        for i, node in enumerate(nodes):
            token_count = TokenCounter.count(node.text)
            
            # Enforce limits (simplified)
            text_to_use = node.text
            if token_count > 1024:
                text_to_use = TokenCounter.split_to_fit(node.text, 1024)[0]
                token_count = TokenCounter.count(text_to_use)
            
            section_header = self._detect_section_header(text_to_use)
            
            metadata = ChunkMetadata(
                chunk_index=i,
                total_chunks=len(nodes),
                page_numbers=[],
                section_header=section_header,
                token_count=token_count,
                character_count=len(text_to_use),
                strategy="semantic",
                overlap_with_previous=False,
                document_metadata=document.metadata
            )
            
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_chunk_id=None,
                text=text_to_use,
                metadata=metadata
            ))
            
        logger.info("Semantic chunking complete", chunks_produced=len(chunks))
        return chunks
