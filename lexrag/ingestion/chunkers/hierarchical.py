import uuid
from typing import List

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import Document as LlamaDocument

from lexrag.ingestion.parsers.base import ParsedDocument
from lexrag.logger import get_logger
from .base import BaseChunker, Chunk, ChunkMetadata
from .token_counter import TokenCounter

logger = get_logger(__name__)

class HierarchicalChunker(BaseChunker):
    """Hierarchical chunking: large parent chunks and smaller child chunks."""
    
    def __init__(self) -> None:
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512],
            chunk_overlap=50
        )
        
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        doc_id = document.metadata.content_hash or str(uuid.uuid4())
        logger.info("Hierarchical chunking started", document_id=doc_id)
        
        llama_doc = LlamaDocument(text=document.content, id_=doc_id)
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])
        
        leaf_nodes = get_leaf_nodes(nodes)
        parent_nodes = {n.node_id: n for n in nodes if n not in leaf_nodes}
        
        chunks = []
        for i, node in enumerate(leaf_nodes):
            parent_id = node.parent_node.node_id if node.parent_node else None
            parent_text = parent_nodes[parent_id].text if parent_id and parent_id in parent_nodes else None
            
            section_header = self._detect_section_header(node.text)
            if not section_header and parent_text:
                section_header = self._detect_section_header(parent_text)
            
            metadata = ChunkMetadata(
                chunk_index=i,
                total_chunks=len(leaf_nodes),
                page_numbers=[], # Would require mapping back to original text ranges
                section_header=section_header,
                token_count=TokenCounter.count(node.text),
                character_count=len(node.text),
                strategy="hierarchical",
                overlap_with_previous=True if i > 0 else False,
                document_metadata=document.metadata,
                window_text=parent_text
            )
            
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                parent_chunk_id=parent_id,
                text=node.text,
                metadata=metadata
            ))
            
        logger.info("Hierarchical chunking complete", chunks_produced=len(chunks))
        return chunks
