import hashlib
import uuid
from typing import Any, Dict, List
import numpy as np

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from lexrag.config import settings
from lexrag.logger import get_logger
from lexrag.exceptions import RetrievalError
from lexrag.ingestion.chunkers.base import Chunk

logger = get_logger(__name__)

class QdrantVectorStore:
    """Manages connection and operations with Qdrant vector database."""
    
    def __init__(self, host: str = settings.qdrant_host, port: int = settings.qdrant_port) -> None:
        self.client = AsyncQdrantClient(host=host, port=port)
        self.collection_name = settings.qdrant_collection_name
        
    async def create_collection(self) -> None:
        """Create the Qdrant collection with dense and sparse vectors and HNSW config."""
        logger.info("Ensuring Qdrant collection exists", collection=self.collection_name)
        
        try:
            collections = await self.client.get_collections()
            if any(c.name == self.collection_name for c in collections.collections):
                logger.debug("Collection already exists")
                return
                
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=settings.qdrant_vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                },
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100
                ),
                on_disk_payload=True
            )
            
            # Create payload indexes for filtering
            await self.client.create_payload_index(self.collection_name, "document_id", field_schema=models.PayloadSchemaType.KEYWORD)
            await self.client.create_payload_index(self.collection_name, "section_header", field_schema=models.PayloadSchemaType.KEYWORD)
            await self.client.create_payload_index(self.collection_name, "page_numbers", field_schema=models.PayloadSchemaType.INTEGER)
            await self.client.create_payload_index(self.collection_name, "chunk_index", field_schema=models.PayloadSchemaType.INTEGER)
            
            logger.info("Collection created with indexes")
        except Exception as e:
            raise RetrievalError(f"Failed to create collection: {str(e)}")

    async def upsert_chunks(self, chunks: List[Chunk], dense_embeddings: np.ndarray, sparse_embeddings: List[Dict[str, Any]]) -> None:
        """Upsert chunks with both dense and sparse vectors in batches."""
        logger.info("Upserting chunks to Qdrant", count=len(chunks))
        
        points = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_embeddings, sparse_embeddings):
            # Qdrant requires UUIDs or integers for point IDs
            try:
                point_id = str(uuid.UUID(chunk.chunk_id))
            except ValueError:
                hash_val = hashlib.md5(chunk.chunk_id.encode()).hexdigest()
                point_id = str(uuid.UUID(hash_val))

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "parent_chunk_id": chunk.parent_chunk_id,
                "text": chunk.text,
                "chunk_index": chunk.metadata.chunk_index,
                "total_chunks": chunk.metadata.total_chunks,
                "section_header": chunk.metadata.section_header,
                "token_count": chunk.metadata.token_count,
                "strategy": chunk.metadata.strategy,
                "window_text": chunk.metadata.window_text
            }
            
            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    )
                },
                payload=payload
            ))
            
        # Batch upsert
        batch_size = 100
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            logger.info("Upsert complete")
        except Exception as e:
            raise RetrievalError(f"Failed to upsert chunks: {str(e)}")
            
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks associated with a specific document_id."""
        logger.info("Deleting document from vector store", document_id=document_id)
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
                wait=True
            )
        except Exception as e:
            raise RetrievalError(f"Failed to delete document: {str(e)}")
