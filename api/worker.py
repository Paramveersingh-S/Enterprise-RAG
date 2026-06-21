import asyncio
from pathlib import Path
from celery import Celery

from lexrag.config import settings
from lexrag.ingestion.pipeline import IngestionPipeline
from lexrag.ingestion.chunkers.factory import ChunkerFactory
from lexrag.embeddings.encoder import BGEEncoder
from lexrag.embeddings.dual_encoder import DualEncoder
from lexrag.retrieval.vector_store import QdrantVectorStore
from lexrag.graph.store import Neo4jGraphStore
from lexrag.graph.extractor import EntityRelationExtractor
from lexrag.logger import get_logger

logger = get_logger(__name__)

celery_app = Celery(
    "lexrag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url
)

def run_async(coro):
    """Wrapper to run async pipeline in a sync celery task."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@celery_app.task(bind=True, name="ingest_documents")
def ingest_documents_task(self, file_paths: list[str]):
    """Background task to ingest documents."""
    logger.info("Starting background ingestion task", task_id=self.request.id, files=len(file_paths))
    
    pipeline = IngestionPipeline()
    chunker = ChunkerFactory.create("hierarchical")
    encoder = BGEEncoder()
    dual_encoder = DualEncoder(encoder)
    vector_store = QdrantVectorStore()
    graph_store = Neo4jGraphStore()
    extractor = EntityRelationExtractor()
    
    async def process():
        await pipeline.initialize()
        
        # Ensure collections and constraints exist
        await vector_store.create_collection()
        await graph_store.create_constraints()
        
        results = []
        for path_str in file_paths:
            logger.info(f"Processing file: {path_str}")
            res = await pipeline.ingest_file(Path(path_str))
            
            if res.status == "success" and res.parsed_document:
                logger.info("Chunking document...")
                chunks = chunker.chunk(res.parsed_document)
                
                logger.info("Extracting graph entities...")
                graph_extracts = extractor.extract(chunks)
                for extract in graph_extracts:
                    for entity in extract.entities:
                        await graph_store.upsert_entity(entity, doc_id=res.document_id)
                    for relation in extract.relations:
                        await graph_store.upsert_relation(relation)
                        
                logger.info("Generating embeddings...")
                texts = [c.text for c in chunks]
                dense_embeddings, sparse_embeddings = await dual_encoder.encode(texts)
                
                logger.info("Upserting to Qdrant...")
                await vector_store.upsert_chunks(chunks, dense_embeddings, sparse_embeddings)
                
            results.append({"file": path_str, "status": res.status})
            
        await graph_store.close()
        return results
        
    return run_async(process())
