import asyncio
from pathlib import Path
from celery import Celery

from lexrag.config import settings
from lexrag.ingestion.pipeline import IngestionPipeline
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
    
    async def process():
        await pipeline.initialize()
        results = []
        for path_str in file_paths:
            res = await pipeline.ingest_file(Path(path_str))
            results.append({"file": path_str, "status": res.status})
        return results
        
    return run_async(process())
