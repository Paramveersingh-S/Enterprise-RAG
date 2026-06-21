from fastapi import APIRouter, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph

from lexrag.generation.state import AgentState
from lexrag.logger import get_logger
from .models import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from .dependencies import get_workflow
from .worker import ingest_documents_task

logger = get_logger(__name__)
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    workflow: CompiledStateGraph = Depends(get_workflow)
):
    """Execute full Graph RAG pipeline for a given query."""
    logger.info("Received API query", query=request.query)
    
    initial_state = AgentState(
        question=request.query,
        documents=[],
        graph_context=None,
        answer="",
        sources=[],
        hallucination_score=0.0,
        relevance_score=0.0,
        rewrite_count=0
    )
    
    try:
        final_state = await workflow.ainvoke(initial_state)
        
        return QueryResponse(
            answer=final_state.get("answer", ""),
            sources=final_state.get("sources", []),
            relevance_score=final_state.get("relevance_score", 0.0),
            hallucination_score=final_state.get("hallucination_score", 1.0)
        )
    except Exception as e:
        logger.error("Error executing query", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """Trigger background ingestion of documents via Celery."""
    logger.info("Triggering ingestion task", file_count=len(request.file_paths))
    
    task = ingest_documents_task.delay(request.file_paths)
    
    return IngestResponse(
        task_id=task.id,
        status="Processing"
    )
