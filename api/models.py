from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for vector search")
    
class SourceDoc(BaseModel):
    document_id: str
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    relevance_score: float
    hallucination_score: float
    
class IngestRequest(BaseModel):
    file_paths: List[str] = Field(..., description="Absolute paths to files to ingest")
    
class IngestResponse(BaseModel):
    task_id: str
    status: str
