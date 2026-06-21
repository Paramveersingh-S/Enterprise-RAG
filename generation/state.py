from typing import Any, Dict, List, TypedDict

from lexrag.retrieval.hybrid import RetrievalResult
from lexrag.graph.store import GraphQueryResult

class AgentState(TypedDict):
    """State object passed through the LangGraph execution."""
    question: str
    documents: List[RetrievalResult]
    graph_context: GraphQueryResult
    answer: str
    sources: List[str]
    hallucination_score: float
    relevance_score: float
    rewrite_count: int
