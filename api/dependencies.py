from functools import lru_cache
from typing import AsyncGenerator

from langgraph.graph.state import CompiledStateGraph

from lexrag.embeddings.encoder import BGEEncoder
from lexrag.embeddings.dual_encoder import DualEncoder
from lexrag.retrieval.vector_store import QdrantVectorStore
from lexrag.retrieval.hybrid import HybridRetriever
from lexrag.retrieval.reranker import CrossEncoderReranker
from lexrag.retrieval.pipeline import RetrievalPipeline
from lexrag.graph.store import Neo4jGraphStore
from lexrag.generation.nodes import RAGNodes
from lexrag.generation.graph import create_rag_graph

class AppDependencies:
    """Singleton container for all stateful application services."""
    def __init__(self) -> None:
        self.encoder = BGEEncoder()
        self.dual_encoder = DualEncoder(self.encoder)
        self.vector_store = QdrantVectorStore()
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.dual_encoder)
        self.reranker = CrossEncoderReranker()
        self.retrieval_pipeline = RetrievalPipeline(self.hybrid_retriever, self.reranker)
        self.graph_store = Neo4jGraphStore()
        self.rag_nodes = RAGNodes(self.retrieval_pipeline, self.graph_store)
        self.workflow = create_rag_graph(self.rag_nodes)

@lru_cache()
def get_app_dependencies() -> AppDependencies:
    return AppDependencies()

async def get_workflow() -> AsyncGenerator[CompiledStateGraph, None]:
    """FastAPI dependency for injecting the LangGraph workflow."""
    deps = get_app_dependencies()
    yield deps.workflow
