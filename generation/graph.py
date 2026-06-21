from typing import Any
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lexrag.logger import get_logger
from .state import AgentState
from .nodes import RAGNodes

logger = get_logger(__name__)

def create_rag_graph(nodes: RAGNodes) -> CompiledStateGraph:
    """Create and compile the LangGraph workflow."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("graph_fetch", nodes.graph_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("evaluate", nodes.evaluate_node)
    workflow.add_node("rewrite", nodes.rewrite_node)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "graph_fetch")
    workflow.add_edge("graph_fetch", "generate")
    workflow.add_edge("generate", "evaluate")
    
    # Conditional routing based on evaluation
    def evaluate_routing(state: AgentState) -> str:
        rel_score = state.get("relevance_score", 0.0)
        hal_score = state.get("hallucination_score", 1.0)
        rewrite_count = state.get("rewrite_count", 0)
        
        # If acceptable or we've retried too many times, finish
        if (rel_score > 0.7 and hal_score < 0.3) or rewrite_count >= 2:
            return "end"
            
        return "rewrite"
        
    workflow.add_conditional_edges(
        "evaluate",
        evaluate_routing,
        {
            "end": END,
            "rewrite": "rewrite"
        }
    )
    
    # Loop back
    workflow.add_edge("rewrite", "retrieve")
    
    # Compile
    return workflow.compile()
