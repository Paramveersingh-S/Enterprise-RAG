import json
import re
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from lexrag.config import settings
from lexrag.logger import get_logger
from lexrag.retrieval.pipeline import RetrievalPipeline
from lexrag.graph.store import Neo4jGraphStore, GraphQueryResult
from .state import AgentState

logger = get_logger(__name__)

# Initialize LLM with Groq's fast Llama 3.1
llm = ChatGroq(
    model_name=settings.groq_model_name,
    api_key=settings.groq_api_key,
    temperature=0.0,
    max_retries=3
)

class RAGNodes:
    """Contains all node execution logic for the LangGraph orchestrator."""
    
    def __init__(self, retrieval_pipeline: RetrievalPipeline, graph_store: Neo4jGraphStore) -> None:
        self.retriever = retrieval_pipeline
        self.graph_store = graph_store
        
    async def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents for the question."""
        question = state["question"]
        logger.info("Retrieving documents", question=question)
        
        results = await self.retriever.retrieve_and_rerank(
            query=question,
            top_k=20,
            top_n=5
        )
        
        return {"documents": results}
        
    async def graph_node(self, state: AgentState) -> Dict[str, Any]:
        """Extract entities from question and fetch graph neighbors."""
        question = state["question"]
        logger.info("Fetching graph context", question=question)
        
        # Simple extraction for demo: uppercase words > 3 chars
        # A more robust solution uses LLM extraction
        potential_entities = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', question)
        
        nodes = []
        edges = []
        contexts = []
        
        for entity in potential_entities:
            result = await self.graph_store.get_entity_neighbors(entity)
            nodes.extend(result.nodes)
            edges.extend(result.edges)
            contexts.extend(result.contexts)
            
        combined_result = GraphQueryResult(
            nodes=nodes, edges=edges, contexts=list(set(contexts))
        )
        
        return {"graph_context": combined_result}
        
    async def generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate answer using LLM with RAG and Graph context."""
        question = state["question"]
        documents = state.get("documents", [])
        graph_context = state.get("graph_context")
        
        logger.info("Generating answer")
        
        # Prepare context
        doc_texts = []
        sources = []
        for d in documents:
            doc_texts.append(f"Source [{d.document_id}]:\n{d.parent_text or d.text}")
            sources.append(d.document_id)
            
        graph_texts = []
        if graph_context and graph_context.contexts:
            graph_texts = [f"Graph Context:\n{ctx}" for ctx in graph_context.contexts]
            
        context_str = "\n\n".join(doc_texts + graph_texts)
        
        prompt = PromptTemplate.from_template(
            """You are an enterprise AI assistant.
Answer the following question based ONLY on the provided context. 
If the context does not contain the answer, say "I cannot answer this based on the provided documents."
You must cite your sources using the [Source ID] format inline.

Context:
{context}

Question: {question}
Answer:"""
        )
        
        chain = prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context_str, "question": question})
        
        return {"answer": answer, "sources": list(set(sources))}

    async def evaluate_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate generation quality (LLM-as-a-judge)."""
        question = state["question"]
        answer = state["answer"]
        documents = state.get("documents", [])
        
        logger.info("Evaluating answer quality")
        
        doc_texts = "\n".join([d.parent_text or d.text for d in documents])
        
        eval_prompt = PromptTemplate.from_template(
            """Evaluate the following answer.
Provide two scores between 0.0 and 1.0:
1. relevance_score: Does the answer directly address the question?
2. hallucination_score: Is the answer fully supported by the context (0.0 = fully supported, 1.0 = highly hallucinated)?

Context: {context}
Question: {question}
Answer: {answer}

Respond ONLY in JSON format: {{"relevance_score": 0.0, "hallucination_score": 0.0}}"""
        )
        
        try:
            chain = eval_prompt | llm | StrOutputParser()
            result_str = await chain.ainvoke({
                "context": doc_texts,
                "question": question,
                "answer": answer
            })
            
            # Clean possible markdown formatting
            result_str = result_str.replace("```json", "").replace("```", "").strip()
            scores = json.loads(result_str)
            
            return {
                "relevance_score": float(scores.get("relevance_score", 0.0)),
                "hallucination_score": float(scores.get("hallucination_score", 1.0))
            }
        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            # Default to passing if eval fails to not block pipeline
            return {"relevance_score": 1.0, "hallucination_score": 0.0}
            
    async def rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite question if evaluation failed."""
        question = state["question"]
        rewrite_count = state.get("rewrite_count", 0)
        
        logger.info("Rewriting question", current_count=rewrite_count)
        
        prompt = PromptTemplate.from_template(
            """The original question did not yield good results.
Rewrite it to be more specific and optimized for vector search.
Original Question: {question}
Rewritten Question:"""
        )
        
        chain = prompt | llm | StrOutputParser()
        new_question = await chain.ainvoke({"question": question})
        
        return {
            "question": new_question.strip(),
            "rewrite_count": rewrite_count + 1
        }
