import json
import uuid
from typing import Dict, Any, List
from pathlib import Path

import mlflow
from datasets import Dataset
from ragas import evaluate

from lexrag.logger import get_logger
from lexrag.generation.graph import create_rag_graph
from lexrag.generation.nodes import RAGNodes
from lexrag.generation.state import AgentState
from .metrics import get_standard_metrics

logger = get_logger(__name__)

class EvaluationRunner:
    """Runs programmatic evaluation using Ragas and logs to MLflow."""
    
    def __init__(self, rag_nodes: RAGNodes, experiment_name: str = "Enterprise_RAG_Eval"):
        self.graph = create_rag_graph(rag_nodes)
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    async def run_evaluation(self, testset_path: Path) -> Dict[str, float]:
        """Run evaluation on a JSONL testset containing question and ground_truth."""
        logger.info("Starting evaluation run", testset_path=str(testset_path))
        
        if not testset_path.exists():
            raise FileNotFoundError(f"Testset not found at {testset_path}")
            
        questions: List[str] = []
        ground_truths: List[str] = []
        
        with open(testset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue
                data = json.loads(line)
                questions.append(data["question"])
                ground_truths.append(data["ground_truth"])
                
        answers: List[str] = []
        contexts: List[List[str]] = []
        
        for q in questions:
            initial_state = AgentState(
                question=q,
                documents=[],
                graph_context=None,
                answer="",
                sources=[],
                hallucination_score=0.0,
                relevance_score=0.0,
                rewrite_count=0
            )
            
            final_state = await self.graph.ainvoke(initial_state)
            answers.append(final_state.get("answer", ""))
            
            # Ragas expects contexts as list of strings
            ctx_texts = [d.text for d in final_state.get("documents", [])]
            contexts.append(ctx_texts)
            
        # Build dataset for RAGAS
        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data_dict)
        
        # Run RAGAS
        logger.info("Running RAGAS evaluation")
        result = evaluate(
            dataset=dataset,
            metrics=get_standard_metrics()
        )
        
        # Log to MLflow
        run_name = f"eval_{uuid.uuid4().hex[:8]}"
        with mlflow.start_run(run_name=run_name):
            for metric_name, score in result.items():
                mlflow.log_metric(metric_name, score)
            logger.info("Metrics logged to MLflow", run_name=run_name)
                
        return result
