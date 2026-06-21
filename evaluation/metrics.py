from typing import List, Any

# RAGAS metrics for evaluation
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

def get_standard_metrics() -> List[Any]:
    """Return the standard suite of RAGAS metrics for Enterprise RAG."""
    return [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

# Note: Custom metrics like EnterpriseComplianceMetric would be defined here
# by inheriting from ragas.metrics.base.Metric and implementing _score()
