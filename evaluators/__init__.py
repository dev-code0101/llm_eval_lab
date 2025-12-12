"""
Evaluation logic for relevance and hallucination detection.
"""

from .llm_evaluator import LLMEvaluator, MockLLMEvaluator

__all__ = [
    "LLMEvaluator",
    "MockLLMEvaluator"
]

