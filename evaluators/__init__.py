"""
Evaluation logic for relevance and hallucination detection.
"""

from .llm_evaluator import LLMEvaluator, MockLLMEvaluator
from .rouge_evaluator import ROUGEEvaluator
from .lettucedetect_evaluator import LettuceDetectEvaluator
from .unified_evaluator import UnifiedEvaluator
from models import ROUGEResult

__all__ = [
    "LLMEvaluator",
    "MockLLMEvaluator",
    "ROUGEEvaluator",
    "LettuceDetectEvaluator",
    "ROUGEResult",
    "UnifiedEvaluator"
]

