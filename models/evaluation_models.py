"""
Data models for evaluation results.
"""

from dataclasses import dataclass, field
from enum import Enum


class EvaluationScore(Enum):
    """Standardized scoring levels"""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    FAIL = 1


@dataclass
class RelevanceResult:
    """Relevance & Completeness evaluation result"""
    score: int  # 1-5
    is_relevant: bool
    is_complete: bool
    relevance_explanation: str
    completeness_explanation: str
    missing_aspects: list[str] = field(default_factory=list)


@dataclass
class HallucinationResult:
    """Hallucination / Factual Accuracy evaluation result"""
    score: int  # 1-5
    has_hallucination: bool
    factual_accuracy: float  # 0.0 - 1.0
    hallucinated_claims: list[dict] = field(default_factory=list)
    verified_claims: list[dict] = field(default_factory=list)
    explanation: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single response"""
    turn_id: int
    user_query: str
    ai_response: str
    relevance: RelevanceResult
    hallucination: HallucinationResult
    overall_score: float
    evaluation_summary: str
    context_used: list[str] = field(default_factory=list)

