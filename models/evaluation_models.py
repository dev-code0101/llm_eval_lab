"""
Data models for evaluation results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
class ROUGEResult:
    """ROUGE evaluation result"""
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None
    rouge_1_precision: Optional[float] = None
    rouge_1_recall: Optional[float] = None
    rouge_2_precision: Optional[float] = None
    rouge_2_recall: Optional[float] = None
    rouge_l_precision: Optional[float] = None
    rouge_l_recall: Optional[float] = None
    average_score: float = 0.0
    explanation: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single response"""
    turn_id: int
    user_query: str
    ai_response: str
    relevance: Optional[RelevanceResult] = None
    hallucination: Optional[HallucinationResult] = None
    rouge: Optional[ROUGEResult] = None
    overall_score: float = 0.0
    evaluation_summary: str = ""
    context_used: list[str] = field(default_factory=list)
    evaluation_note: Optional[str] = None  # Optional note from the conversation record

