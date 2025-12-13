"""
Data models for LLM evaluation pipeline.
"""

from .evaluation_models import (
    EvaluationScore,
    RelevanceResult,
    HallucinationResult,
    ROUGEResult,
    EvaluationResult
)
from .conversation_models import (
    ConversationTurn,
    Conversation,
    VectorData,
    VectorInfo,
    VectorSources,
    ContextVectorsData,
    ContextVectorsResponse
)
from .json_schemas import (
    ConversationTurnDict,
    ConversationDict,
    VectorInfoDict,
    VectorSourcesDict,
    VectorDataDict,
    ContextVectorsDataDict,
    ContextVectorsResponseDict
)

__all__ = [
    # Evaluation models
    "EvaluationScore",
    "RelevanceResult",
    "HallucinationResult",
    "ROUGEResult",
    "EvaluationResult",
    # Conversation dataclass models
    "ConversationTurn",
    "Conversation",
    "VectorData",
    "VectorInfo",
    "VectorSources",
    "ContextVectorsData",
    "ContextVectorsResponse",
    # JSON TypedDict schemas
    "ConversationTurnDict",
    "ConversationDict",
    "VectorInfoDict",
    "VectorSourcesDict",
    "VectorDataDict",
    "ContextVectorsDataDict",
    "ContextVectorsResponseDict",
]

