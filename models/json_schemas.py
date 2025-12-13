"""
TypedDict schemas for raw JSON structures.

These TypedDict definitions represent the structure of JSON data
after parsing but before conversion to dataclass models. They provide
type safety and IDE autocomplete for raw dictionary access.
"""

from typing import TypedDict, Optional, List


# Conversation JSON schemas
class ConversationTurnDict(TypedDict, total=False):
    """Raw JSON structure for a conversation turn"""
    turn: int
    sender_id: int
    role: str
    message: str
    created_at: str
    evaluation_note: Optional[str]


class ConversationDict(TypedDict, total=False):
    """Raw JSON structure for a conversation"""
    chat_id: int
    user_id: int
    conversation_turns: List[ConversationTurnDict]


# Context Vector JSON schemas
class VectorInfoDict(TypedDict, total=False):
    """Raw JSON structure for vector info"""
    score: float
    vector_id: int
    tokens_count: int


class VectorSourcesDict(TypedDict, total=False):
    """Raw JSON structure for vector sources"""
    message_id: int
    vector_ids: List[int]
    vectors_info: List[VectorInfoDict]
    vectors_used: List[int]
    final_response: List[str]


class VectorDataDict(TypedDict, total=False):
    """Raw JSON structure for a single vector"""
    id: int
    text: str
    source_url: Optional[str]
    tokens: int
    created_at: str
    source_type: Optional[int]


class ContextVectorsDataDict(TypedDict, total=False):
    """Raw JSON structure for context vectors data"""
    vector_data: List[VectorDataDict]
    sources: VectorSourcesDict


class ContextVectorsResponseDict(TypedDict, total=False):
    """Raw JSON structure for the full context vectors response"""
    status: str
    status_code: int
    message: str
    data: ContextVectorsDataDict

