"""
Data models for conversation and context vector JSON structures.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ConversationTurn:
    """A single turn in a conversation"""
    turn: int
    sender_id: int
    role: str  # "AI/Chatbot" or "User"
    message: str
    created_at: str  # ISO datetime string
    evaluation_note: Optional[str] = None  # Optional evaluation note


@dataclass
class Conversation:
    """Complete conversation structure"""
    chat_id: int
    user_id: int
    conversation_turns: List[ConversationTurn]

@dataclass
class VectorInfo:
    """Information about a vector's relevance score"""
    score: float
    vector_id: int
    tokens_count: int

@dataclass
class VectorData:
    """A single context vector from the knowledge base"""
    id: int
    text: str
    source_url: Optional[str] = None
    tokens: Optional[int] = None
    created_at: Optional[str] = None
    source_type: Optional[int] = None  # Optional source type field

@dataclass
class VectorSources:
    """Sources information for context vectors"""
    message_id: Optional[int] = None
    vector_ids: List[int] = field(default_factory=list)
    vectors_info: List[VectorInfo] = field(default_factory=list)
    vectors_used: List[int] = field(default_factory=list)
    final_response: List[str] = field(default_factory=list)


@dataclass
class ContextVectorsData:
    """Data section of context vectors response"""
    vector_data: List[VectorData] = field(default_factory=list)
    sources: VectorSources = field(default_factory=VectorSources)


@dataclass
class ContextVectorsResponse:
    """Complete context vectors response structure"""
    status: str
    status_code: int
    message: str
    data: ContextVectorsData

