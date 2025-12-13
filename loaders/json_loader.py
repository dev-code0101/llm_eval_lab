"""
JSON data loader with cleanup for malformed JSON.
"""

import json
import re
from typing import Optional, List, Dict, Any

from models import (
    Conversation,
    ConversationTurn,
    ContextVectorsResponse,
    ContextVectorsData,
    VectorData,
    VectorSources,
    VectorInfo
)


class JSONDataLoader:
    """Loads and parses JSON files with cleanup for common issues"""
    
    def load_conversation(self, conversation_path: str) -> Conversation:
        """
        Load chat conversation from JSON file and return typed Conversation object.
        
        Args:
            conversation_path: Path to conversation JSON file
        
        Returns:
            Conversation object with typed fields
        """
        with open(conversation_path, 'r', encoding='utf-8') as f:
            content = f.read()
        data = self._parse_json_with_cleanup(content)
        
        # Convert to typed Conversation object
        turns = []
        for turn_data in data.get("conversation_turns", []):
            turn = ConversationTurn(
                turn=turn_data.get("turn"),
                sender_id=turn_data.get("sender_id"),
                role=turn_data.get("role"),
                message=turn_data.get("message", ""),
                created_at=turn_data.get("created_at", ""),
                evaluation_note=turn_data.get("evaluation_note")
            )
            turns.append(turn)
        
        return Conversation(
            chat_id=data.get("chat_id"),
            user_id=data.get("user_id"),
            conversation_turns=turns
        )
    
    def load_context_vectors(self, vectors_path: str) -> ContextVectorsResponse:
        """
        Load context vectors from JSON file and return typed ContextVectorsResponse object.
        
        Args:
            vectors_path: Path to context vectors JSON file
        
        Returns:
            ContextVectorsResponse object with typed fields
        """
        with open(vectors_path, 'r', encoding='utf-8') as f:
            content = f.read()
        data = self._parse_json_with_cleanup(content)
        
        # Parse vector_data
        vector_data_list = []
        for vec_data in data.get("data", {}).get("vector_data", []):
            vector = VectorData(
                id=vec_data.get("id"),
                text=vec_data.get("text", ""),
                source_url=vec_data.get("source_url"),
                tokens=vec_data.get("tokens"),
                created_at=vec_data.get("created_at"),
                source_type=vec_data.get("source_type")
            )
            vector_data_list.append(vector)
        
        # Parse sources
        sources_data = data.get("data", {}).get("sources", {})
        vectors_info_list = []
        for vi_data in sources_data.get("vectors_info", []):
            vi = VectorInfo(
                score=vi_data.get("score", 0.0),
                vector_id=vi_data.get("vector_id"),
                tokens_count=vi_data.get("tokens_count", 0)
            )
            vectors_info_list.append(vi)
        
        sources = VectorSources(
            message_id=sources_data.get("message_id"),
            vector_ids=sources_data.get("vector_ids", []),
            vectors_info=vectors_info_list,
            vectors_used=sources_data.get("vectors_used", []),
            final_response=sources_data.get("final_response", [])
        )
        
        vectors_data = ContextVectorsData(
            vector_data=vector_data_list,
            sources=sources
        )
        
        return ContextVectorsResponse(
            status=data.get("status", ""),
            status_code=data.get("status_code", 200),
            message=data.get("message", ""),
            data=vectors_data
        )
    
    def _parse_json_with_cleanup(self, content: str) -> dict:
        """Parse JSON with cleanup for common issues"""
        # Remove // comments (not inside strings)
        lines = []
        for line in content.split('\n'):
            stripped = line.strip()
            # Skip pure comment lines
            if stripped.startswith('//'):
                continue
            lines.append(line)
        content = '\n'.join(lines)
        
        # Remove trailing commas before ] or }
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try with strict=False to allow control characters
            try:
                return json.loads(content, strict=False)
            except json.JSONDecodeError:
                raise e
    
    def extract_context_for_turn(
        self,
        vectors_data: ContextVectorsResponse,
        turn_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant context vectors for a specific turn.
        
        Returns a list of dictionaries compatible with the evaluator interface.
        """
        # Get all vector data
        all_vectors = vectors_data.data.vector_data
        
        # Get vectors that were actually used (if available)
        sources = vectors_data.data.sources
        vectors_used = sources.vectors_used
        
        if vectors_used:
            # Prioritize vectors that were actually used
            used_vectors = [v for v in all_vectors if v.id in vectors_used]
            # Add remaining vectors sorted by score
            vectors_info = {vi.vector_id: vi.score 
                          for vi in sources.vectors_info}
            other_vectors = [v for v in all_vectors if v.id not in vectors_used]
            other_vectors.sort(key=lambda x: vectors_info.get(x.id, 0), reverse=True)
            selected_vectors = used_vectors + other_vectors[:10]
        else:
            selected_vectors = all_vectors[:15]
        
        # Convert to dict format for evaluator compatibility
        return [
            {
                "id": vec.id,
                "source_url": vec.source_url,
                "text": vec.text,
                "tokens": vec.tokens,
                "created_at": vec.created_at
            }
            for vec in selected_vectors
        ]
    
    def get_ai_responses_with_context(self, conversation: Conversation) -> List[Dict[str, Any]]:
        """
        Extract AI response turns that follow user messages.
        
        Returns a list of dictionaries with turn information for evaluation.
        """
        turns = conversation.conversation_turns
        ai_responses = []
        
        for i, turn in enumerate(turns):
            if turn.role == "AI/Chatbot" and i > 0:
                # Find the preceding user message
                prev_turn = turns[i - 1]
                if prev_turn.role == "User":
                    ai_responses.append({
                        "turn_id": turn.turn,
                        "user_query": prev_turn.message,
                        "ai_response": turn.message,
                        "timestamp": turn.created_at,
                        "evaluation_note": turn.evaluation_note  # Include evaluation note if present
                    })
        
        return ai_responses

