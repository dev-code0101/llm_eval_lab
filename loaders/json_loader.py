"""
JSON data loader with cleanup for malformed JSON.
"""

import json
import re
from typing import Optional


class JSONDataLoader:
    """Loads and parses JSON files with cleanup for common issues"""
    
    def load_conversation(self, conversation_path: str) -> dict:
        """Load chat conversation from JSON file"""
        with open(conversation_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._parse_json_with_cleanup(content)
    
    def load_context_vectors(self, vectors_path: str) -> dict:
        """Load context vectors from JSON file"""
        with open(vectors_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._parse_json_with_cleanup(content)
    
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
            except:
                raise e
    
    def extract_context_for_turn(
        self,
        vectors_data: dict,
        turn_id: Optional[int] = None
    ) -> list[dict]:
        """Extract relevant context vectors for a specific turn"""
        # Get all vector data
        all_vectors = vectors_data.get("data", {}).get("vector_data", [])
        
        # Get vectors that were actually used (if available)
        sources = vectors_data.get("data", {}).get("sources", {})
        vectors_used = sources.get("vectors_used", [])
        
        if vectors_used:
            # Prioritize vectors that were actually used
            used_vectors = [v for v in all_vectors if v.get("id") in vectors_used]
            # Add remaining vectors sorted by score
            vectors_info = {vi["vector_id"]: vi["score"] 
                          for vi in sources.get("vectors_info", [])}
            other_vectors = [v for v in all_vectors if v.get("id") not in vectors_used]
            other_vectors.sort(key=lambda x: vectors_info.get(x.get("id"), 0), reverse=True)
            return used_vectors + other_vectors[:10]  # Limit for context window
        
        return all_vectors[:15]  # Default: top 15 vectors
    
    def get_ai_responses_with_context(self, conversation: dict) -> list[dict]:
        """Extract AI response turns that follow user messages"""
        turns = conversation.get("conversation_turns", [])
        ai_responses = []
        
        for i, turn in enumerate(turns):
            if turn.get("role") == "AI/Chatbot" and i > 0:
                # Find the preceding user message
                prev_turn = turns[i - 1]
                if prev_turn.get("role") == "User":
                    ai_responses.append({
                        "turn_id": turn.get("turn"),
                        "user_query": prev_turn.get("message", ""),
                        "ai_response": turn.get("message", ""),
                        "timestamp": turn.get("created_at", "")
                    })
        
        return ai_responses

