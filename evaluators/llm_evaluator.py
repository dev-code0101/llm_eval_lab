"""
LLM-based evaluators for relevance and hallucination detection.
"""

import json
import re
from typing import Optional

from models import RelevanceResult, HallucinationResult, EvaluationResult
from prompts import RELEVANCE_EVALUATION_PROMPT, HALLUCINATION_EVALUATION_PROMPT
from clients import LLMClient, OpenAIClient, AnthropicClient


class LLMEvaluator:
    """
    Evaluates LLM responses using an LLM-as-judge approach.
    
    Supports multiple LLM backends:
    - OpenAI (default)
    - Anthropic Claude
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        
        # Initialize appropriate client
        if provider == "openai":
            self.client: LLMClient = OpenAIClient(model=model, api_key=api_key)
        elif provider == "anthropic":
            self.client: LLMClient = AnthropicClient(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _parse_json_response(self, response: str) -> dict:
        """Safely parse JSON from LLM response"""
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                return json.loads(json_match.group(1))
            # Try to find JSON object pattern
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")
    
    def _format_context(self, context_vectors: list[dict]) -> str:
        """Format context vectors into readable text for evaluation"""
        formatted_parts = []
        for i, vector in enumerate(context_vectors, 1):
            source = vector.get("source_url", "Unknown source")
            text = vector.get("text", "")
            formatted_parts.append(f"[Source {i}] {source}\n{text}")
        return "\n\n---\n\n".join(formatted_parts)
    
    def evaluate_relevance(
        self,
        user_query: str,
        ai_response: str,
        context_vectors: list[dict]
    ) -> RelevanceResult:
        """Evaluate response relevance and completeness"""
        context = self._format_context(context_vectors)
        
        prompt = RELEVANCE_EVALUATION_PROMPT.format(
            context=context,
            user_query=user_query,
            ai_response=ai_response
        )
        
        response = self.client.call(prompt)
        result = self._parse_json_response(response)
        
        # Calculate combined score
        avg_score = (result["relevance_score"] + result["completeness_score"]) / 2
        
        return RelevanceResult(
            score=round(avg_score),
            is_relevant=result["is_relevant"],
            is_complete=result["is_complete"],
            relevance_explanation=result["relevance_explanation"],
            completeness_explanation=result["completeness_explanation"],
            missing_aspects=result.get("missing_aspects", [])
        )
    
    def evaluate_hallucination(
        self,
        user_query: str,
        ai_response: str,
        context_vectors: list[dict]
    ) -> HallucinationResult:
        """Evaluate response for hallucinations and factual accuracy"""
        context = self._format_context(context_vectors)
        
        prompt = HALLUCINATION_EVALUATION_PROMPT.format(
            context=context,
            user_query=user_query,
            ai_response=ai_response
        )
        
        response = self.client.call(prompt)
        result = self._parse_json_response(response)
        
        return HallucinationResult(
            score=result["hallucination_score"],
            has_hallucination=result["has_hallucination"],
            factual_accuracy=result["factual_accuracy"],
            hallucinated_claims=result.get("hallucinated_claims", []),
            verified_claims=result.get("verified_claims", []),
            explanation=result.get("explanation", "")
        )
    
    def evaluate_response(
        self,
        turn_id: int,
        user_query: str,
        ai_response: str,
        context_vectors: list[dict]
    ) -> EvaluationResult:
        """Run complete evaluation on a single response"""
        
        # Run both evaluations
        relevance = self.evaluate_relevance(user_query, ai_response, context_vectors)
        hallucination = self.evaluate_hallucination(user_query, ai_response, context_vectors)
        
        # Calculate overall score (weighted average)
        overall_score = (relevance.score * 0.4 + hallucination.score * 0.6)
        
        # Generate summary
        summary_parts = []
        if relevance.is_relevant and relevance.is_complete:
            summary_parts.append("✅ Response is relevant and complete.")
        else:
            if not relevance.is_relevant:
                summary_parts.append("⚠️ Response lacks relevance.")
            if not relevance.is_complete:
                summary_parts.append("⚠️ Response is incomplete.")
        
        if hallucination.has_hallucination:
            summary_parts.append(f"🚨 Detected {len(hallucination.hallucinated_claims)} hallucination(s).")
        else:
            summary_parts.append("✅ No hallucinations detected.")
        
        summary_parts.append(f"Overall Score: {overall_score:.1f}/5.0")
        
        return EvaluationResult(
            turn_id=turn_id,
            user_query=user_query,
            ai_response=ai_response,
            relevance=relevance,
            hallucination=hallucination,
            overall_score=overall_score,
            evaluation_summary="\n".join(summary_parts),
            context_used=[v.get("source_url", "") for v in context_vectors[:5]]
        )


class MockLLMEvaluator(LLMEvaluator):
    """
    Mock evaluator for testing without API calls.
    Uses rule-based heuristics for basic evaluation.
    """
    
    def __init__(self):
        # Don't call super().__init__ to avoid client initialization
        self.provider = "mock"
        self.model = "mock"
        self.client = None
    
    def _call_llm_mock(self, prompt: str) -> str:
        """Simulate LLM evaluation with heuristics"""
        # Extract response from prompt for analysis
        response_match = re.search(r"AI Response to Evaluate:\n(.+?)(?:\n\n|$)", prompt, re.DOTALL)
        ai_response = response_match.group(1).strip() if response_match else ""
        
        context_match = re.search(r"(?:Ground Truth Context|Context Information).*?:\n(.+?)(?:\n\n##|$)", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        
        # Simple heuristic analysis
        response_words = set(ai_response.lower().split())
        context_words = set(context.lower().split())
        overlap = len(response_words & context_words) / max(len(response_words), 1)
        
        if "hallucination" in prompt.lower():
            # Hallucination evaluation
            has_hallucination = overlap < 0.3
            score = 5 if overlap > 0.5 else (3 if overlap > 0.3 else 2)
            return json.dumps({
                "hallucination_score": score,
                "has_hallucination": has_hallucination,
                "factual_accuracy": overlap,
                "hallucinated_claims": [{"claim": "Mock detection", "category": "unsupported", "severity": "low"}] if has_hallucination else [],
                "verified_claims": [],
                "explanation": f"Heuristic analysis: {overlap:.0%} word overlap with context"
            })
        else:
            # Relevance evaluation
            return json.dumps({
                "relevance_score": 4 if overlap > 0.2 else 3,
                "relevance_explanation": "Heuristic relevance analysis",
                "is_relevant": overlap > 0.1,
                "completeness_score": 4,
                "completeness_explanation": "Heuristic completeness analysis",
                "is_complete": len(ai_response) > 50,
                "missing_aspects": []
            })
    
    def evaluate_relevance(
        self,
        user_query: str,
        ai_response: str,
        context_vectors: list[dict]
    ) -> RelevanceResult:
        """Evaluate response relevance and completeness (mock)"""
        from prompts import RELEVANCE_EVALUATION_PROMPT
        
        context = self._format_context(context_vectors)
        prompt = RELEVANCE_EVALUATION_PROMPT.format(
            context=context,
            user_query=user_query,
            ai_response=ai_response
        )
        
        response = self._call_llm_mock(prompt)
        result = self._parse_json_response(response)
        
        avg_score = (result["relevance_score"] + result["completeness_score"]) / 2
        
        return RelevanceResult(
            score=round(avg_score),
            is_relevant=result["is_relevant"],
            is_complete=result["is_complete"],
            relevance_explanation=result["relevance_explanation"],
            completeness_explanation=result["completeness_explanation"],
            missing_aspects=result.get("missing_aspects", [])
        )
    
    def evaluate_hallucination(
        self,
        user_query: str,
        ai_response: str,
        context_vectors: list[dict]
    ) -> HallucinationResult:
        """Evaluate response for hallucinations (mock)"""
        from prompts import HALLUCINATION_EVALUATION_PROMPT
        
        context = self._format_context(context_vectors)
        prompt = HALLUCINATION_EVALUATION_PROMPT.format(
            context=context,
            user_query=user_query,
            ai_response=ai_response
        )
        
        response = self._call_llm_mock(prompt)
        result = self._parse_json_response(response)
        
        return HallucinationResult(
            score=result["hallucination_score"],
            has_hallucination=result["has_hallucination"],
            factual_accuracy=result["factual_accuracy"],
            hallucinated_claims=result.get("hallucinated_claims", []),
            verified_claims=result.get("verified_claims", []),
            explanation=result.get("explanation", "")
        )

