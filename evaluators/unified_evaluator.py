"""
Unified evaluator that conditionally runs metrics based on configuration.
"""

from typing import Optional
from models import (
    RelevanceResult, 
    HallucinationResult, 
    ROUGEResult, 
    EvaluationResult
)
from evaluators import (
    LLMEvaluator, 
    MockLLMEvaluator, 
    ROUGEEvaluator,
    LettuceDetectEvaluator
)
from parsers import MetricsConfig


class UnifiedEvaluator:
    """
    Unified evaluator that runs enabled metrics based on configuration.
    """
    
    def __init__(
        self,
        llm_evaluator: LLMEvaluator,
        rouge_evaluator: Optional[ROUGEEvaluator] = None,
        lettucedetect_evaluator: Optional[LettuceDetectEvaluator] = None,
        config: Optional[MetricsConfig] = None
    ):
        self.llm_evaluator = llm_evaluator
        self.rouge_evaluator = rouge_evaluator or ROUGEEvaluator()
        self.lettucedetect_evaluator = lettucedetect_evaluator
        self.config = config or MetricsConfig()
        
        # Initialize LettuceDetect evaluator if needed
        if (self.config.hallucination and 
            self.config.hallucination_method == "lettucedetect" and 
            self.lettucedetect_evaluator is None):
            self.lettucedetect_evaluator = LettuceDetectEvaluator(
                model_path=self.config.lettucedetect_model
            )
    
    def evaluate_response(
        self,
        turn_id: int,
        user_query: str,
        ai_response: str,
        context_vectors: list[dict]
    ) -> EvaluationResult:
        """
        Run complete evaluation on a single response based on config.
        
        Only runs metrics that are enabled in the configuration.
        """
        relevance = None
        hallucination = None
        rouge = None
        
        # Evaluate relevance (includes both relevance and completeness)
        if self.config.response_relevance or self.config.response_completeness:
            relevance = self.llm_evaluator.evaluate_relevance(
                user_query, ai_response, context_vectors
            )
        
        # Evaluate hallucination (using configured method)
        if self.config.hallucination:
            if self.config.hallucination_method == "lettucedetect":
                # Use LettuceDetect transformer-based detection
                hallucination = self.lettucedetect_evaluator.evaluate(
                    user_query, ai_response, context_vectors
                )
            elif self.config.hallucination_method == "llm_judge":
                # Use LLM-as-judge
                hallucination = self.llm_evaluator.evaluate_hallucination(
                    user_query, ai_response, context_vectors
                )
            else:
                # Fallback to LLM-as-judge for unknown methods
                hallucination = self.llm_evaluator.evaluate_hallucination(
                    user_query, ai_response, context_vectors
                )
        
        # Evaluate ROUGE (using configured method)
        if self.config.rouge:
            if self.config.rouge_method == "rouge":
                # Use ROUGE n-gram metrics
                rouge = self.rouge_evaluator.evaluate(
                    ai_response, 
                    context_vectors,
                    rouge_types=self.config.rouge_types
                )
            elif self.config.rouge_method == "llm_judge":
                # Use LLM-as-judge for ROUGE evaluation
                # Format context for LLM evaluation
                context_text = self.llm_evaluator._format_context(context_vectors)
                rouge = self._evaluate_rouge_with_llm(
                    user_query, ai_response, context_text
                )
            else:
                # Fallback to ROUGE n-gram
                rouge = self.rouge_evaluator.evaluate(
                    ai_response, 
                    context_vectors,
                    rouge_types=self.config.rouge_types
                )
        
        # Calculate overall score using configured weights
        overall_score = 0.0
        total_weight = 0.0
        
        if relevance:
            # Relevance and completeness are combined in one result
            # Split the weight between them
            if self.config.response_relevance:
                overall_score += (relevance.score / 5.0) * self.config.response_relevance_weight
                total_weight += self.config.response_relevance_weight
            if self.config.response_completeness:
                overall_score += (relevance.score / 5.0) * self.config.response_completeness_weight
                total_weight += self.config.response_completeness_weight
        
        if hallucination:
            overall_score += (hallucination.score / 5.0) * self.config.hallucination_weight
            total_weight += self.config.hallucination_weight
        
        if rouge:
            overall_score += rouge.average_score * self.config.rouge_weight
            total_weight += self.config.rouge_weight
        
        # Normalize to 0-5 scale
        if total_weight > 0:
            overall_score = (overall_score / total_weight) * 5.0
        else:
            overall_score = 0.0
        
        # Generate summary
        summary_parts = []
        
        if relevance:
            if self.config.response_relevance:
                if relevance.is_relevant:
                    summary_parts.append("✅ Response is relevant.")
                else:
                    summary_parts.append("⚠️ Response lacks relevance.")
            
            if self.config.response_completeness:
                if relevance.is_complete:
                    summary_parts.append("✅ Response is complete.")
                else:
                    summary_parts.append("⚠️ Response is incomplete.")
        
        if hallucination:
            if hallucination.has_hallucination:
                summary_parts.append(
                    f"🚨 Detected {len(hallucination.hallucinated_claims)} hallucination(s)."
                )
            else:
                summary_parts.append("✅ No hallucinations detected.")
        
        if rouge:
            summary_parts.append(
                f"📊 ROUGE Average: {rouge.average_score:.3f}"
            )
        
        summary_parts.append(f"Overall Score: {overall_score:.1f}/5.0")
        
        return EvaluationResult(
            turn_id=turn_id,
            user_query=user_query,
            ai_response=ai_response,
            relevance=relevance,
            hallucination=hallucination,
            rouge=rouge,
            overall_score=overall_score,
            evaluation_summary="\n".join(summary_parts),
            context_used=[v.get("source_url", "") for v in context_vectors[:5]]
        )
    
    def _evaluate_rouge_with_llm(
        self,
        user_query: str,
        ai_response: str,
        context_text: str
    ) -> ROUGEResult:
        """
        Evaluate ROUGE using LLM-as-judge approach.
        This provides semantic evaluation rather than n-gram overlap.
        """
        # Use a modified prompt for ROUGE evaluation
        prompt = f"""You are an expert evaluator assessing how well an AI response captures information from the provided context.

## Context Information (Reference):
{context_text}

## User Query:
{user_query}

## AI Response to Evaluate:
{ai_response}

## Evaluation Task:
Evaluate how well the AI response captures and reflects the information from the context using ROUGE-like metrics:

1. **ROUGE-1 (Unigram Coverage)**: How many key words/concepts from the context appear in the response? (0.0-1.0)
2. **ROUGE-2 (Bigram Coverage)**: How well do key phrases from the context appear in the response? (0.0-1.0)
3. **ROUGE-L (Longest Common Subsequence)**: How well does the response maintain the semantic flow and key information sequences from the context? (0.0-1.0)

## Output Format (JSON):
{{
    "rouge_1": <0.0-1.0>,
    "rouge_2": <0.0-1.0>,
    "rouge_l": <0.0-1.0>,
    "rouge_1_precision": <0.0-1.0>,
    "rouge_1_recall": <0.0-1.0>,
    "rouge_2_precision": <0.0-1.0>,
    "rouge_2_recall": <0.0-1.0>,
    "rouge_l_precision": <0.0-1.0>,
    "rouge_l_recall": <0.0-1.0>,
    "explanation": "<brief explanation of scores>"
}}

Respond ONLY with valid JSON."""
        
        try:
            response = self.llm_evaluator.client.call(prompt)
            result = self.llm_evaluator._parse_json_response(response)
            
            rouge_1 = result.get("rouge_1", 0.0)
            rouge_2 = result.get("rouge_2", 0.0)
            rouge_l = result.get("rouge_l", 0.0)
            
            enabled_scores = [s for s in [rouge_1, rouge_2, rouge_l] if s is not None]
            avg_score = sum(enabled_scores) / len(enabled_scores) if enabled_scores else 0.0
            
            return ROUGEResult(
                rouge_1=rouge_1,
                rouge_2=rouge_2,
                rouge_l=rouge_l,
                rouge_1_precision=result.get("rouge_1_precision"),
                rouge_1_recall=result.get("rouge_1_recall"),
                rouge_2_precision=result.get("rouge_2_precision"),
                rouge_2_recall=result.get("rouge_2_recall"),
                rouge_l_precision=result.get("rouge_l_precision"),
                rouge_l_recall=result.get("rouge_l_recall"),
                average_score=avg_score,
                explanation=result.get("explanation", "LLM-as-judge ROUGE evaluation")
            )
        except Exception as e:
            # Fallback to ROUGE n-gram on error
            return self.rouge_evaluator.evaluate(
                ai_response,
                [{"text": context_text}],
                rouge_types=self.config.rouge_types
            )

