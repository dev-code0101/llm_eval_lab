"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics evaluator.

ROUGE metrics measure n-gram overlap between the generated response and reference context.
"""

from typing import Optional, List

from models import ROUGEResult


class ROUGEEvaluator:
    """
    Evaluates responses using ROUGE metrics.
    
    ROUGE metrics compare the AI response against the reference context
    to measure how well the response captures information from the context.
    """
    
    def __init__(self, rouge_types: List[str] = None):
        """
        Initialize ROUGE evaluator.
        
        Args:
            rouge_types: List of ROUGE metrics to compute. 
                        Options: ['rouge-1', 'rouge-2', 'rouge-l']
        """
        self.rouge_types = rouge_types or ["rouge-1", "rouge-2", "rouge-l"]
        self._rouge_scorer = None
    
    def _get_rouge_scorer(self):
        """Lazy initialization of ROUGE scorer"""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True
                )
            except ImportError:
                raise ImportError(
                    "ROUGE evaluation requires rouge-score package.\n"
                    "Install with: pip install rouge-score\n"
                    "Or: pip install -r requirements.txt"
                )
        return self._rouge_scorer
    
    def _extract_reference_text(self, context_vectors: List[dict]) -> str:
        """
        Extract reference text from context vectors.
        Combines all context text into a single reference string.
        """
        reference_parts = []
        for vector in context_vectors:
            text = vector.get("text", "")
            if text:
                reference_parts.append(text)
        return " ".join(reference_parts)
    
    def evaluate(
        self,
        ai_response: str,
        context_vectors: List[dict],
        rouge_types: Optional[List[str]] = None
    ) -> ROUGEResult:
        """
        Evaluate response using ROUGE metrics.
        
        Args:
            ai_response: The AI-generated response to evaluate
            context_vectors: List of context vectors (ground truth/reference)
            rouge_types: Override default ROUGE types for this evaluation
        
        Returns:
            ROUGEResult with ROUGE scores
        """
        if not ai_response or not context_vectors:
            return ROUGEResult(
                explanation="Cannot compute ROUGE: missing response or context"
            )
        
        rouge_types_to_use = rouge_types or self.rouge_types
        
        # Extract reference text from context
        reference_text = self._extract_reference_text(context_vectors)
        
        if not reference_text:
            return ROUGEResult(
                explanation="Cannot compute ROUGE: no reference text found in context"
            )
        
        # Compute ROUGE scores
        scorer = self._get_rouge_scorer()
        scores = scorer.score(reference_text, ai_response)
        
        result = ROUGEResult()
        enabled_scores = []
        
        # Extract ROUGE-1 scores
        if "rouge-1" in rouge_types_to_use or "rouge1" in rouge_types_to_use:
            rouge1 = scores['rouge1']
            result.rouge_1 = rouge1.fmeasure
            result.rouge_1_precision = rouge1.precision
            result.rouge_1_recall = rouge1.recall
            enabled_scores.append(rouge1.fmeasure)
        
        # Extract ROUGE-2 scores
        if "rouge-2" in rouge_types_to_use or "rouge2" in rouge_types_to_use:
            rouge2 = scores['rouge2']
            result.rouge_2 = rouge2.fmeasure
            result.rouge_2_precision = rouge2.precision
            result.rouge_2_recall = rouge2.recall
            enabled_scores.append(rouge2.fmeasure)
        
        # Extract ROUGE-L scores
        if "rouge-l" in rouge_types_to_use or "rougel" in rouge_types_to_use:
            rougeL = scores['rougeL']
            result.rouge_l = rougeL.fmeasure
            result.rouge_l_precision = rougeL.precision
            result.rouge_l_recall = rougeL.recall
            enabled_scores.append(rougeL.fmeasure)
        
        # Calculate average
        if enabled_scores:
            result.average_score = sum(enabled_scores) / len(enabled_scores)
            result.explanation = (
                f"ROUGE scores computed: {len(enabled_scores)} metric(s). "
                f"Average F1: {result.average_score:.3f}"
            )
        else:
            result.explanation = "No ROUGE metrics enabled"
        
        return result

