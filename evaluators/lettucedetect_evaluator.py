"""
LettuceDetect-based hallucination detection evaluator.

LettuceDetect uses transformer-based models to detect hallucinations
by identifying which parts of the answer are not supported by the context.
"""

from typing import Optional, List
from models import HallucinationResult
import string

class LettuceDetectEvaluator:
    """
    Evaluates hallucinations using LettuceDetect transformer models.
    
    LettuceDetect provides span-level predictions indicating which parts
    of the answer are considered hallucinated based on the context.
    """
    
    def __init__(self, model_path: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"):
        """
        Initialize LettuceDetect evaluator.
        
        Args:
            model_path: Path to the LettuceDetect model
        """
        self.model_path = model_path
        self._detector = None
    
    def _get_detector(self):
        """Lazy initialization of LettuceDetect detector"""
        if self._detector is None:
            try:
                from lettucedetect.models.inference import HallucinationDetector
                self._detector = HallucinationDetector(
                    method="transformer",
                    model_path=self.model_path
                )
            except ImportError:
                raise ImportError(
                    "LettuceDetect evaluation requires lettucedetect package.\n"
                    "Install with: pip install lettucedetect\n"
                    "Or: pip install -r requirements.txt"
                )
        return self._detector
    
    def _extract_context_text(self, context_vectors: List[dict]) -> List[str]:
        """
        Extract context text from context vectors.
        Returns a list of context strings.
        """
        contexts = []
        for vector in context_vectors:
            text = vector.get("text", "")
            if text:
                contexts.append(text)
        return contexts if contexts else [""]
    
    def evaluate(
        self,
        user_query: str,
        ai_response: str,
        context_vectors: List[dict]
    ) -> HallucinationResult:
        """
        Evaluate response for hallucinations using LettuceDetect.
        
        Args:
            user_query: The user's question/query
            ai_response: The AI-generated response to evaluate
            context_vectors: List of context vectors (ground truth/reference)
        
        Returns:
            HallucinationResult with detected hallucinations
        """
        if not ai_response or not context_vectors:
            return HallucinationResult(
                score=1,
                has_hallucination=False,
                factual_accuracy=0.0,
                explanation="Cannot evaluate: missing response or context"
            )
        
        # Extract contexts
        contexts = self._extract_context_text(context_vectors)
        
        if not contexts or not contexts[0]:
            return HallucinationResult(
                score=1,
                has_hallucination=False,
                factual_accuracy=0.0,
                explanation="Cannot evaluate: no context text found"
            )
        
        try:
            detector = self._get_detector()
            
            # Get span-level predictions
            predictions = detector.predict(
                context=contexts,
                question=user_query,
                answer=ai_response,
                output_format="spans"
            )
            
            # Process predictions
            hallucinated_spans = []
            total_length = len(ai_response)
            hallucinated_length = 0
            
            if predictions:
                for pred in predictions:
                    start = pred.get('start', 0)
                    end = pred.get('end', len(ai_response))
                    confidence = pred.get('confidence', 0.0)
                    text : str = pred.get('text', ai_response[start:end])
                    
                    if len(text.strip().replace(string.punctuation, "")) < 5 or \
                       confidence < 0.8:
                        continue
                    
                    hallucinated_spans.append({
                        'claim': text.strip(),
                        'category': 'fabricated',  # LettuceDetect identifies unsupported claims
                        'explanation': f"Detected unsupported claim with confidence {confidence:.2f}",
                        'severity': 'high' if confidence > 0.8 else ('medium' if confidence > 0.5 else 'low'),
                        'start': start,
                        'end': end,
                        'confidence': confidence
                    })
                    
                    hallucinated_length += (end - start)
            
            # Calculate factual accuracy (1 - proportion of hallucinated text)
            if total_length > 0:
                factual_accuracy = 1.0 - (hallucinated_length / total_length)
            else:
                factual_accuracy = 0.0
            
            # Calculate score (1-5 scale based on factual accuracy and number of hallucinations)
            has_hallucination = len(hallucinated_spans) > 0
            
            if not has_hallucination:
                score = 5
            elif factual_accuracy > 0.9:
                score = 4
            elif factual_accuracy > 0.7:
                score = 3
            elif factual_accuracy > 0.5:
                score = 2
            else:
                score = 1
            
            # Create verified claims (non-hallucinated parts)
            verified_claims = []
            if not has_hallucination:
                verified_claims.append({
                    'claim': ai_response,
                    'source_snippet': 'Full response verified against context'
                })
            
            explanation = (
                f"LettuceDetect analysis: {len(hallucinated_spans)} hallucinated span(s) detected. "
                f"Factual accuracy: {factual_accuracy:.1%}"
            )
            
            return HallucinationResult(
                score=score,
                has_hallucination=has_hallucination,
                factual_accuracy=factual_accuracy,
                hallucinated_claims=hallucinated_spans,
                verified_claims=verified_claims,
                explanation=explanation
            )
            
        except Exception as e:
            return HallucinationResult(
                score=3,
                has_hallucination=False,
                factual_accuracy=0.5,
                explanation=f"LettuceDetect evaluation error: {str(e)}"
            )

