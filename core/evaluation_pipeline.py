"""
Pipeline orchestration for evaluating chat conversations.
"""

from typing import Optional

from models import EvaluationResult
from evaluators import LLMEvaluator
from loaders import JSONDataLoader
from generators import TextReportGenerator, JSONReportGenerator


class EvaluationPipeline:
    """
    Orchestrates the evaluation of chat conversations against context vectors.
    """
    
    def __init__(
        self,
        evaluator: LLMEvaluator,
        data_loader: Optional[JSONDataLoader] = None,
        text_report_generator: Optional[TextReportGenerator] = None,
        json_report_generator: Optional[JSONReportGenerator] = None
    ):
        self.evaluator = evaluator
        self.data_loader = data_loader or JSONDataLoader()
        self.text_report_generator = text_report_generator or TextReportGenerator()
        self.json_report_generator = json_report_generator or JSONReportGenerator()
        self.results: list[EvaluationResult] = []
    
    def evaluate_conversation(
        self,
        conversation_path: str,
        vectors_path: str,
        target_turn: Optional[int] = None
    ) -> list[EvaluationResult]:
        """
        Evaluate all AI responses in a conversation.
        
        Args:
            conversation_path: Path to conversation JSON
            vectors_path: Path to context vectors JSON
            target_turn: If specified, only evaluate this turn
        
        Returns:
            List of evaluation results
        """
        conversation = self.data_loader.load_conversation(conversation_path)
        vectors_data = self.data_loader.load_context_vectors(vectors_path)
        
        ai_responses = self.data_loader.get_ai_responses_with_context(conversation)
        
        if target_turn:
            ai_responses = [r for r in ai_responses if r["turn_id"] == target_turn]
        
        self.results = []
        
        for response_data in ai_responses:
            context_vectors = self.data_loader.extract_context_for_turn(
                vectors_data, 
                response_data["turn_id"]
            )
            
            result = self.evaluator.evaluate_response(
                turn_id=response_data["turn_id"],
                user_query=response_data["user_query"],
                ai_response=response_data["ai_response"],
                context_vectors=context_vectors
            )
            
            self.results.append(result)
            print(f"Evaluated turn {response_data['turn_id']}: {result.overall_score:.1f}/5.0")
        
        return self.results
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a detailed evaluation report"""
        return self.text_report_generator.generate(self.results, output_path)
    
    def export_results_json(self, output_path: str):
        """Export results as JSON for further processing"""
        self.json_report_generator.generate(self.results, output_path)

