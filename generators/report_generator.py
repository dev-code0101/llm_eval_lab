"""
Report generators for evaluation results.
"""

import json
from typing import Optional

from models import EvaluationResult
from parsers import EvaluationConfig


class TextReportGenerator:
    """Generates human-readable text reports from evaluation results"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config
    
    def set_config(self, config: EvaluationConfig):
        """Set configuration for report generation"""
        self.config = config
    
    def generate(self, results: list[EvaluationResult], output_path: Optional[str] = None) -> str:
        """Generate a detailed evaluation report"""
        if not results:
            return "No evaluation results available."
        
        report_lines = [
            "=" * 80,
            "LLM RESPONSE EVALUATION REPORT",
            "=" * 80,
            ""
        ]
        
        # Summary statistics
        metrics_config = self.config.metrics if self.config else None
        
        avg_overall = sum(r.overall_score for r in results) / len(results) if results else 0
        
        # Calculate averages only for enabled metrics
        avg_relevance = 0.0
        avg_hallucination = 0.0
        avg_rouge = 0.0
        total_hallucinations = 0
        
        relevance_count = 0
        hallucination_count = 0
        rouge_count = 0
        
        for r in results:
            if r.relevance and (not metrics_config or metrics_config.response_relevance or metrics_config.response_completeness):
                avg_relevance += r.relevance.score
                relevance_count += 1
            if r.hallucination and (not metrics_config or metrics_config.hallucination):
                avg_hallucination += r.hallucination.score
                hallucination_count += 1
                total_hallucinations += len(r.hallucination.hallucinated_claims)
            if r.rouge and (not metrics_config or metrics_config.rouge):
                avg_rouge += r.rouge.average_score
                rouge_count += 1
        
        if relevance_count > 0:
            avg_relevance /= relevance_count
        if hallucination_count > 0:
            avg_hallucination /= hallucination_count
        if rouge_count > 0:
            avg_rouge /= rouge_count
        
        report_lines.extend([
            "SUMMARY STATISTICS",
            "-" * 40,
            f"Total Responses Evaluated: {len(results)}",
            f"Average Overall Score: {avg_overall:.2f}/5.0",
            ""
        ])
        
        # Only show enabled metrics
        if not metrics_config or metrics_config.response_relevance or metrics_config.response_completeness:
            report_lines.append(f"Average Relevance Score: {avg_relevance:.2f}/5.0")
        if not metrics_config or metrics_config.hallucination:
            report_lines.append(f"Average Factual Accuracy Score: {avg_hallucination:.2f}/5.0")
            report_lines.append(f"Total Hallucinations Detected: {total_hallucinations}")
        if not metrics_config or metrics_config.rouge:
            report_lines.append(f"Average ROUGE Score: {avg_rouge:.3f}")
        
        report_lines.append("")
        
        # Individual results
        report_lines.extend([
            "DETAILED RESULTS",
            "-" * 40,
            ""
        ])
        
        for result in results:
            report_lines.extend([
                f"Turn {result.turn_id}",
                f"  User Query: {result.user_query[:100]}...",
                f"  Overall Score: {result.overall_score:.1f}/5.0",
            ])
            
            # Show evaluation note from conversation if present
            if result.evaluation_note:
                report_lines.append(f"  📝 Conversation Note: {result.evaluation_note}")
            
            report_lines.append("")
            
            # Relevance & Completeness
            if result.relevance and (not metrics_config or metrics_config.response_relevance or metrics_config.response_completeness):
                report_lines.append("  Relevance & Completeness:")
                if metrics_config and metrics_config.response_relevance:
                    report_lines.append(f"    Relevance Score: {result.relevance.score}/5")
                    report_lines.append(f"    Is Relevant: {'Yes' if result.relevance.is_relevant else 'No'}")
                    if self.config and self.config.report.include_detailed_explanations:
                        report_lines.append(f"    Explanation: {result.relevance.relevance_explanation}")
                
                if metrics_config and metrics_config.response_completeness:
                    report_lines.append(f"    Completeness Score: {result.relevance.score}/5")
                    report_lines.append(f"    Is Complete: {'Yes' if result.relevance.is_complete else 'No'}")
                    if self.config and self.config.report.include_detailed_explanations:
                        report_lines.append(f"    Explanation: {result.relevance.completeness_explanation}")
                
                if result.relevance.missing_aspects:
                    report_lines.append(f"    Missing: {', '.join(result.relevance.missing_aspects)}")
                report_lines.append("")
            
            # Hallucination
            if result.hallucination and (not metrics_config or metrics_config.hallucination):
                report_lines.extend([
                    "  Hallucination & Factual Accuracy:",
                    f"    Score: {result.hallucination.score}/5",
                    f"    Has Hallucinations: {'Yes' if result.hallucination.has_hallucination else 'No'}",
                    f"    Factual Accuracy: {result.hallucination.factual_accuracy:.0%}",
                ])
                
                if self.config and self.config.report.include_detailed_explanations:
                    report_lines.append(f"    Explanation: {result.hallucination.explanation}")
                
                if result.hallucination.hallucinated_claims:
                    report_lines.append("    Hallucinated Claims:")
                    for claim in result.hallucination.hallucinated_claims:
                        report_lines.append(
                            f"      ⚠️ [{claim.get('severity', 'unknown')}] {claim.get('claim', '')[:80]}"
                        )
                        report_lines.append(f"         Category: {claim.get('category', 'unknown')}")
                report_lines.append("")
            
            # ROUGE
            if result.rouge and (not metrics_config or metrics_config.rouge):
                report_lines.extend([
                    "  ROUGE Metrics:",
                    f"    Average Score: {result.rouge.average_score:.3f}",
                ])
                
                if result.rouge.rouge_1 is not None:
                    report_lines.append(
                        f"    ROUGE-1: {result.rouge.rouge_1:.3f} "
                        f"(P: {result.rouge.rouge_1_precision:.3f}, "
                        f"R: {result.rouge.rouge_1_recall:.3f})"
                    )
                if result.rouge.rouge_2 is not None:
                    report_lines.append(
                        f"    ROUGE-2: {result.rouge.rouge_2:.3f} "
                        f"(P: {result.rouge.rouge_2_precision:.3f}, "
                        f"R: {result.rouge.rouge_2_recall:.3f})"
                    )
                if result.rouge.rouge_l is not None:
                    report_lines.append(
                        f"    ROUGE-L: {result.rouge.rouge_l:.3f} "
                        f"(P: {result.rouge.rouge_l_precision:.3f}, "
                        f"R: {result.rouge.rouge_l_recall:.3f})"
                    )
                
                if self.config and self.config.report.include_detailed_explanations:
                    report_lines.append(f"    Explanation: {result.rouge.explanation}")
                report_lines.append("")
            
            report_lines.extend(["-" * 40, ""])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report


class JSONReportGenerator:
    """Generates JSON reports from evaluation results"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config
    
    def set_config(self, config: EvaluationConfig):
        """Set configuration for report generation"""
        self.config = config
    
    def generate(self, results: list[EvaluationResult], output_path: str):
        """Export results as JSON for further processing"""
        metrics_config = self.config.metrics if self.config else None
        
        # Calculate summary statistics
        summary = {
            "total_evaluated": len(results),
            "avg_overall_score": sum(r.overall_score for r in results) / len(results) if results else 0,
        }
        
        # Only include enabled metrics in summary
        if not metrics_config or metrics_config.response_relevance or metrics_config.response_completeness:
            relevance_results = [r for r in results if r.relevance]
            if relevance_results:
                summary["avg_relevance_score"] = sum(r.relevance.score for r in relevance_results) / len(relevance_results)
        
        if not metrics_config or metrics_config.hallucination:
            hallucination_results = [r for r in results if r.hallucination]
            if hallucination_results:
                summary["avg_factual_score"] = sum(r.hallucination.score for r in hallucination_results) / len(hallucination_results)
        
        if not metrics_config or metrics_config.rouge:
            rouge_results = [r for r in results if r.rouge]
            if rouge_results:
                summary["avg_rouge_score"] = sum(r.rouge.average_score for r in rouge_results) / len(rouge_results)
        
        export_data = {
            "summary": summary,
            "results": []
        }
        
        for r in results:
            result_dict = {
                "turn_id": r.turn_id,
                "user_query": r.user_query,
                "ai_response": r.ai_response,
                "overall_score": r.overall_score,
            }
            
            if r.evaluation_note:
                result_dict["evaluation_note"] = r.evaluation_note
            
            if r.relevance and (not metrics_config or metrics_config.response_relevance or metrics_config.response_completeness):
                result_dict["relevance"] = {
                    "score": r.relevance.score,
                    "is_relevant": r.relevance.is_relevant,
                    "is_complete": r.relevance.is_complete,
                    "relevance_explanation": r.relevance.relevance_explanation,
                    "completeness_explanation": r.relevance.completeness_explanation,
                    "missing_aspects": r.relevance.missing_aspects
                }
            
            if r.hallucination and (not metrics_config or metrics_config.hallucination):
                result_dict["hallucination"] = {
                    "score": r.hallucination.score,
                    "has_hallucination": r.hallucination.has_hallucination,
                    "factual_accuracy": r.hallucination.factual_accuracy,
                    "hallucinated_claims": r.hallucination.hallucinated_claims,
                    "verified_claims": r.hallucination.verified_claims,
                    "explanation": r.hallucination.explanation
                }
            
            if r.rouge and (not metrics_config or metrics_config.rouge):
                result_dict["rouge"] = {
                    "rouge_1": r.rouge.rouge_1,
                    "rouge_2": r.rouge.rouge_2,
                    "rouge_l": r.rouge.rouge_l,
                    "rouge_1_precision": r.rouge.rouge_1_precision,
                    "rouge_1_recall": r.rouge.rouge_1_recall,
                    "rouge_2_precision": r.rouge.rouge_2_precision,
                    "rouge_2_recall": r.rouge.rouge_2_recall,
                    "rouge_l_precision": r.rouge.rouge_l_precision,
                    "rouge_l_recall": r.rouge.rouge_l_recall,
                    "average_score": r.rouge.average_score,
                    "explanation": r.rouge.explanation
                }
            
            export_data["results"].append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Results exported to: {output_path}")

