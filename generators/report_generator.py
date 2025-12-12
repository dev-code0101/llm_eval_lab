"""
Report generators for evaluation results.
"""

import json
from typing import Optional

from models import EvaluationResult


class TextReportGenerator:
    """Generates human-readable text reports from evaluation results"""
    
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
        avg_overall = sum(r.overall_score for r in results) / len(results)
        avg_relevance = sum(r.relevance.score for r in results) / len(results)
        avg_hallucination = sum(r.hallucination.score for r in results) / len(results)
        total_hallucinations = sum(
            len(r.hallucination.hallucinated_claims) for r in results
        )
        
        report_lines.extend([
            "SUMMARY STATISTICS",
            "-" * 40,
            f"Total Responses Evaluated: {len(results)}",
            f"Average Overall Score: {avg_overall:.2f}/5.0",
            f"Average Relevance Score: {avg_relevance:.2f}/5.0",
            f"Average Factual Accuracy Score: {avg_hallucination:.2f}/5.0",
            f"Total Hallucinations Detected: {total_hallucinations}",
            ""
        ])
        
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
                "",
                f"  Relevance & Completeness:",
                f"    Score: {result.relevance.score}/5",
                f"    Is Relevant: {'Yes' if result.relevance.is_relevant else 'No'}",
                f"    Is Complete: {'Yes' if result.relevance.is_complete else 'No'}",
                f"    Explanation: {result.relevance.relevance_explanation}",
            ])
            
            if result.relevance.missing_aspects:
                report_lines.append(f"    Missing: {', '.join(result.relevance.missing_aspects)}")
            
            report_lines.extend([
                "",
                f"  Hallucination & Factual Accuracy:",
                f"    Score: {result.hallucination.score}/5",
                f"    Has Hallucinations: {'Yes' if result.hallucination.has_hallucination else 'No'}",
                f"    Factual Accuracy: {result.hallucination.factual_accuracy:.0%}",
            ])
            
            if result.hallucination.hallucinated_claims:
                report_lines.append("    Hallucinated Claims:")
                for claim in result.hallucination.hallucinated_claims:
                    report_lines.append(
                        f"      ⚠️ [{claim.get('severity', 'unknown')}] {claim.get('claim', '')[:80]}"
                    )
                    report_lines.append(f"         Category: {claim.get('category', 'unknown')}")
            
            report_lines.extend(["", "-" * 40, ""])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report


class JSONReportGenerator:
    """Generates JSON reports from evaluation results"""
    
    def generate(self, results: list[EvaluationResult], output_path: str):
        """Export results as JSON for further processing"""
        export_data = {
            "summary": {
                "total_evaluated": len(results),
                "avg_overall_score": sum(r.overall_score for r in results) / len(results) if results else 0,
                "avg_relevance_score": sum(r.relevance.score for r in results) / len(results) if results else 0,
                "avg_factual_score": sum(r.hallucination.score for r in results) / len(results) if results else 0,
            },
            "results": []
        }
        
        for r in results:
            export_data["results"].append({
                "turn_id": r.turn_id,
                "user_query": r.user_query,
                "ai_response": r.ai_response,
                "overall_score": r.overall_score,
                "relevance": {
                    "score": r.relevance.score,
                    "is_relevant": r.relevance.is_relevant,
                    "is_complete": r.relevance.is_complete,
                    "relevance_explanation": r.relevance.relevance_explanation,
                    "completeness_explanation": r.relevance.completeness_explanation,
                    "missing_aspects": r.relevance.missing_aspects
                },
                "hallucination": {
                    "score": r.hallucination.score,
                    "has_hallucination": r.hallucination.has_hallucination,
                    "factual_accuracy": r.hallucination.factual_accuracy,
                    "hallucinated_claims": r.hallucination.hallucinated_claims,
                    "verified_claims": r.hallucination.verified_claims,
                    "explanation": r.hallucination.explanation
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Results exported to: {output_path}")

