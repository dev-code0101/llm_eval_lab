#!/usr/bin/env python3
"""
Example usage of the LLM Evaluation Pipeline.

This script demonstrates how to evaluate the sample conversation
against the context vectors using either:
1. Real LLM evaluation (requires API key)
2. Mock evaluation (no API needed, for testing)
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluators import LLMEvaluator, MockLLMEvaluator
from core import EvaluationPipeline


def run_mock_evaluation():
    """
    Run evaluation with mock evaluator (no API calls).
    Useful for testing the pipeline structure.
    """
    print("=" * 60)
    print("Running MOCK Evaluation (No API calls)")
    print("=" * 60)
    
    evaluator = MockLLMEvaluator()
    pipeline = EvaluationPipeline(evaluator)
    
    # Evaluate all AI responses
    results = pipeline.evaluate_conversation(
        conversation_path="sample-chat-conversation-01.json",
        vectors_path="sample_context_vectors-01.json"
    )
    
    # Generate report
    report = pipeline.generate_report()
    print("\n" + report)
    
    # Export JSON results
    pipeline.export_results_json("evaluation_results_mock.json")
    
    return results


def run_openai_evaluation(target_turn: int = None):
    """
    Run evaluation with OpenAI GPT-4o-mini.
    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return None
    
    print("=" * 60)
    print("Running OpenAI GPT-4o-mini Evaluation")
    print("=" * 60)
    
    evaluator = LLMEvaluator(
        provider="openai",
        model="gpt-4o-mini",  # Cost-effective option; use gpt-4o for best results
        api_key=api_key
    )
    pipeline = EvaluationPipeline(evaluator)
    
    # Evaluate specific turn or all turns
    results = pipeline.evaluate_conversation(
        conversation_path="sample-chat-conversation-01.json",
        vectors_path="sample_context_vectors-01.json",
        target_turn=target_turn
    )
    
    # Generate report
    report = pipeline.generate_report("evaluation_report.txt")
    print("\n" + report)
    
    # Export JSON results
    pipeline.export_results_json("evaluation_results.json")
    
    return results


def evaluate_single_response_demo():
    """
    Demonstrate evaluating a single response programmatically.
    This shows how to use the evaluator for real-time evaluation.
    """
    print("=" * 60)
    print("Single Response Evaluation Demo (Mock)")
    print("=" * 60)
    
    # Sample data
    user_query = "Do you have an idea how much their rooms cost per night?"
    
    ai_response = """For Gopal Mansion, an air-conditioned room with TV and bath is Rs 800 per night. 
    We also offer specially subsidized air-conditioned rooms at our clinic for Rs 2000 (US $50) per night, 
    and non-AC rooms for Rs 1500 (US$ 40) per night, including free breakfast.
    Happy Home Hotel, which is a 5-minute walk from our clinic, offers single rooms for Rs 1400 
    and double rooms for Rs 2000."""
    
    # Context from knowledge base
    context_vectors = [
        {
            "id": 27025,
            "source_url": "https://www.drmalpani.com/hotels",
            "text": """Hotels Near Malpani Infertility Clinic. Your best choice would be 
            Gopal Mansion. Their website is www.gopalmansion.com. This is a building with 
            50 rooms, about 30 min away from the clinic. An airconditioned room with TV 
            and bath is only Rs 800 per night."""
        },
        {
            "id": 28960,
            "source_url": "https://www.drmalpani.com/hotels",
            "text": """Happy Home Hotel, 66, Woodehouse Road, Opp. Chiragdin Showroom, 
            Colaba, Mumbai - 4000005 A/C Room Attached Bathroom Food service. 
            Room Charges 1400/- Single Room, 2000/- Double Room. 
            This is a 5 min walk away from the clinic."""
        }
    ]
    
    evaluator = MockLLMEvaluator()
    
    # Run evaluation
    result = evaluator.evaluate_response(
        turn_id=14,
        user_query=user_query,
        ai_response=ai_response,
        context_vectors=context_vectors
    )
    
    print(f"\nTurn ID: {result.turn_id}")
    print(f"Overall Score: {result.overall_score:.1f}/5.0")
    print(f"\n{result.evaluation_summary}")
    print(f"\nRelevance: {result.relevance.score}/5")
    print(f"Hallucination Score: {result.hallucination.score}/5")
    print(f"Has Hallucination: {result.hallucination.has_hallucination}")
    
    # Note: The mock evaluator uses simple heuristics
    # The real evaluator would catch the hallucination about "subsidized rooms at clinic"
    print("\n⚠️  Note: Mock evaluation uses heuristics. Real LLM evaluation")
    print("    would detect the hallucination about clinic rooms.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM Evaluation Pipeline Demo")
    parser.add_argument(
        "--mode", 
        choices=["mock", "openai", "demo"],
        default="mock",
        help="Evaluation mode: mock (no API), openai (requires key), demo (single response)"
    )
    parser.add_argument(
        "--turn", "-t",
        type=int,
        help="Evaluate specific turn only (for openai mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "mock":
        run_mock_evaluation()
    elif args.mode == "openai":
        run_openai_evaluation(target_turn=args.turn)
    elif args.mode == "demo":
        evaluate_single_response_demo()

