#!/usr/bin/env python3
"""
LLM Response Evaluation Pipeline - Main Entry Point

CLI interface for evaluating LLM responses.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import EvaluationPipeline
from evaluators import LLMEvaluator, MockLLMEvaluator


def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses for relevance and hallucination"
    )
    parser.add_argument(
        "conversation_file",
        help="Path to conversation JSON file"
    )
    parser.add_argument(
        "context_vectors_file", 
        help="Path to context vectors JSON file"
    )
    parser.add_argument(
        "--turn", "-t",
        type=int,
        help="Evaluate only specific turn ID"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "mock"],
        default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for report file"
    )
    parser.add_argument(
        "--json-output", "-j",
        help="Output path for JSON results"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock evaluator (no API calls)"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    if args.mock:
        evaluator = MockLLMEvaluator()
    else:
        evaluator = LLMEvaluator(provider=args.provider, model=args.model)
    
    # Run pipeline
    pipeline = EvaluationPipeline(evaluator)
    
    try:
        pipeline.evaluate_conversation(
            args.conversation_file,
            args.context_vectors_file,
            target_turn=args.turn
        )
        
        # Generate and display report
        report = pipeline.generate_report(args.output)
        print("\n" + report)
        
        # Export JSON if requested
        if args.json_output:
            pipeline.export_results_json(args.json_output)
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
