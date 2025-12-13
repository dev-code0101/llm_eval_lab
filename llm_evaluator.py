#!/usr/bin/env python3
"""
LLM Response Evaluation Pipeline - Main Entry Point

CLI interface for evaluating LLM responses.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import EvaluationPipeline
from evaluators import LLMEvaluator, MockLLMEvaluator
from parsers import ConfigLoader


def ensure_reports_dir() -> Path:
    """Ensure the reports directory exists and return its path"""
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


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
        choices=["openai", "anthropic", "huggingface", "mock"],
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
    parser.add_argument(
        "--config", "-c",
        help="Path to config YAML file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader.load(args.config)
    
    # Override config with CLI arguments if provided
    if args.provider != "openai" or args.model != "gpt-4o-mini":
        config.llm_provider.provider = args.provider
        config.llm_provider.model = args.model
    
    # Initialize evaluator
    if args.mock:
        evaluator = MockLLMEvaluator()
    else:
        evaluator = LLMEvaluator(
            provider=config.llm_provider.provider,
            model=config.llm_provider.model
        )
    
    # Run pipeline with config
    pipeline = EvaluationPipeline(evaluator, config=config)
    
    try:
        pipeline.evaluate_conversation(
            args.conversation_file,
            args.context_vectors_file,
            target_turn=args.turn
        )
        
        # Generate and display report
        if args.output:
            report_path = args.output
        else:
            reports_dir = ensure_reports_dir()
            report_path = str(reports_dir / "evaluation_report.txt")
        
        report = pipeline.generate_report(report_path)
        print("\n" + report)
        print(f"Report saved to: {report_path}")
        
        # Export JSON
        if args.json_output:
            json_path = args.json_output
        else:
            reports_dir = ensure_reports_dir()
            json_path = str(reports_dir / "evaluation_results.json")
        
        pipeline.export_results_json(json_path)
        print(f"JSON results saved to: {json_path}")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
