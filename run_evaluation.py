#!/usr/bin/env python3
"""
LLM Evaluation Pipeline Runner

This script evaluates AI responses from recorded conversations.
It supports multiple evaluation modes:

1. from_recorded: Evaluate AI responses from a recorded conversation JSON
   - Treats the conversation as a recorded session
   - Evaluates each AI/Chatbot response against context vectors
   - Supports both mock (no API) and real LLM evaluation

2. openai: Real LLM evaluation (requires API key)

3. demo: Single response evaluation demo

The script follows the configuration in config.yaml to determine
which metrics to evaluate and which methods to use.
"""

import os
import sys
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env file from the script's directory
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    # dotenv is optional - if not installed, just use system environment variables
    pass

from evaluators import LLMEvaluator, MockLLMEvaluator
from core import EvaluationPipeline
from parsers import ConfigLoader


def run_from_recorded(
    conversation_path: str = "sample-chat-conversation-01.json",
    vectors_path: str = "sample_context_vectors-01.json",
    use_mock: bool = False,
    target_turn: Optional[int] = None
):
    """
    Evaluate AI responses from a recorded conversation.
    
    This mode treats the conversation JSON as a recorded conversation,
    evaluating each AI/Chatbot response against the context vectors
    that were retrieved for that turn.
    
    Args:
        conversation_path: Path to conversation JSON file
        vectors_path: Path to context vectors JSON file
        use_mock: If True, force mock evaluator. If False, use real LLM.
                  If None, auto-detect based on config (use real LLM if any metric uses llm_judge)
        target_turn: If specified, only evaluate this specific turn
    
    Returns:
        List of evaluation results
    """
    print("=" * 60)
    print("Running FROM_RECORDED Evaluation")
    print("=" * 60)
    print(f"Conversation: {conversation_path}")
    print(f"Context Vectors: {vectors_path}")
    if target_turn:
        print(f"Target Turn: {target_turn} (evaluating only this turn)")
    print("=" * 60)
    
    # Load configuration
    config = ConfigLoader.load()
    print_config_summary(config)
    
    # Check if any enabled metric uses llm_judge
    metrics = config.metrics
    needs_llm = False
    if use_mock is None or not use_mock:
        # Check if any enabled metric requires LLM
        if (metrics.response_relevance and metrics.response_relevance_method == "llm_judge") or \
           (metrics.response_completeness and metrics.response_completeness_method == "llm_judge") or \
           (metrics.hallucination and metrics.hallucination_method == "llm_judge") or \
           (metrics.rouge and metrics.rouge_method == "llm_judge"):
            needs_llm = True
    
    # Determine evaluator type
    if use_mock is True:
        # Force mock mode
        evaluator = MockLLMEvaluator()
        print("Evaluator: Mock (no API) [forced]")
    elif needs_llm or use_mock is False:
        # Need real LLM (clients will auto-resolve API keys from env vars)
        provider = config.llm_provider.provider
        evaluator = LLMEvaluator(
            provider=provider,
            model=config.llm_provider.model
        )
        print(f"Evaluator: LLM ({config.llm_provider.provider}/{config.llm_provider.model})")
        if needs_llm and use_mock is None:
            print("  [Auto-selected: LLM-as-judge required by config]")
    else:
        # No LLM needed, use mock
        evaluator = MockLLMEvaluator()
        print("Evaluator: Mock (no API) [no LLM metrics enabled]")
    
    print("=" * 60)
    
    # Create pipeline with config
    pipeline = EvaluationPipeline(evaluator, config=config)
    
    # Evaluate all AI responses from the recorded conversation
    print("\nEvaluating recorded conversation...")
    results = pipeline.evaluate_conversation(
        conversation_path=conversation_path,
        vectors_path=vectors_path,
        target_turn=target_turn
    )
    
    # Generate report
    print("\nGenerating report...")
    report_filename = f"evaluation_from_recorded_turn{target_turn}.txt" if target_turn else "evaluation_from_recorded.txt"
    report = pipeline.generate_report(report_filename)
    print("\n" + report)
    
    # Export JSON results
    json_filename = f"evaluation_results_from_recorded_turn{target_turn}.json" if target_turn else "evaluation_results_from_recorded.json"
    pipeline.export_results_json(json_filename)
    
    return results


def run_mock_evaluation():
    """
    Run evaluation with mock evaluator (no API calls).
    Useful for testing the pipeline structure.
    Follows config.yaml for metric selection and methods.
    """
    return run_from_recorded(use_mock=True)


def run_openai_evaluation(target_turn: int = None):
    """
    Run evaluation with configured LLM provider.
    Requires appropriate API key environment variable based on provider.
    Follows config.yaml for metric selection and methods.
    """
    # Load configuration first to determine provider
    config = ConfigLoader.load()
    provider = config.llm_provider.provider
    
    print("=" * 60)
    print(f"Running {provider.upper()} Evaluation")
    print("=" * 60)
    print_config_summary(config)
    
    # Create LLM evaluator (clients will auto-resolve API keys from env vars)
    evaluator = LLMEvaluator(
        provider=provider,
        model=config.llm_provider.model
    )
    
    # Create pipeline with config
    pipeline = EvaluationPipeline(evaluator, config=config)
    
    # Evaluate specific turn or all turns
    print("\nEvaluating conversation...")
    results = pipeline.evaluate_conversation(
        conversation_path="sample-chat-conversation-01.json",
        vectors_path="sample_context_vectors-01.json",
        target_turn=target_turn
    )
    
    # Generate report
    print("\nGenerating report...")
    report = pipeline.generate_report("evaluation_report.txt")
    print("\n" + report)
    
    # Export JSON results
    pipeline.export_results_json("evaluation_results.json")
    
    return results


def evaluate_single_response_demo():
    """
    Demonstrate evaluating a single response programmatically.
    This shows how to use the evaluator for real-time evaluation.
    Uses the pipeline with config to follow the configured methods.
    """
    print("=" * 60)
    print("Single Response Evaluation Demo (Mock)")
    print("=" * 60)
    
    # Load configuration
    config = ConfigLoader.load()
    print_config_summary(config)
    
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
    
    # Create evaluator and pipeline with config
    evaluator = MockLLMEvaluator()
    pipeline = EvaluationPipeline(evaluator, config=config)
    
    # Run evaluation using the pipeline (follows config)
    result = pipeline.evaluator.evaluate_response(
        turn_id=14,
        user_query=user_query,
        ai_response=ai_response,
        context_vectors=context_vectors
    )
    
    print(f"\nTurn ID: {result.turn_id}")
    print(f"Overall Score: {result.overall_score:.1f}/5.0")
    print(f"\n{result.evaluation_summary}")
    
    if result.relevance:
        print(f"\nRelevance: {result.relevance.score}/5")
        print(f"  Is Relevant: {result.relevance.is_relevant}")
        print(f"  Is Complete: {result.relevance.is_complete}")
    
    if result.hallucination:
        print(f"\nHallucination Score: {result.hallucination.score}/5")
        print(f"Has Hallucination: {result.hallucination.has_hallucination}")
        print(f"Factual Accuracy: {result.hallucination.factual_accuracy:.1%}")
    
    if result.rouge:
        print(f"\nROUGE Average: {result.rouge.average_score:.3f}")
        if result.rouge.rouge_1 is not None:
            print(f"  ROUGE-1: {result.rouge.rouge_1:.3f}")
        if result.rouge.rouge_2 is not None:
            print(f"  ROUGE-2: {result.rouge.rouge_2:.3f}")
        if result.rouge.rouge_l is not None:
            print(f"  ROUGE-L: {result.rouge.rouge_l:.3f}")
    
    # Note: The mock evaluator uses simple heuristics
    # The real evaluator would catch the hallucination about "subsidized rooms at clinic"
    print("\n⚠️  Note: Mock evaluation uses heuristics. Real LLM evaluation")
    print("    would detect the hallucination about clinic rooms.")


def print_config_summary(config):
    """Print a summary of the loaded configuration"""
    print("\nConfiguration Summary:")
    print("-" * 60)
    metrics = config.metrics
    
    print(f"Response Relevance: {'✓' if metrics.response_relevance else '✗'} "
          f"(method: {metrics.response_relevance_method})")
    print(f"Response Completeness: {'✓' if metrics.response_completeness else '✗'} "
          f"(method: {metrics.response_completeness_method})")
    print(f"Hallucination: {'✓' if metrics.hallucination else '✗'} "
          f"(method: {metrics.hallucination_method})")
    print(f"ROUGE: {'✓' if metrics.rouge else '✗'} "
          f"(method: {metrics.rouge_method})")
    
    if metrics.rouge and metrics.rouge_method == "rouge":
        print(f"  ROUGE Types: {', '.join(metrics.rouge_types)}")
    
    print(f"\nLLM Provider: {config.llm_provider.provider} ({config.llm_provider.model})")
    print("-" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run LLM Evaluation Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from recorded conversation (mock, no API calls)
  python run_evaluation.py --mode from_recorded
  
  # Evaluate from recorded conversation (with real LLM)
  export OPENAI_API_KEY='your-key'
  python run_evaluation.py --mode from_recorded --no-mock
  
  # Use custom conversation files
  python run_evaluation.py --mode from_recorded --conversation my_conv.json --vectors my_vectors.json
  
  # Evaluate specific turn
  python run_evaluation.py --mode from_recorded --turn 14
  
  # Run OpenAI evaluation (requires API key)
  export OPENAI_API_KEY='your-key'
  python run_evaluation.py --mode openai
  
  # Single response demo
  python run_evaluation.py --mode demo
        """
    )
    parser.add_argument(
        "--mode", 
        choices=["from_recorded", "mock", "openai", "demo"],
        default="from_recorded",
        help="Evaluation mode: from_recorded (evaluate recorded conversation), mock (legacy), openai (requires key), demo (single response)"
    )
    parser.add_argument(
        "--turn", "-t",
        type=int,
        help="Evaluate specific turn only"
    )
    parser.add_argument(
        "--conversation", "-c",
        default="sample-chat-conversation-01.json",
        help="Path to conversation JSON file (default: sample-chat-conversation-01.json)"
    )
    parser.add_argument(
        "--vectors", "-v",
        default="sample_context_vectors-01.json",
        help="Path to context vectors JSON file (default: sample_context_vectors-01.json)"
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Use real LLM evaluator instead of mock (requires API key)"
    )
    parser.add_argument(
        "--force-mock",
        action="store_true",
        help="Force mock evaluator even if LLM-as-judge is enabled (for testing)"
    )
    parser.add_argument(
        "--config",
        help="Path to custom config file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "from_recorded":
            # Determine mock setting: force_mock > no_mock > auto-detect
            if args.force_mock:
                use_mock = True
            elif args.no_mock:
                use_mock = False
            else:
                use_mock = None  # Auto-detect based on config
            
            run_from_recorded(
                conversation_path=args.conversation,
                vectors_path=args.vectors,
                use_mock=use_mock,
                target_turn=args.turn
            )
        elif args.mode == "mock":
            run_mock_evaluation()
        elif args.mode == "openai":
            run_openai_evaluation(target_turn=args.turn)
        elif args.mode == "demo":
            evaluate_single_response_demo()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

