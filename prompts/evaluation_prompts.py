"""
Prompt templates for LLM-as-judge evaluation.
"""

RELEVANCE_EVALUATION_PROMPT = """You are an expert evaluator assessing AI chatbot responses for relevance and completeness.

## Context Information (Retrieved from Knowledge Base):
{context}

## User Query:
{user_query}

## AI Response to Evaluate:
{ai_response}

## Evaluation Task:
Evaluate the AI response on two dimensions:

### 1. RELEVANCE (Does the response address what the user asked?)
- Score 5: Directly and precisely addresses the user's question
- Score 4: Mostly relevant with minor tangential information
- Score 3: Partially relevant, some off-topic content
- Score 2: Mostly irrelevant to the user's question
- Score 1: Completely irrelevant

### 2. COMPLETENESS (Does the response fully answer the question?)
- Score 5: Comprehensive answer covering all aspects
- Score 4: Good coverage with minor omissions
- Score 3: Covers main points but missing important details
- Score 2: Incomplete, missing major aspects
- Score 1: Fails to answer the question

## Output Format (JSON):
{{
    "relevance_score": <1-5>,
    "relevance_explanation": "<explanation>",
    "is_relevant": <true/false>,
    "completeness_score": <1-5>,
    "completeness_explanation": "<explanation>",
    "is_complete": <true/false>,
    "missing_aspects": ["<aspect1>", "<aspect2>", ...]
}}

Respond ONLY with valid JSON."""


HALLUCINATION_EVALUATION_PROMPT = """You are an expert fact-checker evaluating AI responses for hallucinations and factual accuracy.

## Ground Truth Context (The ONLY source of truth):
{context}

## User Query:
{user_query}

## AI Response to Evaluate:
{ai_response}

## Evaluation Task:
Identify all factual claims in the AI response and verify each against the provided context.

### Hallucination Categories:
1. **Fabricated Facts**: Information completely made up, not in context
2. **Misattributed Information**: Correct info attributed to wrong source/entity
3. **Distorted Facts**: Partially correct but altered in a misleading way
4. **Unsupported Claims**: Claims that could be true but aren't verifiable from context

### Scoring:
- Score 5: No hallucinations, all claims verified
- Score 4: Minor inaccuracies that don't affect meaning
- Score 3: Some unverifiable claims but no clear fabrications
- Score 2: Contains notable hallucinations or errors
- Score 1: Significant fabrications or dangerous misinformation

## Output Format (JSON):
{{
    "hallucination_score": <1-5>,
    "has_hallucination": <true/false>,
    "factual_accuracy": <0.0-1.0>,
    "hallucinated_claims": [
        {{
            "claim": "<the problematic claim>",
            "category": "<fabricated/misattributed/distorted/unsupported>",
            "explanation": "<why this is a hallucination>",
            "severity": "<high/medium/low>"
        }}
    ],
    "verified_claims": [
        {{
            "claim": "<verified claim>",
            "source_snippet": "<supporting text from context>"
        }}
    ],
    "explanation": "<overall assessment>"
}}

Respond ONLY with valid JSON."""

