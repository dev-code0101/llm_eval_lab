pip install uptrain
pip install rouge_score  # Required for some evaluation metrics

Step 2: Import Required Libraries

Create a new Python file or Jupyter notebook and import the necessary modules:

from uptrain import EvalLLM, Evals, Settings
import json

Step 3: Prepare Your Evaluation Data

Create a list of dictionaries containing your evaluation data. Each dictionary should include:

- question: The original question
- context: The context provided to the LLM
- response: The LLM's response

data = [
    {
        'question': 'What is the largest ocean on Earth?',
        'context': """The Earth's oceans cover over 70% of its surface, and there are five major oceans: the Pacific, Atlantic, Indian, Southern, and Arctic Oceans. The Pacific Ocean is by far the largest, stretching from the Arctic in the north to the Southern Ocean in the south. It covers an area of more than 63 million square miles, roughly double the size of the next largest ocean, the Atlantic.""",
        'response': 'The Pacific Ocean is the largest ocean, covering over 63 million square miles'
    }
]

Step 4: Configure UpTrain Settings

Set up your evaluation settings with your API key and model choice:

settings = Settings(
    model='mistral/mistral-tiny',
    mistral_api_key='YOUR_MISTRAL_API_KEY'
)
eval_llm = EvalLLM(settings)

Step 5: Run the Evaluation

Execute the evaluation with your chosen metrics:

results = eval_llm.evaluate(

    data=data,
    checks=[
        Evals.CONTEXT_RELEVANCE,    # Checks if context matches the question
        Evals.RESPONSE_RELEVANCE,   # Checks if response addresses the question
        Evals.RESPONSE_COMPLETENESS # Checks if response is complete
    ]

)

Step 6: Analyze the Results

Print and analyze the evaluation results:

print(json.dumps(results, indent=2))

The results will include scores and explanations for each metric:

- `score_context_relevance`: How relevant the context is to the question (0-1)
- `score_response_relevance`: How relevant the response is to the question (0-1)
- `score_response_completeness`: How complete the response is (0-1)

Decoding UpTrain Output Scores

Testing with relevant Context

When evaluating our first example with matching context about Earth’s oceans, UpTrain demonstrates perfect scores across all metrics. The context relevance (1.0), response relevance (1.0), and response completeness (1.0) indicate an ideal scenario where the context perfectly aligns with the question and the response effectively uses this information.

data_relevant =[
  {
    "question": "What is the largest ocean on Earth?",
    "context": "The Earth's oceans cover over 70% of its surface...",
    "response": "The Pacific Ocean is the largest ocean...",
    "score_context_relevance": 1.0,
    "explanation_context_relevance": "The context provides complete information...",
    "score_response_relevance": 1.0,
    "explanation_response_relevance": "The response directly answers the question...",
    "score_response_completeness": 1.0,
    "explanation_response_completeness": "The response includes all necessary information..."
  }
]

Testing with Irrelevant Context

In our second test, despite having irrelevant context about Indian trees, UpTrain shows its analytical precision. While the context relevance drops to 0.0 (indicating mismatched context), the response relevance and completeness maintain perfect scores (1.0) as the answer remains accurate regardless of the provided context, highlighting UpTrain’s nuanced evaluation capabilities.

data_irrelevant = [
    {
        'question': 'What is the largest ocean on Earth?',
        'context': ‘The most abundant tree which is found in the entire India….’,
        'response': 'The Pacific Ocean is the largest ocean, covering over 63 mil…'
        "score_context_relevance": 0.0,
        "explanation_context_relevance": "The extracted context does not...}",
        "score_response_relevance": 1.0,
        "explanation_response_relevance": "The given response provides the correct..”                  
         "score_response_completeness": 1.0,
         "explanation_response_completeness": "The given response is complete for the given question because it…”
    }
]

UpTrain’s 21 Evaluation Metrics

Let’s explore other powerful evaluation metrics from UpTrain that help assess different aspects of LLM responses. These metrics cover everything from factual accuracy to conversation satisfaction, helping ensure high-quality AI interactions.
Criterion	Description
context_relevance	Assesses if context matches the question.
factual_accuracy	Checks if the information provided is factually correct.
response_relevance	Evaluates if the answer addresses the question appropriately.
critique_language	Reviews the language quality and appropriateness of the response.
response_completeness	Check if the answer covers all aspects of the question.
response_completeness_wrt_context	Verifies if the answer uses all the relevant context effectively.
response_consistency	Ensures that the answer is internally consistent.
response_conciseness	Check if the answer is succinct and to the point.
valid_response	Confirms that the response meets the basic requirements of the question.
response_alignment_with_scenario	Matches the response to the given scenario.
response_sincerity_with_scenario	Evaluates authenticity and sincerity in the context of the scenario.
Prompt_injection	Identifies attempts to make the LLM reveal its system prompts.
code_hallucination	Checks if the code presented in the response is grounded in the context.
sub_query_completeness	Ensures that all parts of a multi-part question are answered.
context_reranking	Orders context by relevance priority.
context_conciseness	Check if the context provided is brief and efficient.
GuidelineAdherence	Ensures the response adheres to any specified rules or guidelines.
CritiqueTone	Evaluates if the tone of the response is appropriate for the situation.
ResponseMatching	Compares the response against the expected or ideal answer.
JailbreakDetection	Identifies any attempts to bypass safeguards or rules.
ConversationSatisfaction	Measures the overall quality and satisfaction of the conversation.