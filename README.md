# LLM Evaluation Lab

## Run pipeline on specified JSON file using:-

```bash
 python3 run_evaluation.py --mode from_recorded
```

## Current Approach Used

- Avoid relying completely LLM-as-judge as much as possible. This is to make the script feasible and scalable to handle thousands of interactions.
- Modular codebase with SOLID-like architecture
- Used LLM-as-judge for response completenes and relevance metrics. Used LettuceDetect for haluccination detection (transformers model that can be run offline and hence improv scalability)

## Next Steps

- Wrap modules with FastAPI/Flask REST API servr (they have good async support). To deploy it as a service and make it easy to us as a tool in a lab.
- Make use of ROUGE, BLEU, CHRF and TER scores to mix traditional NLP approach with LLM-as-judge. This will improve quality of score estimation since LLM are not good at generating scores beyond a range
- Add other hallucination techniques such as COT based approach proposed by OpenAI and prometheus model designed specifically for evaluation
- Add job queing suport using RabbitMQ/Celery. The current processing pipeline could highly benefit from an async architecture since some steps are I/O bound
- Create or find a fine-tuned model specifically for syntax similarity (different ones for scoring and RAG). Most models are designed with a hug knowledge base in mind. RAG does not require this. This will also reduce/eliminate hallucination
- Add simlation mode wherein different models will answer the user's qustion and be scored simultaneously
- Improve sample used for context at each step with simulated RAG approach where different embedding transformer (tokenisation + vectorisation) can be tsted and evaluated
