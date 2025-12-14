# LLM Evaluation Lab

## Current Approach Used

- Avoid relying completely LLM-as-judge as much as possible. This is to make the script feasible and scalable to hand thousands of interactions.
- Modular codebase with SOLID-like architecture

## Next Steps

- Make use of ROUGE, BLEU, CHRF and TER scores to mix traditional NLP approach with LLM-as-judge. This will improve quality of score estimation since LLM are not good at generating scores beyond a range
- Wrap modules with FastAPI/Flask REST API servr (they have good async support). To deploy it as a service and make it easy to us as a tool in a lab.
- Add job queing suport using RabbitMQ/Celery. The current processing pipeline could highly benefit from an async architecture since some steps are I/O bound
- Create or find a fine-tuned model specifically for syntax similarity (different for scoring and RAG). Most models are designed with a hug knowledge base in mind. RAG does not require this. This will also reduce/eliminate hallucination
