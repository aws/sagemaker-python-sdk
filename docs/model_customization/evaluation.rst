Model Evaluation Job Submission
=================================

Launch evaluation jobs with four options:

  * **LLM as a Judge (LLMAJ) Evaluation** - Use large language models to assess model outputs
  * **Custom Scorer Evaluation** - Apply previously defined evaluator functions
  * **Benchmark Evaluation** - Run standardized performance benchmarks
  * **Multi-Turn RL (Agentic) Evaluation** - Evaluate multi-turn agent models with rollout-based metrics (pass@k, mean reward)

.. toctree::
   :maxdepth: 1

   ../../v3-examples/model-customization-examples/llm_as_judge_demo
   ../../v3-examples/model-customization-examples/custom_scorer_demo
   ../../v3-examples/model-customization-examples/benchmark_demo
   mtrl_evaluation