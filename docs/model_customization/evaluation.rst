Model Evaluation
=================

Launch evaluation jobs with the following options:

  * **LLM as a Judge (LLMAJ) Evaluation** - Use large language models to assess model outputs
  * **InspectAI Evaluation** - Run open-source InspectAI benchmark tasks on SageMaker infrastructure
  * **Custom Scorer Evaluation** - Apply previously defined evaluator functions
  * **Benchmark Evaluation** - Run standardized performance benchmarks
  * **Multi-Turn RL (Agentic) Evaluation** - Evaluate multi-turn agent models with rollout-based metrics (pass@k, mean reward)

.. toctree::
   :maxdepth: 1

   ../../v3-examples/model-customization-examples/llm_as_judge_demo
   ../../v3-examples/model-customization-examples/inspect_ai_evaluation_demo
   ../../v3-examples/model-customization-examples/custom_scorer_demo
   ../../v3-examples/model-customization-examples/benchmark_demo
   ../../v3-examples/nova-examples/evaluation-benchmark-and-custom-scorer


Dry-Run Validation
-------------------

Pass ``dry_run=True`` to ``evaluate()`` to validate your evaluation configuration without
submitting a job or consuming compute. The SDK runs all validation (IAM role resolution,
model resolution, dataset path existence) and then stops before launching the evaluation
pipeline. Returns ``None`` on success, raises ``ValueError`` on validation failure.

Supported on all evaluators: ``BenchMarkEvaluator``, ``CustomScorerEvaluator``, and
``LLMAsJudgeEvaluator``.

.. code-block:: python

   from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks

   Benchmark = get_benchmarks()

   evaluator = BenchMarkEvaluator(
       benchmark=Benchmark.MMLU,
       model="arn:aws:sagemaker:us-east-1:123456789012:model-package/my-models/3",
       s3_output_path="s3://my-bucket/eval-output/",
   )

   # Validate without launching — returns None on success
   evaluator.evaluate(dry_run=True)

.. code-block:: python

   from sagemaker.train.evaluate import CustomScorerEvaluator

   evaluator = CustomScorerEvaluator(
       model="my-model-package-arn",
       evaluation_dataset="s3://my-bucket/eval-data.jsonl",
       s3_output_path="s3://my-bucket/custom-eval-output/",
       scorer_function=my_scorer,
   )

   # Raises ValueError if dataset path does not exist
   evaluator.evaluate(dry_run=True)
