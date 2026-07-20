AI Model Customization Job Submission
======================================

SageMaker Python SDK V3 provides specialized trainer classes for different model customization
approaches, along with advanced features like data mixing, recipe overrides, evaluation, and
deployment.

Compute Options
----------------

By default, training jobs run on **serverless** compute — fully managed infrastructure that
abstracts away instance provisioning and scaling. You do not need to specify a ``compute``
parameter to use serverless.

If you need dedicated instances, you can optionally specify:

- **TrainingJobCompute** — serverful SageMaker Training Job instances (see :doc:`finetuning_serverful`)
- **HyperPodCompute** — SageMaker HyperPod managed clusters for multi-node distributed training (see :doc:`finetuning_hyperpod`)


Trainer Classes
----------------

**SFTTrainer (Supervised Fine-Tuning)**
  Traditional fine-tuning with labeled datasets for task-specific adaptation

**CPTTrainer (Continued Pre-Training)**
  Continue pre-training on a raw corpus to extend model knowledge in a specific domain

**DPOTrainer (Direct Preference Optimization)**
  Fine-tune models using human preference data without reinforcement learning complexity

**RLAIFTrainer (Reinforcement Learning from AI Feedback)**
  Use AI-generated feedback to improve model behavior and alignment

**RLVRTrainer (Reinforcement Learning from Verifiable Rewards)**
  Fine-tune with verifiable reward signals for objective optimization

**MultiTurnRLTrainer (Agentic Reinforcement Fine-Tuning)**
  Fine-tune models for multi-turn agent interactions using reinforcement learning from environment feedback


Key Features
-------------

**Data Mixing** (Nova, serverless only)
  Blend your custom data with Nova's curated synthetic datasets (code, math, chat, planning,
  reasoning, etc.) to prevent catastrophic forgetting while specializing on your domain.
  See :doc:`nova_data_mixing` for details.

**Recipe Overrides**
  Layer training configuration from multiple sources (YAML recipes, override dicts, SDK defaults)
  with clear precedence. Use ``get_resolved_recipe()`` to inspect the merged configuration
  before job submission. See :doc:`finetuning_serverful` and :doc:`finetuning_hyperpod` for examples.

**Dry-Run Validation**
  Pass ``dry_run=True`` to ``train()`` to run the validation steps without submitting a job 
  or consuming compute. Returns ``None`` on success, raises ``ValueError`` on validation failure.

  Supported on all trainers (SFT, DPO, RLVR, RLAIF, CPT) and ``ModelTrainer.train()``.
  Works across serverless, serverful (``TrainingJobCompute``), and HyperPod
  (``HyperPodCompute``) compute modes. Validates S3 URIs, DataSet ARNs, and ``DataSet``
  objects.

  Also available on evaluators — see :doc:`evaluation` for details.

  .. code-block:: python

     from sagemaker.train import SFTTrainer
     from sagemaker.train.common import TrainingType

     trainer = SFTTrainer(
         model="meta-textgeneration-llama-3-2-1b-instruct",
         training_type=TrainingType.LORA,
         model_package_group="my-finetuned-models",
         training_dataset="s3://my-bucket/train.jsonl",
         accept_eula=True,
     )

     # Validate without submitting — returns None on success
     trainer.train(dry_run=True)


.. toctree::
   :maxdepth: 1
   :caption: Customization Techniques

   SFT Finetuning <../../v3-examples/model-customization-examples/sft_finetuning_example_notebook_pysdk_prod_v3>
   DPOTrainer Finetuning <../../v3-examples/model-customization-examples/dpo_trainer_example_notebook_v3_prod>
   RLAIF Finetuning <../../v3-examples/model-customization-examples/rlaif_finetuning_example_notebook_v3_prod>
   RLVR Finetuning <../../v3-examples/model-customization-examples/rlvr_finetuning_example_notebook_v3_prod>
   CPT Training on HyperPod <../../v3-examples/model-customization-examples/cpt_data_mixing_hyperpod>
   Fine-Tuning with Serverful Training Jobs <finetuning_serverful>
   Fine-Tuning with HyperPod <finetuning_hyperpod>
