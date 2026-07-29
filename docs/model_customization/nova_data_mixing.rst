Nova Data Mixing
=================

Data mixing blends your custom training data with Nova's curated synthetic datasets (code, math,
chat, planning, instruction-following, reasoning, etc.) to prevent catastrophic forgetting while
specializing the model on your domain.

.. important::

   Data mixing is only supported with **serverless** compute type. It is not available for
   serverful training jobs (SMTJ) or HyperPod clusters.

How Data Mixing Works
----------------------

During fine-tuning, data mixing controls the proportion of training samples that come from your
custom dataset versus Nova's internal curated datasets. By mixing in curated data, the base model
retains broad capabilities (e.g., code generation, mathematical reasoning) while adapting to your
specific task.

.. code-block:: text

   Your data (customer_data_percent%) + Nova curated data (remaining%)
                                            ├── code
                                            ├── math
                                            ├── chat
                                            ├── planning
                                            ├── instruction-following
                                            ├── reasoning
                                            ├── stem
                                            ├── rag
                                            └── factuality


Configuration
--------------

Use ``DataMixingConfig`` to specify the blend:

.. code-block:: python

   from sagemaker.train.data_mixing_config import DataMixingConfig

   data_mixing_config = DataMixingConfig(
       customer_data_percent=70.0,
       nova_data_percentages={
           "code": 30.0,
           "math": 70.0,
       },
   )

**Parameters:**

- ``customer_data_percent`` — percentage of training data that comes from your dataset (0–100)
- ``nova_data_percentages`` — distribution of Nova's curated datasets for the remaining portion.
  The values represent the relative weight among the selected Nova categories.
  Available categories include: ``code``, ``math``, ``chat``, ``planning``,
  ``instruction-following``, ``reasoning``, ``stem``, ``rag``, ``factuality``, etc.

.. note::

   The values in ``nova_data_percentages`` must sum to 100.


Example: SFT with Data Mixing (Serverless)
--------------------------------------------

.. code-block:: python

   import json
   import boto3

   REGION = "us-east-1"
   ROLE_ARN = "arn:aws:iam::123456789012:role/MySageMakerRole"
   S3_BUCKET = "sagemaker-us-east-1-123456789012"
   S3_OUTPUT_PATH = f"s3://{S3_BUCKET}/sft-data-mixing/output"
   TRAINING_DATASET = f"s3://{S3_BUCKET}/datasets/sft_training_data.jsonl"

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.common import TrainingType
   from sagemaker.train.data_mixing_config import DataMixingConfig

   data_mixing_config = DataMixingConfig(
       customer_data_percent=70.0,
       nova_data_percentages={
           "code": 30.0,
           "math": 70.0,
       },
   )

   sft_trainer = SFTTrainer(
       model="amazon.nova-2-lite-v1",
       training_type=TrainingType.LORA,
       training_dataset=TRAINING_DATASET,
       s3_output_path=S3_OUTPUT_PATH,
       role=ROLE_ARN,
       data_mixing_config=data_mixing_config,
       base_job_name="sft-datamix",
   )

   # Set hyperparameters
   sft_trainer.hyperparameters.max_steps = 50
   sft_trainer.hyperparameters.learning_rate = 5e-6
   sft_trainer.hyperparameters.global_batch_size = 32

   # Submit (non-blocking)
   training_job = sft_trainer.train(wait=False)
   print(f"Training job submitted: {training_job}")


Tips for Choosing Data Mix Percentages
---------------------------------------

- **High customer_data_percent (80–90%)** — Use when your domain-specific task is well-defined
  and you have enough training data. The model will specialize quickly but may lose some
  general capabilities.

- **Balanced customer_data_percent (50–70%)** — Good default for most use cases. Preserves
  broad capabilities while still adapting to your domain.

- **Low customer_data_percent (20–40%)** — Use when you want to mostly preserve the base
  model's capabilities with light specialization, or when your training dataset is small.

- **Nova category selection** — Choose categories that complement your task. For example,
  if fine-tuning for a coding assistant, include ``code`` and ``reasoning`` in the Nova mix
  to reinforce those capabilities.


Interactive Notebook
---------------------

For a complete walkthrough, see the
:doc:`Data Mixing notebook <../../v3-examples/model-customization-examples/nova_data_mixing>`.
