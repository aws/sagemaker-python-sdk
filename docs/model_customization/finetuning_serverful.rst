Fine-Tuning with Serverful Training Jobs
==========================================

Fine-tune Amazon Nova models using ``SFTTrainer`` on **serverful SageMaker Training Job** instances
with **recipe overrides**.

Key Concepts
------------

Recipe Override Precedence
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training configuration is resolved with clear precedence:

.. code-block:: text

   overrides dict  >  recipe YAML  >  Hub/SDK defaults

You can provide a YAML recipe for bulk configuration, then surgically override individual keys.
Use ``get_resolved_recipe()`` to inspect the fully merged recipe before job submission.


Setup
------

.. code-block:: python

   import json
   import boto3

   REGION = "us-east-1"
   ROLE_ARN = "arn:aws:iam::123456789012:role/MySageMakerRole"
   S3_BUCKET = "sagemaker-us-east-1-123456789012"
   S3_OUTPUT_PATH = f"s3://{S3_BUCKET}/sft-smtj/output"
   TRAINING_DATASET = f"s3://{S3_BUCKET}/datasets/sft_training_data.jsonl"


Create Trainer
---------------

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.common import TrainingType
   from sagemaker.core.training.configs import TrainingJobCompute

   sft_trainer = SFTTrainer(
       model="amazon.nova-2-lite-v1",
       training_type=TrainingType.LORA,
       training_dataset=TRAINING_DATASET,
       s3_output_path=S3_OUTPUT_PATH,
       compute=TrainingJobCompute(
           instance_type="ml.p5.48xlarge",
           instance_count=2,
       ),
       role=ROLE_ARN,
       base_job_name="sft-smtj",
   )


Using Recipe Overrides
-----------------------

You can provide a YAML recipe file with your training configuration, then selectively override
specific parameters via the ``overrides`` dict. The override takes precedence over values in
the recipe file. This is useful when you want a shared base recipe but need to experiment with
specific hyperparameters (e.g., learning rate or number of epochs) without modifying the file.

.. code-block:: python

   import yaml

   # Write a custom recipe YAML
   recipe_config = {
       "training": {
           "learning_rate": 1e-5,
           "num_epochs": 3,
           "batch_size": 8,
           "sequence_length": 2048,
       }
   }

   with open("my_sft_recipe.yaml", "w") as f:
       yaml.dump(recipe_config, f)

   # Create trainer with recipe + overrides
   # Here we override learning_rate (1e-5 → 5e-6) and num_epochs (3 → 5)
   # from the recipe file above, while keeping batch_size and sequence_length unchanged
   sft_trainer_with_recipe = SFTTrainer(
       model="amazon.nova-2-lite-v1",
       training_type=TrainingType.LORA,
       training_dataset=TRAINING_DATASET,
       s3_output_path=S3_OUTPUT_PATH,
       compute=TrainingJobCompute(
           instance_type="ml.p5.48xlarge",
           instance_count=2,
       ),
       role=ROLE_ARN,
       recipe="my_sft_recipe.yaml",
       overrides={"training_config": {"learning_rate": 5e-6, "num_epochs": 5}},
       base_job_name="sft-recipe-override-smtj",
   )

   # Inspect the resolved recipe to confirm overrides were applied
   resolved = sft_trainer_with_recipe.get_resolved_recipe()
   print(json.dumps(resolved.get("training_config", resolved), indent=2))


Set Hyperparameters and Submit
-------------------------------

.. code-block:: python

   sft_trainer.hyperparameters.max_steps = 50
   sft_trainer.hyperparameters.learning_rate = 5e-6
   sft_trainer.hyperparameters.global_batch_size = 32

   # Submit (non-blocking)
   training_job = sft_trainer.train(wait=False)
   print(f"Training job submitted: {training_job}")


Monitor the Job
----------------

.. code-block:: python

   from sagemaker.core.resources import TrainingJob

   job = TrainingJob.get(training_job_name=training_job.training_job_name)
   print(f"Status: {job.training_job_status}")
   print(f"Secondary Status: {job.secondary_status}")


Interactive Notebook
---------------------

.. toctree::
   :maxdepth: 1

   ../../v3-examples/model-customization-examples/sft_finetuning_serverful_smtj
