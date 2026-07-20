Fine-Tuning with HyperPod
===========================

Fine-tune Amazon Nova models using ``SFTTrainer`` on **SageMaker HyperPod** managed clusters
with **recipe overrides**.

HyperPod provides managed cluster orchestration with support for multi-node distributed training.

Key Concepts
------------

Recipe Override Precedence
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training configuration is resolved with clear precedence:

.. code-block:: text

   overrides dict  >  recipe YAML  >  Hub/SDK defaults

You can provide a YAML recipe for bulk configuration, then surgically override individual keys.
Use ``get_resolved_recipe()`` to inspect the fully merged recipe before job submission.


Prerequisites: HyperPod CLI Installation
------------------------------------------

HyperPod-based training requires the
`SageMaker HyperPod CLI <https://github.com/aws/sagemaker-hyperpod-cli/>`_ to connect
to clusters and start jobs.

**1. Install Helm 3** (required):

.. code-block:: bash

   curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
   chmod 700 get_helm.sh
   ./get_helm.sh
   rm -f ./get_helm.sh
   helm version  # Verify installation

**2. Install the HyperPod CLI:**

.. code-block:: bash

   git clone -b release_v2 https://github.com/aws/sagemaker-hyperpod-cli.git
   cd sagemaker-hyperpod-cli
   pip install .

**3. Verify the installation:**

.. code-block:: bash

   hyperpod --help

.. note::

   If you are a Nova Forge customer, download the HyperPod CLI with Forge feature support
   from S3 instead. See the `Nova Forge SDK documentation
   <https://github.com/aws/nova-forge-sdk#hyperpod-cli>`_ for details.


Setup
------

.. code-block:: python

   import json
   import os

   # Required for HyperPod CLI recipe resolution
   os.environ["PYTHONPATH"] = (
       "<path-to-your-hyperpod-cli>/hyperpod_cli/"
       "sagemaker_hyperpod_recipes/launcher/nemo/nemo_framework_launcher/launcher_scripts:"
       + os.environ.get("PYTHONPATH", "")
   )

   REGION = "us-east-1"
   S3_BUCKET = "sagemaker-us-east-1-123456789012"
   S3_OUTPUT_PATH = f"s3://{S3_BUCKET}/sft-hyperpod/output"
   TRAINING_DATASET = f"s3://{S3_BUCKET}/datasets/sft_training_data.jsonl"
   CLUSTER_NAME = "my-cluster"
   NAMESPACE = "kubeflow"


Create Trainer
---------------

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.common import TrainingType
   from sagemaker.core.training.configs import HyperPodCompute

   compute = HyperPodCompute(
       cluster_name=CLUSTER_NAME,
       namespace=NAMESPACE,
       instance_type="ml.p5.48xlarge",
       node_count=2,
   )

   sft_trainer = SFTTrainer(
       model="nova-textgeneration-micro",
       training_type=TrainingType.LORA,
       training_dataset=TRAINING_DATASET,
       s3_output_path=S3_OUTPUT_PATH,
       compute=compute,
       base_job_name="sft-hp",
   )


Using Recipe Overrides
-----------------------

You can provide a YAML recipe file with your training configuration, then selectively override
specific parameters via the ``overrides`` dict. The override takes precedence over values in
the recipe file. This is useful when you want a shared base recipe but need to experiment with
specific hyperparameters (e.g., learning rate or number of epochs) without modifying the file.

.. code-block:: python

   import yaml

   # Create a custom recipe YAML
   recipe_config = {
       "training": {
           "learning_rate": 1e-5,
           "num_epochs": 3,
           "batch_size": 8,
           "sequence_length": 2048,
       }
   }

   with open("my_sft_recipe_hp.yaml", "w") as f:
       yaml.dump(recipe_config, f)

   # Create trainer with recipe + overrides
   # Here we override learning_rate (1e-5 → 5e-6) and num_epochs (3 → 5)
   # from the recipe file above, while keeping batch_size and sequence_length unchanged
   sft_trainer_with_recipe = SFTTrainer(
       model="nova-textgeneration-micro",
       training_type=TrainingType.LORA,
       training_dataset=TRAINING_DATASET,
       s3_output_path=S3_OUTPUT_PATH,
       compute=HyperPodCompute(
           cluster_name=CLUSTER_NAME,
           namespace=NAMESPACE,
           instance_type="ml.p5.48xlarge",
           node_count=1,
       ),
       recipe="my_sft_recipe_hp.yaml",
       overrides={"training_config": {"learning_rate": 5e-6, "num_epochs": 5}},
       base_job_name="sft-recipe-override-hp",
   )

   # Inspect the resolved recipe to confirm overrides were applied
   resolved = sft_trainer_with_recipe.get_resolved_recipe()
   print(json.dumps(resolved, indent=2))


Set Hyperparameters and Submit
-------------------------------

.. code-block:: python

   sft_trainer.hyperparameters.max_steps = 10
   sft_trainer.hyperparameters.global_batch_size = 64

   # Submit (non-blocking)
   sft_job = sft_trainer.train(wait=False)
   print(f"HyperPod job submitted: {sft_job}")


Monitor the Job
----------------
show_metrics()
~~~~~~~~~~~~~~

Plot training metrics parsed from CloudWatch logs for your HyperPod cluster:

.. code-block:: python

   # After training completes
   df = trainer.show_metrics()

   # Plot specific metrics
   df = trainer.show_metrics(metrics=["training_loss", "reward_score"])

   # Filter by step range
   df = trainer.show_metrics(starting_step=50, ending_step=200)

   # Filter by time window, this can help speed up completion
   from datetime import datetime
   df = trainer.show_metrics(
       start_time=datetime(2026, 7, 1, 10, 0, 0),
       end_time=datetime(2026, 7, 2, 12, 0, 0),
   )

.. code-block:: python

   # After a kernel restart, set up a trainer with the compute config to retrieve metrics.
   from sagemaker.train import SFTTrainer
   from sagemaker.core.training.configs import HyperPodCompute

   # Create trainer with the same compute config (not used for training)
   trainer = SFTTrainer(
      model="nova-textgeneration-micro",
      training_dataset="s3://dataset-unused-for-metrics",  # not used for metrics
      compute=HyperPodCompute(
         cluster_name="my-cluster", 
         instance_type="ml.p5.48xlarge",
      )
   )

   # Set the job name manually
   trainer._latest_training_job = "my-hp-job-20260716153000"

   # Now show_metrics works — it uses compute.cluster_name to find logs
   df = trainer.show_metrics()


stream_logs()
~~~~~~~~~~~~~

Stream CloudWatch logs from the HyperPod cluster in real-time:

.. code-block:: python

   # Start training
   job = trainer.train(wait=False)

   # Stream logs (blocks until user manually runs Ctrl+C)
   trainer.stream_logs()

   # Custom polling interval in seconds
   trainer.stream_logs(poll=10)

   # Stream from a specific start time
   trainer.stream_logs(start_time=datetime(2026, 1, 1, 15, 0, 0))

.. note::

   HyperPod log streaming runs until you press Ctrl+C (unlike SMTJ which
   auto-stops when the job reaches a terminal state). Logs may take a few
   minutes to propagate to CloudWatch.


Interactive Notebook
---------------------

.. toctree::
   :maxdepth: 1

   ../../v3-examples/model-customization-examples/sft_finetuning_hyperpod
