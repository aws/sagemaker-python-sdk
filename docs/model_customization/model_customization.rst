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

**CPTTrainer (Continued Pre-Training)** *(Nova models only)*
  Continue pre-training on a raw corpus to extend model knowledge in a specific domain.
  See the :doc:`Nova section <nova>` for CPT examples.

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

**Job Notifications**
  Receive SNS notifications when training jobs complete, fail, or stop. Uses EventBridge
  rules to route SageMaker Training Job status changes to your SNS topic. 

  The SDK creates one rule per unique config (topic + events + prefix). Re-running with 
  the same config reuses the existing rule. Different configs create separate rules.

  .. note::
    Supported for SMTJ (serverful and serverless) compute only. HyperPod is not currently supported.

  .. code-block:: python

    trainer = SFTTrainer(
        model="nova-textgeneration-micro",
        training_dataset="s3://my-bucket/train.jsonl",
        accept_eula=True,
        notifications={
            "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:my-topic",  # Required
            "events": ["Completed", "Failed"],    # Optional (default: Completed, Failed, Stopped)
            "job_name_prefix": "my-team-sft-",    # Optional: filter by job name
            "event_bus_arn": "arn:aws:events:us-east-1:123456789012:event-bus/custom-bus" # Optional
        },
    )
    job = trainer.train(wait=False)  # Notification sent on completion

    # Access the rule ARN
    print(trainer.notification_rule_arn)

    # List and manage rules
    rules = trainer.list_notification_rules()
    trainer.delete_notification_rule(rule_arn=trainer.notification_rule_arn)

  **Prerequisites:**

  - An SNS topic (and subscription) with a resource policy allowing ``events.amazonaws.com``. 
    To set up a topic and subscription, see `Creating an SNS topic and subscription <https://docs.aws.amazon.com/sns/latest/dg/sns-create-subscribe-endpoint-to-topic.html>`_. 
  - IAM permissions: ``events:PutRule``, ``events:PutTargets``, ``events:ListRules``,
    ``events:RemoveTargets``, ``events:DeleteRule``

**Monitoring: show_metrics()**
  Plot training metrics after a job completes. Works across all compute types.

  - **Nova models**: Metrics parsed from CloudWatch logs.
  - **OSS models**: Metrics pulled from MLflow.

  .. code-block:: python

    # Plot all available metrics
    df = trainer.show_metrics()

    # Plot specific metrics
    df = trainer.show_metrics(metrics=["training_loss", "lr"])

    # Filter by step range
    df = trainer.show_metrics(starting_step=10, ending_step=100)

    # Filter by time window
    from datetime import datetime
    df = trainer.show_metrics(
        start_time=datetime(2026, 1, 1, 10, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 0),
    )

  **After a kernel restart:**

  .. code-block:: python

    # Standalone (SMTJ)
    from sagemaker.train import plot_training_metrics
    plot_training_metrics("my-sft-job")

    # Re-attach (HyperPod — needs cluster name for log group resolution)
    from sagemaker.train import SFTTrainer
    from sagemaker.core.training.configs import HyperPodCompute

    trainer = SFTTrainer(
        model="nova-textgeneration-micro",
        training_dataset="s3://unused",
        compute=HyperPodCompute(cluster_name="my-cluster", instance_type="ml.p5.48xlarge")
    )
    trainer._latest_training_job = "my-hp-job"
    df = trainer.show_metrics()

**Monitoring: stream_logs()**
  Stream CloudWatch logs in real-time while a job is running.

  .. code-block:: python

    # Start training non-blocking
    job = trainer.train(wait=False)

    # Stream logs (blocks until job completes or Ctrl+C)
    trainer.stream_logs()

    # Custom polling interval (seconds)
    trainer.stream_logs(poll=10)

    # Stream from a specific start time - providing this will speed up execution.
    from datetime import datetime
    trainer.stream_logs(start_time=datetime(2026, 1, 1, 15, 0, 0))

  .. note::

    - **SMTJ**: Streaming auto-stops when the job reaches a terminal state.
    - **HyperPod**: Streaming runs until you press Ctrl+C. Logs may take a few minutes
      to propagate to CloudWatch on first run.


----


.. toctree::
   :maxdepth: 1
   :caption: Customization Techniques

   SFT Finetuning <../../v3-examples/model-customization-examples/sft_finetuning_example_notebook_pysdk_prod_v3>
   DPOTrainer Finetuning <../../v3-examples/model-customization-examples/dpo_trainer_example_notebook_v3_prod>
   RLAIF Finetuning <../../v3-examples/model-customization-examples/rlaif_finetuning_example_notebook_v3_prod>
   RLVR Finetuning <../../v3-examples/model-customization-examples/rlvr_finetuning_example_notebook_v3_prod>
   Fine-Tuning with Serverful Training Jobs <finetuning_serverful>
   Fine-Tuning with HyperPod <finetuning_hyperpod>
