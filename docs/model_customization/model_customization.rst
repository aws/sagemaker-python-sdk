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
