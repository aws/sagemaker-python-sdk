Model Customization
===================

.. raw:: html

   <div class="v3-exclusive-feature">
   <strong>🆕 V3 EXCLUSIVE FEATURE</strong><br>
   Model customization with specialized trainers is only available in SageMaker Python SDK V3. 
   This powerful capability was built from the ground up for foundation model fine-tuning.
   </div>

SageMaker Python SDK V3 revolutionizes foundation model fine-tuning with specialized trainer classes, making it easier than ever to customize large language models and foundation models for your specific use cases. This modern approach provides powerful fine-tuning capabilities while maintaining simplicity and performance.

Key Benefits of V3 Model Customization
--------------------------------------

* **Specialized Trainers**: Purpose-built classes for different fine-tuning approaches (SFT, DPO, RLAIF, RLVR)
* **Foundation Model Focus**: Optimized for large language models and transformer architectures
* **Advanced Techniques**: Support for cutting-edge fine-tuning methods like RLHF and preference optimization
* **Production Ready**: Built-in evaluation, monitoring, and deployment capabilities

Quick Start Example
-------------------

Here's how model customization works in V3:

**Supervised Fine-Tuning (SFT):**

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.common import TrainingType

   # Create SFT trainer for foundation model fine-tuning
   trainer = SFTTrainer(
       model="meta-llama/Llama-2-7b-hf",
       training_type=TrainingType.LORA,
       model_package_group_name="my-custom-models",
       training_dataset="s3://my-bucket/training-data.jsonl"
   )

   # Start fine-tuning
   training_job = trainer.train()

**Direct Preference Optimization (DPO):**

.. code-block:: python

   from sagemaker.train import DPOTrainer

   # Create DPO trainer for preference-based fine-tuning
   dpo_trainer = DPOTrainer(
       model="my-base-model",
       preference_dataset="s3://my-bucket/preference-data.jsonl",
       training_type=TrainingType.LORA
   )

   # Train with human preferences
   dpo_job = dpo_trainer.train()

Fine-Tuning Trainers Overview
-----------------------------

SageMaker Python SDK V3 provides four specialized trainer classes for different model customization approaches:

**SFTTrainer (Supervised Fine-Tuning)**
  Traditional fine-tuning with labeled datasets for task-specific adaptation

**DPOTrainer (Direct Preference Optimization)**
  Fine-tune models using human preference data without reinforcement learning complexity

**RLAIFTrainer (Reinforcement Learning from AI Feedback)**
  Use AI-generated feedback to improve model behavior and alignment

**RLVRTrainer (Reinforcement Learning from Verifiable Rewards)**
  Fine-tune with verifiable reward signals for objective optimization

.. code-block:: python

   from sagemaker.train import SFTTrainer, DPOTrainer, RLAIFTrainer, RLVRTrainer
   from sagemaker.train.common import TrainingType
   from sagemaker.train.configs import LoRAConfig

   # Configure LoRA for parameter-efficient fine-tuning
   lora_config = LoRAConfig(
       rank=16,
       alpha=32,
       dropout=0.1,
       target_modules=["q_proj", "v_proj"]
   )

   # Choose your fine-tuning approach
   sft_trainer = SFTTrainer(
       model="huggingface-model-id",
       training_dataset="s3://bucket/sft-data.jsonl",
       lora_config=lora_config
   )

   # Or use preference optimization
   dpo_trainer = DPOTrainer(
       model="base-model",
       preference_dataset="s3://bucket/preferences.jsonl",
       lora_config=lora_config
   )

Model Customization Capabilities
--------------------------------

Advanced Fine-Tuning Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

V3 supports state-of-the-art fine-tuning methods for foundation models:

* **LoRA (Low-Rank Adaptation)** - Parameter-efficient fine-tuning with minimal memory requirements
* **Full Fine-Tuning** - Complete model parameter updates for maximum customization
* **Preference Learning** - Train models using human feedback and preference data
* **Reinforcement Learning** - Advanced alignment techniques for improved model behavior

**Parameter-Efficient Fine-Tuning Example:**

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.configs import LoRAConfig, TrainingConfig

   # Configure LoRA for efficient fine-tuning
   lora_config = LoRAConfig(
       rank=8,
       alpha=16,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
   )

   # Set up training configuration
   training_config = TrainingConfig(
       learning_rate=2e-4,
       batch_size=4,
       gradient_accumulation_steps=4,
       max_steps=1000
   )

   trainer = SFTTrainer(
       model="microsoft/DialoGPT-medium",
       training_dataset="s3://bucket/conversation-data.jsonl",
       lora_config=lora_config,
       training_config=training_config
   )

Key Model Customization Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Parameter-Efficient Training** - LoRA and other techniques reduce memory usage by up to 90% while maintaining performance quality
* **Multi-GPU Support** - Distributed training across multiple GPUs with automatic parallelization and gradient synchronization
* **Custom Evaluation Metrics** - Built-in support for 11 evaluation benchmarks including BLEU, ROUGE, perplexity, and domain-specific metrics
* **MLflow Integration** - Comprehensive experiment tracking with real-time metrics, model versioning, and artifact management
* **Flexible Deployment** - Deploy fine-tuned models to SageMaker endpoints, Bedrock, or export for external use

Supported Model Customization Scenarios
---------------------------------------

Model Types
~~~~~~~~~~~

* **Large Language Models** - GPT, LLaMA, BERT, T5, and other transformer architectures
* **Conversational AI** - ChatGPT-style models, dialogue systems, and virtual assistants
* **Domain-Specific Models** - Legal, medical, financial, and technical domain adaptation
* **Multimodal Models** - Vision-language models and cross-modal understanding

Fine-Tuning Approaches
~~~~~~~~~~~~~~~~~~~~~~

* **Task-Specific Adaptation** - Fine-tune for specific downstream tasks like summarization, QA, or classification
* **Instruction Following** - Train models to follow complex instructions and multi-step reasoning
* **Safety and Alignment** - Improve model behavior, reduce harmful outputs, and align with human values
* **Style and Persona** - Customize model personality, writing style, and response patterns

Evaluation and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Automated Benchmarking** - Built-in evaluation on standard benchmarks and custom metrics
* **Human Evaluation Integration** - Tools for collecting and incorporating human feedback
* **Performance Monitoring** - Track model quality, safety, and alignment metrics during training
* **A/B Testing Support** - Compare different fine-tuning approaches and model variants

Migration from V2
------------------

V3 introduces entirely new capabilities for model customization that weren't available in V2:

* **New Specialized Trainers**: SFTTrainer, DPOTrainer, RLAIFTrainer, and RLVRTrainer are V3-exclusive
* **Foundation Model Focus**: V2 primarily supported traditional ML models; V3 is optimized for LLMs
* **Advanced Techniques**: Preference learning and RLHF capabilities are new in V3
* **Integrated Evaluation**: Built-in benchmarking and evaluation tools replace manual evaluation workflows

Model Customization Examples
----------------------------

Explore comprehensive model customization examples that demonstrate V3 capabilities:

.. toctree::
   :maxdepth: 1

   ../v3-examples/model-customization-examples/sft_finetuning_example_notebook_pysdk_prod_v3
   ../v3-examples/model-customization-examples/dpo_trainer_example_notebook_v3_prod
   ../v3-examples/model-customization-examples/rlaif_finetuning_example_notebook_v3_prod
   ../v3-examples/model-customization-examples/rlvr_finetuning_example_notebook_v3_prod
   ../v3-examples/model-customization-examples/llm_as_judge_demo
   ../v3-examples/model-customization-examples/custom_scorer_demo
   ../v3-examples/model-customization-examples/benchmark_demo
   ../v3-examples/model-customization-examples/bedrock-modelbuilder-deployment
   ../v3-examples/model-customization-examples/model_builder_deployment_notebook
   ../v3-examples/model-customization-examples/ai_registry_example
