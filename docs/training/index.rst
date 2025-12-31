Training
========

SageMaker Python SDK V3 revolutionizes machine learning training with the unified **ModelTrainer** class, replacing the complex framework-specific estimators from V2. This modern approach provides a consistent interface across all training scenarios while maintaining the power and flexibility you need.

Key Benefits of V3 Training
---------------------------

* **Unified Interface**: Single ``ModelTrainer`` class replaces multiple framework-specific estimators
* **Simplified Configuration**: Object-oriented API with auto-generated configs aligned with AWS APIs
* **Reduced Boilerplate**: Streamlined workflows with intuitive interfaces
* **Enhanced Performance**: Modernized architecture for better training efficiency

Quick Start Example
-------------------

Here's how training has evolved from V2 to V3:

**SageMaker Python SDK V2:**

.. code-block:: python

   from sagemaker.estimator import Estimator
   
   estimator = Estimator(
       image_uri="my-training-image",
       role="arn:aws:iam::123456789012:role/SageMakerRole",
       instance_count=1,
       instance_type="ml.m5.xlarge",
       output_path="s3://my-bucket/output"
   )
   estimator.fit({"training": "s3://my-bucket/train"})

**SageMaker Python SDK V3:**

.. code-block:: python

   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import InputData

   trainer = ModelTrainer(
       training_image="my-training-image",
       role="arn:aws:iam::123456789012:role/SageMakerRole"
   )

   train_data = InputData(
       channel_name="training",
       data_source="s3://my-bucket/train"
   )

   trainer.train(input_data_config=[train_data])

ModelTrainer Overview
--------------------

The ``ModelTrainer`` class is the cornerstone of SageMaker Python SDK V3, providing a unified interface for all training scenarios. This single class replaces the complex web of framework-specific estimators from V2, offering:

**Unified Training Interface**
  One class handles PyTorch, TensorFlow, Scikit-learn, XGBoost, and custom containers

**Intelligent Defaults**
  Automatically configures optimal settings based on your training requirements

**Flexible Configuration**
  Object-oriented design with structured configs that align with AWS APIs

**Seamless Integration**
  Works seamlessly with SageMaker features like distributed training, spot instances, and hyperparameter tuning

.. code-block:: python

   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import InputData, ResourceConfig

   # Create trainer with intelligent defaults
   trainer = ModelTrainer(
       training_image="your-training-image",
       role="your-sagemaker-role"
   )

   # Configure training data
   train_data = InputData(
       channel_name="training",
       data_source="s3://your-bucket/train-data"
   )

   # Start training
   training_job = trainer.train(
       input_data_config=[train_data],
       resource_config=ResourceConfig(
           instance_type="ml.m5.xlarge",
           instance_count=1
       )
   )

Training Capabilities
---------------------

Model Fine-Tuning Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

V3 introduces powerful fine-tuning capabilities with four specialized trainer classes:

* **SFTTrainer** - Supervised fine-tuning for foundation models
* **DPOTrainer** - Direct preference optimization
* **RLAIFTrainer** - Reinforcement Learning from AI Feedback
* **RLVRTrainer** - Reinforcement Learning from Verifiable Rewards

**Quick Fine-Tuning Example:**

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.common import TrainingType

   trainer = SFTTrainer(
       model="meta-llama/Llama-2-7b-hf",
       training_type=TrainingType.LORA,
       model_package_group_name="my-models",
       training_dataset="s3://bucket/train.jsonl"
   )

   training_job = trainer.train()

Key Fine-Tuning Features
~~~~~~~~~~~~~~~~~~~~~~~~

* **LoRA & Full Fine-Tuning Support** - Choose between parameter-efficient LoRA (Low-Rank Adaptation) for faster training with reduced memory requirements, or full fine-tuning for maximum model customization and performance
* **MLflow Integration with Real-Time Metrics** - Monitor training progress with comprehensive metrics tracking, experiment comparison, and model versioning through integrated MLflow support
* **Multi-Platform Deployment** - Seamlessly deploy your fine-tuned models to Amazon SageMaker endpoints for real-time inference or Amazon Bedrock for foundation model serving
* **Comprehensive Evaluation Suite** - Validate model performance with 11 built-in benchmark evaluations including accuracy, perplexity, BLEU scores, and domain-specific metrics
* **Serverless Training Capabilities** - Scale training automatically without managing infrastructure, with pay-per-use pricing and automatic resource optimization

Supported Training Scenarios
----------------------------

Framework Support
~~~~~~~~~~~~~~~~~

* **PyTorch** - Deep learning with dynamic computation graphs
* **TensorFlow** - Production-ready machine learning at scale
* **Scikit-learn** - Classical machine learning algorithms
* **XGBoost** - Gradient boosting for structured data
* **Custom Containers** - Bring your own training algorithms

Training Types
~~~~~~~~~~~~~~

* **Single Instance Training** - Cost-effective training for smaller models
* **Multi-Instance Training** - Distributed training for large-scale models
* **Spot Instance Training** - Cost optimization with managed spot instances
* **Local Mode Training** - Development and debugging on local infrastructure

Advanced Features
~~~~~~~~~~~~~~~~~

* **Automatic Model Tuning** - Hyperparameter optimization at scale
* **Distributed Training** - Multi-node, multi-GPU training strategies
* **Checkpointing** - Resume training from saved states
* **Early Stopping** - Prevent overfitting with intelligent stopping criteria

Migration from V2
------------------

If you're migrating from V2, the key changes are:

* Replace framework-specific estimators (PyTorchEstimator, TensorFlowEstimator, etc.) with ``ModelTrainer``
* Use structured ``InputData`` configs instead of dictionary-based input specifications
* Leverage the new object-oriented API for cleaner, more maintainable code

Training Examples
-----------------

Explore comprehensive training examples that demonstrate V3 capabilities:

.. toctree::
   :maxdepth: 1

   ../v3-examples/training-examples/local-training-example
   ../v3-examples/training-examples/distributed-local-training-example
   ../v3-examples/training-examples/hyperparameter-training-example
   ../v3-examples/training-examples/jumpstart-training-example
   ../v3-examples/training-examples/custom-distributed-training-example
   ../v3-examples/training-examples/aws_batch/sm-training-queues_getting_started_with_model_trainer
