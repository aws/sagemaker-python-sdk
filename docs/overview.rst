Overview
========

.. note::
   This is the documentation for SageMaker Python SDK V3. For legacy V2 documentation, please visit `SageMaker Python SDK V2 Documentation <https://sagemaker.readthedocs.io/en/v2/>`_.

Welcome to SageMaker Python SDK V3 - a revolutionary approach to machine learning on Amazon SageMaker. Version 3.0 represents a significant milestone with modernized architecture, enhanced performance, and powerful new capabilities while maintaining our commitment to user experience and reliability.

What's New in V3
-----------------

.. raw:: html

   <div class="whats-new-container">
     <div class="new-feature-card">
       <h3>Model Customization (V3 Exclusive)</h3>
       <p>Revolutionary foundation model fine-tuning with specialized trainers:</p>
       <ul>
         <li><strong>SFTTrainer</strong> - Supervised fine-tuning for task-specific adaptation</li>
         <li><strong>DPOTrainer</strong> - Direct preference optimization without RL complexity</li>
         <li><strong>RLAIFTrainer</strong> - Reinforcement learning from AI feedback</li>
         <li><strong>RLVRTrainer</strong> - Reinforcement learning from verifiable rewards</li>
       </ul>
       <p><em>Advanced techniques like LoRA, preference optimization, and RLHF that simply don't exist in V2.</em></p>
     </div>

     <div class="new-feature-card">
       <h3>Modular Architecture</h3>
       <p>Separate PyPI packages for specialized capabilities:</p>
       <ul>
         <li><strong>sagemaker-core</strong> - Low-level SageMaker resource management</li>
         <li><strong>sagemaker-train</strong> - Unified training with ModelTrainer</li>
         <li><strong>sagemaker-serve</strong> - Simplified inference with ModelBuilder</li>
         <li><strong>sagemaker-mlops</strong> - ML operations and pipeline management</li>
       </ul>
     </div>

     <div class="new-feature-card">
       <h3>Unified Classes</h3>
       <p>Single classes replace multiple framework-specific implementations:</p>
       <ul>
         <li><strong>ModelTrainer</strong> replaces PyTorchEstimator, TensorFlowEstimator, SKLearnEstimator, etc.</li>
         <li><strong>ModelBuilder</strong> replaces PyTorchModel, TensorFlowModel, SKLearnModel, etc.</li>
       </ul>
     </div>

     <div class="new-feature-card">
       <h3>Object-Oriented API</h3>
       <p>Structured interface with auto-generated configs aligned with AWS APIs for better developer experience.</p>
     </div>
   </div>

Capabilities

Training with ModelTrainer
---------------------------

Unified training interface replacing framework-specific estimators with intelligent defaults and streamlined workflows:

.. code-block:: python

   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import InputData

   trainer = ModelTrainer(
       training_image="your-training-image",
       role="your-sagemaker-role"
   )

   train_data = InputData(
       channel_name="training",
       data_source="s3://your-bucket/train-data"
   )

   training_job = trainer.train(input_data_config=[train_data])

:doc:`Learn more about Training <training/index>`

Inference with ModelBuilder
----------------------------

Simplified model deployment and inference with the V3 workflow: ModelBuilder() → build() → deploy() → invoke():

.. code-block:: python

   from sagemaker.serve import ModelBuilder

   model_builder = ModelBuilder(
       model="your-model",
       model_path="s3://your-bucket/model-artifacts"
   )

   model = model_builder.build(model_name="my-model")
   endpoint = model_builder.deploy(
       endpoint_name="my-endpoint",
       instance_type="ml.m5.xlarge",
       initial_instance_count=1
   )
   result = endpoint.invoke(
       body={"inputs": "your-input-data"},
       content_type="application/json"
   )

:doc:`Learn more about Inference <inference/index>`

ML Operations
-------------

Comprehensive MLOps capabilities for building, deploying, and managing machine learning workflows at scale:

.. code-block:: python

   from sagemaker.mlops import Pipeline, TrainingStep, ModelStep

   pipeline = Pipeline(name="production-ml-pipeline")
   
   training_step = TrainingStep(
       name="train-model",
       training_config=TrainingConfig(
           algorithm_specification={
               "training_image": "your-training-image"
           }
       )
   )
   
   pipeline.add_step(training_step)

:doc:`Learn more about ML Operations <ml_ops/index>`

SageMaker Core
--------------

Low-level, object-oriented access to Amazon SageMaker resources with intelligent defaults and type safety:

.. code-block:: python

   from sagemaker.core.resources import TrainingJob

   training_job = TrainingJob.create(
       training_job_name="my-training-job",
       role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
       input_data_config=[{
           "channel_name": "training",
           "data_source": "s3://my-bucket/train"
       }]
   )

:doc:`Learn more about SageMaker Core <sagemaker_core/index>`

Getting Started
===============

Installation
------------

:doc:`Install SageMaker Python SDK V3 <installation>` to get started

Migration from V2
------------------

Key changes when migrating from V2:

* Replace Estimator classes with ``ModelTrainer``
* Replace Model classes with ``ModelBuilder``
* Use structured config objects instead of parameter dictionaries
* Leverage specialized fine-tuning trainers for foundation models

Next Steps
-----------

**Get Started**: Follow the :doc:`quickstart` guide for a hands-on introduction
