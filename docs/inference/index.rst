Inference
=========

SageMaker Python SDK V3 transforms model deployment and inference with the unified **ModelBuilder** class, replacing the complex framework-specific model classes from V2. This modern approach provides a consistent interface for all inference scenarios while maintaining the flexibility and performance you need.

Key Benefits of V3 Inference
----------------------------

* **Unified Interface**: Single ``ModelBuilder`` class replaces multiple framework-specific model classes
* **Simplified Deployment**: Object-oriented API with intelligent defaults for endpoint configuration
* **Enhanced Performance**: Optimized inference pipelines with automatic scaling and load balancing
* **Multi-Modal Support**: Deploy models for real-time, batch, and serverless inference scenarios

Quick Start Example
-------------------

Here's how inference has evolved from V2 to V3:

**SageMaker Python SDK V2:**

.. code-block:: python

   from sagemaker.model import Model
   from sagemaker.predictor import Predictor
   
   model = Model(
       image_uri="my-inference-image",
       model_data="s3://my-bucket/model.tar.gz",
       role="arn:aws:iam::123456789012:role/SageMakerRole"
   )
   predictor = model.deploy(
       initial_instance_count=1,
       instance_type="ml.m5.xlarge"
   )
   result = predictor.predict(data)

**SageMaker Python SDK V3:**

.. code-block:: python

   from sagemaker.serve import ModelBuilder

   model_builder = ModelBuilder(
       model="my-model",
       model_path="s3://my-bucket/model.tar.gz"
   )

   model = model_builder.build(model_name="my-deployed-model")

   endpoint = model_builder.deploy(
       endpoint_name="my-endpoint",
       instance_type="ml.m5.xlarge",
       initial_instance_count=1
   )

   result = endpoint.invoke(
       body=data,
       content_type="application/json"
   )

ModelBuilder Overview
--------------------

The ``ModelBuilder`` class is the cornerstone of SageMaker Python SDK V3 inference, providing a unified interface for all deployment scenarios. This single class replaces the complex web of framework-specific model classes from V2, offering:

**Unified Deployment Interface**
  One class handles PyTorch, TensorFlow, Scikit-learn, XGBoost, HuggingFace, and custom containers

**Intelligent Optimization**
  Automatically optimizes model serving configuration based on your model characteristics

**Flexible Deployment Options**
  Support for real-time endpoints, batch transform, and serverless inference

**Seamless Integration**
  Works seamlessly with SageMaker features like auto-scaling, multi-model endpoints, and A/B testing

.. code-block:: python

   from sagemaker.serve import ModelBuilder

   model_builder = ModelBuilder(
       model="your-model",
       model_path="s3://your-bucket/model-artifacts",
       role="your-sagemaker-role"
   )

   model = model_builder.build(model_name="my-model")
   
   endpoint = model_builder.deploy(
       endpoint_name="my-endpoint",
       instance_type="ml.m5.xlarge",
       initial_instance_count=1
   )
   
   response = endpoint.invoke(
       body={"inputs": "your-input-data"},
       content_type="application/json"
   )

Inference Capabilities
----------------------

Model Optimization Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

V3 introduces powerful model optimization capabilities for enhanced performance:

* **SageMaker Neo** - Optimize models for specific hardware targets
* **TensorRT Integration** - Accelerate deep learning inference on NVIDIA GPUs
* **ONNX Runtime** - Cross-platform model optimization and acceleration
* **Quantization Support** - Reduce model size and improve inference speed

**Model Optimization Example:**

.. code-block:: python

   from sagemaker.serve import ModelBuilder

   # Create ModelBuilder with optimization settings
   model_builder = ModelBuilder(
       model="huggingface-bert-base",
       role="your-sagemaker-role"
   )

   # Build and deploy with optimization
   model = model_builder.build(model_name="optimized-bert")
   endpoint = model_builder.deploy(
       endpoint_name="bert-endpoint",
       instance_type="ml.inf1.xlarge",
       initial_instance_count=1
   )

Key Inference Features
~~~~~~~~~~~~~~~~~~~~~

* **Multi-Model Endpoints** - Host multiple models on a single endpoint with automatic model loading and unloading for cost optimization
* **Auto-Scaling Integration** - Automatically scale endpoint capacity based on traffic patterns with configurable scaling policies
* **A/B Testing Support** - Deploy multiple model variants with traffic splitting for safe model updates and performance comparison
* **Batch Transform Jobs** - Process large datasets efficiently with automatic data partitioning and parallel processing
* **Serverless Inference** - Pay-per-request pricing with automatic scaling from zero to handle variable workloads

Supported Inference Scenarios
-----------------------------

Deployment Types
~~~~~~~~~~~~~~~

* **Real-Time Endpoints** - Low-latency inference for interactive applications
* **Batch Transform** - High-throughput processing for large datasets
* **Serverless Inference** - Cost-effective inference for variable workloads
* **Multi-Model Endpoints** - Host multiple models on shared infrastructure

Framework Support
~~~~~~~~~~~~~~~~~

* **PyTorch** - Deep learning models with dynamic computation graphs
* **TensorFlow** - Production-ready machine learning models at scale
* **Scikit-learn** - Classical machine learning algorithms
* **XGBoost** - Gradient boosting models for structured data
* **HuggingFace** - Pre-trained transformer models for NLP tasks
* **Custom Containers** - Bring your own inference logic and dependencies

Advanced Features
~~~~~~~~~~~~~~~~

* **Model Monitoring** - Track model performance and data drift in production
* **Endpoint Security** - VPC support, encryption, and IAM-based access control
* **Multi-AZ Deployment** - High availability with automatic failover
* **Custom Inference Logic** - Implement preprocessing, postprocessing, and custom prediction logic

Migration from V2
------------------

If you're migrating from V2, the key changes are:

* Replace framework-specific model classes (PyTorchModel, TensorFlowModel, etc.) with ``ModelBuilder``
* Use structured configuration objects instead of parameter dictionaries
* Leverage the new ``invoke()`` method instead of ``predict()`` for more consistent API
* Take advantage of built-in optimization and auto-scaling features

Inference Examples
-----------------

Explore comprehensive inference examples that demonstrate V3 capabilities:

.. toctree::
   :maxdepth: 1

   Custom InferenceSpec <../v3-examples/inference-examples/inference-spec-example>
   ModelBuilder with JumpStart models <../v3-examples/inference-examples/jumpstart-example>
   Optimize a JumpStart model <../v3-examples/inference-examples/optimize-example>
   Train-to-Inference E2E <../v3-examples/inference-examples/train-inference-e2e-example>
   JumpStart E2E <../v3-examples/inference-examples/jumpstart-e2e-training-example>
   Local Container Mode <../v3-examples/inference-examples/local-mode-example>
   Deploy HuggingFace Models <../v3-examples/inference-examples/huggingface-example>
   ModelBuilder in In-Process mode <../v3-examples/inference-examples/in-process-mode-example>
