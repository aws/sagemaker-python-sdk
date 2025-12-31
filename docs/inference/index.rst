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
   endpoint = model_builder.build()
   result = endpoint.invoke(data)

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
   from sagemaker.serve.configs import EndpointConfig

   # Create model builder with intelligent defaults
   model_builder = ModelBuilder(
       model="your-model",
       model_path="s3://your-bucket/model-artifacts",
       role="your-sagemaker-role"
   )

   # Configure endpoint settings
   endpoint_config = EndpointConfig(
       instance_type="ml.m5.xlarge",
       initial_instance_count=1,
       auto_scaling_enabled=True
   )

   # Deploy model
   endpoint = model_builder.build(endpoint_config=endpoint_config)
   
   # Make predictions
   response = endpoint.invoke({"inputs": "your-input-data"})

Inference Capabilities
----------------------

Model Optimization Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

V3 introduces powerful model optimization capabilities for enhanced performance:

* **SageMaker Neo** - Optimize models for specific hardware targets
* **TensorRT Integration** - Accelerate deep learning inference on NVIDIA GPUs
* **ONNX Runtime** - Cross-platform model optimization and acceleration
* **Quantization Support** - Reduce model size and improve inference speed

**Quick Optimization Example:**

.. code-block:: python

   from sagemaker.serve import ModelBuilder
   from sagemaker.serve.configs import OptimizationConfig

   model_builder = ModelBuilder(
       model="huggingface-bert-base",
       optimization_config=OptimizationConfig(
           target_device="ml_inf1",
           optimization_level="O2",
           quantization_enabled=True
       )
   )

   optimized_endpoint = model_builder.build()

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

   ../v3-examples/inference-examples/inference-spec-example
   ../v3-examples/inference-examples/jumpstart-example
   ../v3-examples/inference-examples/optimize-example
   ../v3-examples/inference-examples/train-inference-e2e-example
   ../v3-examples/inference-examples/jumpstart-e2e-training-example
   ../v3-examples/inference-examples/local-mode-example
   ../v3-examples/inference-examples/huggingface-example
   ../v3-examples/inference-examples/in-process-mode-example
