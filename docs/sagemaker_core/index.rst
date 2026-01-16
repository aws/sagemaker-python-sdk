SageMaker Core
==============

SageMaker Core provides low-level, object-oriented access to Amazon SageMaker resources with intelligent defaults and streamlined workflows. This foundational layer offers direct control over SageMaker services while maintaining the simplicity and power you need for advanced use cases.

Key Benefits of SageMaker Core
------------------------------

* **Direct Resource Access**: Low-level control over SageMaker resources with full API coverage
* **Object-Oriented Design**: Intuitive resource abstractions that map directly to AWS APIs
* **Intelligent Defaults**: Automatic configuration of optimal settings based on resource requirements
* **Type Safety**: Strong typing and validation for better development experience

Quick Start Example
-------------------

Here's how SageMaker Core simplifies resource management:

**Traditional Boto3 Approach:**

.. code-block:: python

   import boto3
   
   client = boto3.client('sagemaker')
   response = client.create_training_job(
       TrainingJobName='my-training-job',
       RoleArn='arn:aws:iam::123456789012:role/SageMakerRole',
       InputDataConfig=[{
           'ChannelName': 'training',
           'DataSource': {
               'S3DataSource': {
                   'S3DataType': 'S3Prefix',
                   'S3Uri': 's3://my-bucket/train',
                   'S3DataDistributionType': 'FullyReplicated'
               }
           }
       }],
       # ... many more required parameters
   )

**SageMaker Core Approach:**

.. code-block:: python

   from sagemaker.core.resources import TrainingJob
   from sagemaker.core.shapes import TrainingJobConfig

   training_job = TrainingJob.create(
       training_job_name="my-training-job",
       role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
       input_data_config=[{
           "channel_name": "training",
           "data_source": "s3://my-bucket/train"
       }]
   )

SageMaker Core Overview
----------------------

SageMaker Core serves as the foundation for all SageMaker Python SDK V3 operations, providing direct access to SageMaker resources through an object-oriented interface:

**Resource Abstractions**
  Direct mapping to SageMaker resources like TrainingJob, Model, Endpoint, and ProcessingJob

**Intelligent Configuration**
  Automatically fills in required parameters with sensible defaults while allowing full customization

**Type-Safe Operations**
  Strong typing and validation prevent common configuration errors

**Seamless Integration**
  Works as the foundation layer for higher-level SDK components

.. code-block:: python

   from sagemaker.core.resources import Endpoint, Model
   from sagemaker.core.shapes import EndpointConfig

   # Create a model resource
   model = Model.create(
       model_name="my-model",
       primary_container={
           "image": "your-inference-image",
           "model_data_url": "s3://your-bucket/model.tar.gz"
       },
       execution_role_arn="your-sagemaker-role"
   )

   # Deploy to an endpoint
   endpoint = Endpoint.create(
       endpoint_name="my-endpoint",
       endpoint_config_name="my-config",
       model_name=model.model_name
   )

   # Make predictions
   response = endpoint.invoke_endpoint(
       body=b'{"instances": [1, 2, 3, 4]}',
       content_type="application/json"
   )

Core Capabilities
-----------------

Resource Management
~~~~~~~~~~~~~~~~~~

SageMaker Core provides comprehensive resource management capabilities:

* **Training Jobs** - Create, monitor, and manage training workloads with full parameter control
* **Models** - Define and register models with custom inference logic and container configurations
* **Endpoints** - Deploy real-time inference endpoints with auto-scaling and monitoring
* **Processing Jobs** - Run data processing and feature engineering workloads at scale

**Resource Lifecycle Management:**

.. code-block:: python

   from sagemaker.core.resources import ProcessingJob

   # Create processing job
   processing_job = ProcessingJob.create(
       processing_job_name="data-preprocessing",
       app_specification={
           "image_uri": "your-processing-image",
           "container_entrypoint": ["python", "preprocess.py"]
       },
       processing_inputs=[{
           "input_name": "raw-data",
           "s3_input": {
               "s3_uri": "s3://your-bucket/raw-data",
               "local_path": "/opt/ml/processing/input"
           }
       }],
       processing_outputs=[{
           "output_name": "processed-data",
           "s3_output": {
               "s3_uri": "s3://your-bucket/processed-data",
               "local_path": "/opt/ml/processing/output"
           }
       }]
   )

Key Core Features
~~~~~~~~~~~~~~~~

* **Direct API Access** - Full coverage of SageMaker APIs with object-oriented abstractions for better usability
* **Intelligent Defaults** - Automatic parameter inference and validation reduces boilerplate while maintaining flexibility
* **Resource Chaining** - Seamlessly connect resources together for complex workflows and dependencies
* **Monitoring Integration** - Built-in support for CloudWatch metrics, logging, and resource status tracking
* **Error Handling** - Comprehensive error handling with detailed feedback for troubleshooting and debugging

Supported Core Scenarios
------------------------

Resource Types
~~~~~~~~~~~~~

* **Training Resources** - TrainingJob, HyperParameterTuningJob, AutoMLJob
* **Inference Resources** - Model, EndpointConfig, Endpoint, Transform
* **Processing Resources** - ProcessingJob, FeatureGroup, Pipeline
* **Monitoring Resources** - ModelQualityJobDefinition, DataQualityJobDefinition

Advanced Features
~~~~~~~~~~~~~~~~

* **Batch Operations** - Efficiently manage multiple resources with batch create, update, and delete operations
* **Resource Tagging** - Comprehensive tagging support for cost allocation, governance, and resource organization
* **Cross-Region Support** - Deploy and manage resources across multiple AWS regions with unified interface
* **Custom Configurations** - Override any default behavior with custom configurations and parameters

Integration Patterns
~~~~~~~~~~~~~~~~~~~

* **Pipeline Integration** - Use Core resources as building blocks for SageMaker Pipelines
* **Event-Driven Workflows** - Integrate with AWS Lambda and EventBridge for automated workflows
* **Multi-Account Deployments** - Deploy resources across multiple AWS accounts with proper IAM configuration

Migration from Boto3
--------------------

If you're migrating from direct Boto3 usage, the key benefits are:

* **Simplified Interface**: Object-oriented resources replace complex dictionary-based API calls
* **Intelligent Defaults**: Automatic parameter inference reduces configuration overhead
* **Type Safety**: Strong typing prevents common configuration errors
* **Better Error Messages**: More descriptive error handling and validation feedback

SageMaker Core Examples
----------------------

Explore comprehensive SageMaker Core examples:

.. toctree::
   :maxdepth: 1

   ../sagemaker-core/example_notebooks/get_started
   ../sagemaker-core/example_notebooks/sagemaker_core_overview
   ../sagemaker-core/example_notebooks/intelligent_defaults_and_logging
