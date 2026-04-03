Implement MLOps
=============

SageMaker Python SDK V3 provides comprehensive MLOps capabilities for building, deploying, and managing machine learning workflows at scale. This includes advanced pipeline orchestration, model monitoring, data quality checks, and automated deployment strategies for production ML systems.

Key Benefits of V3 ML Operations
--------------------------------

* **Unified Pipeline Interface**: Streamlined workflow orchestration with intelligent step dependencies
* **Advanced Monitoring**: Built-in model quality, data drift, and bias detection capabilities
* **Automated Governance**: Model registry integration with approval workflows and lineage tracking
* **Production-Ready**: Enterprise-grade features for compliance, security, and scalability

Quick Start Example
-------------------

Here's how ML Operations workflows are simplified in V3:

**Traditional Pipeline Approach:**

.. code-block:: python

   from sagemaker.workflow.pipeline import Pipeline
   from sagemaker.workflow.steps import TrainingStep, ProcessingStep
   from sagemaker.sklearn.processing import SKLearnProcessor
   
   # Complex setup with multiple framework-specific classes
   processor = SKLearnProcessor(
       framework_version="0.23-1",
       role=role,
       instance_type="ml.m5.xlarge",
       instance_count=1
   )
   
   processing_step = ProcessingStep(
       name="PreprocessData",
       processor=processor,
       # ... many configuration parameters
   )

**SageMaker V3 MLOps Approach:**

.. code-block:: python

   from sagemaker.mlops import Pipeline, ProcessingStep
   from sagemaker.mlops.configs import ProcessingConfig

   # Simplified configuration with intelligent defaults
   pipeline = Pipeline(name="ml-workflow")
   
   processing_step = ProcessingStep(
       name="preprocess-data",
       processing_config=ProcessingConfig(
           image_uri="sklearn-processing-image",
           instance_type="ml.m5.xlarge"
       ),
       inputs={"raw_data": "s3://bucket/raw-data"},
       outputs={"processed_data": "s3://bucket/processed-data"}
   )
   
   pipeline.add_step(processing_step)

MLOps Pipeline Overview
----------------------

SageMaker V3 MLOps provides a unified interface for building and managing end-to-end machine learning workflows:

**Pipeline Orchestration**
  Intelligent step dependencies with automatic resource management and error handling

**Model Registry Integration**
  Seamless model versioning, approval workflows, and deployment automation

**Quality Monitoring**
  Built-in data quality, model performance, and bias detection capabilities

**Governance and Compliance**
  Comprehensive lineage tracking, audit trails, and approval mechanisms

.. code-block:: python

   from sagemaker.mlops import Pipeline, TrainingStep, ModelStep, EndpointStep
   from sagemaker.mlops.configs import ModelConfig, EndpointConfig

   # Create comprehensive ML pipeline
   pipeline = Pipeline(name="production-ml-pipeline")

   # Training step
   training_step = TrainingStep(
       name="train-model",
       training_config=TrainingConfig(
           algorithm_specification={
               "training_image": "your-training-image"
           }
       )
   )

   # Model registration step
   model_step = ModelStep(
       name="register-model",
       model_config=ModelConfig(
           model_package_group_name="production-models",
           approval_status="PendingManualApproval"
       ),
       depends_on=[training_step]
   )

   # Deployment step
   endpoint_step = EndpointStep(
       name="deploy-model",
       endpoint_config=EndpointConfig(
           instance_type="ml.m5.xlarge",
           initial_instance_count=1
       ),
       depends_on=[model_step]
   )

   pipeline.add_steps([training_step, model_step, endpoint_step])

MLOps Capabilities
------------------

Advanced Pipeline Features
~~~~~~~~~~~~~~~~~~~~~~~~~

V3 introduces powerful pipeline capabilities for production ML workflows:

* **Conditional Execution** - Dynamic pipeline paths based on data quality checks and model performance
* **Parallel Processing** - Automatic parallelization of independent pipeline steps for faster execution
* **Resource Optimization** - Intelligent resource allocation and cost optimization across pipeline steps
* **Failure Recovery** - Automatic retry mechanisms and checkpoint-based recovery for robust workflows

**Advanced Pipeline Example:**

.. code-block:: python

   from sagemaker.mlops import Pipeline, ConditionStep, ParallelStep
   from sagemaker.mlops.conditions import ModelAccuracyCondition

   pipeline = Pipeline(name="advanced-ml-pipeline")

   # Conditional model deployment based on accuracy
   accuracy_condition = ModelAccuracyCondition(
       threshold=0.85,
       metric_name="validation:accuracy"
   )

   condition_step = ConditionStep(
       name="check-model-quality",
       condition=accuracy_condition,
       if_steps=[deploy_to_production_step],
       else_steps=[retrain_model_step]
   )

   pipeline.add_step(condition_step)

Key MLOps Features
~~~~~~~~~~~~~~~~~

* **Model Registry Integration** - Centralized model versioning with automated approval workflows and deployment tracking
* **Data Quality Monitoring** - Continuous monitoring of data drift, schema changes, and statistical anomalies in production
* **Model Performance Tracking** - Real-time monitoring of model accuracy, latency, and business metrics with alerting
* **Bias Detection and Fairness** - Built-in bias detection across protected attributes with automated reporting and remediation
* **Automated Retraining** - Trigger-based model retraining based on performance degradation or data drift detection

Supported MLOps Scenarios
-------------------------

Pipeline Types
~~~~~~~~~~~~~

* **Training Pipelines** - End-to-end model training with data preprocessing, feature engineering, and validation
* **Inference Pipelines** - Real-time and batch inference workflows with preprocessing and postprocessing
* **Data Processing Pipelines** - ETL workflows for feature engineering, data validation, and preparation
* **Model Deployment Pipelines** - Automated deployment with A/B testing, canary releases, and rollback capabilities

Monitoring and Governance
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Model Monitoring** - Continuous tracking of model performance, data quality, and operational metrics
* **Compliance Reporting** - Automated generation of audit reports for regulatory compliance and governance
* **Lineage Tracking** - Complete data and model lineage from raw data to production predictions
* **Access Control** - Fine-grained permissions and approval workflows for model deployment and updates

Integration Patterns
~~~~~~~~~~~~~~~~~~~

* **CI/CD Integration** - Seamless integration with GitHub Actions, Jenkins, and other CI/CD platforms
* **Event-Driven Workflows** - Trigger pipelines based on data availability, model performance, or business events
* **Multi-Environment Deployment** - Automated promotion of models across development, staging, and production environments

Migration from V2
------------------

If you're migrating MLOps workflows from V2, the key improvements are:

* **Simplified Pipeline Definition**: Unified interface replaces complex framework-specific configurations
* **Enhanced Monitoring**: Built-in model and data quality monitoring replaces custom solutions
* **Improved Governance**: Integrated model registry and approval workflows streamline compliance
* **Better Resource Management**: Automatic resource optimization and cost management across workflows

Lineage Tracking
~~~~~~~~~~~~~~~~


SageMaker Lineage enables tracing events across your ML workflow via a graph structure. V3 provides lineage tracking through ``sagemaker.core.lineage`` with support for:


- **Contexts** - Logical grouping of lineage entities under workflow contexts
- **Actions** - Recording computational steps like model builds and transformations
- **Artifacts** - Registering data inputs, labels, and trained models
- **Associations** - Directed edges linking entities to form the lineage graph
- **Traversal** - Querying relationships between entities for reporting and analysis

.. code-block:: python

   from sagemaker.core.lineage.context import Context
   from sagemaker.core.lineage.action import Action
   from sagemaker.core.lineage.artifact import Artifact
   from sagemaker.core.lineage.association import Association

   # Create a workflow context
   context = Context.create(
       context_name="my-ml-workflow",
       context_type="MLWorkflow",
       source_uri="workflow-source",
   )

   # Create an action and associate it with the context
   action = Action.create(
       action_name="model-build-step",
       action_type="ModelBuild",
       source_uri="build-source",
   )

   Association.create(
       source_arn=context.context_arn,
       destination_arn=action.action_arn,
       association_type="AssociatedWith",
   )

:doc:`Learn more about Lineage Tracking <lineage>`

ML Operations Examples
----------------------


E2E Pipeline with Model Registry
----------------------------------


Build a SageMaker Pipeline that preprocesses data, trains a model, and registers it to the Model Registry.

.. code-block:: python

   from sagemaker.mlops.workflow.pipeline import Pipeline
   from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
   from sagemaker.mlops.workflow.model_step import ModelStep
   from sagemaker.core.processing import ScriptProcessor
   from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input, ProcessingOutput, ProcessingS3Output
   from sagemaker.core.workflow.parameters import ParameterString
   from sagemaker.core.workflow.pipeline_context import PipelineSession
   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import InputData, Compute
   from sagemaker.serve.model_builder import ModelBuilder

   pipeline_session = PipelineSession()

   # Processing step
   processor = ScriptProcessor(image_uri=sklearn_image, instance_type="ml.m5.xlarge", ...)
   step_process = ProcessingStep(name="Preprocess", step_args=processor.run(...))

   # Training step
   trainer = ModelTrainer(training_image=xgboost_image, compute=Compute(instance_type="ml.m5.xlarge"), ...)
   step_train = TrainingStep(name="Train", step_args=trainer.train())

   # Register model
   model_builder = ModelBuilder(
       s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
       image_uri=xgboost_image, role_arn=role, sagemaker_session=pipeline_session,
   )
   step_register = ModelStep(name="Register", step_args=model_builder.register(
       model_package_group_name="my-group", approval_status="Approved",
   ))

   pipeline = Pipeline(name="my-pipeline", steps=[step_process, step_train, step_register], sagemaker_session=pipeline_session)
   pipeline.upsert(role_arn=role)
   pipeline.start()

:doc:`Full example notebook <../v3-examples/ml-ops-examples/v3-pipeline-train-create-registry>`



Processing Jobs
----------------


Run data preprocessing with ``ScriptProcessor`` (sklearn) or ``FrameworkProcessor`` (PyTorch).

.. code-block:: python

   from sagemaker.core.processing import ScriptProcessor
   from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input, ProcessingOutput, ProcessingS3Output

   processor = ScriptProcessor(
       image_uri=image_uris.retrieve(framework="sklearn", region=region, version="1.2-1", py_version="py3", instance_type="ml.m5.xlarge"),
       instance_type="ml.m5.xlarge", instance_count=1, role=role,
   )

   processor.run(
       inputs=[ProcessingInput(input_name="input-1", s3_input=ProcessingS3Input(s3_uri=input_data, local_path="/opt/ml/processing/input", s3_data_type="S3Prefix"))],
       outputs=[ProcessingOutput(output_name="train", s3_output=ProcessingS3Output(s3_uri="s3://bucket/train", local_path="/opt/ml/processing/train", s3_upload_mode="EndOfJob"))],
       code="code/preprocess.py",
       arguments=["--input-data", input_data],
   )

:doc:`SKLearn example <../v3-examples/ml-ops-examples/v3-processing-job-sklearn>` · :doc:`PyTorch example <../v3-examples/ml-ops-examples/v3-processing-job-pytorch/v3-pytorch-processing-example>`



Batch Transform Jobs
---------------------


Run batch inference on large datasets using ``Transformer``.

.. code-block:: python

   from sagemaker.core.transformer import Transformer
   from sagemaker.serve.model_builder import ModelBuilder

   model_builder = ModelBuilder(image_uri=xgboost_image, s3_model_data_url=model_url, role_arn=role)
   model_builder.build(model_name="my-transform-model")

   transformer = Transformer(
       model_name="my-transform-model", instance_count=1, instance_type="ml.m5.xlarge",
       accept="text/csv", assemble_with="Line", output_path="s3://bucket/output",
   )
   transformer.transform("s3://bucket/input", content_type="text/csv", split_type="Line", input_filter="$[1:]")

:doc:`Full example notebook <../v3-examples/ml-ops-examples/v3-transform-job-example>`



Hyperparameter Tuning
----------------------


Optimize hyperparameters with ``HyperparameterTuner`` using ``ContinuousParameter`` and ``CategoricalParameter`` ranges.

.. code-block:: python

   from sagemaker.train.tuner import HyperparameterTuner
   from sagemaker.core.parameter import ContinuousParameter, CategoricalParameter
   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import InputData

   trainer = ModelTrainer(training_image=pytorch_image, source_code=source_code, compute=compute, hyperparameters={"epochs": 1})

   tuner = HyperparameterTuner(
       model_trainer=trainer,
       objective_metric_name="average test loss",
       hyperparameter_ranges={"lr": ContinuousParameter(0.001, 0.1), "batch-size": CategoricalParameter([32, 64, 128])},
       metric_definitions=[{"Name": "average test loss", "Regex": "Test set: Average loss: ([0-9\\.]+)"}],
       max_jobs=3, max_parallel_jobs=2, strategy="Random", objective_type="Minimize",
   )

   tuner.tune(inputs=[InputData(channel_name="training", data_source=s3_data_uri)], wait=False)

:doc:`Standalone example <../v3-examples/ml-ops-examples/v3-hyperparameter-tuning-example/v3-hyperparameter-tuning-example>` · :doc:`Pipeline example <../v3-examples/ml-ops-examples/v3-hyperparameter-tuning-example/v3-hyperparameter-tuning-pipeline>`



Model Registry
---------------


Register models, create models from registry entries, and manage approval workflows.

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.core.resources import Model, ModelPackage

   # Register from artifact
   model_builder = ModelBuilder(s3_model_data_url=s3_url, image_uri=image_uri, role_arn=role)
   model_builder.build(model_name="my-model")
   model_builder.register(model_package_group_name="my-group", content_types=["application/json"], response_types=["application/json"], approval_status="Approved")

   # Create model from registry
   model_package = ModelPackage.get(model_package_name=registered_arn)
   model_builder = ModelBuilder(
       s3_model_data_url=model_package.inference_specification.containers[0].model_data_url,
       image_uri=model_package.inference_specification.containers[0].image, role_arn=role,
   )
   model_builder.build(model_name="model-from-registry")

:doc:`Full example notebook <../v3-examples/ml-ops-examples/v3-model-registry-example/v3-model-registry-example>`



Clarify Bias and Explainability
--------------------------------


Run pre-training bias analysis and SHAP explainability using ``SageMakerClarifyProcessor``.

.. code-block:: python

   from sagemaker.core.clarify import SageMakerClarifyProcessor, DataConfig, BiasConfig, SHAPConfig

   data_config = DataConfig(s3_data_input_path=data_uri, s3_output_path=output_uri, label="target", headers=headers, dataset_type="text/csv")
   bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="gender", facet_values_or_threshold=[1])

   clarify_processor = SageMakerClarifyProcessor(role=role, instance_count=1, instance_type="ml.m5.large")
   clarify_processor.run_pre_training_bias(data_config=data_config, data_bias_config=bias_config, methods=["CI", "DPL"])

:doc:`Full example notebook <../v3-examples/ml-ops-examples/v3-sagemaker-clarify>`



EMR Serverless Pipeline Step
-----------------------------


Run PySpark jobs on EMR Serverless within a SageMaker Pipeline.

.. code-block:: python

   from sagemaker.mlops.workflow.emr_serverless_step import EMRServerlessStep, EMRServerlessJobConfig
   from sagemaker.mlops.workflow.pipeline import Pipeline

   job_config = EMRServerlessJobConfig(
       job_driver={"sparkSubmit": {"entryPoint": script_uri, "entryPointArguments": ["--input", input_uri, "--output", output_uri]}},
       execution_role_arn=emr_role,
   )

   step = EMRServerlessStep(
       name="SparkJob", job_config=job_config,
       application_config={"name": "spark-app", "releaseLabel": "emr-6.15.0", "type": "SPARK"},
   )

   pipeline = Pipeline(name="EMRPipeline", steps=[step], sagemaker_session=pipeline_session)
   pipeline.upsert(role_arn=role)
   pipeline.start()

:doc:`Full example notebook <../v3-examples/ml-ops-examples/v3-emr-serverless-step-example>`



MLflow Integration
-------------------


Train with MLflow metric tracking and deploy from the MLflow model registry.

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer
   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.serve.mode.function_pointers import Mode

   # Train (script logs to MLflow internally)
   trainer = ModelTrainer(training_image=pytorch_image, source_code=SourceCode(source_dir=code_dir, entry_script="train.py", requirements="requirements.txt"))
   trainer.train()

   # Deploy from MLflow registry
   model_builder = ModelBuilder(
       mode=Mode.SAGEMAKER_ENDPOINT,
       schema_builder=schema_builder,
       model_metadata={"MLFLOW_MODEL_PATH": "models:/my-model/1", "MLFLOW_TRACKING_ARN": tracking_arn},
   )
   model_builder.build(model_name="mlflow-model")
   model_builder.deploy(endpoint_name="mlflow-endpoint")

:doc:`Full example notebook <../v3-examples/ml-ops-examples/v3-mlflow-train-inference-e2e-example>`



Migration from V2
------------------


MLOps Classes and Imports
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - V2
     - V3
   * - ``sagemaker.workflow.pipeline.Pipeline``
     - ``sagemaker.mlops.workflow.pipeline.Pipeline``
   * - ``sagemaker.workflow.steps.ProcessingStep``
     - ``sagemaker.mlops.workflow.steps.ProcessingStep``
   * - ``sagemaker.workflow.steps.TrainingStep``
     - ``sagemaker.mlops.workflow.steps.TrainingStep``
   * - ``sagemaker.workflow.step_collections.RegisterModel``
     - ``sagemaker.mlops.workflow.model_step.ModelStep`` + ``model_builder.register()``
   * - ``sagemaker.workflow.model_step.ModelStep``
     - ``sagemaker.mlops.workflow.model_step.ModelStep``
   * - ``sagemaker.sklearn.processing.SKLearnProcessor``
     - ``sagemaker.core.processing.ScriptProcessor``
   * - ``sagemaker.processing.ScriptProcessor``
     - ``sagemaker.core.processing.ScriptProcessor``
   * - ``sagemaker.processing.FrameworkProcessor``
     - ``sagemaker.core.processing.FrameworkProcessor``
   * - ``sagemaker.processing.ProcessingInput``
     - ``sagemaker.core.shapes.ProcessingInput`` + ``ProcessingS3Input``
   * - ``sagemaker.processing.ProcessingOutput``
     - ``sagemaker.core.shapes.ProcessingOutput`` + ``ProcessingS3Output``
   * - ``sagemaker.tuner.HyperparameterTuner``
     - ``sagemaker.train.tuner.HyperparameterTuner``
   * - ``sagemaker.parameter.ContinuousParameter``
     - ``sagemaker.core.parameter.ContinuousParameter``
   * - ``sagemaker.transformer.Transformer``
     - ``sagemaker.core.transformer.Transformer``
   * - ``sagemaker.clarify.SageMakerClarifyProcessor``
     - ``sagemaker.core.clarify.SageMakerClarifyProcessor``
   * - ``sagemaker.workflow.parameters.ParameterString``
     - ``sagemaker.core.workflow.parameters.ParameterString``
   * - ``sagemaker.workflow.pipeline_context.PipelineSession``
     - ``sagemaker.core.workflow.pipeline_context.PipelineSession``
   * - ``sagemaker.lineage.context.Context``
     - ``sagemaker.core.lineage.context.Context``


V3 Package Structure
---------------------


.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - V3 Package
     - MLOps Components
   * - ``sagemaker-core``
     - ScriptProcessor, FrameworkProcessor, Transformer, Clarify, lineage, pipeline context, parameters, image URIs
   * - ``sagemaker-train``
     - ModelTrainer, HyperparameterTuner, InputData, Compute, SourceCode
   * - ``sagemaker-serve``
     - ModelBuilder (build, register, deploy)
   * - ``sagemaker-mlops``
     - Pipeline, ProcessingStep, TrainingStep, ModelStep, TuningStep, EMRServerlessStep, CacheConfig


Explore comprehensive MLOps examples:

.. toctree::
   :maxdepth: 1

   lineage
   ../v3-examples/ml-ops-examples/v3-sagemaker-clarify
   ../v3-examples/ml-ops-examples/v3-pipeline-train-create-registry
   ../v3-examples/ml-ops-examples/v3-transform-job-example
   ../v3-examples/ml-ops-examples/v3-hyperparameter-tuning-example/v3-hyperparameter-tuning-example
   ../v3-examples/ml-ops-examples/v3-hyperparameter-tuning-example/v3-hyperparameter-tuning-pipeline
   ../v3-examples/ml-ops-examples/v3-model-registry-example/v3-model-registry-example
   ../v3-examples/ml-ops-examples/v3-processing-job-pytorch/v3-pytorch-processing-example
   ../v3-examples/ml-ops-examples/v3-processing-job-sklearn
   ../v3-examples/ml-ops-examples/v3-emr-serverless-step-example
   ../v3-examples/ml-ops-examples/v3-mlflow-train-inference-e2e-example
