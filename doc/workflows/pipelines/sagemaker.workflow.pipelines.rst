Pipelines
=========

ConditionStep
-------------

.. autoclass:: sagemaker.workflow.condition_step.ConditionStep
.. deprecated:: sagemaker.workflow.condition_step.JsonGet

Conditions
----------

.. autoclass:: sagemaker.workflow.conditions.ConditionTypeEnum

.. autoclass:: sagemaker.workflow.conditions.Condition

.. autoclass:: sagemaker.workflow.conditions.ConditionComparison

.. autoclass:: sagemaker.workflow.conditions.ConditionEquals

.. autoclass:: sagemaker.workflow.conditions.ConditionGreaterThan

.. autoclass:: sagemaker.workflow.conditions.ConditionGreaterThanOrEqualTo

.. autoclass:: sagemaker.workflow.conditions.ConditionLessThan

.. autoclass:: sagemaker.workflow.conditions.ConditionLessThanOrEqualTo

.. autoclass:: sagemaker.workflow.conditions.ConditionIn

.. autoclass:: sagemaker.workflow.conditions.ConditionNot

.. autoclass:: sagemaker.workflow.conditions.ConditionOr

CheckJobConfig
--------------

.. autoclass:: sagemaker.workflow.check_job_config.CheckJobConfig

Entities
--------

.. autoclass:: sagemaker.workflow.entities.Entity

.. autoclass:: sagemaker.workflow.entities.DefaultEnumMeta

.. autoclass:: sagemaker.workflow.entities.Expression

.. autoclass:: sagemaker.workflow.entities.PipelineVariable

Execution Variables
-------------------

.. autoclass:: sagemaker.workflow.execution_variables.ExecutionVariable

.. autoclass:: sagemaker.workflow.execution_variables.ExecutionVariables
    :members: START_DATETIME, CURRENT_DATETIME, PIPELINE_EXECUTION_ID, PIPELINE_EXECUTION_ARN, PIPELINE_NAME, PIPELINE_ARN, TRAINING_JOB_NAME, PROCESSING_JOB_NAME

Functions
---------

.. autoclass:: sagemaker.workflow.functions.Join

.. autoclass:: sagemaker.workflow.functions.JsonGet

Parameters
----------

.. autoclass:: sagemaker.workflow.parameters.ParameterTypeEnum

.. autoclass:: sagemaker.workflow.parameters.Parameter

.. autoclass:: sagemaker.workflow.parameters.ParameterString

.. autoclass:: sagemaker.workflow.parameters.ParameterInteger

.. autoclass:: sagemaker.workflow.parameters.ParameterFloat

.. autoclass:: sagemaker.workflow.parameters.ParameterBoolean

Pipeline
--------

.. autoclass:: sagemaker.workflow.pipeline.Pipeline
    :members:

.. autoclass:: sagemaker.workflow.pipeline._PipelineExecution
    :members:

Pipeline Context
------------------

.. autoclass:: sagemaker.workflow.pipeline_context.PipelineSession
    :members:

.. autoclass:: sagemaker.workflow.pipeline_context.LocalPipelineSession
    :members:

Pipeline Schedule
-----------------

.. autoclass:: sagemaker.workflow.triggers.PipelineSchedule

Parallelism Configuration
-------------------------

.. autoclass:: sagemaker.workflow.parallelism_config.ParallelismConfiguration
    :members:

Pipeline Definition Config
--------------------------

.. autoclass:: sagemaker.workflow.pipeline_definition_config.PipelineDefinitionConfig

Pipeline Experiment Config
--------------------------

.. autoclass:: sagemaker.workflow.pipeline_experiment_config.PipelineExperimentConfig

.. autoclass:: sagemaker.workflow.pipeline_experiment_config.PipelineExperimentConfigProperty

Selective Execution Config
--------------------------

.. autoclass:: sagemaker.workflow.selective_execution_config.SelectiveExecutionConfig

Properties
----------

.. autoclass:: sagemaker.workflow.properties.PropertiesMeta

.. autoclass:: sagemaker.workflow.properties.Properties

.. autoclass:: sagemaker.workflow.properties.PropertiesList

.. autoclass:: sagemaker.workflow.properties.PropertyFile

Step Collections
----------------

.. autoclass:: sagemaker.workflow.step_collections.StepCollection

.. autoclass:: sagemaker.workflow.step_collections.RegisterModel

.. autoclass:: sagemaker.workflow.step_collections.EstimatorTransformer

.. autoclass:: sagemaker.workflow.model_step.ModelStep

.. autoclass:: sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep

Steps
-----

.. autoclass:: sagemaker.workflow.steps.StepTypeEnum

.. autoclass:: sagemaker.workflow.steps.Step

.. autoclass:: sagemaker.workflow.steps.TrainingStep

.. autoclass:: sagemaker.workflow.steps.TuningStep

.. autofunction:: sagemaker.workflow.steps.TuningStep.get_top_model_s3_uri

.. autoclass:: sagemaker.workflow.steps.TransformStep

.. autoclass:: sagemaker.workflow.steps.ProcessingStep

.. autoclass:: sagemaker.workflow.notebook_job_step.NotebookJobStep

.. autoclass:: sagemaker.workflow.steps.CreateModelStep

.. autoclass:: sagemaker.workflow.callback_step.CallbackStep

.. autoclass:: sagemaker.workflow.steps.CacheConfig

.. autoclass:: sagemaker.workflow.lambda_step.LambdaStep

.. autoclass:: sagemaker.workflow.quality_check_step.QualityCheckConfig

.. autoclass:: sagemaker.workflow.quality_check_step.QualityCheckStep

.. autoclass:: sagemaker.workflow.clarify_check_step.ClarifyCheckConfig

.. autoclass:: sagemaker.workflow.clarify_check_step.ClarifyCheckStep

.. autoclass:: sagemaker.workflow.fail_step.FailStep

.. autoclass:: sagemaker.workflow.emr_step.EMRStepConfig

.. autoclass:: sagemaker.workflow.emr_step.EMRStep

.. autoclass:: sagemaker.workflow.automl_step.AutoMLStep

@step decorator
---------------

.. automethod:: sagemaker.workflow.function_step.step

.. autoclass:: sagemaker.workflow.function_step.DelayedReturn

.. autoclass:: sagemaker.workflow.step_outputs.StepOutput

.. autofunction:: sagemaker.workflow.step_outputs.get_step
