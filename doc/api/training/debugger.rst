Debugger
--------

Amazon SageMaker Debugger provides full visibility
into training jobs of state-of-the-art machine learning models.
This SageMaker Debugger module provides high-level methods
to set up Debugger configurations to
monitor, profile, and debug your training job.
Configure the Debugger-specific parameters when constructing
a SageMaker estimator to gain visibility and insights
into your training job.

.. currentmodule:: sagemaker.debugger

.. autoclass:: get_rule_container_image_uri
    :show-inheritance:

.. autoclass:: get_default_profiler_rule
    :show-inheritance:

.. class:: sagemaker.debugger.rule_configs

    A helper module to configure the SageMaker Debugger built-in rules with
    the :class:`~sagemaker.debugger.Rule` classmethods and
    and the :class:`~sagemaker.debugger.ProfilerRule` classmethods.

    For a full list of built-in rules, see
    `List of Debugger Built-in Rules
    <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.

    This module is imported from the Debugger client library for rule configuration.
    For more information, see
    `Amazon SageMaker Debugger RulesConfig
    <https://github.com/awslabs/sagemaker-debugger-rulesconfig>`_.

.. autoclass:: RuleBase
    :show-inheritance:

.. autoclass:: Rule
    :show-inheritance:
    :inherited-members:

.. autoclass:: ProfilerRule
    :show-inheritance:
    :inherited-members:

.. autoclass:: CollectionConfig
    :show-inheritance:

.. autoclass:: DebuggerHookConfig
    :show-inheritance:

.. autoclass:: TensorBoardOutputConfig
    :show-inheritance:

.. autoclass:: ProfilerConfig
    :show-inheritance:
    
.. warning::
    
		SageMaker Debugger deprecates the framework profiling feature starting from TensorFlow 2.11 and PyTorch 2.0. You can still use the feature in the previous versions of the frameworks and SDKs as follows. 
		
		* SageMaker Python SDK <= v2.130.0
		* PyTorch >= v1.6.0, < v2.0
		* TensorFlow >= v2.3.1, < v2.11
	
		With the deprecation, SageMaker Debugger discontinues support for the APIs below this note.
		
		See also `Amazon SageMaker Debugger Release Notes: March 16, 2023 <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-release-notes.html#debugger-release-notes-20230315>`_.

.. autoclass:: FrameworkProfile
    :show-inheritance:

.. autoclass:: DetailedProfilingConfig
    :show-inheritance:

.. autoclass:: DataloaderProfilingConfig
    :show-inheritance:

.. autoclass:: PythonProfilingConfig
    :show-inheritance:

.. autoclass:: PythonProfiler
    :show-inheritance:

.. autoclass:: cProfileTimer
    :show-inheritance:

.. automodule:: sagemaker.debugger.metrics_config
    :members: StepRange, TimeRange
    :undoc-members:
