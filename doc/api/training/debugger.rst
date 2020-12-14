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
