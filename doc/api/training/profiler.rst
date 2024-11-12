Profiler
--------

Amazon SageMaker Profiler provides full visibility
into provisioned compute resources for training
state-of-the-art deep learning models.
The following SageMaker Profiler classes are
for activating SageMaker Profiler while creating
an estimator object of `:class:sagemaker.pytorch.estimator.PyTorch`
or `:class:sagemaker.tensorflow.estimator.TensorFlow`.

.. contents::

.. currentmodule:: sagemaker.debugger

Profiler configuration modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: sagemaker.Profiler(cpu_profiling_duration=3600)

    A configuration class to activate
    `Amazon SageMaker Profiler <https://docs.aws.amazon.com/sagemaker/latest/dg/train-profile-computational-performance.html>`_.

    To adjust the Profiler configuration instead of using the default configuration, use the following parameters.

    **Parameters:**

        - **cpu_profiling_duration** (*str*): Specify the time duration in seconds for
          profiling CPU activities. The default value is 3600 seconds.

    **Example usage:**

    .. code:: python

        import sagemaker
        from sagemaker.pytorch import PyTorch
        from sagemaker import ProfilerConfig, Profiler

        profiler_config = ProfilerConfig(
            profiler_params = Profiler(cpu_profiling_duration=3600)
        )

        estimator = PyTorch(
            framework_version="2.0.0",
            ... # Set up other essential parameters for the estimator class
            profiler_config=profiler_config
        )

    For a complete instruction on activating and using SageMaker Profiler, see
    `Use Amazon SageMaker Profiler to profile activities on AWS compute resources
    <https://docs.aws.amazon.com/sagemaker/latest/dg/train-profile-computational-performance.html>`_.

.. autoclass:: sagemaker.ProfilerConfig


Profiler Rule APIs
~~~~~~~~~~~~~~~~~~

The following API is for setting up SageMaker Debugger's profiler rules
to detect computational performance issues from training jobs.

.. autoclass:: ProfilerRule
    :inherited-members:


Debugger Configuration APIs for Framework Profiling (Deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

  In favor of `Amazon SageMaker Profiler <https://docs.aws.amazon.com/sagemaker/latest/dg/train-profile-computational-performance.html>`_,
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
