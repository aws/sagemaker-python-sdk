Distributed Training APIs
-------------------------
SageMaker distributed training libraries offer both data parallel and model parallel training strategies.
They combine software and hardware technologies to improve inter-GPU and inter-node communications.
They extend SageMaker’s training capabilities with built-in options that require only small code changes to your training scripts.

.. _sdp_api_docs_toc:

The SageMaker Distributed Data Parallel Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    smd_data_parallel
    sdp_versions/latest
    smd_data_parallel_use_sm_pysdk
    smd_data_parallel_release_notes/smd_data_parallel_change_log

.. _smp_api_docs_toc:

The SageMaker Distributed Model Parallel Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    Since the release of the SageMaker model parallelism (SMP) version 2 in December 2023,
    this documentation is no longer supported for maintenence.
    The live documentation is available at
    `SageMaker model parallelism library v2
    <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-v2.html>`_
    in the *Amazon SageMaker User Guide*.

    The documentation for the SMP library v1.x is archived and available at
    `Run distributed training with the SageMaker model parallelism library
    <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_
    in the *Amazon SageMaker User Guide*,
    and the SMP v1.x API reference is available in the
    `SageMaker Python SDK v2.199.0 documentation
    <https://sagemaker.readthedocs.io/en/v2.199.0/api/training/distributed.html#the-sagemaker-distributed-model-parallel-library>`_.
