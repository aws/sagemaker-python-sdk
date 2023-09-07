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

.. toctree::
   :maxdepth: 2

   smd_model_parallel
   smp_versions/latest
   smd_model_parallel_general
   smd_model_parallel_release_notes/smd_model_parallel_change_log
