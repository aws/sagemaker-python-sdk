.. _sdp_api_docs:

###############################################
Use the Library's API to Adapt Training Scripts
###############################################

This section contains the SageMaker distributed data parallel API documentation.
If you are a new user of this library, it is recommended you use this guide alongside
`SageMaker's Distributed Data Parallel Library
<https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html>`_.

The library provides framework-specific APIs for TensorFlow and PyTorch.

Select the latest or one of the previous versions of the API documentation
depending on the version of the library you use.

.. important::
   The distributed data parallel library only supports training jobs using CUDA 11. When you define a PyTorch or TensorFlow
   ``Estimator`` with ``dataparallel`` parameter ``enabled`` set to ``True``,
   it uses CUDA 11. When you extend or customize your own training image
   you must use a CUDA 11 base image. See
   `SageMaker Python SDK's distributed data parallel library APIs
   <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api>`_
   for more information.

Version 1.4.0 (Latest)
======================

.. toctree::
   :maxdepth: 1

   latest/smd_data_parallel_pytorch
   latest/smd_data_parallel_tensorflow

To find archived API documentation for the previous versions of the library,
see the following link:

Documentation Archive
=====================

.. toctree::
   :maxdepth: 1

   archives
