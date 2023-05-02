.. _sdp_api_docs:

#############################################
Use the Library to Adapt Your Training Script
#############################################

This section contains the SageMaker distributed data parallel API documentation.
If you are a new user of this library, it is recommended you use this guide alongside
`SageMaker's Distributed Data Parallel Library
<https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html>`_.

The library provides framework-specific APIs for TensorFlow and PyTorch.

Select the latest or one of the previous versions of the API documentation
depending on the version of the library you use.

.. important::

   The distributed data parallel library supports training jobs using CUDA 11 or later.
   When you define a :class:`sagemaker.tensorflow.estimator.TensorFlow` or
   :class:`sagemaker.pytorch.estimator.PyTorch`
   estimator with the data parallel library enabled,
   SageMaker uses CUDA 11. When you extend or customize your own training image,
   you must use a base image with CUDA 11 or later. See
   `SageMaker Python SDK's distributed data parallel library APIs
   <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api>`_
   for more information.

For versions between 1.4.0 and 1.8.0 (Latest)
=============================================

.. toctree::
   :maxdepth: 1

   latest/smd_data_parallel_pytorch
   latest/smd_data_parallel_tensorflow

Documentation Archive
=====================

To find the API documentation for the previous versions of the library,
choose one of the following:

.. toctree::
   :maxdepth: 1

   archives
