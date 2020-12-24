Distributed model parallel
--------------------------

The Amazon SageMaker distributed model parallel library is a model parallelism library for training
large deep learning models that were previously difficult to train due to GPU memory limitations.
The library automatically and efficiently splits a model across multiple GPUs and instances and coordinates model training,
allowing you to increase prediction accuracy by creating larger models with more parameters.

You can use the library to automatically partition your existing TensorFlow and PyTorch workloads
across multiple GPUs with minimal code changes. The library's API can be accessed through the Amazon SageMaker SDK.

Use the following sections to learn more about the model parallelism and the library.

.. important::
   The model parallel library only supports training jobs using CUDA 11. When you define a PyTorch or TensorFlow
   ``Estimator`` with ``modelparallel`` parameter ``enabled`` set to ``True``,
   it uses CUDA 11. When you extend or customize your own training image
   you must use a CUDA 11 base image. See
   `Extend or Adapt A Docker Container that Contains the Model Parallel Library
   <https://integ-docs-aws.amazon.com/sagemaker/latest/dg/model-parallel-use-api.html#model-parallel-customize-container>`__
   for more information.

How to Use this Guide
=====================

The library contains a Common API that is shared across frameworks, as well as APIs
that are specific to supported frameworks, TensorFlow and PyTorch. To use the library, reference the
**Common API** documentation alongside the framework specific API documentation.

.. toctree::
   :maxdepth: 1

   smd_model_parallel_general
   smd_model_parallel_common_api
   smd_model_parallel_pytorch
   smd_model_parallel_tensorflow

It is recommended to use this documentation alongside `SageMaker Distributed Model Parallel
<http://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`__ in the Amazon SageMaker
developer guide. This developer guide documentation includes:

   -  An overview of model parallelism and the library
      `core features <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html>`__
   -  Instructions on how to modify `TensorFlow
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script.html#model-parallel-customize-training-script-tf>`__
      and `PyTorch
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script.html#model-parallel-customize-training-script-pt>`__
      training scripts
   -  `Configuration tips and pitfalls
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-tips-pitfalls.html>`__

Latest Updates
==============

New features, bug fixes, and improvements are regularly made to the SageMaker distributed model parallel library.

To see the the latest changes made to the library, refer to the library
`Release Notes
<https://github.com/aws/sagemaker-python-sdk/blob/master/doc/api/training/smd_model_parallel_release_notes/>`_.

