The SageMaker Distributed Model Parallel Library Overview
---------------------------------------------------------

The Amazon SageMaker distributed model parallel library is a model parallelism library for training
large deep learning models that were previously difficult to train due to GPU memory limitations.
The library automatically and efficiently splits a model across multiple GPUs and instances and coordinates model training,
allowing you to increase prediction accuracy by creating larger models with more parameters.

You can use the library to automatically partition your existing TensorFlow and PyTorch workloads
across multiple GPUs with minimal code changes. The library's API can be accessed through the Amazon SageMaker SDK.

.. tip::

  We recommended using this API documentation with the conceptual guide at
  `SageMaker's Distributed Model Parallel
  <http://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_
  in the *Amazon SageMaker developer guide*. This developer guide documentation includes:

  - An overview of model parallelism, and the library's
    `core features <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html>`_,
    and `extended features for PyTorch <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch.html>`_.
  - Instructions on how to modify `TensorFlow
    <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script-tf.html>`_
    and `PyTorch
    <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script-pt.html>`_
    training scripts.
  - Instructions on how to `run a distributed training job using the SageMaker Python SDK
    and the SageMaker model parallel library
    <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html>`_.
  - `Configuration tips and pitfalls
    <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-tips-pitfalls.html>`_.


.. important::
   The model parallel library only supports training jobs using CUDA 11. When you define a PyTorch or TensorFlow
   ``Estimator`` with ``modelparallel`` parameter ``enabled`` set to ``True``,
   it uses CUDA 11. When you extend or customize your own training image
   you must use a CUDA 11 base image. See
   `Extend or Adapt A Docker Container that Contains the Model Parallel Library
   <https://integ-docs-aws.amazon.com/sagemaker/latest/dg/model-parallel-use-api.html#model-parallel-customize-container>`__
   for more information.
