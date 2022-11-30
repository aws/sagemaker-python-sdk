The SageMaker Distributed Model Parallel Library Overview
---------------------------------------------------------

The Amazon SageMaker distributed model parallel library is a model parallelism library for training
large deep learning models that were previously difficult to train due to GPU memory limitations.
The library automatically and efficiently splits a model across multiple GPUs and instances and coordinates model training,
allowing you to increase prediction accuracy by creating larger models with more parameters.

You can use the library to automatically partition your existing TensorFlow and PyTorch workloads
across multiple GPUs with minimal code changes. The library's API can be accessed through the Amazon SageMaker SDK.

.. tip::

  We recommend that you use this API documentation along with the conceptual guide at
  `SageMaker's Distributed Model Parallel
  <http://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_
  in the *Amazon SageMaker developer guide*.
  The conceptual guide includes the following topics:

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
   The model parallel library only supports SageMaker training jobs using CUDA 11.
   Make sure you use the pre-built Deep Learning Containers.
   If you want to extend or customize your own training image,
   you must use a CUDA 11 base image. For more information, see `Extend a Prebuilt Docker
   Container that Contains SageMaker's Distributed Model Parallel Library
   <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-customize-container>`_
   and `Create Your Own Docker Container with the SageMaker Distributed Model Parallel Library
   <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-bring-your-own-container>`_.
