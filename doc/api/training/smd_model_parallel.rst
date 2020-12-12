Distributed model parallel
--------------------------

Amazon SageMaker Distributed Model Parallel (SMP) is a model parallelism library for training
large deep learning models that were previously difficult to train due to GPU memory limitations.
SMP automatically and efficiently splits a model across multiple GPUs and instances and coordinates model training,
allowing you to increase prediction accuracy by creating larger models with more parameters.

You can use SMP to automatically partition your existing TensorFlow and PyTorch workloads
across multiple GPUs with minimal code changes. The SMP API can be accessed through the Amazon SageMaker SDK.

Use the following sections to learn more about the model parallelism and the SMP library.

.. important::
   SMP only supports training jobs using CUDA 11. When you define a PyTorch or TensorFlow
   ``Estimator`` with ``smdistributed`` ``enabled``,
   it uses CUDA 11. When you extend or customize your own training image
   you must use a CUDA 11 base image. See
   `Extend or Adapt A Docker Container that Contains SMP
   <https://integ-docs-aws.amazon.com/sagemaker/latest/dg/model-parallel-use-api.html#model-parallel-customize-container>`__
   for more information.

It is recommended to use this documentation alongside `SageMaker Distributed Model Parallel
<http://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`__ in the Amazon SageMaker
developer guide. This developer guide documentation includes:

   -  An overview of model parallelism and the SMP library
      `core features <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html>`__
   -  Instructions on how to modify `TensorFlow
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script.html#model-parallel-customize-training-script-tf>`__
      and `PyTorch
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script.html#model-parallel-customize-training-script-pt>`__
      training scripts
   -  `Configuration tips and pitfalls
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-tips-pitfalls.html>`__

**How to Use this Guide**

The SMP library contains a Common API that is shared across frameworks, as well as APIs
that are specific to supported frameworks, TensorFlow and PyTroch. To use SMP, reference the
**Common API** documentation alongside framework specific API documentation.


.. toctree::
   :maxdepth: 1

   smd_model_parallel_general
   smd_model_parallel_common_api
   smd_model_parallel_pytorch
   smd_model_parallel_tensorflow
