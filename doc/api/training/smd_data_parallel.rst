##########################
Distributed data parallel
##########################

SageMaker's distributed data parallel library extends SageMaker’s training
capabilities on deep learning models with near-linear scaling efficiency,
achieving fast time-to-train with minimal code changes.

When training a model on a large amount of data, machine learning practitioners
will often turn to distributed training to reduce the time to train.
In some cases, where time is of the essence,
the business requirement is to finish training as quickly as possible or at
least within a constrained time period.
Then, distributed training is scaled to use a cluster of multiple nodes,
meaning not just multiple GPUs in a computing instance, but multiple instances
with multiple GPUs. However, as the cluster size increases, it is possible to see a significant drop
in performance due to communications overhead between nodes in a cluster.

SageMaker's distributed data parallel library addresses communications overhead in two ways:

1. The library performs AllReduce, a key operation during distributed training that is responsible for a
   large portion of communication overhead.
2. The library performs optimized node-to-node communication by fully utilizing AWS’s network
   infrastructure and Amazon EC2 instance topology.

To learn more about the core features of this library, see
`Introduction to SageMaker's Distributed Data Parallel Library
<https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-intro.html>`_
in the SageMaker Developer Guide.

Use with the SageMaker Python SDK
=================================

To use the SageMaker distributed data parallel library with the SageMaker Python SDK, you will need the following:

-  A TensorFlow or PyTorch training script that is
   adapted to use the distributed data parallel library. The :ref:`sdp_api_docs` includes
   framework specific examples of training scripts that are adapted to use this library.
-  Your input data must be in an S3 bucket or in FSx in the AWS region
   that you will use to launch your training job. If you use the Jupyter
   notebooks provided, create a SageMaker notebook instance in the same
   region as the bucket that contains your input data. For more
   information about storing your training data, refer to
   the `SageMaker Python SDK data
   inputs <https://sagemaker.readthedocs.io/en/stable/overview.html#use-file-systems-as-training-inputs>`__ documentation.

When you define
a Pytorch or TensorFlow ``Estimator`` using the SageMaker Python SDK,
you must select ``dataparallel`` as your ``distribution`` strategy:

.. code::

   distribution = { "smdistributed": { "dataparallel": { "enabled": True } } }

We recommend you use one of the example notebooks as your template to launch a training job. When
you use an example notebook you’ll need to swap your training script with the one that came with the
notebook and modify any input functions as necessary. For instructions on how to get started using a
Jupyter Notebook example, see `Distributed Training Jupyter Notebook Examples
<https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training-notebook-examples.html>`_.

Once you have launched a training job, you can monitor it using CloudWatch. To learn more, see
`Monitor and Analyze Training Jobs Using Metrics
<https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html>`_.


After you train a model, you can see how to deploy your trained model to an endpoint for inference by
following one of the `example notebooks for deploying a model
<https://sagemaker-examples.readthedocs.io/en/latest/inference/index.html>`_.
For more information, see `Deploy Models for Inference
<https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html>`_.

.. _sdp_api_docs:

API Documentation
=================

This section contains the SageMaker distributed data parallel API documentation. If you are a
new user of this library, it is recommended you use this guide alongside
`SageMaker's Distributed Data Parallel Library
<https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html>`_.

Select a version to see the API documentation for version.

.. toctree::
   :maxdepth: 1

   sdp_versions/latest.rst
   sdp_versions/v1_1_x.rst
   sdp_versions/v1_0_0.rst

.. important::
   The distributed data parallel library only supports training jobs using CUDA 11. When you define a PyTorch or TensorFlow
   ``Estimator`` with ``dataparallel`` parameter ``enabled`` set to ``True``,
   it uses CUDA 11. When you extend or customize your own training image
   you must use a CUDA 11 base image. See
   `SageMaker Python SDK's distributed data parallel library APIs
   <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api>`_
   for more information.


Release Notes
=============

New features, bug fixes, and improvements are regularly made to the SageMaker
distributed data parallel library.

.. toctree::
   :maxdepth: 1

   smd_data_parallel_release_notes/smd_data_parallel_change_log
