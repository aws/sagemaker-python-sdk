Sagemaker Distributed Data Parallel 1.2.1 Release Notes
=======================================================

*Date: June. 29. 2021*

**New Features:**

-  Added support for TensorFlow 2.5.0.

**Improvements**

-  Improved performance on a single node and small clusters (2-4 nodes).

**Bug fixes**

-  Enable ``sparse_as_dense`` by default for SageMaker distributed data
   parallel library for TensorFlow APIs: ``DistributedGradientTape`` and
   ``DistributedOptimizer``.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers:

- TensorFlow 2.5.0 DLC release: `v1.0-tf-2.5.0-tr-py37
  <https://github.com/aws/deep-learning-containers/releases/tag/v1.0-tf-2.5.0-tr-py37>`__

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/tensorflow-training:2.5.0-gpu-py37-cu112-ubuntu18.04-v1.0

----

Release History
===============

Sagemaker Distributed Data Parallel 1.2.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  New features
-  Bug Fixes

**New features:**

-  Support of `EFA network
   interface <https://aws.amazon.com/hpc/efa/>`__ for distributed
   AllReduce. For best performance, it is recommended you use an
   instance type that supports Amazon Elastic Fabric Adapter
   (ml.p3dn.24xlarge and ml.p4d.24xlarge) when you train a model using
   Sagemaker Distributed data parallel.

**Bug Fixes:**

-  Improved performance on single node and small clusters.

----

Sagemaker Distributed Data Parallel 1.1.2 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Bug Fixes
-  Known Issues

**Bug Fixes:**

-  Fixed a bug that caused some TensorFlow operations to not work with
   certain data types. Operations forwarded from C++ have been extended
   to support every dtype supported by NCCL.

**Known Issues:**

-  Sagemaker Distributed data parallel has slower throughput than NCCL
   when run using a single node. For the best performance, use
   multi-node distributed training with smdistributed.dataparallel. Use
   a single node only for experimental runs while preparing your
   training pipeline.

----

Sagemaker Distributed Data Parallel 1.1.1 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  New Features
-  Bug Fixes
-  Known Issues

**New Features:**

-  Adds support for PyTorch 1.8.1

**Bug Fixes:**

-  Fixes a bug that was causing gradients from one of the worker nodes
   to be added twice resulting in incorrect ``all_reduce`` results under
   some conditions.

**Known Issues:**

-  SageMaker distributed data parallel still is not efficient when run
   using a single node. For the best performance, use multi-node
   distributed training with ``smdistributed.dataparallel``. Use a
   single node only for experimental runs while preparing your training
   pipeline.

----

Sagemaker Distributed Data Parallel 1.1.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  New Features
-  Bug Fixes
-  Improvements
-  Known Issues

**New Features:**

-  Adds support for PyTorch 1.8.0 with CUDA 11.1 and CUDNN 8

**Bug Fixes:**

-  Fixes crash issue when importing ``smdataparallel`` before PyTorch

**Improvements:**

-  Update ``smdataparallel`` name in python packages, descriptions, and
   log outputs

**Known Issues:**

-  SageMaker DataParallel is not efficient when run using a single node.
   For the best performance, use multi-node distributed training with
   ``smdataparallel``. Use a single node only for experimental runs
   while preparing your training pipeline.

Getting Started

For getting started, refer to SageMaker Distributed Data Parallel Python
SDK Guide
(https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api).

----

Sagemaker Distributed Data Parallel 1.0.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  First Release
-  Getting Started

First Release
-------------

SageMaker’s distributed data parallel library extends SageMaker’s
training capabilities on deep learning models with near-linear scaling
efficiency, achieving fast time-to-train with minimal code changes.
SageMaker Distributed Data Parallel:

-  optimizes your training job for AWS network infrastructure and EC2
   instance topology.
-  takes advantage of gradient update to communicate between nodes with
   a custom AllReduce algorithm.

The library currently supports TensorFlow v2 and PyTorch via `AWS Deep
Learning
Containers <https://aws.amazon.com/machine-learning/containers/>`__.

Getting Started
---------------

For getting started, refer to `SageMaker Distributed Data Parallel
Python SDK
Guide <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api>`__.
