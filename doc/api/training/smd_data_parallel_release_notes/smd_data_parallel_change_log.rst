.. _sdp_release_note:

#############
Release Notes
#############

New features, bug fixes, and improvements are regularly made to the SageMaker
data parallelism library.

SageMaker Distributed Data Parallel 1.8.0 Release Notes
=======================================================

*Date: Apr. 17. 2023*

**Currency Updates**

* Added support for PyTorch 2.0.0.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- PyTorch 2.0.0 DLC

  .. code::

    763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker

Binary file of this version of the library for custom container users:

  .. code::

    https://smdataparallel.s3.amazonaws.com/binary/pytorch/2.0.0/cu118/2023-03-20/smdistributed_dataparallel-1.8.0-cp310-cp310-linux_x86_64.whl


----

Release History
===============

SageMaker Distributed Data Parallel 1.7.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Date: Feb. 10. 2023*

**Currency Updates**

* Added support for PyTorch 1.13.1.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- PyTorch 1.13.1 DLC

  .. code::

    763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker

Binary file of this version of the library for custom container users:

  .. code::

    https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.13.1/cu117/2023-01-09/smdistributed_dataparallel-1.7.0-cp39-cp39-linux_x86_64.whl

SageMaker Distributed Data Parallel 1.6.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Date: Dec. 15. 2022*

**New Features**

* New optimized SMDDP AllGather collective to complement the sharded data parallelism technique
  in the SageMaker model parallelism library. For more information, see `Sharded data parallelism with SMDDP Collectives
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html#model-parallel-extended-features-pytorch-sharded-data-parallelism-smddp-collectives>`_
  in the *Amazon SageMaker Developer Guide*.
* Added support for Amazon EC2 ``ml.p4de.24xlarge`` instances. You can run data parallel training jobs
  on ``ml.p4de.24xlarge`` instances with the SageMaker data parallelism library’s AllReduce collective.

**Improvements**

* General performance improvements of the SMDDP AllReduce collective communication operation.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- SageMaker training container for PyTorch v1.12.1

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker


Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-bring-your-own-container>`_ users:

  .. code::

    https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.12.1/cu113/2022-12-05/smdistributed_dataparallel-1.6.0-cp38-cp38-linux_x86_64.whl


SageMaker Distributed Data Parallel 1.5.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Date: Jul. 26. 2022*

**Currency Updates**

* Added support for PyTorch 1.12.0.

**Bug Fixes**

* Improved stability for long-running training jobs.


**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- PyTorch 1.12.0 DLC

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker

Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-bring-your-own-container>`_ users:

  .. code::

    https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.12.0/cu113/2022-07-01/smdistributed_dataparallel-1.5.0-cp38-cp38-linux_x86_64.whl

SageMaker Distributed Data Parallel 1.4.1 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Date: May. 3. 2022*

**Currency Updates**

* Added support for PyTorch 1.11.0

**Known Issues**

* The library currently does not support the PyTorch sub-process groups API
  (`torch.distributed.new_group
  <https://pytorch.org/docs/stable/distributed.html#torch.distributed.new_group>`_).


**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- PyTorch 1.11.0 DLC

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker

Binary file of this version of the library for custom container users:

  .. code::

    https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.11.0/cu113/2022-04-14/smdistributed_dataparallel-1.4.1-cp38-cp38-linux_x86_64.whl


SageMaker Distributed Data Parallel 1.4.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Date: Feb. 24. 2022*

**New Features**

* Integrated to PyTorch DDP as a backend option
* Added support for PyTorch 1.10.2

**Breaking Changes**

* As the library is migrated into the PyTorch distributed package as a backend,
  the following smdistributed implementation APIs are deprecated in
  the SageMaker data parallal library v1.4.0 and later.
  Please use the `PyTorch distributed APIs <https://pytorch.org/docs/stable/distributed.html>`_ instead.

  * ``smdistributed.dataparallel.torch.distributed``
  * ``smdistributed.dataparallel.torch.parallel.DistributedDataParallel``
  * Please note the slight differences between the deprecated
    ``smdistributed.dataparallel.torch`` APIs and the
    `PyTorch distributed APIs <https://pytorch.org/docs/stable/distributed.html>`_.

    * `torch.distributed.barrier <https://pytorch.org/docs/master/distributed.html#torch.distributed.barrier)>`_
      takes ``device_ids``, which the ``smddp`` backend does not support.
    * The ``gradient_accumulation_steps`` option in
      ``smdistributed.dataparallel.torch.parallel.DistributedDataParallel``
      is no longer supported. Please use the PyTorch
      `no_sync <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=no_sync#torch.nn.parallel.DistributedDataParallel.no_sync>`_ API.


* If you want to find documentation for the previous versions of the library
  (v1.3.0 or before), see the `archived SageMaker distributed data parallel library documentation <https://sagemaker.readthedocs.io/en/stable/api/training/sdp_versions/latest.html#documentation-archive>`_.

**Improvements**

* Support for AllReduce Large Tensors
* Support for the following new arguments in the `PyTorch DDP class
  <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_.

  * ``broadcast_buffers``
  * ``find_unused_parameters``
  * ``gradient_as_bucket_view``

**Bug Fixes**

* Fixed stalling issues when training on ``ml.p3.16xlarge``.

**Known Issues**

* The library currently does not support the PyTorch sub-process groups API (`torch.distributed.new_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.new_group>`_).
  This means that you cannot use the ``smddp`` backend concurrently with other
  process group backends such as NCCL and Gloo.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- PyTorch 1.10.2 DLC

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker


SageMaker Distributed Data Parallel 1.2.2 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Date: November. 24. 2021*

**New Features**

* Added support for PyTorch 1.10
* PyTorch ``no_sync`` API support for DistributedDataParallel
* Timeout when training stalls due to allreduce and broadcast collective calls

**Bug Fixes**

* Fixed a bug that would impact correctness in the mixed dtype case
* Fixed a bug related to the timeline writer that would cause a crash when SageMaker Profiler is enabled for single node jobs.

**Improvements**

* Performance optimizations for small models on small clusters

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers:

- PyTorch 1.10 DLC release: `v1.0-pt-sagemaker-1.10.0-py38 <https://github.com/aws/deep-learning-containers/releases/tag/v1.0-pt-sagemaker-1.10.0-py38>`_

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.10.0-gpu-py38-cu113-ubuntu20.04-sagemaker


SageMaker Distributed Data Parallel 1.2.1 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


SageMaker Distributed Data Parallel 1.2.0 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  New features
-  Bug Fixes

**New features:**

-  Support of `EFA network
   interface <https://aws.amazon.com/hpc/efa/>`__ for distributed
   AllReduce. For best performance, it is recommended you use an
   instance type that supports Amazon Elastic Fabric Adapter
   (ml.p3dn.24xlarge and ml.p4d.24xlarge) when you train a model using
   SageMaker Distributed data parallel.

**Bug Fixes:**

-  Improved performance on single node and small clusters.

----

SageMaker Distributed Data Parallel 1.1.2 Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Bug Fixes
-  Known Issues

**Bug Fixes:**

-  Fixed a bug that caused some TensorFlow operations to not work with
   certain data types. Operations forwarded from C++ have been extended
   to support every dtype supported by NCCL.

**Known Issues:**

-  SageMaker Distributed data parallel has slower throughput than NCCL
   when run using a single node. For the best performance, use
   multi-node distributed training with smdistributed.dataparallel. Use
   a single node only for experimental runs while preparing your
   training pipeline.

----

SageMaker Distributed Data Parallel 1.1.1 Release Notes
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

SageMaker Distributed Data Parallel 1.1.0 Release Notes
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

SageMaker Distributed Data Parallel 1.0.0 Release Notes
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
