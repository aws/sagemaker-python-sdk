#############
Release Notes
#############

New features, bug fixes, and improvements are regularly made to the SageMaker
model parallelism library.


SageMaker Distributed Model Parallel 1.15.0 Release Notes
=========================================================

*Date: Apr. 27. 2023*

**Currency Updates**

* Added support for PyTorch v2.0.0.
  Note that the library does not support ``torch.compile`` in this release.

**New Features**

* Using sharded data parallelism with tensor parallelism together is now
  available for PyTorch 1.13.1. It allows you to train with smaller global batch
  sizes while scaling up to large clusters. For more information, see `Sharded
  data parallelism with tensor parallelism <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html#model-parallel-extended-features-pytorch-sharded-data-parallelism-with-tensor-parallelism>`_
  in the *Amazon SageMaker Developer Guide*.
* Added support for saving and loading full model checkpoints when using sharded
  data parallelism. This is enabled by using the standard checkpointing API,
  ``smp.save_checkpoint`` with ``partial=False``.
  Before, full checkpoints needed to be created by merging partial checkpoint
  files after training finishes.
* `DistributedTransformer <https://sagemaker.readthedocs.io/en/stable/api/training/smp_versions/latest/smd_model_parallel_pytorch_tensor_parallel.html#smdistributed.modelparallel.torch.nn.DistributedTransformerLayer>`_
  now supports the ALiBi position embeddings.
  When using DistributedTransformer, you can set the ``use_alibi`` parameter
  to ``True`` to use the Triton-based flash attention kernels. This helps
  evaluate sequences longer than those used for training.

**Bug Fixes**

* When using tensor parallelism, parameters were initialized multiple times
  unncessarily. This release fixed the multiple initialization of parameters
  so that each parameter is initialized exactly once.
  It not only saves time, but also ensures that the random generator behavior
  is similar to the non-tensor parallelism case.

**Known issues**

* Model initialization might take longer with PyTorch 2.0 than that with PyTorch 1.13.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- SageMaker training container for PyTorch v2.0.0

  .. code::

    763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker

- SageMaker training container for PyTorch v1.13.1

  .. code::

    763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker

Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-bring-your-own-container>`_ users:

- For PyTorch v2.0.0

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-2.0.0/build-artifacts/2023-04-14-20-14/smdistributed_modelparallel-1.15.0-cp310-cp310-linux_x86_64.whl

- For PyTorch v1.13.1

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.13.1/build-artifacts/2023-04-17-15-49/smdistributed_modelparallel-1.15.0-cp39-cp39-linux_x86_64.whl

----

Release History
===============

SageMaker Distributed Model Parallel 1.14.0 Release Notes
---------------------------------------------------------

*Date: Jan. 30. 2023*

**Currency Updates**

* Added support for PyTorch v1.13.1

**Improvements**

* Upgraded the flash-attention (https://github.com/HazyResearch/flash-attention) library to  v0.2.6.post1

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- SageMaker training container for PyTorch v1.13.1

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker


Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-bring-your-own-container>`_ users:

- For PyTorch 1.13.1

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.13.1/build-artifacts/2023-01-19-18-35/smdistributed_modelparallel-1.14.0-cp39-cp39-linux_x86_64.whl


SageMaker Distributed Model Parallel 1.13.0 Release Notes
---------------------------------------------------------

*Date: Dec. 15. 2022*

**New Features**

* Sharded data parallelism now supports a new backend for collectives called *SMDDP Collectives*.
  For supported scenarios, SMDDP Collectives are on by default for the AllGather operation.
  For more information, see
  `Sharded data parallelism with SMDDP Collectives
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html#model-parallel-extended-features-pytorch-sharded-data-parallelism-smddp-collectives>`_
  in the *Amazon SageMaker Developer Guide*.
* Introduced FlashAttention for DistributedTransformer to improve memory usage and computational
  performance of models such as GPT2, GPTNeo, GPTJ, GPTNeoX, BERT, and RoBERTa.

**Bug Fixes**

* Fixed initialization of ``lm_head`` in DistributedTransformer to use a provided range
  for initialization, when weights are not tied with the embeddings.

**Improvements**

* When a module has no parameters, we have introduced an optimization to execute
  such a module on the same rank as its parent during pipeline parallelism.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- SageMaker training container for PyTorch v1.12.1

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker


Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-bring-your-own-container>`_ users:

- For PyTorch 1.12.1

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.12.1/build-artifacts/2022-12-08-21-34/smdistributed_modelparallel-1.13.0-cp38-cp38-linux_x86_64.whl


SageMaker Distributed Model Parallel 1.11.0 Release Notes
---------------------------------------------------------

*Date: August. 17. 2022*

**New Features**

The following new features are added for PyTorch.

* The library implements sharded data parallelism, which is a memory-saving
  distributed training technique that splits the training state of a model
  (model parameters, gradients, and optimizer states) across data parallel groups.
  With sharded data parallelism, you can reduce the per-GPU memory footprint of
  a model by sharding the training state over multiple GPUs. To learn more,
  see `Sharded Data Parallelism
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html>`_
  in the *Amazon SageMaker Developer Guide*.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- DLC for PyTorch 1.12.0

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker

Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-bring-your-own-container>`_ users:

- For PyTorch 1.12.0

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.12.0/build-artifacts/2022-08-12-16-58/smdistributed_modelparallel-1.11.0-cp38-cp38-linux_x86_64.whl

SageMaker Distributed Model Parallel 1.10.1 Release Notes
---------------------------------------------------------

*Date: August. 8. 2022*

**Currency Updates**

* Added support for Transformers v4.21.


**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- DLC for PyTorch 1.11.0

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker


Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-bring-your-own-container>`_ users:

- For PyTorch 1.11.0

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.11.0/build-artifacts/2022-07-28-23-07/smdistributed_modelparallel-1.10.1-cp38-cp38-linux_x86_64.whl



SageMaker Distributed Model Parallel 1.10.0 Release Notes
---------------------------------------------------------

*Date: July. 19. 2022*

**New Features**

The following new features are added for PyTorch.

* Added support for FP16 training by implementing smdistributed.modelparallel
  modification of Apex FP16_Module and FP16_Optimizer. To learn more, see
  `FP16 Training with Model Parallelism
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-fp16.html>`_.
* New checkpoint APIs for CPU memory usage optimization. To learn more, see
  `Checkpointing Distributed Models and Optimizer States
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-checkpoint.html>`_.

**Improvements**

* The SageMaker distributed model parallel library manages and optimizes CPU
  memory by garbage-collecting non-local parameters in general and during checkpointing.
* Changes in the `GPT-2 translate functions
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-hugging-face.html>`_
  (``smdistributed.modelparallel.torch.nn.huggingface.gpt2``)
  to save memory by not maintaining two copies of weights at the same time.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- DLC for PyTorch 1.11.0

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker

- DLC for PyTorch 1.12.0

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker

Binary file of this version of the library for `custom container
<https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html#model-parallel-bring-your-own-container>`_ users:

- For PyTorch 1.11.0

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.11.0/build-artifacts/2022-07-11-19-23/smdistributed_modelparallel-1.10.0-cp38-cp38-linux_x86_64.whl

- For PyTorch 1.12.0

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.12.0/build-artifacts/2022-07-11-19-23/smdistributed_modelparallel-1.10.0-cp38-cp38-linux_x86_64.whl


SageMaker Distributed Model Parallel 1.9.0 Release Notes
--------------------------------------------------------

*Date: May. 3. 2022*

**Currency Updates**

* Added support for PyTorch 1.11.0

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers (DLC):

- PyTorch 1.11.0 DLC

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker

Binary file of this version of the library for custom container users:

  .. code::

    https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.11.0/build-artifacts/2022-04-20-17-05/smdistributed_modelparallel-1.9.0-cp38-cp38-linux_x86_64.whl



SageMaker Distributed Model Parallel 1.8.1 Release Notes
--------------------------------------------------------

*Date: April. 23. 2022*

**New Features**

* Added support for more configurations of the Hugging Face Transformers GPT-2 and GPT-J models
  with tensor parallelism: ``scale_attn_weights``, ``scale_attn_by_inverse_layer_idx``,
  ``reorder_and_upcast_attn``. To learn more about these features, please refer to
  the following model configuration classes
  in the *Hugging Face Transformers documentation*:

  * `transformers.GPT2Config <https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config>`_
  * `transformers.GPTJConfig <https://huggingface.co/docs/transformers/model_doc/gptj#transformers.GPTJConfig>`_

* Added support for activation checkpointing of modules which pass keyword value arguments
  and arbitrary structures in their forward methods. This helps support
  activation checkpointing with Hugging Face Transformers models even
  when tensor parallelism is not enabled.

**Bug Fixes**

* Fixed a correctness issue with tensor parallelism for GPT-J model
  which was due to improper scaling during gradient reduction
  for some layer normalization modules.
* Fixed the creation of unnecessary additional processes which take up some
  GPU memory on GPU 0 when the :class:`smp.allgather` collective is called.

**Improvements**

* Improved activation offloading so that activations are preloaded on a
  per-layer basis as opposed to all activations for a micro batch earlier.
  This not only improves memory efficiency and performance, but also makes
  activation offloading a useful feature for non-pipeline parallelism cases.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers:

* HuggingFace 4.17.0 DLC with PyTorch 1.10.2

    .. code::

      763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04


* The binary file of this version of the library for custom container users

    .. code::

      https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.10.0/build-artifacts/2022-04-14-03-58/smdistributed_modelparallel-1.8.1-cp38-cp38-linux_x86_64.whl


SageMaker Distributed Model Parallel 1.8.0 Release Notes
--------------------------------------------------------

*Date: March. 23. 2022*

**New Features**

* Added tensor parallelism support for the `GPT-J model
  <https://huggingface.co/docs/transformers/model_doc/gptj>`_.
  When using the GPT-J model of Hugging Face Transformers v4.17.0 with
  tensor parallelism, the SageMaker model parallel library automatically
  replaces the model with a tensor parallel distributed GPT-J model.
  For more information, see `Support for Hugging Face Transformer Models
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-hugging-face.html>`_
  in the *Amazon SageMaker Model Parallel Training developer guide*.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers:

* HuggingFace 4.17.0 DLC with PyTorch 1.10.2

    .. code::

      763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04


The binary file of this version of the library for custom container users:

    .. code::

      https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.10.0/build-artifacts/2022-03-12-00-33/smdistributed_modelparallel-1.8.0-cp38-cp38-linux_x86_64.whl


SageMaker Distributed Model Parallel 1.7.0 Release Notes
--------------------------------------------------------

*Date: March. 07. 2022*

**Currency Updates**

* Support for PyTorch 1.10.2
* Support for Hugging Face Transformers 4.16.2

**Improvements**

* Additional support for the :ref:`smdmp-pytorch-tensor-parallel`.

  * Added support for FP32 residual addition to avoid overflow (NaN loss values)
    for large models with more than 100 billion parameters when using FP16.
    This is integrated to the following module:

      * :class:`smp.nn.DistributedTransformerOutputLayer`


  * Added support for the following two `NVIDIA Megatron fused kernels
    <https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/fused_kernels>`_:

    * Fusion of attention masking and softmax (``fused_softmax``)
    * Fusion of bias addition and Gelu activation (``fused_bias_gelu``)

    To learn more about these options and how to use them,
    see the :class:`smp.tensor_parallelism` context manager.



**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following AWS Deep Learning Containers:


* PyTorch 1.10.2

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker


SageMaker Distributed Model Parallel 1.6.0 Release Notes
--------------------------------------------------------

*Date: December. 20. 2021*

**New Features**

- **PyTorch**

  - Added extended memory-saving features for PyTorch 1.8.1:

    - `Tensor parallelism <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-tensor-parallelism.html>`_
    - `Optimizer state sharding <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-optimizer-state-sharding.html>`_
    - `Activation checkpointing <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-checkpointing.html>`_
    - `Activation offloading <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-offloading.html>`_

    For more information, see the following documentation:

    - `SageMaker distributed model parallel developer guide <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch.html>`_
    - `SageMaker distributed model parallel API documentation for v1.6.0 <https://sagemaker.readthedocs.io/en/stable/api/training/smp_versions/latest.html>`_

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following
AWS Deep Learning Container(s):

- Deep Learning Container for PyTorch 1.8.1:

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04



SageMaker Distributed Model Parallel 1.5.0 Release Notes
--------------------------------------------------------

*Date: November. 03. 2021*

**New Features**

- **PyTorch**

  - Currency update for PyTorch 1.10.0

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following
AWS Deep Learning Containers:

- Deep Learning Container for PyTorch 1.10.0:

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.10.0-gpu-py38-cu113-ubuntu20.04-sagemaker

----

SageMaker Distributed Model Parallel 1.4.0 Release Notes
--------------------------------------------------------

*Date: June. 29. 2021*

**New Features**

- **TensorFlow**

  - Added support for TensorFlow v2.5.0.
  - Added support for ``keras.model.fit()``.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following
AWS Deep Learning Containers:

- Deep Learning Container for TensorFlow 2.5.0:

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/tensorflow-training:2.5.0-gpu-py37-cu112-ubuntu18.04-v1.0

- Deep Learning Container for PyTorch 1.9.1:

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:1.9.1-gpu-py38-cu111-ubuntu20.04

----

SageMaker Distributed Model Parallel 1.3.1 Release Notes
--------------------------------------------------------

-  New Features
-  Bug Fixes
-  Known Issues

**New Features**

- **TensorFlow**

  -  Exposes a new decorator ``register_post_partition_hook``. This allows
     invoking the decorated methods just after model partition but before
     executing the first step. For example loading a checkpoint. Refer to
     the `SageMaker distributed model parallel API
     documentation <https://sagemaker.readthedocs.io/en/stable/api/training/smp_versions/latest/smd_model_parallel_tensorflow.html>`__
     for more information.

**Bug Fixes**

- **PyTorch**

  -  Improved memory efficiency when using active microbatches by clearing
     activations at end of each microbatch.

- **TensorFlow**

  -  Fixed issue that caused hangs when training some models with XLA
     enabled.

**Known Issues**

- **PyTorch**

  -  A crash was observed when ``optimizer.step()`` was called for certain
     optimizers such as AdaDelta, when the partition on which this method
     was called has no local parameters assigned to it after partitioning.
     This is due to a bug in PyTorch which `has since been
     fixed <https://github.com/pytorch/pytorch/pull/52944>`__. Till that
     makes its way to the next release of PyTorch, only call
     ``optimizer.step()`` on processes which have at least one local
     parameter. This can be checked like this
     ``len(list(model.local_parameters())) > 0``.

  -  A performance regression still exists when training on SMP with
     PyTorch 1.7.1 compared to 1.6. The rootcause was found to be the
     slowdown in performance of ``.grad`` method calls in PyTorch 1.7.1
     compared to 1.6. See the related discussion:
     https://github.com/pytorch/pytorch/issues/50636. This issue does not
     exist with PyTorch 1.8.

----

SageMaker Distributed Model Parallel 1.3.0 Release Notes
--------------------------------------------------------

-  New Features
-  Bug Fixes
-  Known Issues

.. _new-features-1:

**New Features**

.. _pytorch-2:

- **PyTorch**

  Add support for PyTorch 1.8

  -  Adds a new method to DistributedModel ``register_comm_hook`` (for
     PyTorch 1.8 and newer only). This method behaves the same as the
     corresponding method with the same name in
     ``torch.DistributedDataParallel`` API. Refer to the `SageMaker
     distributed model parallel API
     documentation <https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_pytorch.html#smp.DistributedModel>`__
     for more information.

**Improvements**

-  Adds a configuration ``active_microbatches`` to the SageMaker SDK API
   for launching jobs, to control the number of active microbatches
   during training. This helps limit memory usage in cases where the
   number of microbatches is high. Refer to the `SageMaker Python SDK
   parameters API
   documentation <https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_general.html>`__
   for more information.

-  Adds a configuration ``deterministic_server`` to the SageMaker SDK
   API for launching jobs, which ensures that the execution server for
   pipeline parallelism processes requests in a deterministic order
   across data parallel ranks. Refer to the `SageMaker Python SDK
   parameters API
   documentation <https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_general.html>`__
   for more information.

-  Parameter passing is now supported in ``module.forward`` methods for
   DistributedModel and its submodules. This removes the restriction of
   having to pass ``nn.Parameter`` to the ``__init__`` call and making
   it a member of the module to use it. ## Bug Fixes

.. _pytorch-3:

- **PyTorch**

  -  Fixed a case where training hangs due to a module having computation
     which requires grads that is not used by the final output of the
     module. Now such a situtation raises an error with suggestions on
     making such computation compatible.

  -  Fixed an issue with buffers which caused the buffers to not be on the
     correct device after a model is partitioned, and not be synchronized
     across steps (when ``broadcast_buffers`` is True). This could have
     caused correctness issues in models with buffers.

.. _known-issues-1:

**Known Issues**

.. _pytorch-4:

- **PyTorch**

  -  ``mp_barrier`` and ``get_mp_process_group`` are wrongly marked as
     deprecated methods. Ignore the deprecation warning.

  -  A crash was observed when ``optimizer.step()`` was called for certain
     optimizers such as AdaDelta, when the partition on which this method
     was called has no local parameters assigned to it after partitioning.
     This is due to a bug in PyTorch which `has since been
     fixed <https://github.com/pytorch/pytorch/pull/52944>`__. Till that
     makes its way to the next release of PyTorch, only call
     ``optimizer.step()`` on processes which have at least one local
     parameter. This can be checked like this
     ``len(list(model.local_parameters())) > 0``.

  -  A performance regression still exists when training on SMP with
     PyTorch 1.7.1 compared to 1.6. The rootcause was found to be the
     slowdown in performance of ``.grad`` method calls in PyTorch 1.7.1
     compared to 1.6. See the related discussion:
     https://github.com/pytorch/pytorch/issues/50636. This issue does not
     exist with PyTorch 1.8.

----

SageMaker Distributed Model Parallel 1.2.0 Release Notes
--------------------------------------------------------

-  New Features
-  Bug Fixes
-  Known Issues

.. _new-features-2:

**New Features**

.. _pytorch-5:

- **PyTorch**

  Add support for PyTorch 1.7.1

  -  Adds support for ``gradient_as_bucket_view`` (PyTorch 1.7.1 only),
     ``find_unused_parameters`` (PyTorch 1.7.1 only) and
     ``broadcast_buffers`` options to ``smp.DistributedModel``. These
     options behave the same as the corresponding options (with the same
     names) in ``torch.DistributedDataParallel`` API. Refer to the
     `SageMaker distributed model parallel API
     documentation <https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_pytorch.html#smp.DistributedModel>`__
     for more information.

  -  Adds support for ``join`` (PyTorch 1.7.1 only) context manager, which
     is to be used in conjunction with an instance of
     ``smp.DistributedModel`` to be able to train with uneven inputs
     across participating processes.

  -  Adds support for ``_register_comm_hook`` (PyTorch 1.7.1 only) which
     will register the callable as a communication hook for DDP. NOTE:
     Like in DDP, this is an experimental API and subject to change.

.. _tensorflow-2:

- **Tensorflow**

  -  Adds support for Tensorflow 2.4.1

.. _bug-fixes-1:

**Bug Fixes**

.. _pytorch-6:

- **PyTorch**

  -  ``Serialization``: Fix a bug with serialization/flattening where
     instances of subclasses of dict/OrderedDicts were
     serialized/deserialized or internally flattened/unflattened as
     regular dicts.

.. _tensorflow-3:

- **Tensorflow**

  -  Fix a bug that may cause a hang during evaluation when there is no
     model input for one partition.

.. _known-issues-2:

**Known Issues**

.. _pytorch-7:

- **PyTorch**

  -  A performance regression was observed when training on SMP with
     PyTorch 1.7.1 compared to 1.6.0. The rootcause was found to be the
     slowdown in performance of ``.grad`` method calls in PyTorch 1.7.1
     compared to 1.6.0. See the related discussion:
     https://github.com/pytorch/pytorch/issues/50636.

----

SageMaker Distributed Model Parallel 1.1.0 Release Notes
--------------------------------------------------------

-  New Features
-  Bug Fixes
-  Improvements
-  Performance
-  Known Issues

.. _new-features-3:

**New Features**

The following sections describe new feature releases that are common
across frameworks and that are framework specific.

**Common across frameworks***

- Custom slicing support (``smp_slice`` method) for objects passed to ``smp.step`` decorated functions

  To pass an object to ``smp.step`` that contains tensors that needs to be
  split across microbatches and is not an instance of list, dict, tuple or
  set, you should implement ``smp_slice`` method for the object.

  Below is an example of how to use this with PyTorch:

  .. code-block:: python

    class CustomType:
        def __init__(self, tensor):
            self.data = tensor

        # SMP will call this to invoke slicing on the object passing in total microbatches (num_mb)
        # and the current microbatch index (mb).
        def smp_slice(self, num_mb, mb, axis):
            dim_size = list(self.data.size())[axis]

            split_size = dim_size // num_mb
            sliced_tensor = self.data.narrow(axis, mb * split_size, split_size)
            return CustomType(sliced_tensor, self.other)

    custom_obj = CustomType(torch.ones(4,))

    @smp.step()
    def step(custom_obj):
        loss = model(custom_obj)
        model.backward(loss)
        return loss

.. _pytorch-8:

- **PyTorch**

  - Add support for smp.DistributedModel.cpu()

    ``smp.DistributedModel.cpu()``
    `allgather <https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_common_api.html#smp.allgather>`__\ s
    parameters and buffers across all ``mp_ranks`` and moves them to the
    CPU.

  - Add ``trace_memory_usage`` option to ``smp.DistributedModel`` to measure memory usage per module

    Adds ``trace_memory_usage`` option to ``smp.DistributedModel``. This
    attempts to measure memory usage per module during tracing. If this is
    disabled, memory usage is estimated through the sizes of tensors
    returned from the module. This option is disabled by default.

.. _bug-fixes-2:

**Bug Fixes**

.. _pytorch-9:

- **PyTorch**

  -  ``torch.nn.Sequential``: Fix a bug with ``torch.nn.Sequential`` which
     causes a failure with the error message :
     ``shouldnt go less than 0, there is a bug`` when the inputs to the
     first module don’t require grads.

  -  ``smp.DistributedModel``: Fix a bug with ``DistributedModel``
     execution when a module has multiple parents. The bug surfaces with
     the error message:
     ``actual_parent should be different than module_execution_stack parent only for torch.nn.ModuleList``

  -  ``apex.optimizers.FusedNovoGrad``: Fix a bug with
     ``apex.optimizers.FusedNovoGrad`` which surfaces with the error
     message: ``KeyError: 'exp_avg_sq'``

**Improvements**

*Usability*

.. _pytorch-10:

- **PyTorch**

  -  ``smp.DistributedModel``: Improve the error message when the forward
     pass on ``smp.DistributedModel`` is called outside the ``smp.step``
     decorated function.

  -  ``smp.load``: Add user friendly error messages when loading
     checkpoints with ``smp.load``.

*Partitioning Algorithm*

.. _pytorch-11:

- **PyTorch**

  -  Better memory balancing by taking into account the existing modules
     already assigned to the parent, while partitioning the children of a
     given module.

**Performance**

.. _tensorflow-4:

- **Tensorflow**

  -  Addresses long pre-processing times introduced by SMP XLA optimizer
     when dealing with large graphs and large number of microbatches. BERT
     (large) preprocessing time goes down from 40 minutes to 6 minutes on
     p3.16xlarge.

.. _known-issues-3:

**Known Issues**

.. _pytorch-12:

- **PyTorch**

  -  Serialization for Torch in SMP overwrites instances of dict subclass
     to be dict itself, instead of the instances of subclass. One of the
     use cases which fails because of this issue is when a user implements
     a subclass of OrderedDict with the ``__getitem__`` method. After
     serialization/deserialization in SMP, indexing on the object will
     lead to errors. A workaround is to use the dict keys to access the
     underlying item.
