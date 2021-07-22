Sagemaker Distributed Model Parallel 1.4.0 Release Notes
========================================================

*Date: June. 29. 2021*

**New Features**

- **TensorFlow**

  - Added support for TensorFlow v2.5.0.
  - Added support for ``keras.model.fit()``.

**Migration to AWS Deep Learning Containers**

This version passed benchmark testing and is migrated to the following
AWS Deep Learning Containers:

- TensorFlow 2.5.0 DLC release: `v1.0-tf-2.5.0-tr-py37
  <https://github.com/aws/deep-learning-containers/releases/tag/v1.0-tf-2.5.0-tr-py37>`__

  .. code::

    763104351884.dkr.ecr.<region>.amazonaws.com/tensorflow-training:2.5.0-gpu-py37-cu112-ubuntu18.04-v1.0

----

Release History
===============

Sagemaker Distributed Model Parallel 1.3.1 Release Notes
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

Sagemaker Distributed Model Parallel 1.3.0 Release Notes
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

Sagemaker Distributed Model Parallel 1.2.0 Release Notes
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

Sagemaker Distributed Model Parallel 1.1.0 Release Notes
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
