.. admonition:: Contents

   - :ref:`pytorch_saving_loading`
   - :ref:`pytorch_saving_loading_instructions`

PyTorch API
===========

**Supported versions: 1.7.1, 1.6**

This API document assumes you use the following import statements in your training scripts.

.. code:: python

   import smdistributed.modelparallel.torch as smp


.. tip::

   Refer to
   `Modify a PyTorch Training Script
   <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script.html#model-parallel-customize-training-script-pt>`_
   to learn how to use the following API in your PyTorch training script.

.. class:: smp.DistributedModel

   A sub-class of ``torch.nn.Module`` which specifies the model to be
   partitioned. Accepts a ``torch.nn.Module`` object ``module`` which is
   the model to be partitioned. The returned ``DistributedModel`` object
   internally manages model parallelism and data parallelism. Only one
   model in the training script can be wrapped with
   ``smp.DistributedModel``.


   **Example:**

   .. code:: python

      model = smp.DistributedModel(model)

   **Important**: The ``__call__`` and  ``backward`` method calls on the
   ``smp.DistributedModel`` object (in the following example, the object
   is \ ``model``) can only be made inside a ``smp.step``-decorated
   function.


   Since ``DistributedModel``  is a ``torch.nn.Module``, a forward pass can
   be performed by calling the \ ``DistributedModel`` object on the input
   tensors.

   .. code:: python

      predictions = model(inputs)   # model is a smp.DistributedModel object

   For a backward pass, one needs to call the backward function on
   the \ ``DistributedModel`` object, with tensors and gradients as
   arguments, replacing the PyTorch operations \ ``torch.Tensor.backward``
   or ``torch.autograd.backward``.


   The API for ``model.backward`` is very similar to
   ``torch.autograd.backward``. For example, the following
   ``backward`` calls:

   .. code:: python

      torch.autograd.backward(loss) or loss.backward()

   should be replaced with:

   .. code:: python

      model.backward(loss) # loss is a tensor with only one element as its data

   Similarly, for non-scalar tensors, replace the following
   ``backward`` call containing incoming gradient arguments:

   .. code:: python

      torch.autograd.backward(outputs, out_grads)

   with the following line:

   .. code:: python

      model.backward(outputs, out_grads)

   In these examples, all ``__call__``  and ``backward`` method calls on
   the model objects (``model(inputs)`` and ``model.backward(loss)``) must be made inside
   a ``smp.step``-decorated function.

   **Parameters**

   -  ``module`` (``torch.nn.Module``): Module to be distributed (data parallelism and model parallelism).

   -  ``trace_device`` (``"cpu"`` or ``"gpu"``) (default: ``"gpu"``)
      Whether to perform the tracing step on the GPU or CPU. The tracing step gathers
      information on the order of execution of modules, the shapes of
      intermediate outputs, and execution times, to be used by the
      partitioning algorithm. If ``trace_device`` is set to GPU, accurate
      module execution times can be gathered during tracing for potentially
      improved partitioning decision. However, if the model is too large to
      fit in a single GPU, then ``trace_device`` should be set to CPU.

   -  ``trace_execution_times`` (``bool``) (default: ``False``): If ``True``,
      the library profiles the execution time of each module during tracing, and uses
      it in the partitioning decision. This improves the partitioning
      decision, but it might make the tracing slower. It may also introduce
      some degree of non-determinism in partitioning results, because of the
      inherent randomness in module execution times. Must be ``False`` if
      ``trace_device`` is ``"cpu"``.

   -  ``overlapping_allreduce`` (``bool``) (default: ``True``): This is only
      applicable for hybrid data parallelism/model parallelism use cases (when
      ``ddp`` is set to ``True`` while launching training). The library uses this flag
      to decide whether to do overlapping allreduce whenever a parameter
      gradients are ready. This leads to overlapping of communication and
      computation and can improve performance. If this is set to ``False`` ,
      allreduce is performed at the end of the step.

   -  ``backward_passes_per_step`` (``int``) (default: 1): This is only
      applicable for hybrid data parallelism/model parallelism use cases (when
      ``ddp`` is set to ``True`` in config). This parameter indicates the
      number of backward passes to perform before calling allreduce on DDP.
      This allows accumulating updates over multiple mini-batches before
      reducing and applying them.

   -  ``average_grads_across_microbatches`` (``bool``) (default: ``True``):
      Whether or not the computed gradients should be averaged across
      microbatches. If ``False``, the computed gradients will be summed across
      microbatches, but not divided by the number of microbatches. In typical
      use case where the computed loss is averaged over the mini-batch, this
      should be left as ``True``. If you use a loss function that only sums
      the per-sample loss across the batch (and not divide by the batch size),
      then this must be set to ``False`` for correctness.

   -  ``bucket_cap_mb`` (default: 25): \ ``DistributedDataParallel`` buckets
      parameters into multiple buckets so that gradient reduction of each
      bucket can potentially overlap with backward
      computation. \ ``bucket_cap_mb``\ controls the bucket size in MegaBytes
      (MB).

    - ``trace_memory_usage`` (default: False): When set to True, the library attempts
      to measure memory usage per module during tracing. If this is disabled,
      memory usage will be estimated through the sizes of tensors returned from
      the module.

    - ``broadcast_buffers`` (default: True): Flag to be used with ``ddp=True``.
      This parameter is forwarded to the underlying ``DistributedDataParallel`` wrapper.
      Please see: `broadcast_buffer <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`__.

    - ``gradient_as_bucket_view (PyTorch 1.7 only)`` (default: False): To be
      used with ``ddp=True``. This parameter is forwarded to the underlying
      ``DistributedDataParallel`` wrapper. Please see `gradient_as_bucket_view <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`__.

   **Properties**

   -  ``partitioned``: Is ``True`` if the model is partitioned, ``False``
      otherwise. Initialized to ``False`` when ``DistributedModel`` is first
      created. It becomes be ``True`` during the first call
      to ``smp.step``-decorated function. Once the model is partitioned, the
      local parameters or local ``state_dict`` can be fetched using the
      following methods.

   **Methods**

   .. function:: backward(tensors, grad_tensors)

      Triggers a distributed backward
      pass across model partitions. Example usage provided in the previous
      section. The API is very similar
      to https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward.
      ``retain_grad`` and ``create_graph``  flags are not supported.

   .. function:: local_buffers( )

      Returns an iterator over buffers for the modules in
      the partitioned model that have been assigned to the current process.

   .. function:: local_named_buffers( )

      Returns an iterator over buffers for the
      modules in the partitioned model that have been assigned to the current
      process. This yields both the name of the buffer as well as the buffer
      itself.

   .. function:: local_parameters( )

      Returns an iterator over parameters for the
      modules in the partitioned model that have been assigned to the current
      process.

   .. function:: local_named_parameters( )

      Returns an iterator over parameters for
      the modules in the partitioned model that have been assigned to the
      current process. This yields both the name of the parameter as well as
      the parameter itself.

   .. function:: local_modules( )

      Returns an iterator over the modules in the
      partitioned model that have been assigned to the current process.

   .. function:: local_named_modules( )

      Returns an iterator over the modules in the
      partitioned model that have been assigned to the current process. This
      yields both the name of the module as well as the module itself.

   .. function:: local_state_dict( )

      Returns the ``state_dict`` that contains local
      parameters that belong to the current \ ``mp_rank``. This ``state_dict``
      contains a key \ ``_smp_is_partial`` to indicate this is a
      partial \ ``state_dict``, which indicates whether the
      ``state_dict`` contains elements corresponding to only the current
      partition, or to the entire model.

   .. function:: state_dict( )

      Returns the ``state_dict`` that contains parameters
      for the entire model. It first collects the \ ``local_state_dict``  and
      gathers and merges the \ ``local_state_dict`` from all ``mp_rank``\ s to
      create a full ``state_dict``.

   .. function:: load_state_dict( )

      Same as the ``torch.module.load_state_dict()`` ,
      except: It first gathers and merges the ``state_dict``\ s across
      ``mp_rank``\ s, if they are partial. The actual loading happens after the
      model partition so that each rank knows its local parameters.

   .. function:: register_post_partition_hook(hook)

      Registers a callable ``hook`` to
      be executed after the model is partitioned. This is useful in situations
      where an operation needs to be executed after the model partition during
      the first call to ``smp.step``, but before the actual execution of the
      first forward pass. Returns a ``RemovableHandle`` object ``handle``,
      which can be used to remove the hook by calling ``handle.remove()``.

   .. function:: cpu( )

      Allgathers parameters and buffers across all ``mp_rank``\ s and moves them
      to the CPU.

   .. function:: join( )

      **Available for PyTorch 1.7 only**
      A context manager to be used in conjunction with an instance of
      ``smp.DistributedModel``to be able to train with uneven inputs across
      participating processes. This is only supported when ``ddp=True`` for
      ``smp.DistributedModel``. This will use the join with the wrapped
      ``DistributedDataParallel`` instance. Please see: `join <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join>`__.


.. class:: smp.DistributedOptimizer

   **Parameters**
   - ``optimizer``

   An optimizer wrapper for saving/loading optimizer states. This wrapper
   returns ``optimizer`` with the following methods overridden:

   .. function:: state_dict( )

      Returns the ``state_dict`` that contains optimizer state for the entire model.
      It first collects the ``local_state_dict`` and gathers and merges
      the ``local_state_dict`` from all ``mp_rank``s to create a full
      ``state_dict``.

   .. function::  load_state_dict( )

      Same as the ``torch.optimizer.load_state_dict()`` , except:

         -  It first gathers and merges the local ``state_dict``\ s if they are
            partial.
         -  The actual loading happens after the model partition so that each
            rank knows its local parameters.

   .. function::  local_state_dict( )

      Returns the ``state_dict`` that contains the
      local optimizer state that belongs to the current \ ``mp_rank``. This
      ``state_dict`` contains a key \ ``_smp_is_partial`` to indicate this is
      a partial \ ``state_dict``, which indicates whether the
      ``state_dict`` contains elements corresponding to only the current
      partition, or to the entire model.

   ​
.. function:: smp.partition(index)
   :noindex:

   **Inputs**

   -  ``index`` (int) - The index of the partition.

   A context manager which places all modules defined inside into the
   partition with ID ``index``.  The ``index`` argument must be less than
   the number of partitions.

   Use ``smp.partition`` to implement manual partitioning.
   If ``"auto_partition"`` is ``True``, then the
   ``smp.partition`` contexts are ignored. Any module that is not placed in
   any ``smp.partition`` context is placed in the
   ``default_partition`` defined through the SageMaker Python SDK.

   When ``smp.partition`` contexts are nested, the innermost context
   overrides the rest (see the following example). In PyTorch, manual
   partitioning should be done inside the module \ ``__init__``, and the
   partition assignment applies to the modules that are *created* inside
   the ``smp.partition`` context.

   Example:

   .. code:: python

      class Model(torch.nn.Module):
          def __init__(self):
              with smp.partition(1):
                  self.child0 = Child0()            # child0 on partition 1
                  with smp.partition(2):
                      self.child1 = Child1()        # child1 on partition 2
                  self.child2 = Child2()            # child2 on partition 1
              self.child3 = Child3()                # child3 on default_partition

.. function:: smp.get_world_process_group( )

   Returns a ``torch.distributed`` ``ProcessGroup`` that consists of all
   processes, which can be used with the ``torch.distributed`` API.
   Requires ``"ddp": True`` in SageMaker Python SDK parameters.

.. function:: smp.get_mp_process_group( )

   Returns a ``torch.distributed`` ``ProcessGroup`` that consists of the
   processes in the ``MP_GROUP`` which contains the current process, which
   can be used with the \ ``torch.distributed`` API. Requires
   ``"ddp": True`` in SageMaker Python SDK parameters.

.. function:: smp.get_dp_process_group( )

   Returns a ``torch.distributed`` ``ProcessGroup`` that consists of the
   processes in the ``DP_GROUP`` which contains the current process, which
   can be used with the \ ``torch.distributed`` API. Requires
   ``"ddp": True`` in SageMaker Python SDK parameters.

.. function:: smp.is_initialized( )

   Returns ``True`` if ``smp.init`` has already been called for the
   process, and ``False`` otherwise.

.. function::smp.is_tracing( )

   Returns ``True`` if the current process is running the tracing step, and
   ``False`` otherwise.

.. data:: smp.nn.FusedLayerNorm

   `Apex Fused Layer Norm <https://nvidia.github.io/apex/layernorm.html>`__ is currently not
   supported by the library. ``smp.nn.FusedLayerNorm`` replaces ``apex``
   ``FusedLayerNorm`` and provides the same functionality. This requires
   ``apex`` to be installed on the system.

.. data:: smp.optimizers.FusedNovoGrad


   `Fused Novo Grad optimizer <https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedNovoGrad>`__ is
   currently not supported by the library. ``smp.optimizers.FusedNovoGrad`` replaces ``apex`` ``FusedNovoGrad``
   optimizer and provides the same functionality. This requires ``apex`` to
   be installed on the system.

.. data:: smp.optimizers.FusedLamb


   `FusedLamb optimizer <https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedLAMB>`__
   currently doesn’t work with the library. ``smp.optimizers.FusedLamb`` replaces
   ``apex`` ``FusedLamb`` optimizer and provides the same functionality.
   This requires ``apex`` to be installed on the system.

.. data:: smp.amp.GradScaler

   `Torch AMP Gradscaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`__
   currently doesn’t work with the library. ``smp.amp.GradScaler`` replaces
   ``torch.amp.GradScaler`` and provides the same functionality.

.. _pytorch_saving_loading:

APIs for Saving and Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: smp.save( )

   Saves an object. This operation is similar to ``torch.save()``, except
   it has an additional keyword argument, ``partial``, and accepts only
   string type for the argument ``f`` (file). If ``partial=True``, each
   ``mp_rank`` saves a separate checkpoint file and the library adds an ``mp_rank``
   index to your saved file.

   **Parameters**

   -  ``obj`` (dict): A saved object.
   -  ``f`` (str): A string containing a file name.
   -  ``partial`` (bool, default= ``True``):  When set to ``True``, each
      ``mp_rank`` saves a separate checkpoint file and the library adds an
      ``mp_rank`` index to the saved file. If you want to be able to load
      and further train a model that you save with ``smp.save()``, you must
      set ``partial=True``.
   -  ``pickle_module`` (picklemodule, default = module ``"pickle"`` from ``"/opt/conda/lib/python3.6/pickle.py"``):
      A module used for pickling metadata and objects.
   -  ``pickle_protocol``  (int, default=2): Can be specified to
      override the defaultprotocol.

.. function:: smp.load( )

   Loads an object saved with ``smp.save()`` from a file.

   Similar to, `torch.load() <https://pytorch.org/docs/stable/generated/torch.load.html>`__,
   except it has an additional keyword argument, ``partial``, and accepts
   only string type for the argument ``f`` (file). If \ ``partial=True``,
   then each ``mp_rank`` loads a separate checkpoint file.

   **Parameters**

   -  ``f`` (string): A string containing a file name.
   -  ``map_location`` (function): A function
      `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`__,
      a string, or a dict specifying how to remap storage locations.
   -  ``pickle_module`` (pickle module): A module used for unpickling
      metadata and objects (has to match the \ ``pickle_module``\ used to
      serialize file).
   -  ``pickle_load_args`` (Python 3 only): Optional keyword arguments
      passed to ``pickle_module.load()`` and ``pickle_module.Unpickler()``.
   -  ``partial`` (bool, default= ``True``): When set to ``True``, each
      ``mp_rank`` loads the checkpoint corresponding to the ``mp_rank``.
      Should be used when loading a model trained with the library.

.. _pytorch_saving_loading_instructions:

General Instruction For Saving and Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library can save partial or full checkpoints.

-  For partial checkpoints, each ``mp_rank`` saves its own checkpoint
   file with only the parameters that belong to that rank.
-  For full checkpoints, the library saves a single checkpoint that contains
   entire model parameters.

When **saving** using ``smp.save()``, each rank only holds its own
parameters. If you want to save the full model, there will be some
communication between the ranks to create the full model. If you save
checkpoints often, you should save partial checkpoints for best
performance.

When **loading** using ``smp.load()``, the library can load either partial or |
full checkpoints or full checkpoints saved by a non-model-parallel model. If you
want to resume training with a non-model-parallel model or do inference, you need
a full checkpoint.

The following is an example of how you can save and load a checkpoint:

.. code:: python

   # Original model and optimizer
   model = MyModel(...)
   optimizer = MyOpt(...)

   # model parallel wrapper
   model = smp.DistributedModel(model)
   optimizer = smp.DistributedOptimizer(optimizer)

   # To save, always save on dp_rank 0 to avoid data racing
   if partial:
       # To save the partial model on each mp rank
       # the library will create `checkpoint.pt_{mprank}` for each mp rank
       if save_partial_model:
           if smp.dp_rank() == 0:
               model_dict = model.local_state_dict() # save the partial model
               opt_dict = optimizer.local_state_dict() # save the partial optimizer state
               smp.save(
                   {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
                   f"/checkpoint.pt",
                   partial=True,
               )

       # To save the full model
       if save_full_model:
           if smp.dp_rank() == 0:
               model_dict = model.state_dict() # save the full model
               opt_dict = optimizer.state_dict() # save the full optimizer state
               smp.save(
                   {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
                   "/checkpoint.pt",
                   partial=False,
               )

   # To load, load on all ranks.
   # The only difference for partial/full loading is the partial flag in smp.load
   # Load partial checkpoint
   if partial_checkpoint:
       checkpoint = smp.load("/checkpoint.pt", partial=True)
       model.load_state_dict(checkpoint["model_state_dict"])
       optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
   # Load full checkpoint
   if full_checkpoint:
       checkpoint = smp.load("/checkpoint.pt", partial=False)
       model.load_state_dict(checkpoint["model_state_dict"])
       optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
