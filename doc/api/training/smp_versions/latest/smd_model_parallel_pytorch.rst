PyTorch API
===========

To use the PyTorch-specific APIs for SageMaker distributed model parallism,
import the ``smdistributed.modelparallel.torch`` package at the top of your training script.

.. code:: python

   import smdistributed.modelparallel.torch as smp


.. tip::

   Refer to
   `Modify a PyTorch Training Script
   <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script-pt.html>`_
   to learn how to use the following API in your PyTorch training script.

.. contents:: Topics
  :depth: 1
  :local:

smdistributed.modelparallel.torch.DistributedModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: smdistributed.modelparallel.torch.DistributedModel

   A sub-class of ``torch.nn.Module`` which specifies the model to be
   partitioned. Accepts a ``torch.nn.Module`` object ``module`` which is
   the model to be partitioned. The returned ``DistributedModel`` object
   internally manages model parallelism and data parallelism. Only one
   model in the training script can be wrapped with
   ``smdistributed.modelparallel.torch.DistributedModel``.

   **Example:**

   .. code:: python

      import smdistributed.modelparallel.torch as smp

      model = smp.DistributedModel(model)

   **Important**: The ``__call__`` and  ``backward`` method calls on the
   ``smdistributed.modelparallel.torch.DistributedModel`` object (in the following example, the object
   is \ ``model``) can only be made inside a ``smdistributed.modelparallel.torch.step``-decorated
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
   a ``smdistributed.modelparallel.torch.step``-decorated function.

   **Using DDP**

   If DDP is enabled with the SageMaker model parallel library, do not not place a PyTorch
   ``DistributedDataParallel`` wrapper around the ``DistributedModel`` because
   the ``DistributedModel`` wrapper will also handle data parallelism.

   Unlike the original DDP wrapper, when you use ``DistributedModel``,
   model parameters and buffers are not immediately broadcast across
   processes when the wrapper is called. Instead, the broadcast is deferred to the first call of the
   ``smdistributed.modelparallel.torch.step``-decorated function when the partition is done.

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

   -  ``trace_memory_usage`` (default: False): When set to True, the library attempts
      to measure memory usage per module during tracing. If this is disabled,
      memory usage will be estimated through the sizes of tensors returned from
      the module.

   -  ``broadcast_buffers`` (default: True): Flag to be used with ``ddp=True``.
      This parameter is forwarded to the underlying ``DistributedDataParallel`` wrapper.
      Please see: `broadcast_buffer <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`__.

   -  ``gradient_as_bucket_view`` (default: False): To be
      used with ``ddp=True``. This parameter is forwarded to the underlying
      ``DistributedDataParallel`` wrapper. Please see `gradient_as_bucket_view <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`__.

   **Properties**

   -  ``partitioned``: Is ``True`` if the model is partitioned, ``False``
      otherwise. Initialized to ``False`` when ``DistributedModel`` is first
      created. It becomes be ``True`` during the first call
      to ``smdistributed.modelparallel.torch.step``-decorated function. Once the model is partitioned, the
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
      create a full ``state_dict``. Please note that this needs to be called on all ranks with
      ``dp_rank()==0`` to ensure the gather happens properly.
      If it is only called on all such ranks, it can hang.

   .. function:: load_state_dict( )

      Same as the ``torch.module.load_state_dict()`` ,
      except: It first gathers and merges the ``state_dict``\ s across
      ``mp_rank``\ s, if they are partial. The actual loading happens after the
      model partition so that each rank knows its local parameters.

   .. function:: register_post_partition_hook(hook)

      Registers a callable ``hook`` to
      be executed after the model is partitioned. This is useful in situations
      where an operation needs to be executed after the model partition during
      the first call to ``smdistributed.modelparallel.torch.step``, but before the actual execution of the
      first forward pass. Returns a ``RemovableHandle`` object ``handle``,
      which can be used to remove the hook by calling ``handle.remove()``.

   .. function:: cpu( )

      Allgathers parameters and buffers across all ``mp_rank``\ s and moves them
      to the CPU.

   .. function:: join( )

      A context manager to be used in conjunction with an instance of
      ``smdistributed.modelparallel.torch.DistributedModel`` to be able to train with uneven inputs across
      participating processes. This is only supported when ``ddp=True``. This will use the join with the wrapped
      ``DistributedDataParallel`` instance. For more information, see:
      `join <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join>`__
      in the PyTorch documentation.

   .. function:: register_comm_hook( state, callable )

      **Available for PyTorch 1.8.1 only**
      Registers a communication hook which is an enhancement that provides
      a flexible hook ``callable`` to users where they can specify how
      gradients are aggregated across multiple workers. This method will be called on the wrapped ``DistributedDataParallel`` instance.

      Please note that when you register a comm hook you have full control of how the gradients are processed.
      When using only data parallelism with Torch DDP you are expected to average grads across data parallel replicas within the hook.
      Similarly, when using DistributedModel you have to averaging grads across data parallel replicas within the hook.
      In addition to that, you also have to average grads across microbatches within the hook unless you explicitly desire to not average based on your loss function.
      See ``average_grads_across_microbatches`` for more information about averaging grads across microbatches.

      This is only supported when ``ddp=True`` and ``overlapping_allreduce=True`` (default).
      For more information, see:
      `register_comm_hook <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook>`__
      in the PyTorch documentation.

  **Behavior of** ``smdistributed.modelparallel.torch.DistributedModel`` **with Tensor Parallelism**

  When a model is wrapped by ``smdistributed.modelparallel.torch.DistributedModel``, the library
  immediately traverses the modules of the model object, and replaces the
  modules that are supported for tensor parallelism with their distributed
  counterparts. This replacement happens in place. If there are no other
  references to the original modules in the script, they are
  garbage-collected. The module attributes that previously referred to the
  original submodules now refer to the distributed versions of those
  submodules.

  **Example:**

  .. code:: python

     # register DistributedSubmodule as the distributed version of Submodule
     # (note this is a hypothetical example, smp.nn.DistributedSubmodule does not exist)
     import smdistributed.modelparallel.torch as smp

     smp.tp_register_with_module(Submodule, smp.nn.DistributedSubmodule)

     class MyModule(nn.Module):
         def __init__(self):
             ...

             self.submodule = Submodule()
         ...

     # enabling tensor parallelism for the entire model
     with smp.tensor_parallelism():
         model = MyModule()

     # here model.submodule is still a Submodule object
     assert isinstance(model.submodule, Submodule)

     model = smp.DistributedModel(model)

     # now model.submodule is replaced with an equivalent instance
     # of smp.nn.DistributedSubmodule
     assert isinstance(model.module.submodule, smp.nn.DistributedSubmodule)

  If ``pipeline_parallel_degree`` (equivalently, ``partitions``) is 1, the
  placement of model partitions into GPUs and the initial broadcast of
  model parameters and buffers across data-parallel ranks take place
  immediately. This is because it does not need to wait for the model
  partition when ``smdistributed.modelparallel.torch.DistributedModel`` wrapper is called. For other
  cases with ``pipeline_parallel_degree`` greater than 1, the broadcast
  and device placement will be deferred until the first call of an
  ``smdistributed.modelparallel.torch.step``-decorated function happens. This is because the first
  ``smdistributed.modelparallel.torch.step``-decorated function call is when the model partitioning
  happens if pipeline parallelism is enabled.

  Because of the module replacement during the ``smdistributed.modelparallel.torch.DistributedModel``
  call, any ``load_state_dict`` calls on the model, as well as any direct
  access to model parameters, such as during the optimizer creation,
  should be done **after** the ``smdistributed.modelparallel.torch.DistributedModel`` call.

  Since the broadcast of the model parameters and buffers happens
  immediately during ``smdistributed.modelparallel.torch.DistributedModel`` call when the degree of
  pipeline parallelism is 1, using ``@smp.step`` decorators is not
  required when tensor parallelism is used by itself (without pipeline
  parallelism).

  For more information about the library's tensor parallelism APIs for PyTorch,
  see :ref:`smdmp-pytorch-tensor-parallel`.

  **Additional Methods of** ``smdistributed.modelparallel.torch.DistributedModel`` **for Tensor Parallelism**

  The following are the new methods of ``smdistributed.modelparallel.torch.DistributedModel``, in
  addition to the ones listed in the
  `documentation <https://sagemaker.readthedocs.io/en/stable/api/training/smp_versions/v1.2.0/smd_model_parallel_pytorch.html#smp.DistributedModel>`__.

  .. function:: distributed_modules()

     -  An iterator that runs over the set of distributed
        (tensor-parallelized) modules in the model

  .. function:: is_distributed_parameter(param)

     -  Returns ``True`` if the given ``nn.Parameter`` is distributed over
        tensor-parallel ranks.

  .. function::  is_distributed_buffer(buf)

     -  Returns ``True`` if the given buffer is distributed over
        tensor-parallel ranks.

  .. function::  is_scaled_batch_parameter(param)

     -  Returns ``True`` if the given ``nn.Parameter`` is operates on the
        scaled batch (batch over the entire ``TP_GROUP``, and not only the
        local batch).

  .. function::  is_scaled_batch_buffer(buf)

     -  Returns ``True`` if the parameter corresponding to the given
        buffer operates on the scaled batch (batch over the entire
        ``TP_GROUP``, and not only the local batch).

  .. function::  default_reducer_named_parameters()

     -  Returns an iterator that runs over ``(name, param)`` tuples, for
        ``param`` that is allreduced over the ``DP_GROUP``.

  .. function::  scaled_batch_reducer_named_parameters()

     -  Returns an iterator that runs over ``(name, param)`` tuples, for
        ``param`` that is allreduced over the ``RDP_GROUP``.

smdistributed.modelparallel.torch.DistributedOptimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: smdistributed.modelparallel.torch.DistributedOptimizer(optimizer, static_loss_scale=1.0, dynamic_loss_scale=False, **dynamic_loss_args)

   An optimizer wrapper for saving and loading optimizer states.

   :param optimizer: An optimizer object.
   :type optimizer: object
   :param static_loss_scale: Effective only for FP16 training. The default value is ``1.0``.
   :type static_loss_scale: float
   :param dynamic_loss_scale: Effective only for FP16 training. Set to ``True`` to
      use dynamic loss scale. The default value is ``False``.
   :type dynamic_loss_scale: boolean
   :param dynamic_loss_args: Effective only for FP16 training.
      If ``dynamic_loss_scale=True``, you can configure additional scale
      parameters for dynamic loss scale.
      The following list shows available parameters.

      * ``"init_scale"``: Default is ``2**32``
      * ``"scale_factor"``: Default is ``2.``
      * ``"scale_window"``: Default is ``1000``
      * ``"min_scale"``: Default is ``1``
      * ``"delayed_shift"``: Default is ``1``
      * ``"consecutive_hysteresis"``: Default is ``False``
   :type dynamic_loss_args: dict

   **Example usage of an FP32 Optimizer:**

   .. code:: python

      optimizer = torch.optim.AdaDelta(...)
      optimizer = smdistributed.modelparallel.torch.DistributedOptimizer(optimizer)

   **Example usage of an FP16 Optimizer with static loss scale:**

   .. code:: python

      optimizer = torch.optim.AdaDelta(...)
      optimizer = smdistributed.modelparallel.torch.DistributedOptimizer(
          optimizer,
          static_loss_scale=1.0
      )

   **Example usage of an FP16 Optimizer with dynamic loss scale:**

   .. code:: python

      optimizer = torch.optim.AdaDelta(...)
      optimizer = smdistributed.modelparallel.torch.DistributedOptimizer(
          optimizer,
          static_loss_scale=None,
          dynamic_loss_scale=True,
          dynamic_loss_args={
              "scale_window": 1000,
              "min_scale": 1,
              "delayed_shift": 2
          }
      )

   .. tip::

      After you modify training scripts with
      :class:`smdistributed.modelparallel.torch.DistributedModel` and
      :class:`smdistributed.modelparallel.torch.DistributedOptimizer`,
      use the SageMaker PyTorch estimator's distribution configuration to enable FP16 training.
      You simply need to add ``"fp16": True`` to the ``smp_options`` config dictionary's
      ``"parameters"`` key as shown in
      `Using the SageMaker TensorFlow and PyTorch Estimators
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html>`_.
      For more information about available parameters for the ``smp_options`` config,
      see :ref:`sm-sdk-modelparallel-general`.

   This wrapper returns an ``optimizer`` object with the following methods overridden:

   .. method:: state_dict( )

      Returns the ``state_dict`` that contains optimizer state for the entire model.
      It first collects the ``local_state_dict`` and gathers and merges
      the ``local_state_dict`` from all ``mp_rank``\ s to create a full
      ``state_dict``.

   .. method::  load_state_dict( )

      Same as the ``torch.optimizer.load_state_dict()`` , except:

         -  It first gathers and merges the local ``state_dict``\ s if they are
            partial.
         -  The actual loading happens after the model partition so that each
            rank knows its local parameters.

   .. method::  local_state_dict( )

      Returns the ``state_dict`` that contains the
      local optimizer state that belongs to the current \ ``mp_rank``. This
      ``state_dict`` contains a key \ ``_smp_is_partial`` to indicate this is
      a partial \ ``state_dict``, which indicates whether the
      ``state_dict`` contains elements corresponding to only the current
      partition, or to the entire model.

smdistributed.modelparallel.torch.nn.FlashAttentionLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: class smdistributed.modelparallel.torch.nn.FlashAttentionLayer(
   attention_dropout_prob=0.1,
   attention_head_size=None,
   scale_attention_scores=True,
   scale_attn_by_layer_idx=False,
   layer_idx=None,
   scale=None,
   triton_flash_attention=False,
   use_alibi=False
)

   This FlashAttentionLayer class supports 
   `FlashAttention <https://github.com/HazyResearch/flash-attention>`_. 
   It takes the ``qkv`` matrix as argument, computes attention scores and probabilities, 
   and then does the matrix multiplication with value layer. 

   Note that custom attention masks such as Attention with 
   Linear Biases (ALiBi) are only supported when 
   ``triton_flash_attention`` and ``use_alibi`` are set to ``True``. 
   
   Note also that Triton flash attention does not support dropout 
   on the attention probabilities. It uses standard lower triangular 
   causal mask when causal mode is enabled. It also runs only 
   on P4d and P4de instances, with fp16 or bf16.

   This class computes the scale factor to apply when computing attention. 
   By default, scale is ``None``, and it's automatically calculated. 
   When ``scale_attention_scores`` is ``True`` (which is default), 
   ``attention_head_size`` must be passed. When ``scale_attn_by_layer_idx`` is True, 
   then ``layer_idx`` must be passed. If both factors are used, they will 
   be multiplied ``(1/(sqrt(attention_head_size) * (layer_idx+1)))``. 
   This scale calculation can be bypassed by passing a custom scaling 
   factor if needed with ``scale`` parameter.

   **Parameters**

   * ``attention_dropout_prob`` (float): (default: 0.1) specifies dropout probability 
     to apply to attention.
   * ``attention_head_size`` (int): Required when scale_attention_scores is True. 
     When ``scale_attention_scores`` is passed, this contributes 
     ``1/sqrt(attention_head_size)`` to the scale factor.
   * ``scale_attention_scores`` (boolean): (default: True) determines whether 
     to multiply 1/sqrt(attention_head_size) to the scale factor.
   * ``layer_idx`` (int): Required when ``scale_attn_by_layer_idx`` is True. 
     The layer id to use for scaling attention by layer id. 
     It contributes 1/(layer_idx + 1) to the scaling factor.
   * ``scale_attn_by_layer_idx`` (boolean): (default: False) determines whether 
     to multiply 1/(layer_idx + 1) to the scale factor.
   * ``scale`` (float) (default: None): If passed, this scale factor will be 
     applied bypassing the above arguments.
   * ``triton_flash_attention`` (bool): (default: False) If passed, Triton 
     implementation of flash attention will be used. This is necessary to supports 
     Attention with Linear Biases (ALiBi) (see next arg). Note that this version of the kernel doesn’t support dropout.
   * ``use_alibi`` (bool): (default: False) If passed, it enables Attention with 
     Linear Biases (ALiBi) using the mask provided. 

   .. method:: forward(self, qkv, attn_mask=None, causal=False)

      Returns a single ``torch.Tensor`` ``(batch_size x num_heads x seq_len x head_size)``, 
      which represents the output of attention computation.

      **Parameters**
      
      * ``qkv``: ``torch.Tensor`` in the form of ``(batch_size x seqlen x 3 x num_heads x head_size)``.
      * ``attn_mask``: ``torch.Tensor`` in the form of ``(batch_size x 1 x 1 x seqlen)``. 
        By default it is ``None``, and usage of this mask needs ``triton_flash_attention`` 
        and ``use_alibi`` to be set. See how to generate the mask in the following code snippet.
      * ``causal``: When passed, it uses the standard lower triangular mask. The default is ``False``.

   When using ALiBi, it needs an attention mask prepared like the following.

   .. code:: python

      def generate_alibi_attn_mask(attention_mask, batch_size, seq_length, 
         num_attention_heads, alibi_bias_max=8):
         
         device, dtype = attention_mask.device, attention_mask.dtype
         alibi_attention_mask = torch.zeros(
            1, num_attention_heads, 1, seq_length, dtype=dtype, device=device
         )

         alibi_bias = torch.arange(1 - seq_length, 1, dtype=dtype, device=device).view(
            1, 1, 1, seq_length
         )
         m = torch.arange(1, num_attention_heads + 1, dtype=dtype, device=device)
         m.mul_(alibi_bias_max / num_attention_heads)
         alibi_bias = alibi_bias * (1.0 / (2 ** m.view(1, num_attention_heads, 1, 1)))

         alibi_attention_mask.add_(alibi_bias)
         alibi_attention_mask = alibi_attention_mask[..., :seq_length, :seq_length]
         if attention_mask is not None and attention_mask.bool().any():
            alibi_attention_mask.masked_fill(
                  attention_mask.bool().view(batch_size, 1, 1, seq_length), float("-inf")
            )

         return alibi_attention_mask






smdistributed.modelparallel.torch Context Managers and Util Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: smdistributed.modelparallel.torch.model_creation(tensor_parallelism=False, dtype=None, **tensor_parallel_config)

   Context manager to create a ``torch`` model. This API combines both the
   :class:`smdistributed.modelparallel.torch.tensor_parallelism` and
   :class:`smdistributed.modelparallel.torch.delay_param_initialization` decorators,
   so you can simply use this single context when creating the torch model.

   :param tensor_parallelism: Whether to enable tensor parallelism during model creation.
   :type tensor_parallelism: boolean
   :param dtype: The dtype to use when creating the model. It has the following rules.

      * If dtype is specified, it will be used during model creation.
      * If dtype is not specified, the default dtype will be used during model creation,
        which is usually FP32. This is for the best performance on CPU.
      * Any model that causes out-of-memory problems with FP32 initialization
        is recommended to be created with
        :class:`smdistributed.modelparallel.torch.delayed_parameter_initialization`.
      * ``FP16_Module`` casts the model back to FP16 if FP16 training is enabled
        with the ``smp`` config. For more inforamtion about FP16 training
        in SageMaker with the model parallel library, see `FP16 Training
        <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-fp16.html>`_
        in the *Amazon SageMaker Developer Guide*.

   :type dtype: ``torch.dtype``
   :param tensor_parallel_config: kwargs to specifiy other tensor parallel configs.
      This is not used if ``tensor_parallelism`` is ``False``.
   :type tensor_parallel_config: dict

   **Example Usage:**

   .. code:: python

      import smdistributed.modelparallel.torch as smp

      with smp.model_creation(
          tensor_parallelism=smp.tp_size() > 1,
          dtype=torch.float16 if args.fp16 else torch.get_default_dtype()
      ):
          model = MyModel(...)

.. function:: smdistributed.modelparallel.torch.partition(index)

   :param index: The index of the partition.
   :type index: int

   A context manager which places all modules defined inside into the
   partition with ID ``index``.  The ``index`` argument must be less than
   the number of partitions.

   Use ``smdistributed.modelparallel.torch.partition`` to implement manual partitioning.
   If ``"auto_partition"`` is ``True``, then the
   ``smdistributed.modelparallel.torch.partition`` contexts are ignored. Any module that is not placed in
   any ``smdistributed.modelparallel.torch.partition`` context is placed in the
   ``default_partition`` defined through the SageMaker Python SDK.

   When ``smdistributed.modelparallel.torch.partition`` contexts are nested, the innermost context
   overrides the rest (see the following example). In PyTorch, manual
   partitioning should be done inside the module \ ``__init__``, and the
   partition assignment applies to the modules that are *created* inside
   the ``smdistributed.modelparallel.torch.partition`` context.

   Example:

   .. code:: python

      import smdistributed.modelparallel.torch as smp

      class Model(torch.nn.Module):
          def __init__(self):
              with smp.partition(1):
                  self.child0 = Child0()            # child0 on partition 1
                  with smp.partition(2):
                      self.child1 = Child1()        # child1 on partition 2
                  self.child2 = Child2()            # child2 on partition 1
              self.child3 = Child3()                # child3 on default_partition

.. data:: smdistributed.modelparallel.torch.amp.GradScaler

   `Torch AMP Gradscaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`__
   currently doesn’t work with the library. ``smdistributed.modelparallel.torch.amp.GradScaler`` replaces
   ``torch.amp.GradScaler`` and provides the same functionality.

.. function:: smdistributed.modelparallel.torch.delay_param_initialization(enabled=True)

   If enabled, it delays the initialization of parameters
   to save CPU memory. That is, parameter initialization takes place
   after the model is partitioned on GPUs.

.. function:: smdistributed.modelparallel.torch.get_world_process_group( )

   Returns a ``torch.distributed`` ``ProcessGroup`` that consists of all
   processes, which can be used with the ``torch.distributed`` API.
   Requires ``"ddp": True`` in SageMaker Python SDK parameters.

.. function:: smdistributed.modelparallel.torch.get_mp_process_group( )

   Returns a ``torch.distributed`` ``ProcessGroup`` that consists of the
   processes in the ``MP_GROUP`` which contains the current process, which
   can be used with the \ ``torch.distributed`` API. Requires
   ``"ddp": True`` in SageMaker Python SDK parameters.

.. function:: smdistributed.modelparallel.torch.get_dp_process_group( )

   Returns a ``torch.distributed`` ``ProcessGroup`` that consists of the
   processes in the ``DP_GROUP`` which contains the current process, which
   can be used with the \ ``torch.distributed`` API. Requires
   ``"ddp": True`` in SageMaker Python SDK parameters.

.. function:: smdistributed.modelparallel.torch.is_initialized( )

   Returns ``True`` if ``smdistributed.modelparallel.torch.init`` has already been called for the
   process, and ``False`` otherwise.

.. function::smp.is_tracing( )

   Returns ``True`` if the current process is running the tracing step, and
   ``False`` otherwise.

.. data:: smdistributed.modelparallel.torch.nn.FusedLayerNorm

   `Apex Fused Layer Norm <https://nvidia.github.io/apex/layernorm.html>`__ is currently not
   supported by the library. ``smdistributed.modelparallel.torch.nn.FusedLayerNorm`` replaces ``apex``
   ``FusedLayerNorm`` and provides the same functionality. This requires
   ``apex`` to be installed on the system.

.. data:: smdistributed.modelparallel.torch.optimizers.FusedNovoGrad


   `Fused Novo Grad optimizer <https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedNovoGrad>`__ is
   currently not supported by the library. ``smdistributed.modelparallel.torch.optimizers.FusedNovoGrad`` replaces ``apex`` ``FusedNovoGrad``
   optimizer and provides the same functionality. This requires ``apex`` to
   be installed on the system.

.. data:: smdistributed.modelparallel.torch.optimizers.FusedLamb


   `FusedLamb optimizer <https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedLAMB>`__
   currently doesn’t work with the library. ``smdistributed.modelparallel.torch.optimizers.FusedLamb`` replaces
   ``apex`` ``FusedLamb`` optimizer and provides the same functionality.
   This requires ``apex`` to be installed on the system.

.. _pytorch_saving_loading:

smdistributed.modelparallel.torch APIs for Saving and Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: smdistributed.modelparallel.torch.save(obj, f, partial=True, pickel_module=picklemodule, pickle_protocol=2, )

   Saves an object. This operation is similar to `torch.save()
   <https://pytorch.org/docs/stable/generated/torch.save.html>`_, except that
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
      and further train a model that you save with ``smdistributed.modelparallel.torch.save()``, you must
      set ``partial=True``.
   -  ``pickle_module`` (picklemodule, default = module ``"pickle"`` from ``"/opt/conda/lib/python3.6/pickle.py"``):
      A module used for pickling metadata and objects.
   -  ``pickle_protocol``  (int, default=2): Can be specified to
      override the defaultprotocol.

.. function:: smdistributed.modelparallel.torch.load(f, map_location, pickle_module, pickle_load_args, partial=True)

   Loads an object saved with ``smdistributed.modelparallel.torch.save()`` from a file.

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

.. function:: smdistributed.modelparallel.torch.save_checkpoint(path, tag, partial=True, model=None, optimizer=None, user_content=None, translate_if_full=True, num_kept_partial_checkpoints=None)

   Saves a checkpoint. While :class:`smdistributed.modelparallel.torch.save` saves
   model and optimizer objects,
   this function checkpoints model and optimizer and saves the checkpoints as separate files.
   It creates checkpoint folders in the following structure.

   .. code:: text

      - path
      - ${tag}_partial        (folder for partial checkpoint)
        - model_rankinfo.pt
        - optimizer_rankinfo.pt
        - fp16_states_rankinfo.pt
        - user_content.pt
      - $tag                  (checkpoint file for full checkpoint)
      - user_content_$tag     (user_content file for full checkpoint)
      - newest                (a file that indicates the newest checkpoint)

   **Parameters**

   * ``path`` (str) (required): Path to save the checkpoint. The library creates
     the directory if it does not already exist.
     For example, ``/opt/ml/checkpoint/model_parallel``.
   * ``tag`` (str) (required): A tag for the current checkpoint, usually the train
     steps. Note: tag needs to be the same across all ranks (GPU workers).
     When ``partial=False`` this will be the checkpoint file name.
   * ``partial`` (boolean) (default: True): Whether to save the partial checkpoint.
   * ``model`` (:class:`smdistributed.modelparallel.torch.DistributedModel`)
     (default: None): The model to save. It needs to an ``smp.DistributedModel`` object.
   * ``optimizer`` (:class:`smdistributed.modelparallel.torch.DistributedOptimizer`)
     (default: None): The optimizer to save. It needs to be an ``smp.DistributedOptimizer`` object.
   * ``user_content`` (any) (default: None): User-defined content to save.
   * ``translate_if_full`` (boolean) (default: True): Whether to translate the
     full ``state_dict`` to HF ``state_dict`` if possible.
   * ``num_kept_partial_checkpoints`` (int) (default: None): The maximum number
     of partial checkpoints to keep on disk.

.. function:: smdistributed.modelparallel.torch.resume_from_checkpoint(path, tag=None, partial=True, strict=True, load_optimizer=True, load_sharded_optimizer_state=True, translate_function=None)

   While :class:`smdistributed.modelparallel.torch.load` loads saved
   model and optimizer objects, this function resumes from a saved checkpoint file.

   **Parameters**

   * ``path`` (str) (required): Path to load the checkpoint.
   * ``tag`` (str) (default: None): Tag of the checkpoint to resume. If not provided,
     the library tries to locate the newest checkpoint from the saved newest file.
   * ``partial`` (boolean) (default: True): Whether to load the partial checkpoint.
   * ``strict`` (boolean) (default: True): Load with strict load, no extra key or
     missing key is allowed.
   * ``load_optimizer`` (boolean) (default: True): Whether to load ``optimizer``.
   * ``load_sharded_optimizer_state`` (boolean) (default: True): Whether to load
     the sharded optimizer state of a model.
     It can be used only when you activate
     the `sharded data parallelism
     <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html>`_
     feature of the SageMaker model parallel library.
     When this is ``False``, the library only loads the FP16
     states, such as FP32 master parameters and the loss scaling factor,
     not the sharded optimizer states.
   * ``translate_function`` (function) (default: None): function to translate the full
     checkpoint into smdistributed.modelparallel format.
     For supported models, this is not required.

   **Example usage**

   .. code:: python

     # Save
     smp.save_checkpoint(
         checkpoint_dir,
         tag=f"total_steps{total_steps}",
         partial=True,
         model=model,
         optimizer=optimizer,
         user_content=user_content
         num_kept_partial_checkpoints=args.num_kept_checkpoints)

     # Load: this will automatically load the newest checkpoint
     user_content = smp.resume_from_checkpoint(path, partial=partial)

.. _pytorch_saving_loading_instructions:

General instruction on saving and loading
-----------------------------------------

The library can save partial or full checkpoints.

-  For partial checkpoints, each ``mp_rank`` saves its own checkpoint
   file with only the parameters that belong to that rank.
-  For full checkpoints, the library saves a single checkpoint that contains
   entire model parameters.

When **saving** using ``smdistributed.modelparallel.torch.save()``, each rank only holds its own
parameters. If you want to save the full model, there will be some
communication between the ranks to create the full model. If you save
checkpoints often, you should save partial checkpoints for best
performance.

When **loading** using ``smdistributed.modelparallel.torch.load()``, the library can load either partial or |
full checkpoints or full checkpoints saved by a non-model-parallel model. If you
want to resume training with a non-model-parallel model or do inference, you need
a full checkpoint.

The following is an example of how you can save and load a checkpoint:

.. code:: python

   import smdistributed.modelparallel.torch as smp
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
