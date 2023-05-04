.. _smdmp-pytorch-tensor-parallel:

PyTorch API for Tensor Parallelism
==================================

SageMaker distributed tensor parallelism works by replacing specific submodules
in the model with their distributed implementations. The distributed modules
have their parameters and optimizer states partitioned across tensor-parallel
ranks. This is to compute the same output as it would have been computed by
the original modules. Since tensor parallelism occurs across data-parallel
ranks, a rank might collect slices of the activations corresponding to the
data shards on other devices that are part of the same tensor parallelism group.

You can enable or disable tensor parallelism for specific parts of the model.
Within the enabled parts, the replacements with distributed modules will take
place on a best-effort basis for those module supported for tensor parallelism.
Alternatively, you can directly import and use the library’s distributed
modules in the model definition.

Some of the supported modules (such as ``smdistributed.modelparallel.torch.nn.Transformer``) are high-level
blocks that contain many operations. Because custom implementations
(as opposed to the built-in PyTorch modules) are typically used for these
high-level blocks, the library offers an API that you can use to register
specific distributed versions with such custom modules (provided that they
are functionally equivalent). This allows the library to automatically replace
the occurrences of such PyTorch modules with their distributed counterparts
provided by the library.
For more information, see the following topics.

.. contents:: Topics
  :depth: 3
  :local:

.. _registering-tp-modules:

Registering Tensor Parallelism Distributed Modules
--------------------------------------------------

Although PyTorch natively provides some of the commonly used (and
tensor-parallelizable) building blocks such as Transformer, users often
use custom implementations for such higher-level modules. To distribute
such modules with tensor parallelism, you need to register the
distributed modules to the custom module implementation in your class,
so that the library knows how to distribute the custom module. When you
register the distributed modules, make sure the custom module that you
use is functionally equivalent to the distributed module. You can verify
this by taking a look at the equivalent reference implementations in the
:ref:`smdmp-tp-appendix`.
These implementations are functionally equivalent to their distributed
versions in ``smdistributed.modelparallel.torch.nn`` module.

.. class:: smdistributed.modelparallel.torch.tp_register(dist_module, init_hook=None, forward_hook=None, return_hook=None)

   -  A decorator class that registers the ``dist_module`` class with
      the module class that it is attached to. The hooks can be used to
      adapt to different interfaces used with ``__init__`` and
      ``forward`` methods.
   -  **Arguments:**

      -  ``dist_module``: A subclass of ``smdistributed.modelparallel.torch.nn.DistributedModule``
         that implements the distributed version of the module class the
         decorator is attached to. Any distributed module class defined
         in ``smdistributed.modelparallel.torch.nn`` module can be used.
      -  ``init_hook``: A callable that translates the arguments of the
         original module ``__init__`` method to an ``(args, kwargs)``
         tuple compatible with the arguments of the corresponding
         distributed module ``__init__`` method. Must return a tuple,
         whose first element is an iterable representing the positional
         arguments, and second element is a ``dict`` representing the
         keyword arguments. The input signature of the ``init_hook``
         must **exactly** match the signature of the original
         ``__init__`` method (including argument order and default
         values), except it must exclude ``self``.
      -  ``forward_hook``: A callable that translates the arguments of
         the original module ``forward`` method to an ``(args, kwargs)``
         tuple compatible with the arguments of the corresponding
         distributed module ``forward`` method. Must return a tuple,
         whose first element is an iterable representing the positional
         arguments, and second element is a ``dict`` representing the
         keyword arguments. The input signature of the ``init_hook``
         must **exactly** match the signature of the original
         ``forward`` method (including argument order and default
         values), except it must exclude ``self``.
      -  ``return_hook``: A callable that translates the object returned
         from the distributed module to the return object expected of
         the original module.

   -  **Example:**

      .. code:: python

         import smdistributed.modelparallel.torch as smp

         init_hook = lambda config: ((), config.to_dict())

         # register smp.nn.DistributedTransformer
         # as the distributed version of MyTransformer
         @smp.tp_register(smp.nn.DistributedTransformer, init_hook=init_hook)
         class MyTransformer(nn.Module):
             def __init__(self, config):
                 ...

             def forward(self, hidden_states, attention_mask):
                 ...

.. function:: smdistributed.modelparallel.torch.tp_register_with_module(module_cls, dist_module, init_hook=None, forward_hook=None, return_hook=None)

   -  When you do not have direct access to model definition code, you
      can use this API to similarly register a distributed module with
      an existing module class.

   -  **Arguments:**

      -  ``module_cls``: The existing module class that will be
         distributed.
      -  ``dist_module``: A subclass of ``smdistributed.modelparallel.torch.nn.DistributedModule``
         that implements the distributed version of the module class the
         decorator is attached to. Any distributed module class defined
         in ``smdistributed.modelparallel.torch.nn`` module can be used.
      -  ``init_hook``: A callable that translates the arguments of the
         original module ``__init__`` method to an ``(args, kwargs)``
         tuple compatible with the arguments of the corresponding
         distributed module ``__init__`` method. Must return a tuple,
         whose first element is an iterable representing the positional
         arguments, and second element is a ``dict`` representing the
         keyword arguments. The input signature of the ``init_hook``
         must **exactly** match the signature of the original
         ``__init__`` method (including argument order and default
         values), except it must exclude ``self``.
      -  ``forward_hook``: A callable that translates the arguments of
         the original module ``forward`` method to an ``(args, kwargs)``
         tuple compatible with the arguments of the corresponding
         distributed module ``forward`` method. Must return a tuple,
         whose first element is an iterable representing the positional
         arguments, and second element is a ``dict`` representing the
         keyword arguments. The input signature of the ``init_hook``
         must **exactly** match the signature of the original
         ``forward`` method (including argument order and default
         values), except it must exclude ``self``.
      -  ``return_hook``: A callable that translates the object returned
         from the distributed module to the return object expected of
         the original module.

   -  **Example:**

      .. code:: python

         import smdistributed.modelparallel.torch as smp

         from somelibrary import MyTransformer

         init_hook = lambda config: ((), config.to_dict())

         # register smp.nn.DistributedTransformer as the distributed version of MyTransformer
         smp.tp_register_with_module(MyTransformer,
                                     smp.nn.DistributedTransformer,
                                     init_hook=init_hook)

.. _smdmp-supported-modules-for-tp:

Supported Modules for Tensor Parallelism
----------------------------------------

The following modules are supported for tensor parallelism.

.. contents:: Topics
  :depth: 3
  :local:

.. _tp-module-api:

Tensor Parallelism Module APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  :class:`smdistributed.modelparallel.torch.nn.DistributedLinear` (implements ``nn.Linear``)
-  :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLMHead`
-  :class:`smdistributed.modelparallel.torch.nn.DistributedTransformer`
-  :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLayer`
-  :class:`smdistributed.modelparallel.torch.nn.DistributedAttentionLayer`
-  :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerOutputLayer`
-  :class:`smdistributed.modelparallel.torch.nn.DistributedEmbedding`

.. class:: smdistributed.modelparallel.torch.nn.DistributedLinear(in_features, out_features)

    Tensor-parallel implementation of the ``nn.Linear`` class.
    Functionally equivalent to an ``nn.Linear`` module with the same
    ``in_features`` and ``out_features``. In other words,
    ``in_features`` and ``out_features`` are the number of *global*
    channels across tensor-parallel ranks.

    For more information about what's the reference implementation of this module,
    see :ref:`smdmp-tp-appendix`.


    -  **Arguments:**

      -  ``in_features``: The total number of input channels for the
         linear layer across all tensor-parallel ranks.
      -  ``out_features``: The total number of output channels for the
         linear layer across all tensor-parallel ranks.

.. class:: smdistributed.modelparallel.torch.nn.DistributedTransformerLMHead(num_layers=12, num_attention_heads=32, attention_head_size=32, hidden_size=1024, intermediate_size=4096, vocab_size=30522, num_positions=1024, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, num_token_types=0, causal_mask_size=None, add_cross_attention=False, add_lm_head=True,  initializer_range=0.02, use_normal_initialization=False, pre_layernorm=False, post_layernorm=True)

    Constructs a distributed transformer model, including embeddings
    and a single LM head. A word embedding of size
    ``(vocab_size, hidden_size)`` is created, as well as a positional
    embedding of size ``(num_positions, hidden_size)``, and the
    embeddings are added together. If ``num_token_types`` is larger
    than 0, a separate embedding of size
    ``(num_token_types, hidden_size)`` is created, and further added
    on top.

    -  The embeddings are fed through a ``DistributedTransformer``, and
       if ``add_lm_head`` is ``True``, the output passes through a single
       LM head, which is a linear module without bias whose weight is
       tied to the word embeddings.
    -  See :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLayer` for descriptions of the rest
       of the arguments.
    -  **Methods:**

      -  ``forward(self, inputs)``

         -  If ``add_cross_attention`` is ``True``, ``inputs`` must be a
            tuple
            ``(input_ids, attention_mask, token_type_ids, position_ids, cross_states, cross_states, cross_mask, labels)``.
         -  Otherwise, ``inputs`` must be a tuple
            ``(input_ids, attention_mask, token_type_ids, position_ids, labels)``.
         -  If ``token_type_ids`` is ``None``, token type embedding will
            not be used.
         -  ``input_ids`` is assumed to be of shape ``[N, S]``, where
            ``N`` is the batch size and ``S`` is sequence length.
         -  ``attention_mask`` is assumed to be a 0-1 tensor of shape
            ``[N, S]``, where 1 represents a masked position.

.. class:: smdistributed.modelparallel.torch.nn.DistributedTransformer(num_layers=12, num_attention_heads=32, attention_head_size=32, hidden_size=1024, intermediate_size=4096, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, causal_mask_size=None, add_cross_attention=False, pre_layernorm=False, post_layernorm=True)

   A sequence of :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLayer`\ s, whose
   number is given by ``num_layers`` argument. For the other
   arguments and methods, refer to
   :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLayer`.

   If both ``pre_layernorm`` and ``post_layernorm`` are ``True``,
   layer normalization is applied to both the input and the output of
   the ``DistributedTransformer``, in addition to the intermediate
   attention and transformer-output layers.

.. class:: smdistributed.modelparallel.torch.nn.DistributedTransformerLayer(num_attention_heads=32, attention_head_size=32, hidden_size=1024, intermediate_size=4096, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, causal_mask_size=None, add_cross_attention=False, pre_layernorm=False, post_layernorm=True)

   Tensor-parallel implementation of a single transformer layer.
   Number of attention heads, hidden size, and intermediate size
   refer to the global quantities across all tensor-parallel ranks.

   For more information about what's the reference implementation of this module,
   see :ref:`smdmp-tp-appendix`.

   -  **Arguments:**

      -  ``num_attention_heads``: The total number of attention heads
         across tensor-parallel ranks
      -  ``attention_head_size``: The number of channels of a single
         attention head.
      -  ``hidden_size``: The hidden dimension of the transformer. The
         input tensor ``hidden_states`` is assumed to have its last
         dimension size equal to ``hidden_size``.
      -  ``intermediate_size``: The number of output channels in the
         first linear transformation of the transformer output layer.
         ``DistributedTransformerOutputLayer`` first maps
         ``hidden_size`` dimensions of its input tensor into
         ``intermediate_size`` dimensions, and then maps it back into
         ``hidden_size`` dimensions.
      -  ``attention_dropout_prob``: The dropout probability applied to
         the attention probabilities.
      -  ``hidden_dropout_prob``: The dropout probability used in
         dropout layers other than the one applied to the attention
         probabilities.
      -  ``activation``: Choice of activation function to use at the
         output layer. Must be ``"gelu"`` or ``"relu"``.
      -  ``layernorm_epsilon``: The epsilon added to the denominator of
         layer normalization for numerical stability.
      -  ``initializer_range``: If ``use_normal_initialization`` is
         ``True``, the standard deviation of the normal random variable
         to initialize the weights with.
      -  ``use_normal_initialization``: If ``True``, the weights are
         initialized with normal distribution with standard deviation
         given by ``initializer_range``. Otherwise, default PyTorch
         initialization is used.
      -  ``causal_mask_size``: If ``None``, no causal mask is used on
         attentions. Otherwise, should be set to maximum sequence length
         to apply a causal mask to the attention scores. This is used,
         for instance, in GPT-2.
      -  ``add_cross_attention``: If ``True``, a cross-attention layer
         will be added after the self-attention block. The
         cross-attention layer computes the attention keys and values
         based on the ``cross_states`` input (instead of
         ``hidden_states`` input, as in self-attention. This is used in
         the decoder block of encoder-decoder architectures. For
         encoder-only architectures that only use self-attention, this
         should be kept ``False``.
      -  ``pre_layernorm``: If ``True``, inserts layer normalization at
         the input. At least one of ``pre_layernorm`` and
         ``post_layernorm`` must be ``True``.
      -  ``post_layernorm``: If ``True``, inserts layer normalization at
         the output. At least one of ``pre_layernorm`` and
         ``post_layernorm`` must be ``True``.
      -  ``use_alibi`` (bool, default False): Activates Attention with
         Linear Biases (ALiBi) for attention computation.
         ALiBi facilitates efficient extrapolation on input sequences
         and thus improves training efficiency.
         The library enables ALiBi by using the `Triton
         flash attention kernel
         <https://github.com/HazyResearch/flash-attention>`_.
         Refer to https://arxiv.org/abs/2108.12409 for more
         details on the technique.
         (Available from
         the SageMaker model parallelism library v1.15.0.)
      -  ``alibi_bias_max`` (int, default 8): Defines the ALiBi base
         value for mask generation. (Available from
         the SageMaker model parallelism library v1.15.0.)

   -  **Methods:**

      -  ``forward(self, inputs)``: Forward pass for the transformer
         layer.

         -  **Arguments:**

            -  If ``add_cross_attention=False``, ``inputs`` must be a
               tuple ``(hidden_states, attention_mask)``, where
               ``hidden_states`` is assumed to be a tensor of dimensions
               ``[N, S, H]``, where ``N`` is batch size, ``S`` is
               sequence length, and ``H`` is ``hidden_size``.
               ``attention_mask`` is assumed to be a tensor of
               dimensions ``[N, 1, 1, S]``, where ``N`` is the batch
               size, and ``S`` is the sequence length.
            -  If ``add_cross_attention=True``, ``inputs`` must be a
               tuple
               ``(hidden_states, cross_states, attention_mask, cross_mask)``,
               where ``hidden_states`` is assumed to be a tensor of
               dimensions ``[N, S_1, H]``, where ``N`` is batch size,
               ``S_1`` is sequence length, and ``H`` is ``hidden_size``.
               ``cross_states`` is assumed to be a tensor of size
               ``[N, S_2, H]``, similarly interpreted.
               ``attention_mask`` is assumed to be a tensor of
               dimensions ``[N, 1, 1, S_1]``, where ``N`` is the batch
               size, and ``S_1`` is the sequence length, and
               ``cross_mask`` is assumed to be a tensor of size
               ``[N, 1, 1, S_2]``. Keys and values for the attention
               heads in the cross-attention layer (but not the
               self-attention layer) are computed using
               ``cross_states``, and ``cross_mask`` is applied as the
               attention mask in the cross-attention layer (but not the
               self-attention layer).

         -  **Returns:**

            -  If ``add_cross_attention=False``, a tuple
               ``(hidden_states, attention_mask)``, where
               ``hidden_states`` is the output of the transformer, and
               ``attention_mask`` is the same the ``attention_mask``
               argument.
            -  If ``add_cross_attention=True``, a tuple
               ``(hidden_states, cross_states, attention_mask, cross_mask)``,
               where ``hidden_states`` is the output of the transformer,
               and the next three tensors are the same as the input
               arguments.

.. class:: smdistributed.modelparallel.torch.nn.DistributedAttentionLayer(num_attention_heads=32, attention_head_size=32, hidden_size=1024, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, cross_attention=False, causal_mask_size=None, pre_layernorm=False, post_layernorm=True)

   A distributed implementation for the attention block. Includes the
   computation of the self- or cross-attention (context layer),
   followed by a linear mapping and dropout, which is optionally
   followed by the residual-connection and layer normalization.

   For more information about what's the reference implementation of this module,
   see :ref:`smdmp-tp-appendix`.

   -  **Arguments:**

      -  See :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLayer` for descriptions of the
         arguments.
      -  ``cross_attention``: If ``True``, it computes the attentions
         with respect to the ``cross_states`` tensor of the ``forward``
         method input tuple. (Default: ``False``)

   -  **Methods:**

      -  ``forward(self, inputs)``: Forward pass for the attention
         layer.

         -  **Arguments:**

            -  If ``cross_attention=False``, ``inputs`` must be a tuple
               ``(hidden_states, attention_mask)``, where
               ``hidden_states`` is assumed to be a tensor of dimensions
               ``[N, S, H]``, where ``N`` is batch size, ``S`` is
               sequence length, and ``H`` is ``hidden_size``.
               ``attention_mask`` is assumed to be a tensor of
               dimensions ``[N, 1, 1, S]``, where ``N`` is the
               batch size, and ``S`` is the sequence length.
            -  If ``cross_attention=True``, ``inputs`` must be a tuple
               ``(hidden_states, cross_states, attention_mask)``, where
               ``hidden_states`` is assumed to be a tensor of dimensions
               ``[N, S_1, H]``, where ``N`` is batch size, ``S_1`` is
               sequence length, and ``H`` is ``hidden_size``.
               ``cross_states`` is assumed to be a tensor of size
               ``[N, S_2, H]``, similarly interpreted.
               ``attention_mask`` is assumed to be a tensor of
               dimensions ``[N, 1, 1, S_2]``, where ``N`` is the batch
               size, and ``S_2`` is the sequence length. Keys and values
               for the attention heads are computed using
               ``cross_states``.

         -  **Returns:**

            -  A single tensor that is the output of the attention
               layer.

.. class:: smdistributed.modelparallel.torch.nn.DistributedTransformerOutputLayer(hidden_size=1024, intermediate_size=4096,  hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, pre_layernorm=False, post_layernorm=True, fp32_residual_addition=False)

   -  Distributed implementation of a single transformer output layer. A
      single :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLayer` with
      ``add_cross_attention=False`` consists of a single
      ``DistributedAttentionLayer`` immediately followed by a single
      ``DistributedTransformerOutputLayer``. The latter linearly maps
      the last channel of the input tensor from ``hidden_size`` to
      ``intermediate_size``, and then maps it back to ``hidden_size``.

      For more information about what's the reference implementation of this module,
      see :ref:`smdmp-tp-appendix`.

   -  **Arguments:**

      -  See :class:`smdistributed.modelparallel.torch.nn.DistributedTransformerLayer` for descriptions of the
         arguments.
      - ``fp32_residual_addition``: Set to ``True`` if you want to avoid overflow
        (NaN loss values) for large models with more than 100 billion parameters
        when using FP16. (Default: False)

.. class:: smdistributed.modelparallel.torch.nn.DistributedEmbedding(num_embeddings,embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, initializer_range=0.02, _skip_allgather=False,_skip_scatter_and_merge=False,)

   -  Distributed implementation of a single Embedding Layer. Currently
      only supports splitting across the embedding_dim.
   -  **Arguments:**

      -  See :class:`smdistributed.modelparallel.torch.nn.DistributedEmbedding` for descriptions of the
         arguments.

.. _enabling-tp:

Enabling Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways tensor parallelism can be enabled.

First, you can use
the distributed module implementations in ``smdistributed.modelparallel.torch.nn`` module directly in
your model definition. See :ref:`smdmp-supported-modules-for-tp`
for a complete list of built-in distributed modules. Here is an example
of how this can be done:

.. code:: python

   import torch.nn as nn
   import smdistributed.modelparallel.torch as smp

   class TransformerModel:
       def __init__(self):
           self.embedding = nn.Embedding(vocab_size, hidden_size)

           # directly instantiate smp.nn.DistributedTransformer and use it
           self.encoder = smp.nn.DistributedTransformer(num_layers, hidden_size, **kwargs)

           self.pooler = nn.Linear(hidden_size, hidden_size)

       def forward(self, hidden_states):
           emb_out = self.embedding(hidden_states)
           enc_out = self.encoder(emb_out)
           return self.pooler(enc_out)

Second, you can enable tensor parallelism for specific modules or blocks
of code, which will automatically enable tensor parallelism for the
supported modules within that scope. To do this, you can use the
following API:

.. decorator:: smdistributed.modelparallel.torch.tensor_parallelism(enabled=True, **kwargs)

   -  A context manager that enables or disables tensor parallelism for
      any supported module that is created inside. If there are nested
      contexts, the innermost overrides the rest. If there are
      multiple supported modules created within the context, where one
      is the submodule of the other, only the outermost module will be
      distributed. If a supported module shares weights with another
      (supported or unsupported) module, or if its hyperparameters do
      not support distribution (e.g., not divisible by the tensor
      parallelism degree), tensor parallelism will **not** be enabled
      for this module even if this API is used.

      **Example:**

      .. code:: python

         import smdistributed.modelparallel.torch as smp

         with smp.tensor_parallelism():
             self.m0 = nn.Linear(20, 20)                   # will be distributed
             with smp.tensor_parallelism(enabled=False):
                 self.m1 = nn.Linear(20, 20)               # will not be distributed

   - ``kwargs`` - Keyword arguments that can be used to modify the configurations of
     the distributed modules created inside the context.
     If a keyword argument provided through it matches any ``__init__`` method arguments
     of a ``DistributedModule`` that substitutes a module created inside
     the ``smdistributed.modelparallel.torch.tensor_parallelism`` context, this keyword will override
     the value defined in the ``init_hook``.

     - (*For v1.7.0 and later*) Through the following additional keyword arguments,
       the library supports `NVIDIA Megatron’s fused kernels
       <https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/fused_kernels>`_

       - ``fused_softmax`` (bool) - Fusion of attention masking and softmax.
         By default, it is set to ``True``. You can deactivate it by setting
         ``fused_softmax=False`` in the ``smdistributed.modelparallel.torch.tensor_parallelism`` context manager.
       - ``fused_bias_gelu`` (bool) - Fusion of bias addition and Gelu activation.
         By default, it is set to ``False``. You can activate it by setting
         ``fused_bias_gelu=True`` in the ``smdistributed.modelparallel.torch.tensor_parallelism`` context manager.



.. function:: smdistributed.modelparallel.torch.set_tensor_parallelism(module, enabled=True, **kwargs)

   -  Enables or disables tensor parallelism for the supported
      submodules of ``module``. If enabling, the outermost supported
      modules will be distributed. If disabling, tensor parallelism will
      be disabled for the entire module subtree of ``module``. Unlike
      the context manager, this API can be used after the model creation
      (but before wrapping with :class:`smdistributed.modelparallel.torch.DistributedModel`), so direct
      access to model definition code is not required. If a supported
      module shares weights with another (supported or unsupported)
      module, or if its hyperparameters do not support distribution
      (e.g., not divisible by the tensor parallelism degree), tensor
      parallelism will **not** be enabled for this module.
   -  Keyword arguments ``kwargs`` can be used to modify the
      configurations of the distributed modules created inside the
      context. If a keyword argument provided here matches any
      ``__init__`` method arguments of a :class:`smdistributed.modelparallel.torch.DistributedModel` that
      substitutes a module created inside the ``smdistributed.modelparallel.torch.tensor_parallelism``
      context, this keyword will override the value defined in the
      ``init_hook``.
   -  **Example:**

      .. code:: python

         import smdistributed.modelparallel.torch as smp

         model = MyModel()
         smp.set_tensor_parallelism(model.encoder, True)
         smp.set_tensor_parallelism(model.encoder.embedding, True)

         # outermost supported submodules in model.encoder will be distributed, except for
         # model.encoder.embedding
         model = smp.DistributedModel(model)
         optimizer = smp.DistributedOptimizer(optimizer)

.. _activation-checkpointing-api:

Activation Checkpointing APIs
-----------------------------

``smdistributed.modelparallel`` provides three APIs to enable
activation checkpointing: one for checkpointing modules,
one for checkpointing sequential modules, and
one for checkpointing pretrained models.

For a conceptual guide and examples, see
`Activation Checkpointing <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-checkpointing.html>`_
in the *SageMaker's Distributed Model Parallel developer guide*.

.. class:: smdistributed.modelparallel.torch.patches.checkpoint.checkpoint(module, *args, preserve_rng_state=True)

   -  Checkpoints the module passed. Throws error if, during manual
      partitioning, all children of module are not on same rank as the
      module itself, i.e. the module tree is split across multiple
      partitions. During auto-partitioning, if the module is split
      across multiple partitions, then this call is ignored(with a
      warning). Note that this call applies to the module instance only,
      not to the module class.

   -  **Arguments:**

      -  ``module (Instance of nn.Module)``: The module to be
         checkpointed. Note that unlike native checkpointing in
         PyTorch’s, activation checkpointing in
         ``smdistributed.modelparallel`` is at the granularity of a
         module. A generic function cannot be passed here.
      -  ``args``: Tuple containing inputs to the module.
      -  ``preserve_rng_state (bool, default=True)``: Omit stashing and
         restoring the RNG state during each checkpoint.

.. class:: smdistributed.modelparallel.torch.patches.checkpoint.checkpoint_sequential(sequential_module, input, strategy="each", preserve_rng_state=True, pack_args_as_tuple=False)

   -  Checkpoints the modules inside
      `nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`__.
      This can be used even if different layers that are part of the
      sequential container lie on different partitions. Each layer part
      of the sequential module that is checkpointed must lie completely
      within one partition. If this is not the case during manual
      partitioning, then an error will be thrown. If this is not the
      case during auto partitioning, a warning will be raised and this
      module will be run without checkpointing.

   -  **Arguments**

      -  ``sequential_module (nn.Sequential)``: the sequential module to
         be checkpointed.
      -  ``input (torch.Tensor or a tuple of torch.Tensors)``: input to
         the module, which can be a tensor or a tuple of tensors. If a
         tuple is passed, then pack_args_as_tuple should be set to True.
      -  ``strategy (string, default=“each”)`` : Strategy determines how
         many layers part of the sequential module need to be grouped
         together for one checkpointing call. This determines how much
         memory can be reduced. It can take the following values

         -  ``each`` : The default is to checkpoint each module inside
            the sequential separately.
         -  ``contiguous``: Groups consecutive layers on the same
            partition together. For example, if a sequential consists of
            [a, b, c, d] where a,b are on pp_rank0 and c,d are on
            pp_rank 1, then this strategy would checkpoint a,b together
            and then c,d together. This means effectively, inputs of a,
            outputs of b, inputs of c, and outputs of d are in memory;
            the reamining activations are recomputed.
         -  ``group_2, group_3, group_4, etc:`` More generally,
            ``group_x`` where x is an integer. This strategy provides
            more flexibility in how many layers to group together.
            ``group_x`` groups x layers together on a best effort basis.
            It can group x layers together if there are x layers
            consecutively on the same partition. For example:
            [a,b,c,d,e] where a,b are on pp_rank0 and c,d,e are on
            pp_rank 1. If the strategy is ``group_3,`` then a,b are
            checkpointed together on pp_rank0 and c,d,e are checkpointed
            together on pp_rank1.

      -  ``preserve_rng_state (bool, default=True)``: Set to ``False``
         to omit stashing and restoring the RNG state during each
         checkpoint.
      -  ``pack_args_as_tuple (bool, default=False)``: To ensure that
         backward works correctly, the autograd function has to unpack
         any tuples received. If the checkpointed layer takes a tuple as
         input, then this needs to be set to True.

.. class:: smdistributed.modelparallel.torch.set_activation_checkpointing(module, preserve_rng_state=True, pack_args_as_tuple=False, strategy="each")

   -  This API is recommended when importing pretrained models from
      libraries, such as PyTorch and Hugging Face Transformers. This is
      particularly useful when you don’t have access to the model
      definition code and not be able to replace a module call with
      checkpoint.

   -  **Arguments**:

      -  ``module (Instance of nn.Module or nn.Sequential)``: The module
         to checkpoint.
      -  ``preserve_rng_state (bool, default=True)``: Set to ``False``
         to omit stashing and restoring the RNG state during each
         checkpoint.
      -  ``pack_args_as_tuple (bool, default=False)``: *Can only be
         passed when module is a sequential module.* To ensure that
         backward works correctly, the autograd function has to unpack
         any tuples received. If the layer checkpointed takes a tuple as
         input, then this needs to be set to True.
      -  ``strategy: (string, default=“each”)``: *Can only be passed
         when module is a sequential module.* Strategy determines how
         many layers part of the sequential module need to be grouped
         together for one checkpointing call.
      -  This determines how much memory can be reduced. It can take the
         following values

         -  ``each`` : The default is to checkpoint each module inside
            the sequential separately.
         -  ``contiguous``: Groups consecutive layers on the same
            partition together. For example if a sequential consists of
            ``[a, b, c, d]`` where ``a, b`` are on ``pp_rank0`` and ``c, d`` are on
            ``pp_rank 1``, then this strategy would checkpoint a,b together
            and then ``c, d`` together. This means effectively, the inputs of
            ``a``, outputs of ``b``, inputs of ``c``, and outputs of ``d`` are in
            memory, and the rest of the activations are recomputed.
         -  ``group_2, group_3, group_4, etc:`` More generally,
            ``group_x`` where x is an integer. This strategy provides
            more flexibility in how many layers to group together.
            ``group_x`` groups x number of layers together on a best
            effort basis if there are x layers consecutively in the same
            partition. **Example**: Assume a module with layers ``[a, b,
            c, d, e]``. The layers a and b are on pp_rank0, and ``c``, ``d``, and
            ``e`` are on ``pp_rank 1``. If the strategy is ``group_3,`` then ``a``,
            ``b`` are checkpointed together on ``pp_rank0``, and ``c``, ``d``, ``e`` are
            checkpointed together on ``pp_rank1``.

.. _smdmp-tp-appendix:

Appendix: Reference Implementations for Modules
-----------------------------------------------

The following are reference implementations for transformer-related
modules. Note that this is not the actual ``smdistributed`` source code,
but the distributed implementations provided in the library are the
distributed versions of these reference implementations, and can be used
to determine whether the distributed modules perform the same operations
as the custom modules in your script.

To keep the implementations simple, we only assume keyword arguments,
and assume the existence of a method ``parse_args(kwargs)``, which
parses the arguments to ``__init__`` methods and sets the relevant
attributes of the module, such as ``hidden_size`` and
``num_attention_heads``.

``smdistributed.modelparallel.torch.nn.DistributedTransformer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   class Transformer(nn.Module):
       def __init__(self, **kwargs):
           super(Transformer, self).__init__()
           self.parse_args(kwargs)

           self.layers = []
           for l in range(self.num_layers):
               self.layers.append(TransformerLayer(**kwargs))

           self.seq_layers = nn.Sequential(*self.layers)

       def forward(self, inp):
           return self.seq_layers(inp)

``smdistributed.modelparallel.torch.nn.DistributedTransformerLayer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   class TransformerLayer(nn.Module):
       def __init__(self, **kwargs):
           super(TransformerLayer, self).__init__()
           self.parse_args(kwargs)

           self.attention = AttentionLayer(**kwargs)
           self.output = TransformerOutputLayer(**kwargs)

           if self.add_cross_attention:
               self.cross_attention = AttentionLayer(cross_attention=True, **kwargs)

       def forward(self, inp):
           if self.add_cross_attention:
               hidden_states, cross_states, attention_mask, cross_mask = inp
           else:
               hidden_states, attention_mask = inp

           attention_output = self.attention((hidden_states, attention_mask))
           if self.add_cross_attention:
               attention_output = self.cross_attention((attention_output,
                                                        cross_states,
                                                        cross_mask))

           output = self.output(attention_output)

           if self.add_cross_attention:
               return output, cross_states, attention_mask, cross_mask
           else:
               return output, attention_mask

``smdistributed.modelparallel.torch.nn.DistributedAttentionLayer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   class AttentionLayer(nn.Module):
       def __init__(self, **kwargs):
           super(AttentionLayer, self).__init__()
           self.parse_args(kwargs)
           self.attention_head_size = self.hidden_size // self.num_attention_heads

           self.query = nn.Linear(self.hidden_size, self.hidden_size)
           self.key = nn.Linear(self.hidden_size, self.hidden_size)
           self.value = nn.Linear(self.hidden_size, self.hidden_size)
           self.dense = nn.Linear(self.hidden_size, self.hidden_size)

           self.dropout1 = nn.Dropout(self.attention_dropout_prob)
           self.dropout2 = nn.Dropout(self.hidden_dropout_prob)

           if self.pre_layernorm:
               self.pre_layernorm = nn.LayerNorm(self.hidden_size,
                                       eps=self.layernorm_epsilon)

           if self.post_layernorm:
               self.layernorm = nn.LayerNorm(self.hidden_size,
                                       eps=self.layernorm_epsilon)

       def transpose(self, tensor, key=False):
           shape = tensor.size()[:-1] +
                           (self.num_attention_heads, self.attention_head_size)
           tensor = torch.reshape(tensor, shape)
           if key:
               return tensor.permute(0, 2, 3, 1)
           else:
               return tensor.permute(0, 2, 1, 3)

       def forward(self, inp):
           if self.cross_attention:
               hidden_states, cross_states, attention_mask = inp
           else:
               hidden_states, attention_mask = inp

           if self.pre_layernorm:
               norm_states = self.pre_layernorm(hidden_states)
           else:
               norm_states = hidden_states

           query_layer = self.query(norm_states)

           if self.cross_attention:
               key_layer = self.key(cross_states)
               value_layer = self.value(cross_states)
           else:
               key_layer = self.key(norm_states)
               value_layer = self.value(norm_states)

           query_layer = self.transpose(query_layer)
           key_layer = self.transpose(key_layer, key=True)
           value_layer = self.transpose(value_layer)

           attention_scores = torch.matmul(query_layer, key_layer)
           attention_scores = attention_scores / math.sqrt(self.attention_head_size)

           if not self.cross_attention and self.causal_mask is not None:
               attention_scores = self.apply_causal_mask(attention_scores)

           attention_scores = attention_scores + attention_mask

           attention_probs = F.softmax(attention_scores, dim=-1)
           attention_probs = self.dropout1(attention_probs)

           context_layer = torch.matmul(attention_probs, value_layer)
           context_layer = context_layer.permute(0, 2, 1, 3)
           new_context_layer_shape = context_layer.size()[:-2] + \
                                       (self.local_attention_size,)
           context_layer = torch.reshape(context_layer, new_context_layer_shape)

           self_attention = self.dense(context_layer)
           self_attention = self.dropout2(self_attention)

           if self.post_layernorm:
               return self.layernorm(self_attention + hidden_states)
           else:
               return self_attention

``smdistributed.modelparallel.torch.nn.DistributedTransformerOutputLayer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   class TransformerOutputLayer(nn.Module):
       def __init__(self, **kwargs):
           super(TransformerOutputLayer, self).__init__()
           self.parse_args(kwargs)

           self.dense1 = nn.Linear(self.hidden_size, self.intermediate_size)
           self.dense2 = nn.Linear(self.intermediate_size, self.hidden_size)

           self.dropout = nn.Dropout(self.attention_dropout_prob)

           if self.pre_layernorm:
               self.pre_layernorm = nn.LayerNorm(self.hidden_size,
                                       eps=self.layernorm_epsilon)

           if self.post_layernorm:
               self.layernorm = nn.LayerNorm(self.hidden_size,
                                       eps=self.layernorm_epsilon)

       def forward(self, inp):
           if self.pre_layernorm:
               norm_inp = self.pre_layernorm(inp)
           else:
               norm_inp = inp

           dense1_output = self.dense1(norm_inp)
           if self.activation == "gelu":
               act_output = F.gelu(dense1_output)
           else:
               act_output = F.relu(dense1_output)

           dense2_output = self.dense2(act_output)
           output = self.dropout(dense2_output)

           if self.post_layernorm:
               return self.layernorm(inp + output)
           else:
               return output
