.. _smdmp-pytorch-tensor-parallel:

Tensor Parallelism API for PyTorch
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

Some of the supported modules (such as smp.nn.Transformer) are high-level
blocks that contain many operations. Because custom implementations
(as opposed to the built-in PyTorch modules) are typically used for these
high-level blocks, the library offers an API that you can use to register
specific distributed versions with such custom modules (provided that they
are functionally equivalent). This allows the library to automatically replace
the occurrences of such PyTorch modules with their distributed counterparts
provided by the library.
For more information, see :ref:`registering-tp-modules`.

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
`Appendix <https://quip-amazon.com/ZCbgA7XdyJV5/Getting-Started-with-Tensor-Parallelism-in-the-SageMaker-Distributed-Model-Parallelism-Library#HNR9CAPi42F>`__.
These implementations are functionally equivalent to their distributed
versions in ``smp.nn`` module.

.. decorator:: @smp.tp_register(dist_module, init_hook=None, forward_hook=None, return_hook=None)

   -  A class decorator that registers the ``dist_module`` class with
      the module class that it is attached to. The hooks can be used to
      adapt to different interfaces used with ``__init__`` and
      ``forward`` methods.
   -  **Arguments:**

      -  ``dist_module``: A subclass of ``smp.nn.DistributedModule``
         that implements the distributed version of the module class the
         decorator is attached to. Any distributed module class defined
         in ``smp.nn`` module can be used.
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

         init_hook = lambda config: ((), config.to_dict())

         # register smp.nn.DistributedTransformer
         # as the distributed version of MyTransformer
         @smp.tp_register(smp.nn.DistributedTransformer, init_hook=init_hook)
         class MyTransformer(nn.Module):
             def __init__(self, config):
                 ...

             def forward(self, hidden_states, attention_mask):
                 ...

.. function:: smp.tp_register_with_module(module_cls, dist_module, init_hook=None, forward_hook=None, return_hook=None)

   -  When you do not have direct access to model definition code, you
      can use this API to similarly register a distributed module with
      an existing module class.

   -  **Arguments:**

      -  ``module_cls``: The existing module class that will be
         distributed.
      -  ``dist_module``: A subclass of ``smp.nn.DistributedModule``
         that implements the distributed version of the module class the
         decorator is attached to. Any distributed module class defined
         in ``smp.nn`` module can be used.
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

         from somelibrary import MyTransformer

         init_hook = lambda config: ((), config.to_dict())

         # register smp.nn.DistributedTransformer as the distributed version of MyTransformer
         smp.tp_register_with_module(MyTransformer,
                                     smp.nn.DistributedTransformer,
                                     init_hook=init_hook)


Supported Modules for Tensor Parallelism
----------------------------------------

The following modules are supported for tensor
parallelism.

-  ``smp.nn.DistributedLinear`` (implements ``nn.Linear``)
-  ``smp.nn.DistributedTransformerLMHead``
-  ``smp.nn.DistributedTransformer``
-  ``smp.nn.DistributedTransformerLayer``
-  ``smp.nn.DistributedAttentionLayer``
-  ``smp.nn.DistributedTransformerOutputLayer``
-  ``smp.nn.DistributedEmbedding``

For more information about the modules, see :ref:`tp-module-api`.

To find example of using the modules, see :ref:`enabling-tp`.

.. _tp-module-api:

Tensor Parallelism Module APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: smp.nn.DistributedLinear(in_features, out_features)

   -  Tensor-parallel implementation of the ``nn.Linear`` class.
      Functionally equivalent to an ``nn.Linear`` module with the same
      ``in_features`` and ``out_features``. In other words,
      ``in_features`` and ``out_features`` are the number of *global*
      channels across tensor-parallel ranks.
   -  **Arguments:**

      -  ``in_features``: The total number of input channels for the
         linear layer across all tensor-parallel ranks.
      -  ``out_features``: The total number of output channels for the
         linear layer across all tensor-parallel ranks.

.. class:: smp.nn.DistributedTransformerLMHead(num_layers=12, num_attention_heads=32, attention_head_size=32, hidden_size=1024, intermediate_size=4096, vocab_size=30522, num_positions=1024, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, num_token_types=0, causal_mask_size=None, add_cross_attention=False, add_lm_head=True,  initializer_range=0.02, use_normal_initialization=False, pre_layernorm=False, post_layernorm=True)

   -  Constructs a distributed transformer model, including embeddings
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
   -  See ``DistributedTransformerLayer`` for a description of the rest
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

.. class:: smp.nn.DistributedTransformer(num_layers=12, num_attention_heads=32, attention_head_size=32, hidden_size=1024, intermediate_size=4096, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, causal_mask_size=None, add_cross_attention=False, pre_layernorm=False, post_layernorm=True)

   -  A sequence of ``smp.nn.DistributedTransformerLayer``\ s, whose
      number is given by ``num_layers`` argument. For the other
      arguments and methods, refer to
      ``smp.nn.DistributedTransformerLayer``.
   -  If both ``pre_layernorm`` and ``post_layernorm`` are ``True``,
      layer normalization is applied to both the input and the output of
      the ``DistributedTransformer``, in addition to the intermediate
      attention and transformer-output layers.

.. class:: smp.nn.DistributedTransformerLayer(num_attention_heads=32, attention_head_size=32, hidden_size=1024, intermediate_size=4096, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, causal_mask_size=None, add_cross_attention=False, pre_layernorm=False, post_layernorm=True)

   -  Tensor-parallel implementation of a single transformer layer.
      Number of attention heads, hidden size, and intermediate size
      refer to the global quantities across all tensor-parallel ranks.
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

.. class:: smp.nn.DistributedAttentionLayer(num_attention_heads=32, attention_head_size=32, hidden_size=1024, attention_dropout_prob=0.1, hidden_dropout_prob=0.1, layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, cross_attention=False, causal_mask_size=None, pre_layernorm=False, post_layernorm=True)

   -  A distributed implementation for the attention block. Includes the
      computation of the self- or cross-attention (context layer),
      followed by a linear mapping and dropout, which is optionally
      followed by the residual-connection and layer normalization.
   -  **Arguments:**

      -  See ``DistributedTransformerLayer`` for a description of the
         arguments.
      -  If ``cross_attention`` is ``True``, computes the attentions
         with respect to the ``cross_states`` tensor of the ``forward``
         method input tuple.

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
               dimensions ``[N, 1, 1, S]``, \***\* where ``N`` is the
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

.. class:: smp.nn.DistributedTransformerOutputLayer(hidden_size=1024, intermediate_size=4096,  hidden_dropout_prob=0.1, activation="gelu", layernorm_epsilon=1e-5, initializer_range=0.02, use_normal_initialization=False, pre_layernorm=False, post_layernorm=True)

   -  Distributed implementation of a single transformer output layer. A
      single ``DistributedTransformerLayer`` with
      ``add_cross_attention=False`` consists of a single
      ``DistributedAttentionLayer`` immediately followed by a single
      ``DistributedTransformerOutputLayer``. The latter linearly maps
      the last channel of the input tensor from ``hidden_size`` to
      ``intermediate_size``, and then maps it back to ``hidden_size``.
   -  **Arguments:**

      -  See ``DistributedTransformerLayer`` for a description of the
         arguments.

.. class:: smp.nn.DistributedEmbedding(num_embeddings,embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, initializer_range=0.02, _skip_allgather=False,_skip_scatter_and_merge=False,)

   -  Distributed implementation of a single Embedding Layer. Currently
      only supports splitting across the embedding_dim.
   -  **Arguments:**

      -  See ``DistributedEmbedding`` for a description of the
         arguments.

.. _enabling-tp:

Enabling Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways tensor parallelism can be enabled.

First, you can use
the distributed module implementations in ``smp.nn`` module directly in
your model definition. See `Supported
Modules <https://quip-amazon.com/ZCbgA7XdyJV5/Getting-Started-with-SageMaker-Distributed-Model-Parallelism-Library-Private-Preview#HNR9CAYjMQN>`__
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

.. decorator:: smp.tensor_parallelism(enabled=True, **kwargs)

   -  A context manager that enables or disables tensor parallelism for
      any supported module that is created inside. If there are nested
      contexts, the innermost will override the rest. If there are
      multiple supported modules created within the context, where one
      is the submodule of the other, only the outermost module will be
      distributed. If a supported module shares weights with another
      (supported or unsupported) module, or if its hyperparameters do
      not support distribution (e.g., not divisible by the tensor
      parallelism degree), tensor parallelism will **not** be enabled
      for this module even if this API is used.

      **Example:**

      .. code:: python

         with smp.tensor_parallelism():
             self.m0 = nn.Linear(20, 20)                   # will be distributed
             with smp.tensor_parallelism(enabled=False):
                 self.m1 = nn.Linear(20, 20)               # will not be distributed

   - Keyword arguments `kwargs` can be used to modify the configurations of the distributed modules created inside the context. If a keyword argument provided here matches any `__init__` method arguments of a `DistributedModule` that substitutes a module created inside the `smp.tensor_parallelism` context, this keyword will override the value defined in the `init_hook`.

.. function:: smp.set_tensor_parallelism(module, enabled=True, **kwargs)

   -  Enables or disables tensor parallelism for the supported
      submodules of ``module``. If enabling, the outermost supported
      modules will be distributed. If disabling, tensor parallelism will
      be disabled for the entire module subtree of ``module``. Unlike
      the context manager, this API can be used after the model creation
      (but before wrapping with :class:`smp.DistributedModel`), so direct
      access to model definition code is not required. If a supported
      module shares weights with another (supported or unsupported)
      module, or if its hyperparameters do not support distribution
      (e.g., not divisible by the tensor parallelism degree), tensor
      parallelism will **not** be enabled for this module.
   -  Keyword arguments ``kwargs`` can be used to modify the
      configurations of the distributed modules created inside the
      context. If a keyword argument provided here matches any
      ``__init__`` method arguments of a :class:`smp.DistributedModel` that
      substitutes a module created inside the ``smp.tensor_parallelism``
      context, this keyword will override the value defined in the
      ``init_hook``.
   -  **Example:**

      .. code:: python

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

.. class:: smp.set_activation_checkpointing(module, preserve_rng_state=True, pack_args_as_tuple=False, strategy="each")

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
