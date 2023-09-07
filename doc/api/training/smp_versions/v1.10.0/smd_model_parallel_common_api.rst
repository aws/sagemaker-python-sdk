Common API
==========

The following SageMaker distribute model parallel APIs are common across all frameworks.

.. contents:: Table of Contents
  :depth: 3
  :local:

The Library's Core APIs
-----------------------

This API document assumes you use the following import statement in your training scripts.

**TensorFlow**

.. code:: python

   import smdistributed.modelparallel.tensorflow as smp

**PyTorch**

.. code:: python

   import smdistributed.modelparallel.torch as smp


.. function:: smp.init( )
   :noindex:

   Initialize the library. Must be called at the beginning of training script.

.. function:: @smp.step(non_split_inputs, input_split_axes, [*args, **kwargs])
   :noindex:

   A decorator that must be placed over a function that represents a single
   forward and backward pass (for training use cases), or a single forward
   pass (for evaluation use cases). Any computation that is defined inside
   the ``smp.step``-decorated function is executed in a pipelined manner.

   By default, every tensor input to the function is split across its batch
   dimension into a number of microbatches specified while launching the
   training job. This behavior can be customized through the arguments to
   ``smp.step``, described below. The library then orchestrates the execution of
   each microbatch across all partitions, based on the chosen pipeline
   type.

   In a typical use case, forward pass and back-propagation are executed
   inside an \ ``smp.step``-decorated function and gradients, loss, and
   other relevant metrics (such as accuracy, etc.) are returned from
   ``smp.step``-decorated function.

   Any gradient post-processing operation, such as gradient clipping and
   allreduce, as well as ``optimizer.apply_gradients`` calls (for TF) or
   ``optimizer.step`` (for PT) should be applied on the gradients returned
   from the ``smp.step`` function, and not inside the ``smp.step``
   function. This is because every operation inside ``smp.step`` is
   executed once per microbatch, so having these operations inside
   ``smp.step`` can either be inefficient (in the case of allreduce), or
   lead to wrong results (in the case of ``apply_gradients`` /
   ``optimizer.step``).

   If the objects returned from the ``smp.step``-decorated function contain
   ``tf.Tensor``\ s / ``torch.Tensor``\ s, they are converted to
   ``StepOutput`` objects. A ``StepOutput`` object encapsulates all
   versions of the tensor across different microbatches
   (see ``StepOutput`` entry for more information).

   The argument to ``smp.step`` decorated function should either be a tensor
   or an instance of list, tuple, dict or set for it to be split across
   microbatches. If your object doesn't fall into this category, you can make
   the library split your object, by implementing ``smp_slice`` method.

   Below is an example of how to use it with PyTorch.

   .. code:: python

      class CustomType:
          def __init__(self, tensor):
              self.data = tensor

          # The library will call this to invoke slicing on the object passing in total microbatches (num_mb)
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


   **Important:** ``smp.step`` splits the batch into microbatches, and
   executes everything inside the decorated function once per microbatch.
   This might affect the behavior of batch normalization, any operation
   that explicitly uses the batch size information, or any other Python
   code that is expected to run once.

   **TensorFlow-specific behavior**

   ``smp.step`` is a wrapper that
   inherits from and extends the behavior of ``tf.function``, and as such,
   all the caveats that apply to the use of ``tf.function``\ s also apply
   to ``smp.step``. In particular, any operation that is inside
   ``smp.step`` executes in graph mode, and not eager mode.

   In the first call, ``smp.step`` performs tracing of the wrapped function every time
   one of the tensor arguments changes their shape or dtype, or for every
   new value of a Python argument, if there is one. Tracing is expensive,
   so such scenarios should be avoided as much as possible or,
   alternatively, an ``input_signature`` argument must be provided. For
   more information on the usage of ``tf.function``, refer to the
   TensorFlow documentation:

   -  https://www.tensorflow.org/api_docs/python/tf/function\
   -  https://www.tensorflow.org/guide/function\

   Each ``smp.step`` decorated function must have a return value that depends on the
   output of ``smp.DistributedModel``.

   **Common parameters**

   -  ``non_split_inputs`` (``list``): The list of arguments to the decorated function
      that should not be split along the batch dimension. Should be used
      for all input tensors that do not have a batch dimension. Should be a
      list of argument names as ``str``, as they appear in the signature of
      the ``smp.step``-decorated function. By default it is considered an
      empty list.

   -  ``input_split_axes`` (``dict``): A dict that maps the argument name to its batch
      axis. The keys should be the argument names as ``str``, as they
      appear in the signature of the ``smp.step``-decorated function.  By
      default all batch axes are assumed to be the 0-axis.

   **TensorFlow-only parameters**

   -  All arguments of ``tf.function``. Note:
      The \ ``experimental_compile`` argument of ``tf.function`` may not
      work as expected with ``smp.step``, since it interferes with
      pipelining and model partitioning. To enable XLA with the library, you can
      instead use \ ``tf.config.optimizer.set_jit(True)``.

   **PyTorch-only parameters**

   -  ``detach_outputs`` (``bool``) : If ``True``, calls ``torch.Tensor.detach()`` on
      all returned ``torch.Tensor`` outputs. Setting it to ``False``
      increases memory consumption, unless ``detach()`` is manually called
      on the returned tensors, because the model graph is not cleared from
      memory after the training step. Set to \ ``True`` by default.

   **Returns**

   -  The same object(s) returned from the decorated function. All
      returned \ ``tf.Tensor``, \ ``tf.Variable``  objects (for TF) or
      ``torch.Tensor`` objects (for PT) are wrapped inside
      a \ ``StepOutput`` object, even when they are inside a Python
      ``list``, ``tuple``, or ``dict``.



.. class:: StepOutput
   :noindex:


   A class that encapsulates all versions of a ``tf.Tensor``
   or \ ``torch.Tensor`` across all microbatches.

   When a particular ``tf.Tensor`` or ``torch.Tensor`` is computed inside
   ``smp.step``, different versions of the tensor are computed for each
   microbatch.

   When this tensor is returned from ``smp.step`` and is accessed outside
   of the decorated function, it appears as a ``StepOutput`` object, which
   contains all such versions. For example,

   -  In the case of Tensorflow, the gradient for a particular
      ``tf.Variable`` is computed on each microbatch individually, and if
      this gradient is returned from ``smp.step``, all gradients for this
      ``tf.Variable`` become part of the same ``StepOutput`` object. The
      ``StepOutput`` class offers the following API for commonly-used
      post-processing operations on such tensors.
   -  In the case of PyTorch, the loss for each microbatch is computed
      individually and all the ``torch.Tensor``\ s that represent the loss
      for different microbatches become part of same ``StepOutput`` object,
      if loss is returned from the ``smp.step`` function.


   The ``StepOutput`` class offers the following API for commonly-used
   post-processing operations on tensors.

   .. data:: StepOutput.outputs
      :noindex:

      Returns a list of the underlying tensors, indexed by microbatch.

   .. function:: StepOutput.reduce_mean( )
      :noindex:

      Returns a ``tf.Tensor``, ``torch.Tensor`` that averages the constituent ``tf.Tensor`` s
      ``torch.Tensor`` s. This is commonly used for averaging loss and gradients across microbatches.

   .. function:: StepOutput.reduce_sum( )
      :noindex:

      Returns a ``tf.Tensor`` /
      ``torch.Tensor`` that sums the constituent
      ``tf.Tensor``\ s/\ ``torch.Tensor``\ s.

   .. function:: StepOutput.concat( )
      :noindex:

      Returns a
      ``tf.Tensor``/``torch.Tensor`` that concatenates tensors along the
      batch dimension using ``tf.concat`` / ``torch.cat``.

   .. function:: StepOutput.stack( )
      :noindex:

      Applies ``tf.stack`` / ``torch.stack``
      operation to the list of constituent ``tf.Tensor``\ s /
      ``torch.Tensor``\ s.

   **TensorFlow-only methods**

   .. function:: StepOutput.merge( )
      :noindex:

      Returns a ``tf.Tensor`` that
      concatenates the constituent ``tf.Tensor``\ s along the batch
      dimension. This is commonly used for merging the model predictions
      across microbatches.

   .. function:: StepOutput.accumulate(method="variable", var=None)
      :noindex:

      Functionally the same as ``StepOutput.reduce_mean()``. However, it is
      more memory-efficient, especially for large numbers of microbatches,
      since it does not wait for all constituent \ ``tf.Tensor``\ s to be
      ready to start averaging them, thereby saving memory.

      In some cases (XLA for example) ``StepOutput.reduce_mean()`` might end
      up being more memory-efficient than ``StepOutput.accumulate()``.

      **Parameters**

      -  ``method`` (``"add_n"`` or ``"accumulate_n"`` or ``"variable"``):
         If ``"add_n"`` or ``"accumulate_n"``, the library uses
         ``tf.add_n`` and ``tf.accumulate_n``, respectively, to implement
         accumulation. If ``"variable"``, the library uses an internal ``tf.Variable``
         into which to accumulate the tensors. Default is \ ``"variable"``.
         Note: Memory usage behavior of these choices can depend on the model
         and implementation.

      -  ``var``: A ``tf.Variable`` into which, if provided, the library uses to
         accumulate the tensors. If \ ``None``, the library internally creates a
         variable. If ``method`` is not ``"variable"``, this argument is
         ignored.

.. _mpi_basics:
   :noindex:

MPI Basics
----------

The library exposes the following basic MPI primitives to its Python API:

**Global**

-  ``smp.rank()`` : The global rank of the current process.
-  ``smp.size()`` : The total number of processes.
-  ``smp.get_world_process_group()`` :
   ``torch.distributed.ProcessGroup`` that contains all processes.
-  ``smp.CommGroup.WORLD``: The communication group corresponding to all processes.
-  ``smp.local_rank()``: The rank among the processes on the current instance.
-  ``smp.local_size()``: The total number of processes on the current instance.
-  ``smp.get_mp_group()``: The list of ranks over which the current model replica is partitioned.
-  ``smp.get_dp_group()``: The list of ranks that hold different replicas of the same model partition.

**Tensor Parallelism**

-  ``smp.tp_rank()`` : The rank of the process within its
   tensor-parallelism group.
-  ``smp.tp_size()`` : The size of the tensor-parallelism group.
-  ``smp.get_tp_process_group()`` : Equivalent to
   ``torch.distributed.ProcessGroup`` that contains the processes in the
   current tensor-parallelism group.
-  ``smp.CommGroup.TP_GROUP`` : The communication group corresponding to
   the current tensor parallelism group.

**Pipeline Parallelism**

-  ``smp.pp_rank()`` : The rank of the process within its
   pipeline-parallelism group.
-  ``smp.pp_size()`` : The size of the pipeline-parallelism group.
-  ``smp.get_pp_process_group()`` : ``torch.distributed.ProcessGroup``
   that contains the processes in the current pipeline-parallelism group.
-  ``smp.CommGroup.PP_GROUP`` : The communication group corresponding to
   the current pipeline parallelism group.

**Reduced-Data Parallelism**

-  ``smp.rdp_rank()`` : The rank of the process within its
   reduced-data-parallelism group.
-  ``smp.rdp_size()`` : The size of the reduced-data-parallelism group.
-  ``smp.get_rdp_process_group()`` : ``torch.distributed.ProcessGroup``
   that contains the processes in the current reduced data parallelism
   group.
-  ``smp.CommGroup.RDP_GROUP`` : The communication group corresponding
   to the current reduced data parallelism group.

**Model Parallelism**

-  ``smp.mp_rank()`` : The rank of the process within its model-parallelism
   group.
-  ``smp.mp_size()`` : The size of the model-parallelism group.
-  ``smp.get_mp_process_group()`` : ``torch.distributed.ProcessGroup``
   that contains the processes in the current model-parallelism group.
-  ``smp.CommGroup.MP_GROUP`` : The communication group corresponding to
   the current model parallelism group.

**Data Parallelism**

-  ``smp.dp_rank()`` : The rank of the process within its data-parallelism
   group.
-  ``smp.dp_size()`` : The size of the data-parallelism group.
-  ``smp.get_dp_process_group()`` : ``torch.distributed.ProcessGroup``
   that contains the processes in the current data-parallelism group.
-  ``smp.CommGroup.DP_GROUP`` : The communication group corresponding to
   the current data-parallelism group.

.. _communication_api:
   :noindex:

Communication API
-----------------

The library provides a few communication primitives which can be helpful while
developing the training script. These primitives use the following
``enum`` s as arguments to specify which processes the communication
should involve.
​

**Helper structures**

.. data:: smp.CommGroup
   :noindex:

   An ``enum`` that takes the values
   ``CommGroup.WORLD``, ``CommGroup.MP_GROUP``, and ``CommGroup.DP_GROUP``.
   These values can also be accessed as ``smp.WORLD``, ``smp.MP_GROUP``,
   and ``smp.DP_GROUP`` respectively.

   -  ``CommGroup.WORLD``: Represents the entire group of processes used in
      training
   -  ``CommGroup.MP_GROUP``: Represents the group of processes that hold
      the same model replica as the current process. The processes in a
      single ``MP_GROUP`` collectively store an entire replica of the
      model.
   -  ``CommGroup.DP_GROUP``: Represents the group of processes that hold
      the same model partition as the current process. The processes in a
      single ``DP_GROUP`` perform data parallelism/allreduce among
      themselves.

.. data:: smp.RankType
   :noindex:

   An ``enum`` that takes the values
   ``RankType.WORLD_RANK``, ``RankType.MP_RANK``, and ``RankType.DP_RANK``.

   -  ``RankType.WORLD_RANK``: The associated rank is to be interpreted as
      the rank of the process across all processes used in training.
   -  ``RankType.MP_RANK``: The associated rank is to be interpreted as the
      rank of the process within the ``MP_GROUP``.
   -  ``RankType.DP_RANK``: The associated rank is to be interpreted as the
      rank of the process within the ``DP_GROUP``.


**Communication primitives:**

.. function:: smp.broadcast(obj, group)
   :noindex:

   Sends the object to all processes in the
   group. The receiving process must call ``smp.recv_from`` to receive the
   sent object.

   **Inputs**

   -  ``obj``: An arbitrary picklable Python object that will be broadcast.

   -  ``group``: A ``CommGroup`` argument that represents to which group of
      processes the object will be sent.

   **Notes**

   -  When you use ``broadcast`` on the sender process, there needs
      to be an accompanying ``smp.recv_from()`` call on the receiver
      processes.

   -  This is a synchronous call; the ``broadcast`` statement
      returns only after all ranks participating in the call have made a
      matching ``recv_from`` call.

   **Example**

   .. code:: python

      if smp.rank() == 0:
          smp.broadcast(something, group=smp.CommGroup.WORLD)
      else:
          smp.recv_from(0, rank_type=smp.RankType.WORLD_RANK)

.. function:: smp.send(obj, dest_rank, rank_type)
   :noindex:

   Sends the object ``obj`` to
   ``dest_rank``, which is of a type specified by ``rank_type``.

   **Inputs**

   -  ``obj``: An arbitrary picklable Python object that will be sent.

   -  ``dest_rank`` (``int``): An integer denoting the rank of the receiving process.

   -  ``rank_type`` (``enum``): A ``smp.RankType`` ``enum`` that determines how
      ``dest_rank`` is to be interpreted. For example if ``dest_rank`` is 1
      and ``rank_type`` is ``MP_RANK``, then ``obj`` is sent to process
      with ``mp_rank`` 1 in the ``MP_GROUP`` which contains the current
      process.

   **Notes**

   -  Note: \ This is a synchronous call; the ``send`` statement returns
      only after the destination rank has made a matching
      ``recv_from`` call.

.. function:: smp.recv_from(src_rank, rank_type)
   :noindex:

   Receive an object from a peer process. Can be used with a matching
   ``smp.send`` or a ``smp.broadcast`` call.

   **Inputs**

   -  ``src_rank`` (``int``): An integer denoting rank of the sending process.

   -  ``rank_type`` (``enum``): A ``smp.RankType`` ``enum`` that determines how
      ``dest_rank`` is to be interpreted. For example if ``src_rank`` is 1
      and ``rank_type`` is ``MP_RANK``, then the object is received from
      the process with ``mp_rank`` 1 in the ``MP_GROUP`` which contains the
      current process.

   **Returns**

   Returns the python object that is sent by the peer process.

   **Notes**

   -  Note: This is a synchronous call; the ``recv_from`` statement returns
      only after the source rank has made a matching ``send`` or
      ``broadcast`` call, and the object is received.

.. function:: smp.allgather(obj, group)
   :noindex:

   A collective call that gathers all the
   submitted objects across all ranks in the specified ``group``. Returns a
   list whose ``i``\ th index contains the object submitted by the
   ``i``\ th rank in ``group``.

   **Inputs**

   -  ``obj``: An arbitrary picklable Python object that will be
      allgathered.

   -  ``group`` : A ``CommGroup`` argument that represents which group of
      processes participate in ``allgather``.

   **Notes**

   -  Note: This is a synchronous call; the ``allgather`` statement returns
      only after all ranks participating in the call have made a matching
      ``allgather`` call, and all the objects are received at the current
      rank.

   **Examples**

   .. code:: python

      # assuming mp_size() == 2

      if smp.mp_rank() == 0:
          out = smp.allgather(obj1, smp.CommGroup.MP_GROUP)  # returns [obj1, obj2]
      else:
          out = smp.allgather(obj2, smp.CommGroup.MP_GROUP)  # returns [obj1, obj2]

.. function:: smp.barrier(group=smp.WORLD)
   :noindex:

   A statement that hangs until all
   processes in the specified group reach the barrier statement, similar to
   ``MPI_Barrier()``.

   **Inputs**

   -  ``group``: An ``smp.CommGroup`` ``enum`` that specifies the group of
      processes participating in the barrier call. Defaults to
      ``smp.WORLD``.

   **Examples**

   -  Assume there are 8 processes and 2 model partitions, and
      therefore 4 \ ``mp_group``\ s, and 2 ``dp_group``\ s. If
      the \ ``barrier`` call is passed the value ``smp.MP_GROUP`` for its
      group argument, then each process only waits until the other process
      of its own ``mp_group`` reaches that point. It does not wait for
      processes outside that ``mp_group``.

.. function:: smp.dp_barrier()
   :noindex:

   Same as passing ``smp.DP_GROUP``\ to ``smp.barrier()``.
   Waits for the processes in the same \ ``dp_group`` as
   the current process to reach the same point in execution.

.. function:: smp.mp_barrier()
   :noindex:

   Same as passing ``smp.MP_GROUP`` to
   ``smp.barrier()``. Waits for the processes in the same ``mp_group`` as
   the current process to reach the same point in execution.
