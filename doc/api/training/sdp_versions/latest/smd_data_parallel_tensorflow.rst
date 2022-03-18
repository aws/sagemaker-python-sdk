####################
Guide for TensorFlow
####################

Use this guide to learn how to use the SageMaker distributed
data parallel library API for TensorFlow.

.. contents:: Topics
  :depth: 3
  :local:

.. _tensorflow-sdp-modify:

Modify a TensorFlow 2.x training script to use the SageMaker data parallel library
==================================================================================

The following steps show you how to convert a TensorFlow 2.x training
script to utilize the distributed data parallel library.

The distributed data parallel library APIs are designed to be close to Horovod APIs.
See `SageMaker distributed data parallel TensorFlow examples
<https://sagemaker-examples.readthedocs.io/en/latest/training/distributed_training/index.html#tensorflow-distributed>`__
for additional details on how to implement the data parallel library.

-  First import the distributed data parallel library’s TensorFlow client and initialize it:

   .. code:: python

      import smdistributed.dataparallel.tensorflow as sdp
      sdp.init()


-  Pin each GPU to a single smdistributed.dataparallel process
   with ``local_rank`` - this refers to the relative rank of the
   process within a given node. ``sdp.tensorflow.local_rank()`` API
   provides you the local rank of the device. The leader node will be
   rank 0, and the worker nodes will be rank 1, 2, 3, and so on. This is
   invoked in the next code block as ``sdp.local_rank()``.
   ``set_memory_growth`` is not directly related to SMD, but must be set
   for distributed training with TensorFlow.

   .. code:: python

      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      if gpus:
          tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], 'GPU')


-  Scale the learning rate by the number of workers.
   ``sdp.tensorflow.size()`` API provides you number of workers in the
   cluster. This is invoked in the next code block as ``sdp.size()``.

   .. code:: python

      learning_rate = learning_rate * sdp.size()


-  Use the library’s ``DistributedGradientTape`` to optimize AllReduce
   operations during training. This wraps ``tf.GradientTape``.

   .. code:: python

      with tf.GradientTape() as tape:
            output = model(input)
            loss_value = loss(label, output)

      # Wrap tf.GradientTape with the library's DistributedGradientTape
      tape = sdp.DistributedGradientTape(tape)


-  Broadcast initial model variables from the leader node (rank 0) to
   all the worker nodes (ranks 1 through n). This is needed to ensure a
   consistent initialization across all the worker ranks. For this, you
   use ``sdp.tensorflow.broadcast_variables`` API after the
   model and optimizer variables are initialized. This is invoked in the
   next code block as ``sdp.broadcast_variables()``.

   .. code:: python

      sdp.broadcast_variables(model.variables, root_rank=0)
      sdp.broadcast_variables(opt.variables(), root_rank=0)


-  Finally, modify your script to save checkpoints only on the leader
   node. The leader node will have a synchronized model. This also
   avoids worker nodes overwriting the checkpoints and possibly
   corrupting the checkpoints.

   .. code:: python

      if sdp.rank() == 0:
          checkpoint.save(checkpoint_dir)


All put together, the following is an example TensorFlow2 training
script you will have for distributed training with the library.

.. code:: python

   import tensorflow as tf

   # Import the library's TF API
   import smdistributed.dataparallel.tensorflow as sdp

   # Initialize the library
   sdp.init()

   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   if gpus:
       # Pin GPUs to a single process
       tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], 'GPU')

   # Prepare Dataset
   dataset = tf.data.Dataset.from_tensor_slices(...)

   # Define Model
   mnist_model = tf.keras.Sequential(...)
   loss = tf.losses.SparseCategoricalCrossentropy()

   # Scale Learning Rate
   # LR for 8 node run : 0.000125
   # LR for single node run : 0.001
   opt = tf.optimizers.Adam(0.000125 * sdp.size())

   @tf.function
   def training_step(images, labels, first_batch):
       with tf.GradientTape() as tape:
           probs = mnist_model(images, training=True)
           loss_value = loss(labels, probs)

       # Wrap tf.GradientTape with the library's DistributedGradientTape
       tape = sdp.DistributedGradientTape(tape)

       grads = tape.gradient(loss_value, mnist_model.trainable_variables)
       opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

       if first_batch:
          # Broadcast model and optimizer variables
          sdp.broadcast_variables(mnist_model.variables, root_rank=0)
          sdp.broadcast_variables(opt.variables(), root_rank=0)

       return loss_value

   ...

   # Save checkpoints only from master node.
   if sdp.rank() == 0:
       checkpoint.save(checkpoint_dir)


.. _tensorflow-sdp-api:

TensorFlow API
==============

.. function:: smdistributed.dataparallel.tensorflow.init()

   Initialize ``smdistributed.dataparallel``. Must be called at the
   beginning of the training script.


   **Inputs:**

   -  ``None``

   **Returns:**

   -  ``None``


   .. rubric:: Notes

   ``init()`` needs to be called only once. It will throw an error if
   called more than once:

   ``init() called more than once. smdistributed.dataparallel is already initialized.``


.. function:: smdistributed.dataparallel.tensorflow.size()

   The total number of GPUs across all the nodes in the cluster. For
   example, in a 8 node cluster with 8 GPUs each, ``size`` will be equal
   to 64.


   **Inputs:**

   -  ``None``

   **Returns:**

   -  An integer scalar containing the total number of GPUs, across all
      nodes in the cluster.


.. function:: smdistributed.dataparallel.tensorflow.local_size()

   The total number of GPUs on a node. For example, on a node with 8
   GPUs, ``local_size`` will be equal to 8.

   **Inputs:**

   -  ``None``

   **Returns:**

   -  An integer scalar containing the total number of GPUs on itself.


.. function:: smdistributed.dataparallel.tensorflow.rank()

   The rank of the node in the cluster. The rank ranges from 0 to number of
   nodes - 1. This is similar to MPI's World Rank.

   **Inputs:**

   -  ``None``

   **Returns:**

   -  An integer scalar containing the rank of the node.


.. function:: smdistributed.dataparallel.tensorflow.local_rank()

   Local rank refers to the relative rank of the
   GPUs’ ``smdistributed.dataparallel`` processes within the node. For
   example, if a node contains 8 GPUs, it has
   8 ``smdistributed.dataparallel`` processes, then each process will
   get a local rank ranging from 0 to 7.

   **Inputs:**

   -  ``None``

   **Returns:**

   -  An integer scalar containing the rank of the GPU and
      its ``smdistributed.dataparallel`` process.


.. function:: smdistributed.dataparallel.tensorflow.allreduce(tensor, param_index, num_params, compression=Compression.none, op=ReduceOp.AVERAGE)

   Performs an ``allreduce`` operation on a tensor (``tf.Tensor``).

   The ``smdistributed.dataparallel`` package's AllReduce API for TensorFlow to allreduce
   gradient tensors. By default, ``smdistributed.dataparallel`` allreduce averages the
   gradient tensors across participating workers.

   .. note::

    :class:`smdistributed.dataparallel.tensorflow.allreduce()` should
    only be used to allreduce gradient tensors.
    For other (non-gradient) tensors, you must use
    :class:`smdistributed.dataparallel.tensorflow.oob_allreduce()`.
    If you use :class:`smdistributed.dataparallel.tensorflow.allreduce()`
    for non-gradient tensors,
    the distributed training job might stall or stop.

   **Inputs:**

   - ``tensor (tf.Tensor)(required)``: The tensor to be allreduced. The shape of the input must be identical across all ranks.
   - ``param_index (int)(required):`` 0 if you are reducing a single tensor. Index of the tensor if you are reducing a list of tensors.
   - ``num_params (int)(required):`` len(tensor).
   - ``compression (smdistributed.dataparallel.tensorflow.Compression)(optional)``: Compression algorithm used to reduce the amount of data sent and received by each worker node. Defaults to not using compression.

      *   Supported compression types - ``none``, ``fp16``

   - ``op (optional)(smdistributed.dataparallel.tensorflow.ReduceOp)``: The reduction operation to combine tensors across different ranks. Defaults to ``Average`` if None is given.

      *  Supported ops: ``SUM``, ``MIN``, ``MAX``, ``AVERAGE``

   **Returns:**

   -  A tensor of the same shape and type as input ``tensor``, all-reduced across all the processes.


.. function:: smdistributed.dataparallel.tensorflow.broadcast_global_variables(root_rank)

   Broadcasts all global variables from root rank to all other processes.

   **Inputs:**

   -  ``root_rank (int)(required):`` Rank of the process from which global
      variables will be broadcasted to all other processes.

   **Returns:**

   -  ``None``


.. function:: smdistributed.dataparallel.tensorflow.broadcast_variables(variables, root_rank)

   Applicable for TensorFlow 2.x only.
   ​
   Broadcasts variables from root rank to all other processes.
   ​
   With TensorFlow 2.x, ``broadcast_variables`` is used to
   broadcast ``model.variables`` and ``optimizer.variables`` post
   initialization from the leader node to all the worker nodes. This
   ensures a consistent initialization across all the worker ranks.

   **Inputs:**

   -  ``variables (tf.Variable)(required):`` Variables to be broadcasted.
   -  ``root_rank (int)(required):`` Rank of the process from which
      variables will be broadcasted to all other processes.

   **Returns:**

   -  ``None``


.. function:: smdistributed.dataparallel.tensorflow.oob_allreduce(tensor, compression=Compression.none, op=ReduceOp.AVERAGE)

   Out-of-band (oob) AllReduce is simplified AllReduce function for use-cases
   such as calculating total loss across all the GPUs in the training.
   ``oob_allreduce`` average the tensors, as reduction operation, across the
   worker nodes.

   **Inputs:**

   - ``tensor (tf.Tensor)(required)``: The tensor to be all-reduced. The shape of the input must be identical across all worker nodes.
   - ``compression`` (optional): Compression algorithm used to reduce the amount of data sent and received by each worker node. Defaults to not using compression.

      *   Supported compression types - ``none``, ``fp16``

   - ``op (smdistributed.dataparallel.tensorflow.ReduceOp)(optional)``: The reduction operation to combine tensors across different worker nodes. Defaults to ``Average`` if None is given.

      *  Supported ops: ``AVERAGE``

   **Returns:**

   -  ``None``

   .. note::

      In most cases, the :class:`smdistributed.dataparallel.tensorflow.oob_allreduce()`
      function is ~2x slower
      than :class:`smdistributed.dataparallel.tensorflow.allreduce()`. It is not
      recommended to use the :class:`smdistributed.dataparallel.tensorflow.oob_allreduce()`
      function for performing gradient
      reduction during the training process.
      ``smdistributed.dataparallel.tensorflow.oob_allreduce`` internally
      uses NCCL AllReduce with ``ncclSum`` as the reduction operation.

   .. note::

      :class:`smdistributed.dataparallel.tensorflow.oob_allreduce()` should
      only be used to allreduce non-gradient tensors.
      If you use :class:`smdistributed.dataparallel.tensorflow.allreduce()`
      for non-gradient tensors,
      the distributed training job might stall or stop.
      To allreduce gradients, use :class:`smdistributed.dataparallel.tensorflow.allreduce()`.


.. function:: smdistributed.dataparallel.tensorflow.overlap(tensor)

   This function is applicable only for models compiled with XLA. Use this
   function to enable ``smdistributed.dataparallel`` to efficiently
   overlap backward pass with the all reduce operation.

   Example usage:

   .. code:: python

      layer = tf.nn.dropout(...) # Or any other layer
      layer = smdistributed.dataparallel.tensorflow.overlap(layer)

   The overlap operation is inserted into the TF graph as a node. It
   behaves as an identity operation, and helps in achieving the
   communication overlap with backward pass operation.

   **Inputs:**

   -  ``tensor (tf.Tensor)(required):`` The tensor to be all-reduced.

   **Returns:**

   -  ``None``

   .. rubric:: Notes

   This operation helps in speeding up distributed training, as
   the AllReduce operation does not have to wait for all the gradients to
   be ready. Backward propagation proceeds sequentially from the output
   layer of the network to the input layer. When the gradient computation
   for a layer finishes, ``smdistributed.dataparallel`` adds them to a
   fusion buffer. As soon as the size of the fusion buffer reaches a
   predefined threshold (25 Mb), ``smdistributed.dataparallel`` starts
   the AllReduce operation.


.. function:: smdistributed.dataparallel.tensorflow.broadcast(tensor, root_rank)

   Broadcasts the input tensor on root rank to the same input tensor on all
   other ``smdistributed.dataparallel`` processes.
   ​
   The broadcast will not start until all processes are ready to send and
   receive the tensor.

   **Inputs:**

   -  ``tensor (tf.Tensor)(required):`` The tensor to be broadcasted.
   -  ``root_rank (int)(required):`` Rank of the process from which
      tensor will be broadcasted to all other processes.

   **Returns:**

   -  A tensor of the same shape and type as tensor, with the value
      broadcasted from root rank.


.. function:: smdistributed.dataparallel.tensorflow.shutdown()

   Shuts down ``smdistributed.dataparallel``. Optional to call at the end
   of the training script.

   **Inputs:**

   -  ``None``

   **Returns:**

   -  ``None``


.. function:: smdistributed.dataparallel.tensorflow.DistributedOptimizer

   Applicable if you use the ``tf.estimator`` API in TensorFlow 2.x (2.3.1).
   ​
   Construct a new ``DistributedOptimizer`` , which uses TensorFlow
   optimizer under the hood for computing single-process gradient values
   and applying gradient updates after the gradient values have been
   combined across all ``smdistributed.dataparallel`` workers.
   ​
   Example usage:

   .. code:: python

      opt = ... # existing optimizer from tf.train package or your custom optimizer
      opt = smdistributed.dataparallel.tensorflow.DistributedOptimizer(opt)


   - ``optimizer (tf.train.Optimizer)(required):`` TF Optimizer to use for computing gradients and applying updates.

   - ``name (str)(optional):`` Name prefix for the operations created when applying gradients. Defaults to ``smdistributed.dataparallel`` followed by provided optimizer type.

   - ``use_locking (bool)(optional):`` Whether to use locking when updating variables. Defaults to ``False``.

   - ``device_dense:`` Not supported. Raises not supported error.

   - ``device_sparse:`` Not supported. Raises not supported error.

   - ``compression (smdistributed.dataparallel.tensorflow.Compression)(optional)``: Compression algorithm used to reduce the amount of data sent and received by each worker node. Defaults to not using compression.

      *   Supported compression types - ``none``, ``fp16``

   - ``sparse_as_dense:`` Treats sparse gradient tensor as dense tensor. Defaults to ``False``.

   - ``op (smdistributed.dataparallel.tensorflow.ReduceOp)(optional)``: The reduction operation to combine tensors across different ranks. Defaults to ``Average`` if None is given.

      *  Supported ops: ``AVERAGE``

   - ``bucket_cap_mb (int)(optional):`` Size of ``smdistributed.dataparallel`` fusion buffer size. Defaults to 25MB that works optimally for most case. If you provide a value, expects the (value * 1024 * 1024) i.e., bytes to be multiple of 128.


.. function:: smdistributed.dataparallel.tensorflow.DistributedGradientTape

   Applicable to TensorFlow 2.x only.

   Construct a new ``DistributedGradientTape``, which uses
   TensorFlow’s ``GradientTape`` under the hood, using an AllReduce to
   combine gradient values before applying gradients to model weights.
   ​
   Example Usage:

   .. code:: python

      with tf.GradientTape() as tape:
            output = model(input)
            loss_value = loss(label, output)

      # Wrap in smdistributed.dataparallel's DistributedGradientTape
      tape = smdistributed.dataparallel.tensorflow.DistributedGradientTape(tape)


   - ``gradtape (tf.GradientTape)(required):`` GradientTape to use for computing gradients and applying updates.

   - ``device_dense:`` Not supported. Raises not supported error.

   - ``device_sparse:`` Not supported. Raises not supported error.

   - ``compression (smdistributed.dataparallel.tensorflow.Compression)(optional)``: Compression algorithm used to reduce the amount of data sent and received by each worker node. Defaults to not using compression.

      *   Supported compression types - ``none``, ``fp16``

   - ``sparse_as_dense:`` Treats sparse gradient tensor as dense tensor. Defaults to ``False``.

   - ``op (smdistributed.dataparallel.tensorflow.ReduceOp)(optional)``: The reduction operation to combine tensors across different ranks. Defaults to ``Average`` if None is given.

      *  Supported ops: ``AVERAGE``


.. function:: smdistributed.dataparallel.tensorflow.BroadcastGlobalVariablesHook

   Applicable if you use the ``tf.estimator`` API in TensorFlow 2.x (2.3.1).


   ``SessionRunHook`` that will broadcast all global variables from root
   rank to all other processes during initialization.
   ​
   This is necessary to ensure consistent initialization of all workers
   when training is started with random weights or restored from a
   checkpoint.
   ​
   Example Usage:

   .. code:: python

      hooks = [smdistributed.dataparallel.tensorflow.BroadcastGlobalVariablesHook(root_rank=0)]
      ...
      with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                             hooks=hooks,
                                             config=config) as mon_sess:
           ...


   -  ``root_rank (int)(required):`` Rank of the process from which global
      variables will be broadcasted to all other processes.


.. function:: smdistributed.dataparallel.tensorflow.Compression

   Optional Gradient Compression algorithm that can be used in AllReduce
   operation.

   -  ``none``: alias for ``NoneCompression``. Do not compression gradient
      tensors.
   -  ``fp16``: alias for ``FP16Compression``. Compress the floating point
      gradient tensors to 16-bit (FP16)


.. function:: smdistributed.dataparallel.tensorflow.ReduceOp

   Supported reduction operations in ``smdistributed.dataparallel``.

   -  ``AVERAGE``
   -  ``SUM``
   -  ``MIN``
   -  ``MAX``
