TensorFlow API
==============

To use the TensorFlow-specific APIs for SageMaker distributed model parallism,
you need to add the following import statement at the top of your training script.

.. code:: python

   import smdistributed.modelparallel.tensorflow as smp

.. tip::

   Refer to
   `Modify a TensorFlow Training Script
   <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script-tf.html>`_
   to learn how to use the following APIs in your TensorFlow training script.

.. class:: smp.DistributedModel
   :noindex:

   A sub-class of the Keras \ ``Model`` class, which defines the model to
   be partitioned. Model definition is done by sub-classing
   ``smp.DistributedModel`` class, and implementing the ``call()`` method,
   in the same way as the Keras model sub-classing API. Any operation that
   is part of the \ ``smp.DistributedModel.call()`` method is subject to
   partitioning, meaning that every operation placed inside executes in
   exactly one of the devices (the operations outside run on all devices).


   Similar to the regular Keras API, the forward pass is done by directly
   calling the model object on the input tensors. For example:

   .. code:: python

      predictions = model(inputs)   # model is a smp.DistributedModel object

   However, ``model()`` calls can only be made inside a
   ``smp.step``-decorated function.

   The outputs from a ``smp.DistributedModel`` are available in all ranks,
   regardless of which rank computed the last operation.

   **Methods:**

   .. function:: save_model(save_path="/opt/ml/model")
      :noindex:

      **Inputs**
      - ``save_path`` (``string``): A path to save an unpartitioned model with latest training weights.

      Saves the entire,
      unpartitioned model with the latest trained weights to ``save_path`` in
      TensorFlow ``SavedModel`` format. Defaults to ``"/opt/ml/model"``, which
      SageMaker monitors to upload the model artifacts to Amazon S3.

.. function:: smp.partition(index)
   :noindex:

   **Inputs**

   -  ``index`` (``int``): The index of the partition.

   A context manager which places all operations defined inside into the
   partition whose ID is equal to ``index``. When
   ``smp.partition`` contexts are nested, the innermost context overrides
   the rest. The ``index`` argument must be smaller than the number of
   partitions.

   ``smp.partition`` is used in the manual partitioning API;
   if \ ``"auto_partition"`` parameter is set to ``True`` while launching
   training, then ``smp.partition`` contexts are ignored. Any operation
   that is not placed in any ``smp.partition`` context is placed in the
   ``default_partition``, as shown in the following example:

   .. code:: python

      # auto_partition: False
      # default_partition: 0
      smp.init()
      [...]
      x = tf.constant(1.2)                     # placed in partition 0
      with smp.partition(1):
          y = tf.add(x, tf.constant(2.3))      # placed in partition 1
          with smp.partition(3):
              z = tf.reduce_sum(y)             # placed in partition 3


.. function:: register_post_partition_hook(hook)
    :noindex:

    Registers a callable ``hook`` to
    be executed after the model is partitioned. This is useful in situations
    where an operation needs to be executed after the model partition during
    the first call to ``smp.step``, but before the actual execution of the
    first forward pass.

    .. code:: python

        @smp.register_post_partition_hook
        def test_eager():
            # All statements here will be executed right after partition but before the first forward pass
            tf.print("Entered hook through eager context")

.. class:: smp.CheckpointManager
  :noindex:


   A subclass of TensorFlow
   `CheckpointManager <https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager>`__,
   which is used to manage checkpoints. The usage is similar to TensorFlow
   ``CheckpointManager``.

   The following returns a ``CheckpointManager`` object.

   .. code:: python

      smp.CheckpointManager(checkpoint,
                            directory="/opt/ml/checkpoints",
                            max_to_keep=None,
                            checkpoint_name="ckpt")

   **Parameters**

   -  ``checkpoint``: A `tf.train.Checkpoint
      <https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint>`__ instance
      that represents a model checkpoint.

   -  ``directory``: (``str``) The path to a directory in which to write
      checkpoints. A file named "checkpoint" is also written to this
      directory (in a human-readable text format) which contains the state
      of the ``CheckpointManager``. Defaults to
      ``"/opt/ml/checkpoints"``, which is the directory that SageMaker
      monitors for uploading the checkpoints to Amazon S3.
   -  ``max_to_keep`` (``int``): The number of checkpoints to keep. If
      ``None``, all checkpoints are kept.
   -  ``checkpoint_name`` (``str``): Custom name for the checkpoint file.
      Defaults to ``"ckpt"``.


   **Methods:**

   .. function:: save( )
      :noindex:

      Saves a new checkpoint in the specified directory. Internally uses ``tf.train.CheckpointManager.save()``.

   .. function:: restore( )
      :noindex:

      Restores the latest checkpoint in the specified directory.
      Internally uses ``tf.train.CheckpointManager.restore()``.


   **Examples:**

   .. code:: python

      checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
      ckpt_manager = smp.CheckpointManager(checkpoint, max_to_keep=5)  # use /opt/ml/checkpoints

      for inputs in train_ds:
          loss = train_step(inputs)
          # [...]
          ckpt_manager.save()  # save a new checkpoint in /opt/ml/checkpoints

   .. code:: python

      for step, inputs in enumerate(train_ds):
          if step == 0:
              ckpt_manager.restore()
          loss = train_step(inputs)
