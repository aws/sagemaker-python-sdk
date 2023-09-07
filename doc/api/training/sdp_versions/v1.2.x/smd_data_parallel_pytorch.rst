##############################################################
PyTorch Guide to SageMaker's distributed data parallel library
##############################################################

Use this guide to learn about the SageMaker distributed
data parallel library API for PyTorch.

.. contents:: Topics
  :depth: 3
  :local:

.. _pytorch-sdp-modify:
   :noindex:

Modify a PyTorch training script to use SageMaker data parallel
======================================================================

The following steps show you how to convert a PyTorch training script to
utilize SageMaker's distributed data parallel library.

The distributed data parallel library APIs are designed to be close to PyTorch Distributed Data
Parallel (DDP) APIs.
See `SageMaker distributed data parallel PyTorch examples <https://sagemaker-examples.readthedocs.io/en/latest/training/distributed_training/index.html#pytorch-distributed>`__ for additional details on how to implement the data parallel library
API offered for PyTorch.


-  First import the distributed data parallel library’s PyTorch client and initialize it. You also import
   the distributed data parallel library module for distributed training.

   .. code:: python

      import smdistributed.dataparallel.torch.distributed as dist

      from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP

      dist.init_process_group()


-  Pin each GPU to a single distributed data parallel library process with ``local_rank`` - this
   refers to the relative rank of the process within a given node.
   ``smdistributed.dataparallel.torch.get_local_rank()`` API provides
   you the local rank of the device. The leader node will be rank 0, and
   the worker nodes will be rank 1, 2, 3, and so on. This is invoked in
   the next code block as ``dist.get_local_rank()``.

   .. code:: python

      torch.cuda.set_device(dist.get_local_rank())


-  Then wrap the PyTorch model with the distributed data parallel library’s DDP.

   .. code:: python

      model = ...
      # Wrap model with SageMaker's DistributedDataParallel
      model = DDP(model)


-  Modify the ``torch.utils.data.distributed.DistributedSampler`` to
   include the cluster’s information. Set ``num_replicas`` to the
   total number of GPUs participating in training across all the nodes
   in the cluster. This is called ``world_size``. You can get
   ``world_size`` with
   ``smdistributed.dataparallel.torch.get_world_size()`` API. This is
   invoked in the following code as ``dist.get_world_size()``. Also
   supply the node rank using
   ``smdistributed.dataparallel.torch.get_rank()``. This is invoked as
   ``dist.get_rank()``.

   .. code:: python

      train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())


-  Finally, modify your script to save checkpoints only on the leader
   node. The leader node will have a synchronized model. This also
   avoids worker nodes overwriting the checkpoints and possibly
   corrupting the checkpoints.

.. code:: python

   if dist.get_rank() == 0:
      torch.save(...)


All put together, the following is an example PyTorch training script
you will have for distributed training with the distributed data parallel library:

.. code:: python

   # Import distributed data parallel library PyTorch API
   import smdistributed.dataparallel.torch.distributed as dist

   # Import distributed data parallel library PyTorch DDP
   from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP

   # Initialize distributed data parallel library
   dist.init_process_group()

   class Net(nn.Module):
       ...
       # Define model

   def train(...):
       ...
       # Model training

   def test(...):
       ...
       # Model evaluation

   def main():

       # Scale batch size by world size
       batch_size //= dist.get_world_size()
       batch_size = max(batch_size, 1)

       # Prepare dataset
       train_dataset = torchvision.datasets.MNIST(...)

       # Set num_replicas and rank in DistributedSampler
       train_sampler = torch.utils.data.distributed.DistributedSampler(
               train_dataset,
               num_replicas=dist.get_world_size(),
               rank=dist.get_rank())

       train_loader = torch.utils.data.DataLoader(..)

       # Wrap the PyTorch model with distributed data parallel library’s DDP
       model = DDP(Net().to(device))

       # Pin each GPU to a single distributed data parallel library process.
       torch.cuda.set_device(local_rank)
       model.cuda(local_rank)

       # Train
       optimizer = optim.Adadelta(...)
       scheduler = StepLR(...)
       for epoch in range(1, args.epochs + 1):
           train(...)
           if rank == 0:
               test(...)
           scheduler.step()

       # Save model on master node.
       if dist.get_rank() == 0:
           torch.save(...)

   if __name__ == '__main__':
       main()


.. _pytorch-sdp-api:
   :noindex:

PyTorch API
===========

.. class:: smdistributed.dataparallel.torch.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, broadcast_buffers=True, process_group=None, bucket_cap_mb=None)
   :noindex:

   ``smdistributed.dataparallel``'s implementation of distributed data
   parallelism for PyTorch. In most cases, wrapping your PyTorch Module
   with ``smdistributed.dataparallel``'s ``DistributedDataParallel`` (DDP) is
   all you need to do to use ``smdistributed.dataparallel``.

   Creation of this DDP class requires ``smdistributed.dataparallel``
   already initialized
   with ``smdistributed.dataparallel.torch.distributed.init_process_group()``.

   This container parallelizes the application of the given module by
   splitting the input across the specified devices by chunking in the
   batch dimension. The module is replicated on each machine and each
   device, and each such replica handles a portion of the input. During the
   backwards pass, gradients from each node are averaged.

   The batch size should be larger than the number of GPUs used locally.
   ​
   Example usage
   of ``smdistributed.dataparallel.torch.parallel.DistributedDataParallel``:

   .. code:: python

      import torch
      import smdistributed.dataparallel.torch.distributed as dist
      from smdistributed.dataparallel.torch.parallel import DistributedDataParallel as DDP

      dist.init_process_group()

      # Pin GPU to be used to process local rank (one GPU per process)
      torch.cuda.set_device(dist.get_local_rank())

      # Build model and optimizer
      model = ...
      optimizer = torch.optim.SGD(model.parameters(),
                                  lr=1e-3 * dist.get_world_size())
      # Wrap model with smdistributed.dataparallel's DistributedDataParallel
      model = DDP(model)

   **Parameters:**

   -  ``module (torch.nn.Module)(required):`` PyTorch NN Module to be
      parallelized
   -  ``device_ids (list[int])(optional):`` CUDA devices. This should only
      be provided when the input module resides on a single CUDA device.
      For single-device modules,
      the ``ith module replica is placed on device_ids[i]``. For
      multi-device modules and CPU modules, device_ids must be None or an
      empty list, and input data for the forward pass must be placed on the
      correct device. Defaults to ``None``.
   -  ``output_device (int)(optional):`` Device location of output for
      single-device CUDA modules. For multi-device modules and CPU modules,
      it must be None, and the module itself dictates the output location.
      (default: device_ids[0] for single-device modules).  Defaults
      to ``None``.
   -  ``broadcast_buffers (bool)(optional):`` Flag that enables syncing
      (broadcasting) buffers of the module at beginning of the forward
      function. ``smdistributed.dataparallel`` does not support broadcast
      buffer yet. Please set this to ``False``.
   -  ``process_group(smdistributed.dataparallel.torch.distributed.group)(optional):`` Process
      group is not supported in ``smdistributed.dataparallel``. This
      parameter exists for API parity with torch.distributed only. Only
      supported value is
      ``smdistributed.dataparallel.torch.distributed.group.WORLD.`` Defaults
      to ``None.``
   -  ``bucket_cap_mb (int)(optional):`` DistributedDataParallel will
      bucket parameters into multiple buckets so that gradient reduction of
      each bucket can potentially overlap with backward
      computation. ``bucket_cap_mb`` controls the bucket size in
      MegaBytes (MB) (default: 25).

   .. note::

      This module assumes all parameters are registered in the model by the
      time it is created. No parameters should be added nor removed later.

   .. note::

      This module assumes all parameters are registered in the model of
      each distributed processes are in the same order. The module itself
      will conduct gradient all-reduction following the reverse order of
      the registered parameters of the model. In other words, it is users’
      responsibility to ensure that each distributed process has the exact
      same model and thus the exact same parameter registration order.

   .. note::

      You should never change the set of your model’s parameters after
      wrapping up your model with DistributedDataParallel. In other words,
      when wrapping up your model with DistributedDataParallel, the
      constructor of DistributedDataParallel will register the additional
      gradient reduction functions on all the parameters of the model
      itself at the time of construction. If you change the model’s
      parameters after the DistributedDataParallel construction, this is
      not supported and unexpected behaviors can happen, since some
      parameters’ gradient reduction functions might not get called.

   .. method:: no_sync()
      :noindex:

      ``smdistributed.dataparallel`` supports the `PyTorch DDP no_sync() <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync>`_
      context manager. It enables gradient accumulation by skipping AllReduce
      during training iterations inside the context.

      .. note::

        The ``no_sync()`` context manager is available from smdistributed-dataparallel v1.2.2.
        To find the release note, see :ref:`sdp_release_note`.

      **Example:**

      .. code:: python

        # Gradients are accumulated while inside no_sync context
        with model.no_sync():
            ...
            loss.backward()

        # First iteration upon exiting context
        # Incoming gradients are added to the accumulated gradients and then synchronized via AllReduce
        ...
        loss.backward()

        # Update weights and reset gradients to zero after accumulation is finished
        optimizer.step()
        optimizer.zero_grad()


.. function:: smdistributed.dataparallel.torch.distributed.is_available()
   :noindex:

   Check if script started as a distributed job. For local runs user can
   check that is_available returns False and run the training script
   without calls to ``smdistributed.dataparallel``.

   **Inputs:**

   -  ``None``

   **Returns:**

   -  ``True`` if started as a distributed job, ``False`` otherwise


.. function:: smdistributed.dataparallel.torch.distributed.init_process_group(*args, **kwargs)
   :noindex:

   Initialize ``smdistributed.dataparallel``. Must be called at the
   beginning of the training script, before calling any other methods.
   ​
   Process group is not supported in ``smdistributed.dataparallel``. This
   parameter exists for API parity with ``torch.distributed`` only. Only
   supported value is
   ``smdistributed.dataparallel.torch.distributed.group.WORLD.``
   ​
   After this
   call, ``smdistributed.dataparallel.torch.distributed.is_initialized()`` will
   return ``True``.
   ​

   **Inputs:**

   -  ``None``

   **Returns:**

   -  ``None``


.. function:: smdistributed.dataparallel.torch.distributed.is_initialized()
   :noindex:

   Checks if the default process group has been initialized.

   **Inputs:**

   -  ``None``

   **Returns:**

   -  ``True`` if initialized, else ``False``.


.. function:: smdistributed.dataparallel.torch.distributed.get_world_size(group=smdistributed.dataparallel.torch.distributed.group.WORLD)
   :noindex:

   The total number of GPUs across all the nodes in the cluster. For
   example, in a 8 node cluster with 8 GPU each, size will be equal to 64.

   **Inputs:**

   -  ``group (smdistributed.dataparallel.torch.distributed.group) (optional):`` Process
      group is not supported in ``smdistributed.dataparallel``. This
      parameter exists for API parity with torch.distributed only. Only
      supported value is
      ``smdistributed.dataparallel.torch.distributed.group.WORLD.``

   **Returns:**

   -  An integer scalar containing the total number of GPUs in the training
      job, across all nodes in the cluster.


.. function:: smdistributed.dataparallel.torch.distributed.get_rank(group=smdistributed.dataparallel.torch.distributed.group.WORLD)
   :noindex:

   The rank of the node in the cluster. The rank ranges from 0 to number of
   nodes - 1. This is similar to MPI's World Rank.


   **Inputs:**

   -  ``group (smdistributed.dataparallel.torch.distributed.group) (optional):`` Process
      group is not supported in ``smdistributed.dataparallel``. This
      parameter exists for API parity with torch.distributed only. Only
      supported value is
      ``smdistributed.dataparallel.torch.distributed.group.WORLD.``

   **Returns:**

   -  An integer scalar containing the rank of the worker node.


.. function:: smdistributed.dataparallel.torch.distributed.get_local_rank()
   :noindex:

   Local rank refers to the relative rank of
   the ``smdistributed.dataparallel`` process within the node the current
   process is running on. For example, if a node contains 8 GPUs, it has
   8 ``smdistributed.dataparallel`` processes. Each process has
   a ``local_rank`` ranging from 0 to 7.

   **Inputs:**

   -  ``None``

   **Returns:**

   -  An integer scalar containing the rank of the GPU and
      its ``smdistributed.dataparallel`` process.


.. function:: smdistributed.dataparallel.torch.distributed.all_reduce(tensor, op=smdistributed.dataparallel.torch.distributed.ReduceOp.SUM, group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)
   :noindex:

   Performs an all-reduce operation on a tensor (torch.tensor) across
   all ``smdistributed.dataparallel`` workers

   ``smdistributed.dataparallel`` AllReduce API can be used for all
   reducing gradient tensors or any other tensors.  By
   default, ``smdistributed.dataparallel`` AllReduce reduces the tensor
   data across all ``smdistributed.dataparallel`` workers in such a way
   that all get the final result.

   After the call ``tensor`` is going to be bitwise identical in all
   processes.

   **Inputs:**

   - ``tensor (torch.tensor) (required):`` Input and output of the collective. The function operates in-place.

   - ``op (smdistributed.dataparallel.torch.distributed.ReduceOp) (optional)``: The reduction operation to combine tensors across different ranks.  Defaults to ``SUM`` if None is given.

      * Supported ops: ``AVERAGE``, ``SUM``, ``MIN``, ``MAX``

   - ``group (smdistributed.dataparallel.torch.distributed.group) (optional):`` Process group is not supported in ``smdistributed.dataparallel``. This parameter exists for API parity with torch.distributed only.

      * Only supported value is ``smdistributed.dataparallel.torch.distributed.group.WORLD.``

   - ``async_op (bool) (optional):`` Whether this op should be an async op. Defaults to ``False``.

   **Returns:**

   -  Async op work handle, if async_op is set to True. ``None``,
      otherwise.

   .. rubric:: Notes

   ``smdistributed.dataparallel.torch.distributed.allreduce``, in most
   cases, is ~2X slower than all-reducing
   with ``smdistributed.dataparallel.torch.parallel.distributed.DistributedDataParallel`` and
   hence, it is not recommended to be used for performing gradient
   reduction during the training
   process. ``smdistributed.dataparallel.torch.distributed.allreduce`` internally
   uses NCCL AllReduce with ``ncclSum`` as the reduction operation.


.. function:: smdistributed.dataparallel.torch.distributed.broadcast(tensor, src=0, group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)
   :noindex:

   Broadcasts the tensor (torch.tensor) to the whole group.

   ``tensor`` must have the same number of elements as GPUs in the
   cluster.

   **Inputs:**

   -  ``tensor (torch.tensor)(required)``

   -  ``src (int)(optional)``

   -  ``group (smdistributed.dataparallel.torch.distributed.group)(optional):`` Process group is not supported in ``smdistributed.dataparallel``. This parameter exists for API parity with ``torch.distributed`` only.

      * Only supported value is ``smdistributed.dataparallel.torch.distributed.group.WORLD.``

   -  ``async_op (bool)(optional):`` Whether this op should be an async op. Defaults to ``False``.

   **Returns:**

   -  Async op work handle, if async_op is set to True. ``None``, otherwise.


.. function:: smdistributed.dataparallel.torch.distributed.all_gather(tensor_list, tensor, group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)
   :noindex:

   Gathers tensors from the whole group in a list.


   **Inputs:**

   -  ``tensor_list (list[torch.tensor])(required):`` Output list. It
      should contain correctly-sized tensors to be used for output of the
      collective.
   -  ``tensor (torch.tensor)(required):`` Tensor to be broadcast from
      current process.
   -  ``group (smdistributed.dataparallel.torch.distributed.group)(optional):`` Process
      group is not supported in ``smdistributed.dataparallel``. This
      parameter exists for API parity with torch.distributed only. Only
      supported value is
      ``smdistributed.dataparallel.torch.distributed.group.WORLD.``
   -  ``async_op (bool)(optional):`` Whether this op should be an async op.
      Defaults to ``False``.

   **Returns:**

   -  Async op work handle, if async_op is set to True. ``None``,
      otherwise.


.. function:: smdistributed.dataparallel.torch.distributed.all_to_all_single(output_t, input_t, output_split_sizes=None, input_split_sizes=None, group=group.WORLD, async_op=False)
   :noindex:

   Each process scatters input tensor to all processes in a group and return gathered tensor in output.

   **Inputs:**

   -  output_t
   -  input_t
   -  output_split_sizes
   -  input_split_sizes
   -  ``group (smdistributed.dataparallel.torch.distributed.group)(optional):`` Process
      group is not supported in ``smdistributed.dataparallel``. This
      parameter exists for API parity with torch.distributed only. Only
      supported value is
      ``smdistributed.dataparallel.torch.distributed.group.WORLD.``
   -  ``async_op (bool)(optional):`` Whether this op should be an async op.
      Defaults to ``False``.

   **Returns:**

   -  Async op work handle, if async_op is set to True. ``None``,
      otherwise.


.. function:: smdistributed.dataparallel.torch.distributed.barrier(group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)
   :noindex:

   Synchronizes all ``smdistributed.dataparallel`` processes.

   **Inputs:**

   - tensor (torch.tensor)(required): Data to be sent if src is the rank of current process, and tensor to be used to save received data otherwise.

   - src (int)(optional): Source rank.

   -  ``group (smdistributed.dataparallel.torch.distributed.group)(optional):`` Process
      group is not supported in ``smdistributed.dataparallel``. This
      parameter exists for API parity with torch.distributed only.

         * Only supported value is ``smdistributed.dataparallel.torch.distributed.group.WORLD.``

   -  ``async_op (bool)(optional):`` Whether this op should be an async op.
      Defaults to ``False``.

   **Returns:**

   -  Async op work handle, if async_op is set to True. ``None``,
      otherwise.


.. class:: smdistributed.dataparallel.torch.distributed.ReduceOp
   :noindex:

   An enum-like class for supported reduction operations
   in ``smdistributed.dataparallel``.

   The values of this class can be accessed as attributes, for
   example, ``ReduceOp.SUM``. They are used in specifying strategies for
   reduction collectives such as
    ``smdistributed.dataparallel.torch.distributed.all_reduce(...)``.

   -  ``AVERAGE``
   -  ``SUM``
   -  ``MIN``
   -  ``MAX``
