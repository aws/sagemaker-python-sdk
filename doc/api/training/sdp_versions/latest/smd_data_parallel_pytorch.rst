#################
Guide for PyTorch
#################

Use this guide to learn how to use the SageMaker distributed
data parallel library API for PyTorch.

.. contents:: Topics
  :depth: 3
  :local:

.. _pytorch-sdp-modify:

Modify a PyTorch training script to use the SageMaker data parallel library
===========================================================================

The following steps show you how to convert a PyTorch training script to
utilize SageMaker's distributed data parallel library.

The distributed data parallel library works as a backend of the PyTorch distributed package.
See `SageMaker distributed data parallel PyTorch examples <https://sagemaker-examples.readthedocs.io/en/latest/training/distributed_training/index.html#pytorch-distributed>`__ 
for additional details on how to use the library.

1.  Import the SageMaker distributed data parallel library’s PyTorch client.

    .. code:: python

      import smdistributed.dataparallel.torch.torch_smddp

2.  Import the PyTorch distributed modules.

    .. code:: python

      import torch
      import torch.distributed as dist
      from torch.nn.parallel import DistributedDataParallel as DDP

3.  Set the backend of ``torch.distributed`` as ``smddp``.

    .. code:: python

      dist.init_process_group(backend='smddp')

4.  After parsing arguments and defining a batch size parameter
    (for example, ``batch_size=args.batch_size``), add a two-line of code to
    resize the batch size per worker (GPU). PyTorch's DataLoader operation
    does not automatically handle the batch resizing for distributed training.

    .. code:: python

      batch_size //= dist.get_world_size()
      batch_size = max(batch_size, 1)

5.  Pin each GPU to a single SageMaker data parallel library process with
    ``local_rank``. This refers to the relative rank of the process within a given node.

    You can retrieve the rank of the process from the ``LOCAL_RANK`` environment variable.

    .. code:: python

      import os
      local_rank = os.environ["LOCAL_RANK"]
      torch.cuda.set_device(local_rank)

6.  After defining a model, wrap it with the PyTorch DDP.

    .. code:: python

      model = ...

      # Wrap the model with the PyTorch DistributedDataParallel API
      model = DDP(model)

7.  When you call the ``torch.utils.data.distributed.DistributedSampler`` API,
    specify the total number of processes (GPUs) participating in training across
    all the nodes in the cluster. This is called ``world_size``, and you can retrieve
    the number from the ``torch.distributed.get_world_size()`` API. Also, specify
    the rank of each process among all processes using the ``torch.distributed.get_rank()`` API.

    .. code:: python

      train_sampler = DistributedSampler(
          train_dataset,
          num_replicas = dist.get_world_size(),
          rank = dist.get_rank()
      )

8.  Modify your script to save checkpoints only on the leader process (rank 0).
    The leader process has a synchronized model. This also avoids other processes
    overwriting the checkpoints and possibly corrupting the checkpoints.

The following example code shows the structure of a PyTorch training script with DDP and smddp as the backend.

.. code:: python

  import os
  import torch

  # SageMaker data parallel: Import the library PyTorch API
  import smdistributed.dataparallel.torch.torch_smddp

  # SageMaker data parallel: Import PyTorch's distributed API
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP

  # SageMaker data parallel: Initialize the process group
  dist.init_process_group(backend='smddp')

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

      # SageMaker data parallel: Scale batch size by world size
      batch_size //= dist.get_world_size()
      batch_size = max(batch_size, 1)

      # Prepare dataset
      train_dataset = torchvision.datasets.MNIST(...)

      # SageMaker data parallel: Set num_replicas and rank in DistributedSampler
      train_sampler = torch.utils.data.distributed.DistributedSampler(
              train_dataset,
              num_replicas=dist.get_world_size(),
              rank=dist.get_rank())

      train_loader = torch.utils.data.DataLoader(..)

      # SageMaker data parallel: Wrap the PyTorch model with the library's DDP
      model = DDP(Net().to(device))

      # SageMaker data parallel: Pin each GPU to a single library process.
      local_rank = os.environ["LOCAL_RANK"]
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

      # SageMaker data parallel: Save model on the leader node (rank 0).
      if dist.get_rank() == 0:
          torch.save(...)

  if __name__ == '__main__':
      main()


.. _pytorch-sdp-api:

PyTorch API
===========

Since v1.4.0, the SageMaker distributed data parallel library
supports the PyTorch distributed package as a backend option.
To use the library with PyTorch in SageMaker,
you simply specify the backend of
the PyTorch distributed package as ``'smddp'`` when initializing process group.

.. code:: Python

  torch.distributed.init_process_group(backend='smddp')

You don't need to modify your script using
the ``smdistributed`` implementation of the PyTorch distributed modules
that are supported in the library v1.3.0 and before.

.. warning::

  The following APIs for ``smdistributed`` implementation of the PyTorch distributed modules
  are deprecated.


.. class:: smdistributed.dataparallel.torch.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, broadcast_buffers=True, process_group=None, bucket_cap_mb=None)

   .. deprecated:: 1.4.0

      Use the `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
      API instead.


.. function:: smdistributed.dataparallel.torch.distributed.is_available()

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead. For more information, see `Initialization <https://pytorch.org/docs/stable/distributed.html#initialization>`_
      in the *PyTorch documentation*.

.. function:: smdistributed.dataparallel.torch.distributed.init_process_group(*args, **kwargs)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead. For more information, see `Initialization <https://pytorch.org/docs/stable/distributed.html#initialization>`_
      in the *PyTorch documentation*.


.. function:: smdistributed.dataparallel.torch.distributed.is_initialized()

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead. For more information, see `Initialization <https://pytorch.org/docs/stable/distributed.html#initialization>`_
      in the *PyTorch documentation*.


.. function:: smdistributed.dataparallel.torch.distributed.get_world_size(group=smdistributed.dataparallel.torch.distributed.group.WORLD)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead. For more information, see `Post-Initialization <https://pytorch.org/docs/stable/distributed.html#post-initialization>`_
      in the *PyTorch documentation*.


.. function:: smdistributed.dataparallel.torch.distributed.get_rank(group=smdistributed.dataparallel.torch.distributed.group.WORLD)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead. For more information, see `Post-Initialization <https://pytorch.org/docs/stable/distributed.html#post-initialization>`_
      in the *PyTorch documentation*.


.. function:: smdistributed.dataparallel.torch.distributed.get_local_rank()

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead.


.. function:: smdistributed.dataparallel.torch.distributed.all_reduce(tensor, op=smdistributed.dataparallel.torch.distributed.ReduceOp.SUM, group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead.


.. function:: smdistributed.dataparallel.torch.distributed.broadcast(tensor, src=0, group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead.


.. function:: smdistributed.dataparallel.torch.distributed.all_gather(tensor_list, tensor, group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead.

.. function:: smdistributed.dataparallel.torch.distributed.all_to_all_single(output_t, input_t, output_split_sizes=None, input_split_sizes=None, group=group.WORLD, async_op=False)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead.


.. function:: smdistributed.dataparallel.torch.distributed.barrier(group=smdistributed.dataparallel.torch.distributed.group.WORLD, async_op=False)

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead.


.. class:: smdistributed.dataparallel.torch.distributed.ReduceOp

   .. deprecated:: 1.4.0
      Use the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package
      instead.
