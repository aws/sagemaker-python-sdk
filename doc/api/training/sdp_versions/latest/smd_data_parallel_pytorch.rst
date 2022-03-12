#################
Guide for PyTorch
#################

Use this guide to learn how to use the SageMaker distributed
data parallel library API for PyTorch.

.. contents:: Topics
  :depth: 3
  :local:

.. _pytorch-sdp-modify:

Use the SageMaker Distributed Data Parallel Library as a Backend of ``torch.distributed``
===========================================================================================

To use the SageMaker distributed data parallel library,
the only thing you need to do is to import the SageMaker distributed data
parallel library’s PyTorch client (``smdistributed.dataparallel.torch.torch_smddp``).
The client registers ``smddp`` as a backend for PyTorch.
When you initialize the PyTorch distributed process group using
the ``torch.distributed.init_process_group`` API,
make sure you specify ``'smddp'`` to the backend argument.

.. code:: python

  import smdistributed.dataparallel.torch.torch_smddp
  import torch.distributed as dist

  dist.init_process_group(backend='smddp')


If you already have a working PyTorch script and only need to add the
backend specification, you can proceed to :ref:`sdp_api_docs_launch_training_job`.

.. note::

  The ``smddp`` backend currently does not support creating subprocess groups
  with the ``torch.distributed.new_group()`` API.
  You cannot use the ``smddp`` backend concurrently with other backends.

.. seealso::

  If you still need to modify your training script to properly use
  the PyTorch distributed package, see `Preparing a PyTorch Training Script for Distributed Training <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-modify-sdp-pt.html>`_
  in the *Amazon SageMaker Developer Guide*.

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
