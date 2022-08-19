.. _sm-sdk-modelparallel-general:

#############################################################
Run a Distributed Training Job Using the SageMaker Python SDK
#############################################################

Walk through the following pages to learn about the SageMaker model parallel library's APIs
to configure and enable distributed model parallelism
through an Amazon SageMaker estimator.

.. _sm-sdk-modelparallel-params:

Configuration Parameters for ``distribution``
=============================================

Amazon SageMaker's TensorFlow and PyTorch estimator objects contain a ``distribution`` parameter,
which you can use to enable and specify parameters for SageMaker distributed training.
The SageMaker model parallel library internally uses MPI.
To use model parallelism, both ``smdistributed`` and MPI must be enabled
through the ``distribution`` parameter.

The following code example is a template of setting up model parallelism for a PyTorch estimator.

.. code:: python

  import sagemaker
  from sagemaker.pytorch import PyTorch

  smp_options = {
      "enabled":True,
      "parameters": {
          ...
      }
  }

  mpi_options = {
      "enabled" : True,
      ...
  }

  smdmp_estimator = PyTorch(
      ...
      distribution={
          "smdistributed": {"modelparallel": smp_options},
          "mpi": mpi_options
      }
  )

  smdmp_estimator.fit()

.. tip::

  This page provides you a complete list of parameters you can use
  when you construct a SageMaker estimator and configure for distributed training.

  To find examples of how to construct a SageMaker estimator with the distributed training parameters, see
  `Launch a SageMaker Distributed Model Parallel Training Job <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-sm-sdk.html>`_
  in the `SageMaker's Distributed Model Parallel developer guide <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_.

.. contents:: Table of Contents
  :depth: 3
  :local:

Parameters for ``smdistributed``
----------------------------------

You can use the following parameters to initialize the library
configuring a dictionary for ``modelparallel``, which goes
into the ``smdistributed`` option for the ``distribution`` parameter.

.. note::

    ``partitions`` for TensorFlow and ``pipeline_parallel_degree`` for PyTorch are required parameters.
    All other parameters in the following
    table are optional.

Common Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 20 10 60
   :header-rows: 1

   * - Parameter
     - Type / Valid values
     - Default
     - Description
   * - ``partitions`` for TensorFlow and PyTorch with smdistributed-modelparallel<v1.6,
       ``pipeline_parallel_degree`` for PyTorch v1.8.1 with smdistributed-modelparallel>=v1.6)
     - int
     -
     - **Required.** The number of partitions to split the model into.
       In case of ``pipeline_parallel_degree`` for PyTorch, this is the number of devices
       over which pipeline parallelism will be performed.
   * - ``microbatches``
     - int
     - 1
     - The number of microbatches to perform pipelining over. 1 means no pipelining.
       Batch size must be divisible by the number of microbatches.
   * - ``pipeline``
     - ``"interleaved"`` or ``"simple"``
     - ``"interleaved"``
     - The pipeline schedule.
   * - ``optimize``
     - ``"memory"`` or ``"speed"``
     - ``"memory"``
     - Determines the distribution mechanism of transformer layers.
       If optimizing ``speed``, there will be less communication across tensor-parallel ranks
       and layer normalization will not be distributed. However, there will be duplicate activations
       stored across tensor-parallel ranks.
       If optimizing ``memory``, there will be no redundant activations stored,
       but this will result in more communication overhead across tensor parallel ranks.
   * - ``placement_strategy``
     - ``"cluster"``, ``"spread"``, or a permutation of the string ``D``, ``P``, and ``T``.
     - ``"cluster"``
     - Determines the mapping of model partitions onto physical devices.
       When hybrid model/data parallelism is used, ``cluster`` places a single model replica in
       neighboring device IDs. Contrarily, ``spread`` places a model replica as far as possible.
       For more information, see :ref:`ranking-basics`.

       In case of the permutation letters, ``D`` stands for reduced-data parallelism,
       ``P`` stands for pipeline parallelism,
       and ``T`` stands for tensor parallelism.
       ``spread`` is equivalent to ``"TPD"``, and ``cluster`` is equivalent to ``"DPT"``.
       For more information, see :ref:`ranking-basics-tensor-parallelism`.

       Note: For TensorFlow, tensor parallelism is not implemented and
       available parameter values are only ``"spread"`` and ``"cluster"``.
   * - ``auto_partition``
     - bool
     - ``True``
     - Enable auto-partitioning. If disabled, ``default_partition`` parameter must be provided.
   * - ``default_partition``
     - int
     - ``0``
     - **Required** if ``auto_partition`` is false. The partition ID to place operations/modules
       that are not placed in any ``smp.partition`` contexts.

TensorFlow-specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 20 10 60
   :header-rows: 1

   * - Parameter
     - Type / Valid values
     - Default
     - Description
   * - ``contiguous``
     - bool
     - ``True``
     - Whether the model partitions should be contiguous. If true, each partition forms a connected component in the computational graph, unless the graph itself is not connected.
   * - ``horovod``
     - bool
     - ``False``
     - Must be set to ``True`` if hybrid model/data parallelism is used and the data parallelism (DP) framework is Horovod.


PyTorch-specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :widths: 10 20 10 60
  :header-rows: 1

  * - Parameter
    - Type / Valid values
    - Default
    - Description
  * - ``memory_weight``
    - float [0.0, 1.0]
    - ``0.2`` if ``optimize`` is ``"speed"``, else ``0.8``
    - The weight of memory balancing in the auto-partitioni ng objective, as opposed to balancing computational load. If 0.0, the library only tries to balance computation; if 1.0 the library only tries to balance the memory use. Any value in between interpolates between these extremes.
  * - ``ddp``
    - bool
    - ``False``
    - Must be set to True if hybrid model/data parallelism is used with DistributedDataParallel. DistributedDataParallel is used with NCCL backend, and uses the MASTER_PORT provided by SageMaker.
  * - ``active_microbatches`` (**smdistributed-modelparallel**>=v1.3)
    - int
    - ``partitions`` + 2
    - This is the maximum number of microbatches that are simultaneously in execution during pipelining. Jointly scaling batch size and number of microbatches can often mitigate the pipeline bubble overhead, but that can lead to increased memory usage if too many microbatches are simultaneously in execution. In such cases setting the number of active microbatches to a lower number can help control memory usage. By default this is set to two plus the number of partitions of the model.
  * - ``deterministic_server`` (**smdistributed-modelparallel**>=v1.3)
    - bool
    - ``False``
    - Setting this to true ensures that the execution server for pipelining executes requests in the same order across all data parallel ranks.
  * -  ``offload_activations`` (**smdistributed-modelparallel**>=v1.6)
    - bool
    - False
    - Enables activation
      offloading. To improve GPU memory usage, use activation offloading
      only when (1) the ``microbatches`` and ``active_microbatches`` are
      greater than 1, and (2) activation checkpointing is enabled for at
      least one module in the model.
  * - ``activation_loading_horizon`` (**smdistributed-modelparallel**>=v1.6)
    - int
    - 4
    - Specify the number
      of pipeline tasks. This determines how early the activations should
      be loaded back to the GPU, expressed in number of pipeline tasks.
      Smaller value indicates that activations are loaded closer in time to
      when they are needed for backward pass. Setting this value too small
      might improve memory usage, but might potentially cause throughput
      loss and GPU bottlenecks during the CPU-to-GPU data transfer.
  * - ``tensor_parallel_degree`` (**smdistributed-modelparallel**>=v1.6)
    - int
    - 1
    - The number of devices over which the tensor parallel modules will be distributed.
      If ``tensor_parallel_degree`` is greater than 1, then ``ddp`` must be set to ``True``.
  * - ``fp16`` (**smdistributed-modelparallel**>=v1.10)
    - bool
    - ``False``
    - To run FP16 training, add ``"fp16"'": True`` to the smp configuration.
      Other APIs remain the same between FP16 and FP32.
      If ``fp16`` is enabled and when user calls ``smp.DistributedModel``,
      the model will be wrapped with ``FP16_Module``, which converts the model
      to FP16 dtype and deals with forward pass in FP16.
      If ``fp16`` is enabled and when user calls ``smp.DistributedOptimizer``,
      the optimizer will be wrapped with ``FP16_Optimizer``.
  * - ``fp16_params`` (**smdistributed-modelparallel**>=v1.6)
    - bool
    - ``False``
    - If ``True``, the parameters of the distributed modules will be initialized in FP16.
  * - ``shard_optimizer_state`` (**smdistributed-modelparallel**>=v1.6)
    - bool
    - ``False``
    - If ``True``, the library shards the optimizer state of all parameters across
      the data parallel processes which hold the same parameter.
      This optimizer state sharding happens in a balanced manner.
      Note that when sharding optimizer state, full optimizer saving is not currently supported.
      Please save partial optimizer state. For more information about saving and loading checkpoints with
      optimizer state sharding, see `Instructions for Checkpointing with Tensor Parallelism <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-saving-loading-checkpoints.html>`_.
  * - ``prescaled_batch`` (**smdistributed-modelparallel**>=v1.6)
    - bool
    - ``False``
    - If ``True`` and when ``smp.nn.DistributedTransformerLMHead`` is used
      (this is typically used for GPT-2 or GPT-3 models),
      the library assumes that the devices in the same tensor parallelism group
      receive the same input data. Otherwise, it is assumed that they receive
      different examples. To learn more, see :ref:`prescaled-batch`.
  * - ``skip_tracing`` (**smdistributed-modelparallel**>=v1.6)
    - bool
    - False
    - Skips the initial tracing step. This can be useful in very large models
      where even model tracing at the CPU is not possible due to memory constraints.
  * - ``sharded_data_parallel_degree`` (**smdistributed-modelparallel**>=v1.11)
    - int
    - 1
    - To run a training job using sharded data parallelism, add this parameter and specify a number greater than 1.
      Sharded data parallelism is a memory-saving distributed training technique that splits the training state of a model (model parameters, gradients, and optimizer states) across GPUs in a data parallel group.
      For more information, see `Sharded Data Parallelism
      <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html>`_.
  * - ``sdp_reduce_bucket_size`` (**smdistributed-modelparallel**>=v1.11)
    - int
    - 5e8
    - Configuration parameter for sharded data parallelism (for ``sharded_data_parallel_degree > 2``).
      Specifies the size of PyTorch DDP gradient buckets in number of elements of the default dtype.
  * - ``sdp_param_persistence_threshold`` (**smdistributed-modelparallel**>=v1.11)
    - int
    - 1e6
    -  Specifies the size of a parameter tensor in number of elements that can persist at each GPU. Sharded data parallelism splits each parameter tensor across GPUs of a data parallel group. If the number of elements in the parameter tensor is smaller than this threshold, the parameter tensor is not split; this helps reduce communication overhead because the parameter tensor is replicated across data-parallel GPUs.
  * - ``sdp_max_live_parameters`` (**smdistributed-modelparallel**>=v1.11)
    - int
    - 1e9
    - Specifies the maximum number of parameters that can simultaneously be in a recombined training state during the forward and backward pass. Parameter fetching with the AllGather operation pauses when the number of active parameters reaches the given threshold. Note that increasing this parameter increases the memory footprint.
  * - ``sdp_hierarchical_allgather`` (**smdistributed-modelparallel**>=v1.11)
    - bool
    - True
    - If set to True, the AllGather operation runs hierarchically: it runs within each node first, and then runs across nodes. For multi-node distributed training jobs, the hierarchical AllGather operation is automatically activated.
  * - ``sdp_gradient_clipping`` (**smdistributed-modelparallel**>=v1.11)
    - float
    - 1.0
    - Specifies a threshold for gradient clipping the L2 norm of the gradients before propagating them backward through the model parameters. When sharded data parallelism is activated, gradient clipping is also activated. The default threshold is 1.0. Adjust this parameter if you have the exploding gradients problem.


Parameters for ``mpi``
----------------------

For the ``"mpi"`` key, a dict must be passed which contains:

* ``"enabled"``: Set to ``True`` to launch the training job with MPI.

* ``"processes_per_host"``: Specifies the number of processes MPI should launch on each host.
  In SageMaker a host is a single Amazon EC2 ml instance. The SageMaker distributed model parallel library maintains
  a one-to-one mapping between processes and GPUs across model and data parallelism.
  This means that SageMaker schedules each process on a single, separate GPU and no GPU contains more than one process.
  If you are using PyTorch, you must restrict each process to its own device using
  ``torch.cuda.set_device(smp.local_rank())``. To learn more, see
  `Modify a PyTorch Training Script
  <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script.html#model-parallel-customize-training-script-pt-16>`_.

  .. important::
   ``process_per_host`` must be less than or equal to the number of GPUs per instance, and typically will be equal to
   the number of GPUs per instance.

  For example, if you use one instance with 4-way model parallelism and 2-way data parallelism,
  then processes_per_host should be 2 x 4 = 8. Therefore, you must choose an instance that has at least 8 GPUs,
  such as an ml.p3.16xlarge.

  The following image illustrates how 2-way data parallelism and 4-way model parallelism is distributed across 8 GPUs:
  the model is partitioned across 4 GPUs, and each partition is added to 2 GPUs.

  .. image:: smp_versions/model-data-parallel.png
      :width: 650
      :alt: 2-way data parallelism and 4-way model parallelism distributed across 8 GPUs


* ``"custom_mpi_options"``: Use this key to pass any custom MPI options you might need.
  To avoid Docker warnings from contaminating your training logs, we recommend the following flag.
  ```--mca btl_vader_single_copy_mechanism none```


.. _ranking-basics:

Ranking Basics without Tensor Parallelism
=========================================

The library maintains a one-to-one mapping between processes and available GPUs:
for each GPU, there is a corresponding CPU process. Each CPU process
maintains a “rank” assigned by MPI, which is a 0-based unique index for
the process. For instance, if a training job is launched with 4
``p3dn.24xlarge`` instances using all its GPUs, there are 32 processes
across all instances, and the ranks of these processes range from 0 to
31.

The ``local_rank`` of a process is the rank of the process among the
processes in the same instance. This can range from 0 up to the number
of GPUs in the instance, but can be lower if fewer processes than GPUs are
launched in the instance. For instance, in the preceding
example, ``local_rank``\ s of the processes will range from 0 to 7,
since there are 8 GPUs in a ``p3dn.24xlarge`` instance.

When model parallelism is used together with data parallelism (Horovod for TensorFlow
and DDP for PyTorch), the library partitions the set of processes into
disjoint \ ``mp_group``\ s. An ``mp_group`` is a subset of all processes
that together hold a single, partitioned model replica.

For instance, if
a single node job is launched with 8 local processes with
``partitions=2`` (meaning the model will be split into 2), there are
four \ ``mp_group``\ s. The specific sets of processes that form the
``mp_group``\ s can be adjusted by the ``placement_strategy`` option.

- If ``placement_strategy`` is ``spread``, then the four
  ``mp_group``\ s are ``[0, 4], [1, 5], [2, 6], [3, 7]``. The
  ``mp_rank`` is the rank of a process within each ``mp_group``. For example,
  the ``mp_rank`` is 0 for the processes 0, 1, 2, and 3, and the ``mp_rank`` is 1 for
  the processes 4, 5, 6, and 7.

  Analogously, the library defines ``dp_group``\ s as sets of processes that
  all hold the same model partition, and perform data parallelism among
  each other. If ``placement_strategy`` is ``spread``, there are two ``dp_group``\ s:
  ``[0, 1, 2, 3]`` and ``[4, 5, 6, 7]``.

  Since each process within the ``dp_group`` holds the same partition of
  the model, and makes allreduce calls among themselves. Allreduce for
  data parallelism does not take place *across* ``dp_group``\ s.
  ``dp_rank`` is defined as the rank of a process within its ``dp_group``.
  In the preceding example, the \ ``dp_rank`` of process 6 is 2.

- If ``placement_strategy`` is ``cluster``, the four ``mp_group``\ s
  become ``[0, 1], [2, 3], [4, 5], [6, 7]``, and the the two ``dp_group``\ s become
  ``[0, 2, 4, 6]`` and ``[1, 3, 5, 7]``.

.. _ranking-basics-tensor-parallelism:

Placement Strategy with Tensor Parallelism
==========================================

In addition to the two placement strategies introduced in the previous section,
the library provides additional placement strategies for extended tensor parallelism features
for PyTorch. The additional placement strategies (parallelism types) are denoted as follows:

- ``D`` stands for (reduced) data parallelism.
- ``P`` stands for pipeline parallelism.
- ``T`` stands for tensor parallelism.

With given permutation of the tree letters, the library takes the right-most letter
as the first strategy performs over the global ranks in ascending order.
Contrarily, the parallelism type represented by the left-most letter is performed
over the ranks that are as distant as possible.

- **Example:** Given 8 devices with ``tp_size() == 2``,
  ``pp_size() == 2``, ``rdp_size() == 2``

  - ``placement_strategy: "DPT"`` gives

    ==== ======== ======= =======
    rank rdp_rank pp_rank tp_rank
    ==== ======== ======= =======
    0    0        0       0
    1    0        0       1
    2    0        1       0
    3    0        1       1
    4    1        0       0
    5    1        0       1
    6    1        1       0
    7    1        1       1
    ==== ======== ======= =======

  - ``placement_strategy: "PTD"`` gives

    ==== ======== ======= =======
    rank rdp_rank pp_rank tp_rank
    ==== ======== ======= =======
    0    0        0       0
    1    1        0       0
    2    0        0       1
    3    1        0       1
    4    0        1       0
    5    1        1       0
    6    0        1       1
    7    1        1       1
    ==== ======== ======= =======

Because the neighboring ranks are placed on the same instance with
high-bandwidth NVLinks, it is recommended to place the
parallelism type that has higher bandwidth requirements for your model
on the right-most position in the ``placement_strategy`` string. Because
tensor parallelism often requires frequent communication, placing
``T`` in the right-most position is recommended (as in the default
``"cluster"`` strategy). In many large models, keeping the default of
``"cluster"`` would result in the best performance.


.. _prescaled-batch:

Prescaled Batch
===============

``prescaled_batch`` is a configuration parameter that can be useful for
``DistributedTransformerLMHead``, which is used for GPT-2 and GPT-3.

The way tensor parallelism works is that when a module is distributed,
the inputs to the distributed module in different ``tp_rank``\ s gets
shuffled around in a way that is sliced by the hidden dimension and
scaled by the batch dimension. For example, if tensor parallel degree is
8, the inputs to ``DistributedTransformer`` (a tensor with shape
``[B, S, H]`` where ``B``\ =batch size, ``S``\ =sequence length,
``H``\ =hidden width) in different ``tp_rank``\ s will be communicated
around, and the shapes will become ``[8B, S, H/8]``. Each ``tp_rank``
has the batch from all the peer ``tp_rank``\ s, but only the slice that
interacts with their local partition of the module.

By default, the library assumes that each ``tp_rank`` gets assigned a
different batch, and performs the communication described above. If
``prescaled_batch`` is true, then the library assumes that the input
batch is already scaled (and is the same across the ``tp_rank``\ s), and
only does the slicing. In the example above, the library assumes that
input tensor has shape ``[8B, S, H]``, and only converts it into
``[8B, S, H/8]``. So if ``prescaled_batch`` is true, it is the user’s
responsibility to feed the same batch to the ``tp_rank``\ s in the same
``TP_GROUP``. This can be done by doing the data sharding based on
``smp.rdp_size()`` and ``smp.rdp_rank()``, instead of ``smp.dp_size()``
and ``smp.dp_rank()``. When ``prescaled_batch`` is true, the global
batch size is ``smp.rdp_size()`` multiplied by the per-``MP_GROUP``
batch size. When ``prescaled_batch`` is false, global batch size is
``smp.dp_size()`` multiplied by the per-``PP_GROUP`` batch size.

If you use pipeline parallelism degree 1, then you can keep
``prescaled_batch`` false (the default option). If you use a pipeline
parallellism degree more than 1, it is recommended to use
``prescaled_batch`` true, so that you can increase per-``MP_GROUP``
batch size for efficient pipelining, without running into out-of-memory
issues.
