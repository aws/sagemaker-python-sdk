########################################################
The SageMaker Distributed Data Parallel Library Overview
########################################################

SageMaker's distributed data parallel library extends SageMaker’s training
capabilities on deep learning models with near-linear scaling efficiency,
achieving fast time-to-train with minimal code changes.

When training a model on a large amount of data, machine learning practitioners
will often turn to distributed training to reduce the time to train.
In some cases, where time is of the essence,
the business requirement is to finish training as quickly as possible or at
least within a constrained time period.
Then, distributed training is scaled to use a cluster of multiple nodes,
meaning not just multiple GPUs in a computing instance, but multiple instances
with multiple GPUs. However, as the cluster size increases, it is possible to see a significant drop
in performance due to communications overhead between nodes in a cluster.

SageMaker's distributed data parallel library addresses communications overhead in two ways:

1. The library performs AllReduce, a key operation during distributed training that is responsible for a
   large portion of communication overhead.
2. The library performs optimized node-to-node communication by fully utilizing AWS’s network
   infrastructure and Amazon EC2 instance topology.

To learn more about the core features of this library, see
`Introduction to SageMaker's Distributed Data Parallel Library
<https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-intro.html>`_
in the SageMaker Developer Guide.
