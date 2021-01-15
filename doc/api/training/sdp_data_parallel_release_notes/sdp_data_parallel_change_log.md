# Sagemaker Distributed Data Parallel - Release Notes

- First Release
- Getting Started

## First Release
SageMaker's distributed data parallel library extends SageMaker’s training
capabilities on deep learning models with near-linear scaling efficiency,
achieving fast time-to-train with minimal code changes.
SageMaker Distributed Data Parallel :

- optimizes your training job for AWS network infrastructure and EC2 instance topology.
- takes advantage of gradient update to communicate between nodes with a custom AllReduce algorithm.

The library currently supports Tensorflow v2 and PyTorch via [AWS Deep Learning Containers](https://aws.amazon.com/machine-learning/containers/).  

## Getting Started
For getting started, refer to [SageMaker Distributed Data Parallel Python SDK Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api).