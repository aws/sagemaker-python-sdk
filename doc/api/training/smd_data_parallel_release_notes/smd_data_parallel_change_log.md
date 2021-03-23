# Sagemaker Distributed Data Parallel 1.1.0 Release Notes

* New Features
* Bug Fixes
* Improvements
* Known Issues

New Features:

* Adds support for PyTorch 1.8.0 with CUDA 11.1 and CUDNN 8

Bug Fixes:

* Fixes crash issue when importing `smdataparallel` before PyTorch

Improvements:

* Update `smdataparallel` name in python packages, descriptions, and log outputs

Known Issues:

* SageMaker DataParallel is not efficient when run using a single node. For the best performance, use multi-node distributed training with `smdataparallel`. Use a single node only for experimental runs while preparing your training pipeline.

Getting Started

For getting started, refer to SageMaker Distributed Data Parallel Python SDK Guide (https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api).

# Sagemaker Distributed Data Parallel 1.0.0 Release Notes

- First Release
- Getting Started

## First Release

SageMaker's distributed data parallel library extends SageMaker’s training
capabilities on deep learning models with near-linear scaling efficiency,
achieving fast time-to-train with minimal code changes.
SageMaker Distributed Data Parallel:

- optimizes your training job for AWS network infrastructure and EC2 instance topology.
- takes advantage of gradient update to communicate between nodes with a custom AllReduce algorithm.

The library currently supports TensorFlow v2 and PyTorch via [AWS Deep Learning Containers](https://aws.amazon.com/machine-learning/containers/).

## Getting Started

For getting started, refer to [SageMaker Distributed Data Parallel Python SDK Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api).
