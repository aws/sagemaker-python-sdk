===========================================
SageMaker Reinforcement Learning Estimators
===========================================

With Reinforcement Learning (RL) Estimators, you can train reinforcement learning models on Amazon SageMaker.

Supported versions of Coach: ``0.11.1``, ``0.10.1`` with TensorFlow, ``0.11.0`` with TensorFlow or MXNet.
For more information about Coach, see https://github.com/NervanaSystems/coach

Supported versions of Ray: ``0.6.5``, ``0.5.3`` with TensorFlow.
For more information about Ray, see https://github.com/ray-project/ray

For information about using RL with the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/using_rl.html.

SageMaker RL Docker Containers
------------------------------

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several
libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control
the environment your script runs in.

SageMaker runs RL Estimator scripts in either Python 3.5 for MXNet or Python 3.6 for TensorFlow.

The Docker images have the following dependencies installed:

+-------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Dependencies            |      Coach 0.10.1 |      Coach 0.11.0 |      Coach 0.11.1 |         Ray 0.5.3 |         Ray 0.6.5 |
+-------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Python                  |               3.6 |  3.5 (MXNet) or   |               3.6 |               3.6 |               3.6 |
|                         |                   |  3.6 (TensorFlow) |                   |                   |                   |
+-------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| CUDA (GPU image only)   |               9.0 |               9.0 |               9.0 |               9.0 |               9.0 |
+-------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| DL Framework            | TensorFlow-1.11.0 | MXNet-1.3.0 or    | TensorFlow-1.12.0 | TensorFlow-1.11.0 | TensorFlow-1.12.0 |
|                         |                   | TensorFlow-1.11.0 |                   |                   |                   |
+-------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| gym                     |            0.10.5 |            0.10.5 |            0.11.0 |            0.10.5 |            0.12.1 |
+-------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+

The Docker images extend Ubuntu 16.04.

You can select version of  by passing a ``framework_version`` keyword arg to the RL Estimator constructor.
Currently supported versions are listed in the above table. You can also set ``framework_version`` to only specify major and
minor version, which will cause your training script to be run on the latest supported patch version of that minor
version.

Alternatively, you can build your own image by following the instructions in the SageMaker RL containers
repository, and passing ``image_name`` to the RL Estimator constructor.

You can visit `the SageMaker RL containers repository <https://github.com/aws/sagemaker-rl-container>`_.
