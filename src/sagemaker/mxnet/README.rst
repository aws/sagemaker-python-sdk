=========================================
Using MXNet with the SageMaker Python SDK
=========================================

With the SageMaker Python SDK, you can train and host MXNet models on Amazon SageMaker.

Supported versions of MXNet: ``1.4.0``, ``1.3.0``, ``1.2.1``, ``1.1.0``, ``1.0.0``, ``0.12.1``.

Supported versions of MXNet for Elastic Inference: ``1.4.0``, ``1.3.0``.

For information about using MXNet with the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/using_mxnet.html.

SageMaker MXNet Containers
--------------------------

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control the environment your script runs in.

SageMaker runs MXNet scripts in either Python 2.7 or Python 3.6. You can select the Python version by passing a ``py_version`` keyword arg to the MXNet Estimator constructor. Setting this to ``py2`` (the default) will cause your training script to be run on Python 2.7. Setting this to ``py3`` will cause your training script to be run on Python 3.6. This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

Your MXNet training script will be run on version 1.2.1 by default. (See below for how to choose a different version, and currently supported versions.) The decision to use the GPU or CPU version of MXNet is made by the ``train_instance_type``, set on the MXNet constructor. If you choose a GPU instance type, your training job will be run on a GPU version of MXNet. If you choose a CPU instance type, your training job will be run on a CPU version of MXNet. Similarly, when you call deploy, specifying a GPU or CPU deploy_instance_type, will control which MXNet build your Endpoint runs.

The Docker images have the following dependencies installed:

+-------------------------+--------------+-------------+-------------+-------------+-------------+-------------+
| Dependencies            | MXNet 0.12.1 | MXNet 1.0.0 | MXNet 1.1.0 | MXNet 1.2.1 | MXNet 1.3.0 | MXNet 1.4.0 |
+-------------------------+--------------+-------------+-------------+-------------+-------------+-------------+
| Python                  |   2.7 or 3.5 |   2.7 or 3.5|   2.7 or 3.5|   2.7 or 3.5|   2.7 or 3.5|   2.7 or 3.6|
+-------------------------+--------------+-------------+-------------+-------------+-------------+-------------+
| CUDA (GPU image only)   |          9.0 |         9.0 |         9.0 |         9.0 |         9.0 |         9.2 |
+-------------------------+--------------+-------------+-------------+-------------+-------------+-------------+
| numpy                   |       1.13.3 |      1.13.3 |      1.13.3 |      1.14.5 |      1.14.6 |      1.16.3 |
+-------------------------+--------------+-------------+-------------+-------------+-------------+-------------+
| onnx                    |          N/A |         N/A |         N/A |       1.2.1 |       1.2.1 |       1.4.1 |
+-------------------------+--------------+-------------+-------------+-------------+-------------+-------------+
| keras-mxnet             |          N/A |         N/A |         N/A |         N/A |       2.2.2 |     2.2.4.1 |
+-------------------------+--------------+-------------+-------------+-------------+-------------+-------------+

The Docker images extend Ubuntu 16.04.

You can select version of MXNet by passing a ``framework_version`` keyword arg to the MXNet Estimator constructor. Currently supported versions are listed in the above table. You can also set ``framework_version`` to only specify major and minor version, e.g ``1.2``, which will cause your training script to be run on the latest supported patch version of that minor version, which in this example would be 1.2.1.
Alternatively, you can build your own image by following the instructions in the SageMaker MXNet containers repository, and passing ``image_name`` to the MXNet Estimator constructor.

You can visit the SageMaker MXNet container repositories here:

- training: https://github.com/aws/sagemaker-mxnet-container
- serving: https://github.com/aws/sagemaker-mxnet-serving-container
