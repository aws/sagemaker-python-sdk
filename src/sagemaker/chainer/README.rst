=======================================
Chainer SageMaker Estimators and Models
=======================================

With Chainer Estimators, you can train and host Chainer models on Amazon SageMaker.

Supported versions of Chainer: ``4.0.0``, ``4.1.0``, ``5.0.0``

You can visit the Chainer repository at https://github.com/chainer/chainer.

For information about using Chainer with the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/using_chainer.html.

Chainer Training Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

Amazon provides several example Jupyter notebooks that demonstrate end-to-end training on Amazon SageMaker using Chainer.
Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.


SageMaker Chainer Docker containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several
libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control
the environment your script runs in.

SageMaker runs Chainer Estimator scripts in either Python 2.7 or Python 3.5. You can select the Python version by
passing a py_version keyword arg to the Chainer Estimator constructor. Setting this to py3 (the default) will cause your
training script to be run on Python 3.5. Setting this to py2 will cause your training script to be run on Python 2.7
This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

The Chainer Docker images have the following dependencies installed:

+-----------------------------+-------------+-------------+-------------+
| Dependencies                | chainer 4.0 | chainer 4.1 | chainer 5.0 |
+-----------------------------+-------------+-------------+-------------+
| chainer                     | 4.0.0       | 4.1.0       | 5.0.0       |
+-----------------------------+-------------+-------------+-------------+
| chainercv                   | 0.9.0       | 0.10.0      | 0.10.0      |
+-----------------------------+-------------+-------------+-------------+
| chainermn                   | 1.2.0       | 1.3.0       | N/A         |
+-----------------------------+-------------+-------------+-------------+
| CUDA (GPU image only)       | 9.0         | 9.0         | 9.0         |
+-----------------------------+-------------+-------------+-------------+
| cupy                        | 4.0.0       | 4.1.0       | 5.0.0       |
+-----------------------------+-------------+-------------+-------------+
| matplotlib                  | 2.2.0       | 2.2.0       | 2.2.0       |
+-----------------------------+-------------+-------------+-------------+
| mpi4py                      | 3.0.0       | 3.0.0       | 3.0.0       |
+-----------------------------+-------------+-------------+-------------+
| numpy                       | 1.14.3      | 1.15.3      | 1.15.4      |
+-----------------------------+-------------+-------------+-------------+
| opencv-python               | 3.4.0.12    | 3.4.0.12    | 3.4.0.12    |
+-----------------------------+-------------+-------------+-------------+
| Pillow                      | 5.1.0       | 5.3.0       | 5.3.0       |
+-----------------------------+-------------+-------------+-------------+
| Python                      | 2.7 or 3.5  | 2.7 or 3.5  | 2.7 or 3.5  |
+-----------------------------+-------------+-------------+-------------+

The Docker images extend Ubuntu 16.04.

You must select a version of Chainer by passing a ``framework_version`` keyword arg to the Chainer Estimator
constructor. Currently supported versions are listed in the above table. You can also set framework_version to only
specify major and minor version, which will cause your training script to be run on the latest supported patch
version of that minor version.

Alternatively, you can build your own image by following the instructions in the SageMaker Chainer containers
repository, and passing ``image_name`` to the Chainer Estimator constructor.

You can visit the SageMaker Chainer containers repository here: https://github.com/aws/sagemaker-chainer-containers/
