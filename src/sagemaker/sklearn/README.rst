============================================
Scikit-learn SageMaker Estimators and Models
============================================

With Scikit-learn Estimators, you can train and host Scikit-learn models on Amazon SageMaker.

Supported versions of Scikit-learn: ``0.20.0``

You can visit the Scikit-learn repository at https://github.com/scikit-learn/scikit-learn.

For information about using Scikit-learn with the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/using_sklearn.html.

Scikit-learn Training Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Amazon provides an example Jupyter notebook that demonstrate end-to-end training on Amazon SageMaker using Scikit-learn.
Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.


SageMaker Scikit-learn Docker Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several
libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control
the environment your script runs in.

SageMaker runs Scikit-learn Estimator scripts in either Python 2.7 or Python 3.5. You can select the Python version by
passing a py_version keyword arg to the Scikit-learn Estimator constructor. Setting this to py3 (the default) will cause
your training script to be run on Python 3.5. Setting this to py2 will cause your training script to be run on Python 2.7
This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

The Scikit-learn Docker images have the following dependencies installed:

+-----------------------------+-------------+
| Dependencies                | sklearn 0.2 |
+-----------------------------+-------------+
| sklearn                     | 0.20.0      |
+-----------------------------+-------------+
| sagemaker                   | 1.11.3      |
+-----------------------------+-------------+
| sagemaker-containers        | 2.2.4       |
+-----------------------------+-------------+
| numpy                       | 1.15.2      |
+-----------------------------+-------------+
| pandas                      | 0.23.4      |
+-----------------------------+-------------+
| Pillow                      | 3.1.2       |
+-----------------------------+-------------+
| Python                      | 2.7 or 3.5  |
+-----------------------------+-------------+

You can see the full list by calling ``pip freeze`` from the running Docker image.

The Docker images extend Ubuntu 16.04.

You can select version of Scikit-learn by passing a framework_version keyword arg to the Scikit-learn Estimator constructor.
Currently supported versions are listed in the above table. You can also set framework_version to only specify major and
minor version, which will cause your training script to be run on the latest supported patch version of that minor
version.

Alternatively, you can build your own image by following the instructions in the SageMaker Scikit-learn containers
repository, and passing ``image_name`` to the Scikit-learn Estimator constructor.
sagemaker-containers
You can visit the SageMaker Scikit-learn containers repository here: https://github.com/aws/sagemaker-scikit-learn-container/
