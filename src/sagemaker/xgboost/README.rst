============================================
XGBoost SageMaker Estimators and Models
============================================

With XGBoost Estimators, you can train and host XGBoost models on Amazon SageMaker.

Supported versions of SageMaker XGBoost: ``0.90-1``

Note that the first part of the version refers to the upstream module version (aka, 0.90), while the second
part refers to the SageMaker version for the container.

You can visit the XGBoost repository at https://github.com/dmlc/xgboost

For information about using XGBoost with the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/using_xgboost.html.

XGBoost Training Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Amazon provides an example Jupyter notebook that demonstrate end-to-end training on Amazon SageMaker using XGBoost.
Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.


SageMaker XGBoost Docker Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several
libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control
the environment your script runs in.

SageMaker runs XGBoost Estimator scripts in either Python 2.7 or Python 3.5. You can select the Python version by
passing a py_version keyword arg to the XGBoost Estimator constructor. Setting this to py3 (the default) will cause
your training script to be run on Python 3.5. Setting this to py2 will cause your training script to be run on Python 2.7
This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

The XGBoost Docker images have the following dependencies installed:

+-----------------------------+-------------+
| Dependencies                | Version     |
+-----------------------------+-------------+
| xgboost                     | 0.90.0      |
+-----------------------------+-------------+
| matplotlib                  | 3.0.3+      |
+-----------------------------+-------------+
| numpy                       | 1.16.4+     |
+-----------------------------+-------------+
| pandas                      | 0.24.2+     |
+-----------------------------+-------------+
| psutils                     | 5.6.3+      |
+-----------------------------+-------------+
| PyYAML                      | < 4.3       |
+-----------------------------+-------------+
| requests                    | < 2.21      |
+-----------------------------+-------------+
| retrying                    | 1.3.3       |
+-----------------------------+-------------+
| scikit-learn                | 0.21.2+     |
+-----------------------------+-------------+
| scipy                       | 1.3.0+      |
+-----------------------------+-------------+
| sagemaker-containers        | 2.5.1+      |
+-----------------------------+-------------+
| urllib3                     | < 1.25      |
+-----------------------------+-------------+
| Python                      | 2.7 or 3.5  |
+-----------------------------+-------------+

You can see the full list by calling ``pip freeze`` from the running Docker image.

The Docker images extend Ubuntu 16.04.

You can select version of XGBoost by passing a framework_version keyword arg to the XGBoost Estimator constructor.
Currently supported versions are listed in the above table. You can also set framework_version to only specify major and
minor version, which will cause your training script to be run on the latest supported patch version of that minor
version.

Alternatively, you can build your own image by following the instructions in the SageMaker XGBoost containers
repository, and passing ``image_name`` to the XGBoost Estimator constructor.

You can visit the SageMaker XGBoost containers repository here: https://github.com/aws/sagemaker-xgboost-container
