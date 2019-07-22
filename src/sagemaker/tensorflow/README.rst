TensorFlow SageMaker Estimators and Models
==========================================

TensorFlow SageMaker Estimators allow you to run your own TensorFlow
training algorithms on SageMaker Learner, and to host your own TensorFlow
models on SageMaker Hosting.

Documentation of the previous Legacy Mode versions: `1.4.1 <https://github.com/aws/sagemaker-python-sdk/tree/v1.0.0#tensorflow-sagemaker-estimators>`_, `1.5.0 <https://github.com/aws/sagemaker-python-sdk/tree/v1.1.0#tensorflow-sagemaker-estimators>`_, `1.6.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.5.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.7.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.5.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.8.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.5.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.9.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.9.2/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.10.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.10.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| WARNING                                                                                                                                                                     |
+=============================================================================================================================================================================+
| We have added a new format of your TensorFlow training script with TensorFlow version 1.11.                                                                                 |
| This new way gives the user script more flexibility.                                                                                                                        |
| This new format is called Script Mode, as opposed to Legacy Mode, which is what we support with TensorFlow 1.11 and older versions.                                         |
| In addition we are adding Python 3 support with Script Mode.                                                                                                                |
| Last supported version of Legacy Mode will be TensorFlow 1.12.                                                                                                              |
| Script Mode is available with TensorFlow version 1.11 and newer.                                                                                                            |
| Make sure you refer to the correct version of this README when you prepare your script.                                                                                     |
| You can find the Legacy Mode README `here <https://github.com/aws/sagemaker-python-sdk/tree/v1.12.0/src/sagemaker/tensorflow#tensorflow-sagemaker-estimators-and-models>`_. |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Supported versions of TensorFlow for Elastic Inference: ``1.11``, ``1.12``, ``1.13``.

For information about using TensorFlow with the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/using_tf.html.

SageMaker TensorFlow Docker containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The containers include the following Python packages:

+--------------------------------+---------------+
| Dependencies                   | Script Mode   |
+--------------------------------+---------------+
| awscli                         | 1.16.130      |
+--------------------------------+---------------+
| boto3                          | Latest        |
+--------------------------------+---------------+
| botocore                       | Latest        |
+--------------------------------+---------------+
| h5py                           | 2.9.0         |
+--------------------------------+---------------+
| keras                          | 2.2.4         |
+--------------------------------+---------------+
| mpi4py                         | 3.0.1         |
+--------------------------------+---------------+
| numpy                          | 1.16.2        |
+--------------------------------+---------------+
| pandas                         | 0.24.2        |
+--------------------------------+---------------+
| pip                            | 19.1.1        |
+--------------------------------+---------------+
| Pillow                         | 5.4.1         |
+--------------------------------+---------------+
| Python                         | 2.7 or 3.6    |
+--------------------------------+---------------+
| requests                       | 2.21.0        |
+--------------------------------+---------------+
| sagemaker-containers           | >=2.4.6       |
+--------------------------------+---------------+
| sagemaker-tensorflow-container | Latest        |
+--------------------------------+---------------+
| scipy                          | 1.2.1         |
+--------------------------------+---------------+
| sklean                         | 0.20.3        |
+--------------------------------+---------------+
| tensorflow                     | 1.13.0        |
+--------------------------------+---------------+

Script Mode TensorFlow Docker images support both Python 2.7 and Python 3.6. The Docker images extend Ubuntu 16.04.

You can select version of TensorFlow by passing a ``framework_version`` keyword arg to the TensorFlow Estimator constructor. Currently supported versions are listed in the table above. You can also set ``framework_version`` to only specify major and minor version, e.g ``'1.6'``, which will cause your training script to be run on the latest supported patch version of that minor version, which in this example would be 1.6.0.
Alternatively, you can build your own image by following the instructions in the SageMaker TensorFlow containers
repository, and passing ``image_name`` to the TensorFlow Estimator constructor.

For more information on the contents of the images, see the SageMaker TensorFlow containers repositories here:

- training: https://github.com/aws/sagemaker-tensorflow-container
- serving: https://github.com/aws/sagemaker-tensorflow-serving-container
