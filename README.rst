.. image:: https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png
    :height: 100px
    :alt: SageMaker

====================
SageMaker Python SDK
====================

.. image:: https://travis-ci.org/aws/sagemaker-python-sdk.svg?branch=master
   :target: https://travis-ci.org/aws/sagemaker-python-sdk
   :alt: Build Status

.. image:: https://codecov.io/gh/aws/sagemaker-python-sdk/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/aws/sagemaker-python-sdk
   :alt: CodeCov

SageMaker Python SDK is an open source library for training and deploying machine learning models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks **Apache MXNet** and **TensorFlow**.
You can also train and deploy models with **Amazon algorithms**,
which are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training.
If you have **your own algorithms** built into SageMaker compatible Docker containers, you can train and host models using these as well.

For detailed API reference please go to: `Read the Docs <https://sagemaker.readthedocs.io>`_

Table of Contents
-----------------

1. `Installing SageMaker Python SDK <#installing-the-sagemaker-python-sdk>`__
2. `Using the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html>`__
3. `MXNet SageMaker Estimators <#mxnet-sagemaker-estimators>`__
4. `TensorFlow SageMaker Estimators <#tensorflow-sagemaker-estimators>`__
5. `Chainer SageMaker Estimators <#chainer-sagemaker-estimators>`__
6. `PyTorch SageMaker Estimators <#pytorch-sagemaker-estimators>`__
7. `Scikit-learn SageMaker Estimators <#scikit-learn-sagemaker-estimators>`__
8. `SageMaker Reinforcement Learning Estimators <#sagemaker-reinforcement-learning-estimators>`__
9. `SageMaker SparkML Serving <#sagemaker-sparkml-serving>`__
10. `AWS SageMaker Estimators <#aws-sagemaker-estimators>`__
11. `Using SageMaker AlgorithmEstimators <https://sagemaker.readthedocs.io/en/stable/overview.html#using-sagemaker-algorithmestimators>`__
12. `Consuming SageMaker Model Packages <https://sagemaker.readthedocs.io/en/stable/overview.html#consuming-sagemaker-model-packages>`__
13. `BYO Docker Containers with SageMaker Estimators <https://sagemaker.readthedocs.io/en/stable/overview.html#byo-docker-containers-with-sagemaker-estimators>`__
14. `SageMaker Automatic Model Tuning <https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning>`__
15. `SageMaker Batch Transform <https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform>`__
16. `Secure Training and Inference with VPC <https://sagemaker.readthedocs.io/en/stable/overview.html#secure-training-and-inference-with-vpc>`__
17. `BYO Model <https://sagemaker.readthedocs.io/en/stable/overview.html#byo-model>`__
18. `Inference Pipelines <https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines>`__
19. `SageMaker Workflow <#sagemaker-workflow>`__


Installing the SageMaker Python SDK
-----------------------------------

The SageMaker Python SDK is built to PyPI and can be installed with pip as follows:

::

    pip install sagemaker

You can install from source by cloning this repository and running a pip install command in the root directory of the repository:

::

    git clone https://github.com/aws/sagemaker-python-sdk.git
    cd sagemaker-python-sdk
    pip install .

Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Python SDK supports Unix/Linux and Mac.

Supported Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Python SDK is tested on:

- Python 2.7
- Python 3.6

AWS Permissions
~~~~~~~~~~~~~~~

As a managed service, Amazon SageMaker performs operations on your behalf on the AWS hardware that is managed by Amazon SageMaker.
Amazon SageMaker can perform only operations that the user permits.
You can read more about which permissions are necessary in the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.

The SageMaker Python SDK should not require any additional permissions.
However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

Licensing
~~~~~~~~~
SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

Running tests
~~~~~~~~~~~~~

SageMaker Python SDK has unit tests and integration tests.

You can install the libraries needed to run the tests by running :code:`pip install --upgrade .[test]` or, for Zsh users: :code:`pip install --upgrade .\[test\]`

**Unit tests**


We run unit tests with tox, which is a program that lets you run unit tests for multiple Python versions, and also make sure the
code fits our style guidelines. We run tox with Python 2.7 and 3.6, so to run unit tests
with the same configuration we do, you'll need to have interpreters for Python 2.7 and Python 3.6 installed.

To run the unit tests with tox, run:

::

    tox tests/unit

**Integrations tests**

To run the integration tests, the following prerequisites must be met

1. AWS account credentials are available in the environment for the boto3 client to use.
2. The AWS account has an IAM role named :code:`SageMakerRole` with the AmazonSageMakerFullAccess policy attached.

We recommend selectively running just those integration tests you'd like to run. You can filter by individual test function names with:

::

    pytest -k 'test_i_care_about'


You can also run all of the integration tests by running the following command, which runs them in sequence, which may take a while:

::

    pytest tests/integ


You can also run them in parallel:

::

    pytest -n auto tests/integ


Building Sphinx docs
~~~~~~~~~~~~~~~~~~~~

``cd`` into the ``doc`` directory and run:

::

    make html

You can edit the templates for any of the pages in the docs by editing the .rst files in the "doc" directory and then running "``make html``" again.

MXNet SageMaker Estimators
--------------------------

By using MXNet SageMaker ``Estimators``, you can train and host MXNet models on Amazon SageMaker.

Supported versions of MXNet: ``1.3.0``, ``1.2.1``, ``1.1.0``, ``1.0.0``, ``0.12.1``.

Supported versions of MXNet for Elastic Inference: ``1.3.0``

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information, see `Using MXNet with the SageMaker Python SDK`_.

.. _Using MXNet with the SageMaker Python SDK: https://sagemaker.readthedocs.io/en/stable/using_mxnet.html


TensorFlow SageMaker Estimators
-------------------------------

By using TensorFlow SageMaker ``Estimators``, you can train and host TensorFlow models on Amazon SageMaker.

Supported versions of TensorFlow: ``1.4.1``, ``1.5.0``, ``1.6.0``, ``1.7.0``, ``1.8.0``, ``1.9.0``, ``1.10.0``, ``1.11.0``, ``1.12.0``.

Supported versions of TensorFlow for Elastic Inference: ``1.11.0``, ``1.12.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information, see `Using TensorFlow with the SageMaker Python SDK`_.

.. _Using TensorFlow with the SageMaker Python SDK: https://sagemaker.readthedocs.io/en/stable/using_tf.html


Chainer SageMaker Estimators
----------------------------

By using Chainer SageMaker ``Estimators``, you can train and host Chainer models on Amazon SageMaker.

Supported versions of Chainer: ``4.0.0``, ``4.1.0``, ``5.0.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information about Chainer, see https://github.com/chainer/chainer.

For more information about  Chainer SageMaker ``Estimators``, see `Using Chainer with the SageMaker Python SDK`_.

.. _Using Chainer with the SageMaker Python SDK: https://sagemaker.readthedocs.io/en/stable/using_chainer.html


PyTorch SageMaker Estimators
----------------------------

With PyTorch SageMaker ``Estimators``, you can train and host PyTorch models on Amazon SageMaker.

Supported versions of PyTorch: ``0.4.0``, ``1.0.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information about PyTorch, see https://github.com/pytorch/pytorch.

For more information about PyTorch SageMaker ``Estimators``, see `Using PyTorch with the SageMaker Python SDK`_.

.. _Using PyTorch with the SageMaker Python SDK: https://sagemaker.readthedocs.io/en/stable/using_pytorch.html


Scikit-learn SageMaker Estimators
---------------------------------

With Scikit-learn SageMaker ``Estimators``, you can train and host Scikit-learn models on Amazon SageMaker.

Supported versions of Scikit-learn: ``0.20.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information about Scikit-learn, see https://scikit-learn.org/stable/

For more information about Scikit-learn SageMaker ``Estimators``, see `Using Scikit-learn with the SageMaker Python SDK`_.

.. _Using Scikit-learn with the SageMaker Python SDK: https://sagemaker.readthedocs.io/en/stable/using_sklearn.html


SageMaker Reinforcement Learning Estimators
-------------------------------------------

With Reinforcement Learning (RL) Estimators, you can use reinforcement learning to train models on Amazon SageMaker.

Supported versions of Coach: ``0.10.1`` with TensorFlow, ``0.11.0`` with TensorFlow or MXNet.
For more information about Coach, see https://github.com/NervanaSystems/coach

Supported versions of Ray: ``0.5.3`` with TensorFlow.
For more information about Ray, see https://github.com/ray-project/ray

For more information about SageMaker RL ``Estimators``, see `SageMaker Reinforcement Learning Estimators`_.

.. _SageMaker Reinforcement Learning Estimators: src/sagemaker/rl/README.rst


SageMaker SparkML Serving
-------------------------

With SageMaker SparkML Serving, you can now perform predictions against a SparkML Model in SageMaker.
In order to host a SparkML model in SageMaker, it should be serialized with ``MLeap`` library.

For more information on MLeap, see https://github.com/combust/mleap .

Supported major version of Spark: 2.2 (MLeap version - 0.9.6)

Here is an example on how to create an instance of  ``SparkMLModel`` class and use ``deploy()`` method to create an
endpoint which can be used to perform prediction against your trained SparkML Model.

.. code:: python

    sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
    model_name = 'sparkml-model'
    endpoint_name = 'sparkml-endpoint'
    predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

Once the model is deployed, we can invoke the endpoint with a ``CSV`` payload like this:

.. code:: python

    payload = 'field_1,field_2,field_3,field_4,field_5'
    predictor.predict(payload)


For more information about the different ``content-type`` and ``Accept`` formats as well as the structure of the
``schema`` that SageMaker SparkML Serving recognizes, please see `SageMaker SparkML Serving Container`_.

.. _SageMaker SparkML Serving Container: https://github.com/aws/sagemaker-sparkml-serving-container

AWS SageMaker Estimators
------------------------
Amazon SageMaker provides several built-in machine learning algorithms that you can use to solve a variety of problems.

The full list of algorithms is available at: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

The SageMaker Python SDK includes estimator wrappers for the AWS K-means, Principal Components Analysis (PCA), Linear Learner, Factorization Machines,
Latent Dirichlet Allocation (LDA), Neural Topic Model (NTM), Random Cut Forest, k-nearest neighbors (k-NN), Object2Vec, and IP Insights algorithms.

For more information, see `AWS SageMaker Estimators and Models`_.

.. _AWS SageMaker Estimators and Models: src/sagemaker/amazon/README.rst

SageMaker Workflow
------------------

You can use Apache Airflow to author, schedule and monitor SageMaker workflow.

For more information, see `SageMaker Workflow in Apache Airflow`_.

.. _SageMaker Workflow in Apache Airflow: https://sagemaker.readthedocs.io/en/stable/using_workflow.html
