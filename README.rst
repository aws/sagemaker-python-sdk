.. image:: https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png
    :height: 100px
    :alt: SageMaker

====================
SageMaker Python SDK
====================

.. image:: https://img.shields.io/pypi/v/sagemaker.svg
   :target: https://pypi.python.org/pypi/sagemaker
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/sagemaker.svg
   :target: https://pypi.python.org/pypi/sagemaker
   :alt: Supported Python Versions

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

.. image:: https://readthedocs.org/projects/sagemaker/badge/?version=stable
   :target: https://sagemaker.readthedocs.io/en/stable/
   :alt: Documentation Status

SageMaker Python SDK is an open source library for training and deploying machine learning models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks **Apache MXNet** and **TensorFlow**.
You can also train and deploy models with **Amazon algorithms**,
which are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training.
If you have **your own algorithms** built into SageMaker compatible Docker containers, you can train and host models using these as well.

For detailed documentation, including the API reference, see `Read the Docs <https://sagemaker.readthedocs.io>`_.

Table of Contents
-----------------

#. `Installing SageMaker Python SDK <#installing-the-sagemaker-python-sdk>`__
#. `Using the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html>`__
#. `Using MXNet <https://sagemaker.readthedocs.io/en/stable/using_mxnet.html>`__
#. `Using TensorFlow <https://sagemaker.readthedocs.io/en/stable/using_tf.html>`__
#. `Using Chainer <https://sagemaker.readthedocs.io/en/stable/using_chainer.html>`__
#. `Using PyTorch <https://sagemaker.readthedocs.io/en/stable/using_pytorch.html>`__
#. `Using Scikit-learn <https://sagemaker.readthedocs.io/en/stable/using_sklearn.html>`__
#. `Using XGBoost <https://sagemaker.readthedocs.io/en/stable/using_xgboost.html>`__
#. `SageMaker Reinforcement Learning Estimators <https://sagemaker.readthedocs.io/en/stable/using_rl.html>`__
#. `SageMaker SparkML Serving <#sagemaker-sparkml-serving>`__
#. `Amazon SageMaker Built-in Algorithm Estimators <src/sagemaker/amazon/README.rst>`__
#. `Using SageMaker AlgorithmEstimators <https://sagemaker.readthedocs.io/en/stable/overview.html#using-sagemaker-algorithmestimators>`__
#. `Consuming SageMaker Model Packages <https://sagemaker.readthedocs.io/en/stable/overview.html#consuming-sagemaker-model-packages>`__
#. `BYO Docker Containers with SageMaker Estimators <https://sagemaker.readthedocs.io/en/stable/overview.html#byo-docker-containers-with-sagemaker-estimators>`__
#. `SageMaker Automatic Model Tuning <https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning>`__
#. `SageMaker Batch Transform <https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform>`__
#. `Secure Training and Inference with VPC <https://sagemaker.readthedocs.io/en/stable/overview.html#secure-training-and-inference-with-vpc>`__
#. `BYO Model <https://sagemaker.readthedocs.io/en/stable/overview.html#byo-model>`__
#. `Inference Pipelines <https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines>`__
#. `Amazon SageMaker Operators in Apache Airflow <https://sagemaker.readthedocs.io/en/stable/using_workflow.html>`__
#. `SageMaker Autopilot <src/sagemaker/automl/README.rst>`__
#. `Model Monitoring <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html>`__
#. `SageMaker Debugger <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html>`__
#. `SageMaker Processing <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html>`__


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

- Python 3.6
- Python 3.7
- Python 3.8

AWS Permissions
~~~~~~~~~~~~~~~

As a managed service, Amazon SageMaker performs operations on your behalf on the AWS hardware that is managed by Amazon SageMaker.
Amazon SageMaker can perform only operations that the user permits.
You can read more about which permissions are necessary in the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.

The SageMaker Python SDK should not require any additional permissions aside from what is required for using SageMaker.
However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

Licensing
~~~~~~~~~
SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

Running tests
~~~~~~~~~~~~~

SageMaker Python SDK has unit tests and integration tests.

You can install the libraries needed to run the tests by running :code:`pip install --upgrade .[test]` or, for Zsh users: :code:`pip install --upgrade .\[test\]`

**Unit tests**

We run unit tests with tox, which is a program that lets you run unit tests for multiple Python versions, and also make sure the
code fits our style guidelines. We run tox with `all of our supported Python versions <#supported-python-versions>`_, so to run unit tests
with the same configuration we do, you need to have interpreters for those Python versions installed.

To run the unit tests with tox, run:

::

    tox tests/unit

**Integrations tests**

To run the integration tests, the following prerequisites must be met

1. AWS account credentials are available in the environment for the boto3 client to use.
2. The AWS account has an IAM role named :code:`SageMakerRole`.
   It should have the AmazonSageMakerFullAccess policy attached as well as a policy with `the necessary permissions to use Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html>`__.

We recommend selectively running just those integration tests you'd like to run. You can filter by individual test function names with:

::

    tox -- -k 'test_i_care_about'


You can also run all of the integration tests by running the following command, which runs them in sequence, which may take a while:

::

    tox -- tests/integ


You can also run them in parallel:

::

    tox -- -n auto tests/integ


Git Hooks
~~~~~~~~~

to enable all git hooks in the .githooks directory, run these commands in the repository directory:

::

    find .git/hooks -type l -exec rm {} \;
    find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;

To enable an individual git hook, simply move it from the .githooks/ directory to the .git/hooks/ directory.

Building Sphinx docs
~~~~~~~~~~~~~~~~~~~~

Setup a Python environment, and install the dependencies listed in ``doc/requirements.txt``:

::

    # conda
    conda create -n sagemaker python=3.7
    conda activate sagemaker
    conda install sphinx=3.1.1 sphinx_rtd_theme=0.5.0

    # pip
    pip install -r doc/requirements.txt


Clone/fork the repo, and install your local version:

::

    pip install --upgrade .

Then ``cd`` into the ``sagemaker-python-sdk/doc`` directory and run:

::

    make html

You can edit the templates for any of the pages in the docs by editing the .rst files in the ``doc`` directory and then running ``make html`` again.

Preview the site with a Python web server:

::

    cd _build/html
    python -m http.server 8000

View the website by visiting http://localhost:8000

SageMaker SparkML Serving
-------------------------

With SageMaker SparkML Serving, you can now perform predictions against a SparkML Model in SageMaker.
In order to host a SparkML model in SageMaker, it should be serialized with ``MLeap`` library.

For more information on MLeap, see https://github.com/combust/mleap .

Supported major version of Spark: 2.4 (MLeap version - 0.9.6)

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
