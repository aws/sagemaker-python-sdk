.. image:: https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png
    :height: 100px
    :alt: SageMaker

====================
SageMaker Python SDK
====================

.. image:: https://img.shields.io/pypi/v/sagemaker.svg
   :target: https://pypi.python.org/pypi/sagemaker
   :alt: Latest Version

.. image:: https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg
   :target: https://anaconda.org/conda-forge/sagemaker-python-sdk
   :alt: Conda-Forge Version

.. image:: https://img.shields.io/pypi/pyversions/sagemaker.svg
   :target: https://pypi.python.org/pypi/sagemaker
   :alt: Supported Python Versions

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

.. image:: https://readthedocs.org/projects/sagemaker/badge/?version=stable
   :target: https://sagemaker.readthedocs.io/en/stable/
   :alt: Documentation Status

.. image:: https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg
    :target: https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml
    :alt: CI Health

SageMaker Python SDK is an open source library for training and deploying machine learning models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks **Apache MXNet** and **TensorFlow**.
You can also train and deploy models with **Amazon algorithms**,
which are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training.
If you have **your own algorithms** built into SageMaker compatible Docker containers, you can train and host models using these as well.

For detailed documentation, including the API reference, see `Read the Docs <https://sagemaker.readthedocs.io>`_.

Table of Contents
-----------------

#. `Installing SageMaker Python SDK <#installing-the-sagemaker-python-sdk-train>`__
#. `Using the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html>`__


Installing the SageMaker Python SDK Train
-----------------------------------------

You can install from source by cloning this repository and running a pip install command in the root directory of the repository:

::

    git clone https://github.com/aws/sagemaker-python-sdk-staging.git
    cd sagemaker-python-sdk-staging/sagemaker_train
    pip install .

Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Python SDK supports Unix/Linux and Mac.

Supported Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Python SDK is tested on:

- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

Telemetry
~~~~~~~~~~~~~~~

The ``sagemaker`` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the ``TelemetryOptOut`` parameter to ``true`` in the SDK defaults configuration. For detailed instructions, please visit `Configuring and using defaults with the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>`__.

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

**Integration tests**

To run the integration tests, the following prerequisites must be met

1. AWS account credentials are available in the environment for the boto3 client to use.
2. The AWS account has an IAM role named :code:`SageMakerRole`.
   It should have the AmazonSageMakerFullAccess policy attached as well as a policy with `the necessary permissions to use Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei-setup.html>`__.
3. To run remote_function tests, dummy ecr repo should be created. It can be created by running -
    :code:`aws ecr create-repository --repository-name remote-function-dummy-container`

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
