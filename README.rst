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

.. image:: https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg
    :target: https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml
    :alt: CI Health

SageMaker Python SDK is an open source library for training and deploying machine learning models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks **Apache MXNet** and **PyTorch**.
You can also train and deploy models with **Amazon algorithms**,
which are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training.
If you have **your own algorithms** built into SageMaker compatible Docker containers, you can train and host models using these as well.

To install SageMaker Python SDK, see `Installing SageMaker Python SDK <#installing-the-sagemaker-python-sdk>`_.

❗🔥 SageMaker V3 Release
-------------------------

Version 3.0.0 represents a significant milestone in our product's evolution. This major release introduces a modernized architecture, enhanced performance, and powerful new features while maintaining our commitment to user experience and reliability.

**Important: Please review these breaking changes before upgrading.**

* Older interfaces such as Estimator, Model, Predictor and all their subclasses will not be supported in V3. 
* Please see our `V3 examples folder <https://github.com/aws/sagemaker-python-sdk/tree/master/v3-examples>`__ for example notebooks and usage patterns.


Migrating to V3
----------------

**Upgrading to 3.x**

To upgrade to the latest version of SageMaker Python SDK 3.x:

::

    pip install --upgrade sagemaker

If you prefer to downgrade to the 2.x version:

::

    pip install sagemaker==2.*

See `SageMaker V2 Examples <#sagemaker-v2-examples>`__ for V2 documentation and examples.

**Key Benefits of 3.x**

* **Modular Architecture**: Separate PyPI packages for core, training, and serving capabilities

  * `sagemaker-core <https://pypi.org/project/sagemaker-core/>`__
  * `sagemaker-train <https://pypi.org/project/sagemaker-train/>`__
  * `sagemaker-serve <https://pypi.org/project/sagemaker-serve/>`__
  * `sagemaker-mlops <https://pypi.org/project/sagemaker-mlops/>`__

* **Unified Training & Inference**: Single classes (ModelTrainer, ModelBuilder) replace multiple framework-specific classes
* **Object-Oriented API**: Structured interface with auto-generated configs aligned with AWS APIs
* **Simplified Workflows**: Reduced boilerplate and more intuitive interfaces

**Training Experience**

V3 introduces the unified ModelTrainer class to reduce complexity of initial setup and deployment for model training. This replaces the V2 Estimator class and framework-specific classes (PyTorchEstimator, SKLearnEstimator, etc.).

This example shows how to train a model using a custom training container with training data from S3.

*SageMaker Python SDK 2.x:*

.. code:: python

    from sagemaker.estimator import Estimator
    estimator = Estimator(
        image_uri="my-training-image",
        role="arn:aws:iam::123456789012:role/SageMakerRole",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path="s3://my-bucket/output"
    )
    estimator.fit({"training": "s3://my-bucket/train"})

*SageMaker Python SDK 3.x:*

.. code:: python

    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import InputData

    trainer = ModelTrainer(
        training_image="my-training-image",
        role="arn:aws:iam::123456789012:role/SageMakerRole"
    )

    train_data = InputData(
        channel_name="training",
        data_source="s3://my-bucket/train"
    )

    trainer.train(input_data_config=[train_data])

**See more examples:** `SageMaker V3 Examples <#sagemaker-v3-examples>`__

**Inference Experience**

V3 introduces the unified ModelBuilder class for model deployment and inference. This replaces the V2 Model class and framework-specific classes (PyTorchModel, TensorFlowModel, SKLearnModel, XGBoostModel, etc.).

This example shows how to deploy a trained model for real-time inference.

*SageMaker Python SDK 2.x:*

.. code:: python

    from sagemaker.model import Model
    from sagemaker.predictor import Predictor
    model = Model(
        image_uri="my-inference-image",
        model_data="s3://my-bucket/model.tar.gz",
        role="arn:aws:iam::123456789012:role/SageMakerRole"
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge"
    )
    result = predictor.predict(data)

*SageMaker Python SDK 3.x:*

.. code:: python

    from sagemaker.serve import ModelBuilder
    model_builder = ModelBuilder(
        model="my-model",
        model_path="s3://my-bucket/model.tar.gz"
    )
    endpoint = model_builder.build()
    result = endpoint.invoke(...)

**See more examples:** `SageMaker V3 Examples <#sagemaker-v3-examples>`__

SageMaker V3 Examples
---------------------

**Training Examples**

#. `Custom Distributed Training Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/training-examples/custom-distributed-training-example.ipynb>`__
#. `Distributed Local Training Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/training-examples/distributed-local-training-example.ipynb>`__
#. `Hyperparameter Training Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/training-examples/hyperparameter-training-example.ipynb>`__
#. `JumpStart Training Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/training-examples/jumpstart-training-example.ipynb>`__
#. `Local Training Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/training-examples/local-training-example.ipynb>`__

**Inference Examples**

#. `HuggingFace Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/huggingface-example.ipynb>`__
#. `In-Process Mode Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/in-process-mode-example.ipynb>`__
#. `Inference Spec Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/inference-spec-example.ipynb>`__
#. `JumpStart E2E Training Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/jumpstart-e2e-training-example.ipynb>`__
#. `JumpStart Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/jumpstart-example.ipynb>`__
#. `Local Mode Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/local-mode-example.ipynb>`__
#. `Optimize Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/optimize-example.ipynb>`__
#. `Train Inference E2E Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/inference-examples/train-inference-e2e-example.ipynb>`__

**ML Ops Examples**

#. `V3 Hyperparameter Tuning Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-hyperparameter-tuning-example/v3-hyperparameter-tuning-example.ipynb>`__
#. `V3 Hyperparameter Tuning Pipeline <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-hyperparameter-tuning-example/v3-hyperparameter-tuning-pipeline.ipynb>`__
#. `V3 Model Registry Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-model-registry-example/v3-model-registry-example.ipynb>`__
#. `V3 PyTorch Processing Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-processing-job-pytorch/v3-pytorch-processing-example.ipynb>`__
#. `V3 Pipeline Train Create Registry <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-pipeline-train-create-registry.ipynb>`__
#. `V3 Processing Job Sklearn <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-processing-job-sklearn.ipynb>`__
#. `V3 SageMaker Clarify <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-sagemaker-clarify.ipynb>`__
#. `V3 Transform Job Example <https://github.com/aws/sagemaker-python-sdk/blob/master/v3-examples/ml-ops-examples/v3-transform-job-example.ipynb>`__

**Looking for V2 Examples?** See `SageMaker V2 Examples <#sagemaker-v2-examples>`__ below.




Installing the SageMaker Python SDK
-----------------------------------

The SageMaker Python SDK is built to PyPI and the latest version of the SageMaker Python SDK can be installed with pip as follows
::

    pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>

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

Building Sphinx docs
~~~~~~~~~~~~~~~~~~~~

Setup a Python environment, and install the dependencies listed in ``doc/requirements.txt``:

::

    # conda
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

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

Supported major version of Spark: 3.3 (MLeap version - 0.20.0)

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


SageMaker V2 Examples
---------------------

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

🚀 Model Fine-Tuning Support Now Available in V3
-------------------------------------------------

We're excited to announce model fine-tuning capabilities in SageMaker Python SDK V3!

**What's New**

Four new trainer classes for fine-tuning foundation models:

* SFTTrainer - Supervised fine-tuning
* DPOTrainer - Direct preference optimization  
* RLAIFTrainer - RL from AI feedback
* RLVRTrainer - RL from verifiable rewards

**Quick Example**

.. code:: python

    from sagemaker.train import SFTTrainer
    from sagemaker.train.common import TrainingType

    trainer = SFTTrainer(
        model="meta-llama/Llama-2-7b-hf",
        training_type=TrainingType.LORA,
        model_package_group_name="my-models",
        training_dataset="s3://bucket/train.jsonl"
    )

    training_job = trainer.train()

**Key Features**

* ✨ LoRA & full fine-tuning  
* 📊 MLflow integration with real-time metrics  
* 🚀 Deploy to SageMaker or Bedrock  
* 📈 Built-in evaluation (11 benchmarks)  
* ☁️ Serverless training  

**Get Started**

.. code:: python

    pip install sagemaker>=3.1.0

`📓 Example notebooks <https://github.com/aws/sagemaker-python-sdk/tree/master/v3-examples/model-customization-examples>`__