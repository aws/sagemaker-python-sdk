.. image:: branding/icon/sagemaker-banner.png
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

With the SDK, you can train and deploy models using popular deep learning frameworks: **Apache MXNet** and **TensorFlow**. You can also train and deploy models with **Amazon algorithms**, these are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training. If you have **your own algorithms** built into SageMaker compatible Docker containers, you can train and host models using these as well.

For detailed API reference please go to: `Read the Docs <https://readthedocs.org/projects/sagemaker/>`_

Table of Contents
-----------------

1. `Getting SageMaker Python SDK <#getting-sagemaker-python-sdk>`__
2. `SageMaker Python SDK Overview <#sagemaker-python-sdk-overview>`__
3. `MXNet SageMaker Estimators <#mxnet-sagemaker-estimators>`__
4. `TensorFlow SageMaker Estimators <#tensorflow-sagemaker-estimators>`__
5. `Chainer SageMaker Estimators <#chainer-sagemaker-estimators>`__
6. `AWS SageMaker Estimators <#aws-sagemaker-estimators>`__
7. `BYO Docker Containers with SageMaker Estimators <#byo-docker-containers-with-sagemaker-estimators>`__
8. `SageMaker Automatic Model Tuning <#sagemaker-automatic-model-tuning>`__
9. `BYO Model <#byo-model>`__


Getting SageMaker Python SDK
----------------------------

SageMaker Python SDK is built to PyPI and can be installed with pip.

::

    pip install sagemaker

You can install from source by cloning this repository and issuing a pip install command in the root directory of the repository.

::

    git clone https://github.com/aws/sagemaker-python-sdk.git
    python setup.py sdist
    pip install dist/sagemaker-1.5.0.tar.gz

Supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Python SDK is tested on: \* Python 2.7 \* Python 3.5

Licensing
~~~~~~~~~
SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

Running tests
~~~~~~~~~~~~~

SageMaker Python SDK uses tox for running Python tests. You can run the tests by running tox:

::

    tox

Tests are defined in ``tests/`` and includes unit and integ tests. If you just want to run unit tests, then you can issue:

::

    tox tests/unit

To just run integ tests, issue the following command:

::

    pytest tests/integ

You can also filter by individual test function names (usable with any of the previous commands):

::

    pytest -k 'test_i_care_about'

Building Sphinx docs
~~~~~~~~~~~~~~~~~~~~

``cd`` into the ``doc`` directory and run:

::

    make html

You can edit the templates for any of the pages in the docs by editing the .rst files in the "doc" directory and then running "``make html``" again.


SageMaker Python SDK Overview
-----------------------------

SageMaker Python SDK provides several high-level abstractions for working with Amazon SageMaker. These are:

- **Estimators**: Encapsulate training on SageMaker. Can be ``fit()`` to run training, then the resulting model ``deploy()`` ed to a SageMaker Endpoint.
- **Models**: Encapsulate built ML models. Can be ``deploy()`` ed to a SageMaker Endpoint.
- **Predictors**: Provide real-time inference and transformation using Python data-types against a SageMaker Endpoint.
- **Session**: Provides a collection of convenience methods for working with SageMaker resources.

Estimator and Model implementations for MXNet, TensorFlow, and Amazon ML algorithms are included. There's also an Estimator that runs SageMaker compatible custom Docker containers, allowing you to run your own ML algorithms via SageMaker Python SDK.

Later sections of this document explain how to use the different Estimators and Models. These are:

* `MXNet SageMaker Estimators and Models <#mxnet-sagemaker-estimators>`__
* `TensorFlow SageMaker Estimators and Models <#tensorflow-sagemaker-estimators>`__
* `Chainer SageMaker Estimators and Models <#chainer-sagemaker-estimators>`__
* `AWS SageMaker Estimators and Models <#aws-sagemaker-estimators>`__
* `Custom SageMaker Estimators and Models <#byo-docker-containers-with-sagemaker-estimators>`__


Estimator Usage
---------------

Here is an end to end example of how to use a SageMaker Estimator.

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator (no training happens yet)
    mxnet_estimator = MXNet('train.py',
                            train_instance_type='ml.p2.xlarge',
                            train_instance_count = 1)

    # Starts a SageMaker training job and waits until completion.
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploys the model that was generated by fit() to a SageMaker Endpoint
    mxnet_predictor = mxnet_estimator.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge')

    # Serializes data and makes a prediction request to the SageMaker Endpoint
    response = predictor.predict(data)

    # Tears down the SageMaker Endpoint
    mxnet_estimator.delete_endpoint()

Local Mode
~~~~~~~~~~

The SageMaker Python SDK now supports local mode, which allows you to create TensorFlow, MXNet and BYO estimators and
deploy to your local environment. This is a great way to test your deep learning script before running in
SageMaker's managed training or hosting environments.

We can take the example in  `Estimator Usage <#estimator-usage>`__ , and use either ``local`` or ``local_gpu`` as the
instance type.

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator (no training happens yet)
    mxnet_estimator = MXNet('train.py',
                            train_instance_type='local',
                            train_instance_count=1)

    # In Local Mode, fit will pull the MXNet container docker image and run it locally
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Alternatively, you can train using data in your local file system. This is only supported in Local mode.
    mxnet_estimator.fit('file:///tmp/my_training_data')

    # Deploys the model that was generated by fit() to local endpoint in a container
    mxnet_predictor = mxnet_estimator.deploy(initial_instance_count=1, instance_type='local')

    # Serializes data and makes a prediction request to the local endpoint
    response = predictor.predict(data)

    # Tears down the endpoint container
    mxnet_estimator.delete_endpoint()


For detailed examples of running docker in local mode, see:

- `TensorFlow local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb>`__.
- `MXNet local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_mnist/mnist_with_gluon_local_mode.ipynb>`__.

A few important notes:

- Only one local mode endpoint can be running at a time
- If you are using s3 data as input, it will be pulled from S3 to your local environment, please ensure you have sufficient space.
- If you run into problems, this is often due to different docker containers conflicting. Killing these containers and re-running often solves your problems.
- Local Mode requires docker-compose and `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__ for ``local_gpu``.
- Distributed training is not yet supported for ``local_gpu``.


MXNet SageMaker Estimators
--------------------------

With MXNet Estimators, you can train and host MXNet models on Amazon SageMaker.

Supported versions of MXNet: ``1.1.0``, ``1.0.0``, ``0.12.1``.

More details at `MXNet SageMaker Estimators and Models`_.

.. _MXNet SageMaker Estimators and Models: src/sagemaker/mxnet/README.rst


TensorFlow SageMaker Estimators
-------------------------------

TensorFlow SageMaker Estimators allow you to run your own TensorFlow
training algorithms on SageMaker Learner, and to host your own TensorFlow
models on SageMaker Hosting.

Supported versions of TensorFlow: ``1.4.1``, ``1.5.0``, ``1.6.0``, ``1.7.0``, ``1.8.0``.

More details at `TensorFlow SageMaker Estimators and Models`_.

.. _TensorFlow SageMaker Estimators and Models: src/sagemaker/tensorflow/README.rst


Chainer SageMaker Estimators
-------------------------------

With Chainer Estimators, you can train and host Chainer models on Amazon SageMaker.

Supported versions of Chainer: ``4.0.0``

You can visit the Chainer repository at https://github.com/chainer/chainer.

More details at `Chainer SageMaker Estimators and Models`_.

.. _Chainer SageMaker Estimators and Models: src/sagemaker/chainer/README.rst


PyTorch SageMaker Estimators
-------------------------------

With PyTorch Estimators, you can train and host PyTorch models on Amazon SageMaker.

Supported versions of PyTorch: ``0.4.0``

You can visit the PyTorch repository at https://github.com/pytorch/pytorch.

More details at `PyTorch SageMaker Estimators and Models`_.

.. _PyTorch SageMaker Estimators and Models: src/sagemaker/pytorch/README.rst


AWS SageMaker Estimators
------------------------
Amazon SageMaker provides several built-in machine learning algorithms that you can use for a variety of problem types.

The full list of algorithms is available on the AWS website: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

SageMaker Python SDK includes Estimator wrappers for the AWS K-means, Principal Components Analysis(PCA), Linear Learner, Factorization Machines, Latent Dirichlet Allocation(LDA), Neural Topic Model(NTM) and Random Cut Forest algorithms.

More details at `AWS SageMaker Estimators and Models`_.

.. _AWS SageMaker Estimators and Models: src/sagemaker/amazon/README.rst


BYO Docker Containers with SageMaker Estimators
-----------------------------------------------

When you want to use a Docker image prepared earlier and use SageMaker SDK for training the easiest way is to use dedicated ``Estimator`` class. You will be able to instantiate it with desired image and use it in same way as described in previous sections.

Please refer to the full example in the examples repo:

::

    git clone https://github.com/awslabs/amazon-sagemaker-examples.git


The example notebook is is located here:
``advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb``


SageMaker Automatic Model Tuning
--------------------------------

All of the estimators can be used with SageMaker Automatic Model Tuning, which performs hyperparameter tuning jobs.
A hyperparameter tuning job runs multiple training jobs that differ by the values of their hyperparameters to find the best training job.
It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose.
If you're not using an Amazon ML algorithm, then the metric is defined by a regular expression (regex) you provide for going through the training job's logs.
You can read more about SageMaker Automatic Model Tuning in the `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`__.

The SageMaker Python SDK contains a ``HyperparameterTuner`` class for creating and interacting with hyperparameter training jobs.
Here is a basic example of how to use it:

.. code:: python

    from sagemaker.tuner import HyperparameterTuner, ContinuousParameter

    # Configure HyperparameterTuner
    my_tuner = HyperparameterTuner(estimator=my_estimator,  # previously-configured Estimator object
                                   objective_metric_name='validation-accuracy',
                                   hyperparameter_ranges={'learning-rate': ContinuousParameter(0.05, 0.06)},
                                   metric_definitions=[{'Name': 'validation-accuracy', 'Regex': 'validation-accuracy=(\d\.\d+)'}],
                                   max_jobs=100,
                                   max_parallel_jobs=10)

    # Start hyperparameter tuning job
    my_tuner.fit({'train': 's3://my_bucket/my_training_data', 'test': 's3://my_bucket_my_testing_data'})

    # Deploy best model
    my_predictor = my_tuner.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

    # Make a prediction against the SageMaker endpoint
    response = my_predictor.predict(my_prediction_data)

    # Tear down the SageMaker endpoint
    my_tuner.delete_endpoint()

This example shows a hyperparameter tuning job that creates up to 100 training jobs, running up to 10 at a time.
Each training job's learning rate will be a value between 0.05 and 0.06, but this value will differ between training jobs.
You can read more about how these values are chosen in the `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html>`__.

A hyperparameter range can be one of three types: continuous, integer, or categorical.
The SageMaker Python SDK provides corresponding classes for defining these different types.
You can define up to 20 hyperparameters to search over, but each value of a categorical hyperparameter range counts against that limit.

If you are using an Amazon ML algorithm, you don't need to pass in anything for ``metric_definitions``.
In addition, the ``fit()`` call uses a list of ``RecordSet`` objects instead of a dictionary:

.. code:: python

    # Create RecordSet object for each data channel
    train_records = RecordSet(...)
    test_records = RecordSet(...)

    # Start hyperparameter tuning job
    my_tuner.fit([train_records, test_records])

To aid with attaching a previously-started hyperparameter tuning job with a ``HyperparameterTuner`` instance, ``fit()`` injects metadata in the hyperparameters by default.
If the algorithm you are using cannot handle unknown hyperparameters (e.g. an Amazon ML algorithm that does not have a custom estimator in the Python SDK), then you can set ``include_cls_metadata`` to ``False`` when calling fit:

.. code:: python

    my_tuner.fit({'train': 's3://my_bucket/my_training_data', 'test': 's3://my_bucket_my_testing_data'},
                 include_cls_metadata=False)

There is also an analytics object associated with each ``HyperparameterTuner`` instance that presents useful information about the hyperparameter tuning job.
For example, the ``dataframe`` method gets a pandas dataframe summarizing the associated training jobs:

.. code:: python

    # Retrieve analytics object
    my_tuner_analytics = my_tuner.analytics()

    # Look at summary of associated training jobs
    my_dataframe = my_tuner_analytics.dataframe()

For more detailed examples of running hyperparameter tuning jobs, see:

- `Using the TensorFlow estimator with hyperparameter tuning <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/tensorflow_mnist/hpo_tensorflow_mnist.ipynb>`__
- `Bringing your own estimator for hyperparameter tuning <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/r_bring_your_own/hpo_r_bring_your_own.ipynb>`__
- `Analyzing results <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb>`__

For more detailed explanations of the classes that this library provides for automatic model tuning, see:

- `API docs for HyperparameterTuner and parameter range classes <https://sagemaker.readthedocs.io/en/latest/tuner.html>`__
- `API docs for analytics classes <https://sagemaker.readthedocs.io/en/latest/analytics.html>`__


FAQ
---

I want to train a SageMaker Estimator with local data, how do I do this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You'll need to upload the data to S3 before training. You can use the AWS Command Line Tool (the aws cli) to achieve this.

If you don't have the aws cli, you can install it using pip:

::

    pip install awscli --upgrade --user

If you don't have pip or want to learn more about installing the aws cli, please refer to the official `Amazon aws cli installation guide <http://docs.aws.amazon.com/cli/latest/userguide/installing.html>`__.

Once you have the aws cli installed, you can upload a directory of files to S3 with the following command:

::

    aws s3 cp /tmp/foo/ s3://bucket/path

You can read more about using the aws cli for manipulating S3 resources in the `AWS cli command reference <http://docs.aws.amazon.com/cli/latest/reference/s3/index.html>`__.


How do I make predictions against an existing endpoint?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a Predictor object and provide it your endpoint name. Then, simply call its predict() method with your input.

You can either use the generic RealTimePredictor class, which by default does not perform any serialization/deserialization transformations on your input, but can be configured to do so through constructor arguments:
http://sagemaker.readthedocs.io/en/latest/predictors.html

Or you can use the TensorFlow / MXNet specific predictor classes, which have default serialization/deserialization logic:
http://sagemaker.readthedocs.io/en/latest/sagemaker.tensorflow.html#tensorflow-predictor
http://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-predictor

Example code using the TensorFlow predictor:

::

    from sagemaker.tensorflow import TensorFlowPredictor

    predictor = TensorFlowPredictor('myexistingendpoint')
    result = predictor.predict(['my request body'])


BYO Model
-----------------------------------------------
You can also create an endpoint from an existing model rather than training one - i.e. bring your own model.

First, package the files for the trained model into a ``.tar.gz`` file, and upload the archive to S3.

Next, create a ``Model`` object that corresponds to the framework that you are using: `MXNetModel <https://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-model>`__ or `TensorFlowModel <https://sagemaker.readthedocs.io/en/latest/sagemaker.tensorflow.html#tensorflow-model>`__.

Example code using ``MXNetModel``:

.. code:: python

   from sagemaker.mxnet.model import MXNetModel

   sagemaker_model = MXNetModel(model_data='s3://path/to/model.tar.gz',
                                role='arn:aws:iam::accid:sagemaker-role',
                                entry_point='entry_point.py')

After that, invoke the ``deploy()`` method on the ``Model``:

.. code:: python

   predictor = sagemaker_model.deploy(initial_instance_count=1,
                                      instance_type='ml.m4.xlarge')

This returns a predictor the same way an ``Estimator`` does when ``deploy()`` is called. You can now get inferences just like with any other model deployed on Amazon SageMaker.

A full example is available in the `Amazon SageMaker examples repository <https://github.com/ragavvenkatesan/amazon-sagemaker-examples/tree/3c8394f21ee357da0b553b0ab024c5c5e425182a/advanced_functionality/mxnet_mnist_byom>`__.
