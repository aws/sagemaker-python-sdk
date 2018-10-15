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

For detailed API reference please go to: `Read the Docs <https://readthedocs.org/projects/sagemaker/>`_

Table of Contents
-----------------

1. `Installing SageMaker Python SDK <#installing-the-sagemaker-python-sdk>`__
2. `SageMaker Python SDK Overview <#sagemaker-python-sdk-overview>`__
3. `MXNet SageMaker Estimators <#mxnet-sagemaker-estimators>`__
4. `TensorFlow SageMaker Estimators <#tensorflow-sagemaker-estimators>`__
5. `Chainer SageMaker Estimators <#chainer-sagemaker-estimators>`__
6. `PyTorch SageMaker Estimators <#pytorch-sagemaker-estimators>`__
7. `AWS SageMaker Estimators <#aws-sagemaker-estimators>`__
8. `BYO Docker Containers with SageMaker Estimators <#byo-docker-containers-with-sagemaker-estimators>`__
9. `SageMaker Automatic Model Tuning <#sagemaker-automatic-model-tuning>`__
10. `SageMaker Batch Transform <#sagemaker-batch-transform>`__
11. `Secure Training and Inference with VPC <#secure-training-and-inference-with-vpc>`__
12. `BYO Model <#byo-model>`__


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

Supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Python SDK is tested on: \* Python 2.7 \* Python 3.5

Licensing
~~~~~~~~~
SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

Running tests
~~~~~~~~~~~~~

SageMaker Python SDK has unit tests and integration tests.

**Unit tests**

tox is a prerequisite for running unit tests so you need to make sure you have it installed. To run the unit tests:

::

    tox tests/unit

**Integrations tests**

To run the integration tests, the following prerequisites must be met

1. Access to an AWS account to run the tests on
2. AWS account credentials available to boto3 clients used in the tests
3. The AWS account has an IAM role named :code:`SageMakerRole`
4. The libraries listed in the ``extras_require`` object in ``setup.py`` for ``test`` are installed.
   You can do this by running the following command: :code:`pip install --upgrade .[test]`

You can run integ tests by issuing the following command:

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

- **Estimators**: Encapsulate training on SageMaker.
- **Models**: Encapsulate built ML models.
- **Predictors**: Provide real-time inference and transformation using Python data-types against a SageMaker endpoint.
- **Session**: Provides a collection of methods for working with SageMaker resources.

``Estimator`` and ``Model`` implementations for MXNet, TensorFlow, Chainer, PyTorch, and Amazon ML algorithms are included.
There's also an ``Estimator`` that runs SageMaker compatible custom Docker containers, enabling you to run your own ML algorithms by using the SageMaker Python SDK.

The following sections of this document explain how to use the different estimators and models:

* `MXNet SageMaker Estimators and Models <#mxnet-sagemaker-estimators>`__
* `TensorFlow SageMaker Estimators and Models <#tensorflow-sagemaker-estimators>`__
* `Chainer SageMaker Estimators and Models <#chainer-sagemaker-estimators>`__
* `PyTorch SageMaker Estimators <#pytorch-sagemaker-estimators>`__
* `AWS SageMaker Estimators and Models <#aws-sagemaker-estimators>`__
* `Custom SageMaker Estimators and Models <#byo-docker-containers-with-sagemaker-estimators>`__


Using Estimators
----------------

Here is an end to end example of how to use a SageMaker Estimator:

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator (no training happens yet)
    mxnet_estimator = MXNet('train.py',
                            role='SageMakerRole',
                            train_instance_type='ml.p2.xlarge',
                            train_instance_count=1,
                            framework_version='1.2.1')

    # Starts a SageMaker training job and waits until completion.
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploys the model that was generated by fit() to a SageMaker endpoint
    mxnet_predictor = mxnet_estimator.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge')

    # Serializes data and makes a prediction request to the SageMaker endpoint
    response = mxnet_predictor.predict(data)

    # Tears down the SageMaker endpoint
    mxnet_estimator.delete_endpoint()

Local Mode
~~~~~~~~~~

The SageMaker Python SDK supports local mode, which allows you to create estimators and deploy them to your local environment.
This is a great way to test your deep learning scripts before running them in SageMaker's managed training or hosting environments.

We can take the example in  `Using Estimators <#using-estimators>`__ , and use either ``local`` or ``local_gpu`` as the instance type.

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator (no training happens yet)
    mxnet_estimator = MXNet('train.py',
                            role='SageMakerRole',
                            train_instance_type='local',
                            train_instance_count=1,
                            framework_version='1.2.1')

    # In Local Mode, fit will pull the MXNet container Docker image and run it locally
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Alternatively, you can train using data in your local file system. This is only supported in Local mode.
    mxnet_estimator.fit('file:///tmp/my_training_data')

    # Deploys the model that was generated by fit() to local endpoint in a container
    mxnet_predictor = mxnet_estimator.deploy(initial_instance_count=1, instance_type='local')

    # Serializes data and makes a prediction request to the local endpoint
    response = mxnet_predictor.predict(data)

    # Tears down the endpoint container
    mxnet_estimator.delete_endpoint()


If you have an existing model and want to deploy it locally, don't specify a sagemaker_session argument to the ``MXNetModel`` constructor.
The correct session is generated when you call ``model.deploy()``.

Here is an end-to-end example:

.. code:: python

    import numpy
    from sagemaker.mxnet import MXNetModel

    model_location = 's3://mybucket/my_model.tar.gz'
    code_location = 's3://mybucket/sourcedir.tar.gz'
    s3_model = MXNetModel(model_data=model_location, role='SageMakerRole',
                          entry_point='mnist.py', source_dir=code_location)

    predictor = s3_model.deploy(initial_instance_count=1, instance_type='local')
    data = numpy.zeros(shape=(1, 1, 28, 28))
    predictor.predict(data)

    # Tear down the endpoint container
    predictor.delete_endpoint()


If you don't want to deploy your model locally, you can also choose to perform a Local Batch Transform Job. This is
useful if you want to test your container before creating a Sagemaker Batch Transform Job. Note that the performance
will not match Batch Transform Jobs hosted on SageMaker but it is still a useful tool to ensure you have everything
right or if you are not dealing with huge amounts of data.

Here is an end-to-end example:

.. code:: python

    from sagemaker.mxnet import MXNet

    mxnet_estimator = MXNet('train.py',
                            train_instance_type='local',
                            train_instance_count=1,
                            framework_version='1.2.1')

    mxnet_estimator.fit('file:///tmp/my_training_data')
    transformer = mxnet_estimator.transformer(1, 'local', assemble_with='Line', max_payload=1)
    transformer.transform('s3://my/transform/data, content_type='text/csv', split_type='Line')
    transformer.wait()


For detailed examples of running Docker in local mode, see:

- `TensorFlow local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb>`__.
- `MXNet local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_mnist/mnist_with_gluon_local_mode.ipynb>`__.

A few important notes:

- Only one local mode endpoint can be running at a time.
- If you are using S3 data as input, it is pulled from S3 to your local environment. Ensure you have sufficient space to store the data locally.
- If you run into problems it often due to different Docker containers conflicting. Killing these containers and re-running often solves your problems.
- Local Mode requires Docker Compose and `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__ for ``local_gpu``.
- Distributed training is not yet supported for ``local_gpu``.


MXNet SageMaker Estimators
--------------------------

By using MXNet SageMaker ``Estimators``, you can train and host MXNet models on Amazon SageMaker.

Supported versions of MXNet: ``1.2.1``, ``1.1.0``, ``1.0.0``, ``0.12.1``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information, see `MXNet SageMaker Estimators and Models`_.

.. _MXNet SageMaker Estimators and Models: src/sagemaker/mxnet/README.rst


TensorFlow SageMaker Estimators
-------------------------------

By using TensorFlow SageMaker ``Estimators``, you can train and host TensorFlow models on Amazon SageMaker.

Supported versions of TensorFlow: ``1.4.1``, ``1.5.0``, ``1.6.0``, ``1.7.0``, ``1.8.0``, ``1.9.0``, ``1.10.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information, see `TensorFlow SageMaker Estimators and Models`_.

.. _TensorFlow SageMaker Estimators and Models: src/sagemaker/tensorflow/README.rst


Chainer SageMaker Estimators
-------------------------------

By using Chainer SageMaker ``Estimators``, you can train and host Chainer models on Amazon SageMaker.

Supported versions of Chainer: ``4.0.0``, ``4.1.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information about Chainer, see https://github.com/chainer/chainer.

For more information about  Chainer SageMaker ``Estimators``, see `Chainer SageMaker Estimators and Models`_.

.. _Chainer SageMaker Estimators and Models: src/sagemaker/chainer/README.rst


PyTorch SageMaker Estimators
-------------------------------

With PyTorch SageMaker ``Estimators``, you can train and host PyTorch models on Amazon SageMaker.

Supported versions of PyTorch: ``0.4.0``, ``1.0.0.dev`` ("Preview").

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

You can try the "Preview" version of PyTorch by specifying ``'1.0.0.dev'`` for ``framework_version`` when creating your PyTorch estimator.
This will ensure you're using the latest version of ``torch-nightly``.

For more information about PyTorch, see https://github.com/pytorch/pytorch.

For more information about PyTorch SageMaker ``Estimators``, see `PyTorch SageMaker Estimators and Models`_.

.. _PyTorch SageMaker Estimators and Models: src/sagemaker/pytorch/README.rst


AWS SageMaker Estimators
------------------------
Amazon SageMaker provides several built-in machine learning algorithms that you can use to solve a variety of problems.

The full list of algorithms is available at: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

The SageMaker Python SDK includes estimator wrappers for the AWS K-means, Principal Components Analysis (PCA), Linear Learner, Factorization Machines,
Latent Dirichlet Allocation (LDA), Neural Topic Model (NTM) Random Cut Forest and k-nearest neighbors (k-NN) algorithms.

For more information, see `AWS SageMaker Estimators and Models`_.

.. _AWS SageMaker Estimators and Models: src/sagemaker/amazon/README.rst


BYO Docker Containers with SageMaker Estimators
-----------------------------------------------

To use a Docker image that you created and use the SageMaker SDK for training, the easiest way is to use the dedicated ``Estimator`` class.
You can create an instance of the ``Estimator`` class with desired Docker image and use it as described in previous sections.

Please refer to the full example in the examples repo:

::

    git clone https://github.com/awslabs/amazon-sagemaker-examples.git


The example notebook is is located here:
``advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb``


SageMaker Automatic Model Tuning
--------------------------------

All of the estimators can be used with SageMaker Automatic Model Tuning, which performs hyperparameter tuning jobs.
A hyperparameter tuning job finds the best version of a model by running many training jobs on your dataset using the algorithm with different values of hyperparameters within ranges
that you specify. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose.
If you're not using an Amazon SageMaker built-in algorithm, then the metric is defined by a regular expression (regex) you provide.
The hyperparameter tuning job parses the training job's logs to find metrics that match the regex you defined.
For more information about SageMaker Automatic Model Tuning, see `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`__.

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

This example shows a hyperparameter tuning job that creates up to 100 training jobs, running up to 10 training jobs at a time.
Each training job's learning rate is a value between 0.05 and 0.06, but this value will differ between training jobs.
You can read more about how these values are chosen in the `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html>`__.

A hyperparameter range can be one of three types: continuous, integer, or categorical.
The SageMaker Python SDK provides corresponding classes for defining these different types.
You can define up to 20 hyperparameters to search over, but each value of a categorical hyperparameter range counts against that limit.

If you are using an Amazon SageMaker built-in algorithm, you don't need to pass in anything for ``metric_definitions``.
In addition, the ``fit()`` call uses a list of ``RecordSet`` objects instead of a dictionary:

.. code:: python

    # Create RecordSet object for each data channel
    train_records = RecordSet(...)
    test_records = RecordSet(...)

    # Start hyperparameter tuning job
    my_tuner.fit([train_records, test_records])

To help attach a previously-started hyperparameter tuning job to a ``HyperparameterTuner`` instance,
``fit()`` adds the module path of the class used to create the tuner to the list of static hyperparameters by default.
If the algorithm you are using cannot handle unknown hyperparameters
(for example, an Amazon SageMaker built-in algorithm that does not have a custom estimator in the Python SDK),
set ``include_cls_metadata`` to ``False`` when you call ``fit``, so that it does not add the module path as a static hyperparameter:

.. code:: python

    my_tuner.fit({'train': 's3://my_bucket/my_training_data', 'test': 's3://my_bucket_my_testing_data'},
                 include_cls_metadata=False)

There is also an analytics object associated with each ``HyperparameterTuner`` instance that contains useful information about the hyperparameter tuning job.
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


SageMaker Batch Transform
-------------------------

After you train a model, you can use Amazon SageMaker Batch Transform to perform inferences with the model.
Batch Transform manages all necessary compute resources, including launching instances to deploy endpoints and deleting them afterward.
You can read more about SageMaker Batch Transform in the `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html>`__.

If you trained the model using a SageMaker Python SDK estimator,
you can invoke the estimator's ``transformer()`` method to create a transform job for a model based on the training job:

.. code:: python

    transformer = estimator.transformer(instance_count=1, instance_type='ml.m4.xlarge')

Alternatively, if you already have a SageMaker model, you can create an instance of the ``Transformer`` class by calling its constructor:

.. code:: python

    transformer = Transformer(model_name='my-previously-trained-model',
                              instance_count=1,
                              instance_type='ml.m4.xlarge')

For a full list of the possible options to configure by using either of these methods, see the API docs for `Estimator <https://sagemaker.readthedocs.io/en/latest/estimators.html#sagemaker.estimator.Estimator.transformer>`__ or `Transformer <https://sagemaker.readthedocs.io/en/latest/transformer.html#sagemaker.transformer.Transformer>`__.

After you create a ``Transformer`` object, you can invoke ``transform()`` to start a batch transform job with the S3 location of your data.
You can also specify other attributes of your data, such as the content type.

.. code:: python

    transformer.transform('s3://my-bucket/batch-transform-input')

For more details about what can be specified here, see `API docs <https://sagemaker.readthedocs.io/en/latest/transformer.html#sagemaker.transformer.Transformer.transform>`__.


Secure Training and Inference with VPC
--------------------------------------

Amazon SageMaker allows you to control network traffic to and from model container instances using Amazon Virtual Private Cloud (VPC).
You can configure SageMaker to use your own private VPC in order to further protect and monitor traffic.

For more information about Amazon SageMaker VPC features, and guidelines for configuring your VPC,
see the following documentation:

- `Protect Training Jobs by Using an Amazon Virtual Private Cloud <https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html>`__
- `Protect Endpoints by Using an Amazon Virtual Private Cloud <https://docs.aws.amazon.com/sagemaker/latest/dg/host-vpc.html>`__
- `Protect Data in Batch Transform Jobs by Using an Amazon Virtual Private Cloud <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-vpc.html>`__
- `Working with VPCs and Subnets <https://docs.aws.amazon.com/vpc/latest/userguide/working-with-vpcs.html>`__

You can also reference or reuse the example VPC created for integration tests: `tests/integ/vpc_test_utils.py <tests/integ/vpc_test_utils.py>`__

To train a model using your own VPC, set the optional parameters ``subnets`` and ``security_group_ids`` on an ``Estimator``:

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator with subnets and security groups from your VPC
    mxnet_vpc_estimator = MXNet('train.py',
                                train_instance_type='ml.p2.xlarge',
                                train_instance_count=1,
                                framework_version='1.2.1',
                                subnets=['subnet-1', 'subnet-2'],
                                security_group_ids=['sg-1'])

    # SageMaker Training Job will set VpcConfig and container instances will run in your VPC
    mxnet_vpc_estimator.fit('s3://my_bucket/my_training_data/')

When you create a ``Predictor`` from the ``Estimator`` using ``deploy()``, the same VPC configurations will be set on the SageMaker Model:

.. code:: python

    # Creates a SageMaker Model and Endpoint using the same VpcConfig
    # Endpoint container instances will run in your VPC
    mxnet_vpc_predictor = mxnet_vpc_estimator.deploy(initial_instance_count=1,
                                                     instance_type='ml.p2.xlarge')

    # You can also set ``vpc_config_override`` to use a different VpcConfig
    other_vpc_config = {'Subnets': ['subnet-3', 'subnet-4'],
                        'SecurityGroupIds': ['sg-2']}
    mxnet_predictor_other_vpc = mxnet_vpc_estimator.deploy(initial_instance_count=1,
                                                           instance_type='ml.p2.xlarge',
                                                           vpc_config_override=other_vpc_config)

    # Setting ``vpc_config_override=None`` will disable VpcConfig
    mxnet_predictor_no_vpc = mxnet_vpc_estimator.deploy(initial_instance_count=1,
                                                        instance_type='ml.p2.xlarge',
                                                        vpc_config_override=None)

Likewise, when you create ``Transformer`` from the ``Estimator`` using ``transformer()``, the same VPC configurations will be set on the SageMaker Model:

.. code:: python

    # Creates a SageMaker Model using the same VpcConfig
    mxnet_vpc_transformer = mxnet_vpc_estimator.transformer(instance_count=1,
                                                            instance_type='ml.p2.xlarge')

    # Transform Job container instances will run in your VPC
    mxnet_vpc_transformer.transform('s3://my-bucket/batch-transform-input')


FAQ
---

I want to train a SageMaker Estimator with local data, how do I do this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upload the data to S3 before training. You can use the AWS Command Line Tool (the aws cli) to achieve this.

If you don't have the aws cli, you can install it using pip:

::

    pip install awscli --upgrade --user

If you don't have pip or want to learn more about installing the aws cli, see the official `Amazon aws cli installation guide <http://docs.aws.amazon.com/cli/latest/userguide/installing.html>`__.

After you install the AWS cli, you can upload a directory of files to S3 with the following command:

::

    aws s3 cp /tmp/foo/ s3://bucket/path

For more information about using the aws cli for manipulating S3 resources, see `AWS cli command reference <http://docs.aws.amazon.com/cli/latest/reference/s3/index.html>`__.


How do I make predictions against an existing endpoint?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a ``Predictor`` object and provide it with your endpoint name,
then call its ``predict()`` method with your input.

You can use either the generic ``RealTimePredictor`` class, which by default does not perform any serialization/deserialization transformations on your input,
but can be configured to do so through constructor arguments:
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
---------
You can also create an endpoint from an existing model rather than training one.
That is, you can bring your own model:

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

A full example is available in the `Amazon SageMaker examples repository <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/mxnet_mnist_byom>`__.
