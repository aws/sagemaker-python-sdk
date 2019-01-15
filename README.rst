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
2. `SageMaker Python SDK Overview <#sagemaker-python-sdk-overview>`__
3. `MXNet SageMaker Estimators <#mxnet-sagemaker-estimators>`__
4. `TensorFlow SageMaker Estimators <#tensorflow-sagemaker-estimators>`__
5. `Chainer SageMaker Estimators <#chainer-sagemaker-estimators>`__
6. `PyTorch SageMaker Estimators <#pytorch-sagemaker-estimators>`__
7. `Scikit-learn SageMaker Estimators <#scikit-learn-sagemaker-estimators>`__
8. `SageMaker Reinforcement Learning Estimators <#sagemaker-reinforcement-learning-estimators>`__
9. `SageMaker SparkML Serving <#sagemaker-sparkml-serving>`__
10. `AWS SageMaker Estimators <#aws-sagemaker-estimators>`__
11. `Using SageMaker AlgorithmEstimators <#using-sagemaker-algorithmestimators>`__
12. `Consuming SageMaker Model Packages <#consuming-sagemaker-model-packages>`__
13. `BYO Docker Containers with SageMaker Estimators <#byo-docker-containers-with-sagemaker-estimators>`__
14. `SageMaker Automatic Model Tuning <#sagemaker-automatic-model-tuning>`__
15. `SageMaker Batch Transform <#sagemaker-batch-transform>`__
16. `Secure Training and Inference with VPC <#secure-training-and-inference-with-vpc>`__
17. `BYO Model <#byo-model>`__
18. `Inference Pipelines <#inference-pipelines>`__
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
- Python 3.5

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
* `Scikit-learn SageMaker Estimators and Models <#scikit-learn-sagemaker-estimators>`__
* `SageMaker Reinforcement Learning Estimators <#sagemaker-reinforcement-learning-estimators>`__
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

Training Metrics
~~~~~~~~~~~~~~~~
The SageMaker Python SDK allows you to specify a name and a regular expression for metrics you want to track for training.
A regular expression (regex) matches what is in the training algorithm logs, like a search function.
Here is an example of how to define metrics:

.. code:: python

    # Configure an BYO Estimator with metric definitions (no training happens yet)
    byo_estimator = Estimator(image_name=image_name,
                              role='SageMakerRole', train_instance_count=1,
                              train_instance_type='ml.c4.xlarge',
                              sagemaker_session=sagemaker_session,
                              metric_definitions=[{'Name': 'test:msd', 'Regex': '#quality_metric: host=\S+, test msd <loss>=(\S+)'},
                                                  {'Name': 'test:ssd', 'Regex': '#quality_metric: host=\S+, test ssd <loss>=(\S+)'}])

All Amazon SageMaker algorithms come with built-in support for metrics.
You can go to `the AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html>`__ for more details about built-in metrics of each Amazon SageMaker algorithm.

Local Mode
~~~~~~~~~~

The SageMaker Python SDK supports local mode, which allows you to create estimators and deploy them to your local environment.
This is a great way to test your deep learning scripts before running them in SageMaker's managed training or hosting environments.
Local Mode is supported for only frameworks (e.g. TensorFlow, MXNet) and images you supply yourself.

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

Incremental Training
~~~~~~~~~~~~~~~~~~~~

Incremental training allows you to bring a pre-trained model into a SageMaker training job and use it as a starting point for a new model.
There are several situations where you might want to do this:

- You want to perform additional training on a model to improve its fit on your data set.
- You want to import a pre-trained model and fit it to your data.
- You want to resume a training job that you previously stopped.

To use incremental training with SageMaker algorithms, you need model artifacts compressed into a ``tar.gz`` file. These
artifacts are passed to a training job via an input channel configured with the pre-defined settings Amazon SageMaker algorithms require.

To use model files with a SageMaker estimator, you can use the following parameters:

* ``model_uri``: points to the location of a model tarball, either in S3 or locally. Specifying a local path only works in local mode.
* ``model_channel_name``: name of the channel SageMaker will use to download the tarball specified in ``model_uri``. Defaults to 'model'.

This is converted into an input channel with the specifications mentioned above once you call ``fit()`` on the predictor.
In bring-your-own cases, ``model_channel_name`` can be overriden if you require to change the name of the channel while using
the same settings.

If your bring-your-own case requires different settings, you can create your own ``s3_input`` object with the settings you require.

Here's an example of how to use incremental training:

.. code:: python

    # Configure an estimator
    estimator = sagemaker.estimator.Estimator(training_image,
                                              role,
                                              train_instance_count=1,
                                              train_instance_type='ml.p2.xlarge',
                                              train_volume_size=50,
                                              train_max_run=360000,
                                              input_mode='File',
                                              output_path=s3_output_location)

    # Start a SageMaker training job and waits until completion.
    estimator.fit('s3://my_bucket/my_training_data/')

    # Create a new estimator using the previous' model artifacts
    incr_estimator = sagemaker.estimator.Estimator(training_image,
                                                  role,
                                                  train_instance_count=1,
                                                  train_instance_type='ml.p2.xlarge',
                                                  train_volume_size=50,
                                                  train_max_run=360000,
                                                  input_mode='File',
                                                  output_path=s3_output_location,
                                                  model_uri=estimator.model_data)

    # Start a SageMaker training job using the original model for incremental training
    incr_estimator.fit('s3://my_bucket/my_training_data/')

Currently, the following algorithms support incremental training:

- Image Classification
- Object Detection
- Semantic Segmentation


MXNet SageMaker Estimators
--------------------------

By using MXNet SageMaker ``Estimators``, you can train and host MXNet models on Amazon SageMaker.

Supported versions of MXNet: ``1.3.0``, ``1.2.1``, ``1.1.0``, ``1.0.0``, ``0.12.1``.

Supported versions of MXNet for Elastic Inference: ``1.3.0``

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information, see `MXNet SageMaker Estimators and Models`_.

.. _MXNet SageMaker Estimators and Models: src/sagemaker/mxnet/README.rst


TensorFlow SageMaker Estimators
-------------------------------

By using TensorFlow SageMaker ``Estimators``, you can train and host TensorFlow models on Amazon SageMaker.

Supported versions of TensorFlow: ``1.4.1``, ``1.5.0``, ``1.6.0``, ``1.7.0``, ``1.8.0``, ``1.9.0``, ``1.10.0``, ``1.11.0``, ``1.12.0``.

Supported versions of TensorFlow for Elastic Inference: ``1.11.0``, ``1.12.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information, see `TensorFlow SageMaker Estimators and Models`_.

.. _TensorFlow SageMaker Estimators and Models: src/sagemaker/tensorflow/README.rst


Chainer SageMaker Estimators
----------------------------

By using Chainer SageMaker ``Estimators``, you can train and host Chainer models on Amazon SageMaker.

Supported versions of Chainer: ``4.0.0``, ``4.1.0``, ``5.0.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information about Chainer, see https://github.com/chainer/chainer.

For more information about  Chainer SageMaker ``Estimators``, see `Chainer SageMaker Estimators and Models`_.

.. _Chainer SageMaker Estimators and Models: src/sagemaker/chainer/README.rst


PyTorch SageMaker Estimators
----------------------------

With PyTorch SageMaker ``Estimators``, you can train and host PyTorch models on Amazon SageMaker.

Supported versions of PyTorch: ``0.4.0``, ``1.0.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information about PyTorch, see https://github.com/pytorch/pytorch.

For more information about PyTorch SageMaker ``Estimators``, see `PyTorch SageMaker Estimators and Models`_.

.. _PyTorch SageMaker Estimators and Models: src/sagemaker/pytorch/README.rst


Scikit-learn SageMaker Estimators
---------------------------------

With Scikit-learn SageMaker ``Estimators``, you can train and host Scikit-learn models on Amazon SageMaker.

Supported versions of Scikit-learn: ``0.20.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

For more information about Scikit-learn, see https://scikit-learn.org/stable/

For more information about Scikit-learn SageMaker ``Estimators``, see `Scikit-learn SageMaker Estimators and Models`_.

.. _Scikit-learn SageMaker Estimators and Models: src/sagemaker/sklearn/README.rst


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

Using SageMaker AlgorithmEstimators
-----------------------------------

With the SageMaker Algorithm entities, you can create training jobs with just an ``algorithm_arn`` instead of
a training image. There is a dedicated ``AlgorithmEstimator`` class that accepts ``algorithm_arn`` as a
parameter, the rest of the arguments are similar to the other Estimator classes. This class also allows you to
consume algorithms that you have subscribed to in the AWS Marketplace. The AlgorithmEstimator performs
client-side validation on your inputs based on the algorithm's properties.

Here is an example:

.. code:: python

        import sagemaker

        algo = sagemaker.AlgorithmEstimator(
            algorithm_arn='arn:aws:sagemaker:us-west-2:1234567:algorithm/some-algorithm',
            role='SageMakerRole',
            train_instance_count=1,
            train_instance_type='ml.c4.xlarge')

        train_input = algo.sagemaker_session.upload_data(path='/path/to/your/data')

        algo.fit({'training': train_input})
        algo.deploy(1, 'ml.m4.xlarge')

        # When you are done using your endpoint
        algo.delete_endpoint()


Consuming SageMaker Model Packages
----------------------------------

SageMaker Model Packages are a way to specify and share information for how to create SageMaker Models.
With a SageMaker Model Package that you have created or subscribed to in the AWS Marketplace,
you can use the specified serving image and model data for Endpoints and Batch Transform jobs.

To work with a SageMaker Model Package, use the ``ModelPackage`` class.

Here is an example:

.. code:: python

        import sagemaker

        model = sagemaker.ModelPackage(
            role='SageMakerRole',
            model_package_arn='arn:aws:sagemaker:us-west-2:123456:model-package/my-model-package')
        model.deploy(1, 'ml.m4.xlarge', endpoint_name='my-endpoint')

        # When you are done using your endpoint
        model.sagemaker_session.delete_endpoint('my-endpoint')


BYO Docker Containers with SageMaker Estimators
-----------------------------------------------

To use a Docker image that you created and use the SageMaker SDK for training, the easiest way is to use the dedicated ``Estimator`` class.
You can create an instance of the ``Estimator`` class with desired Docker image and use it as described in previous sections.

Please refer to the full example in the examples repo:

::

    git clone https://github.com/awslabs/amazon-sagemaker-examples.git


The example notebook is located here:
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

By default, training job early stopping is turned off. To enable early stopping for the tuning job, you need to set the ``early_stopping_type`` parameter to ``Auto``:

.. code:: python

    # Enable early stopping
    my_tuner = HyperparameterTuner(estimator=my_estimator,  # previously-configured Estimator object
                                   objective_metric_name='validation-accuracy',
                                   hyperparameter_ranges={'learning-rate': ContinuousParameter(0.05, 0.06)},
                                   metric_definitions=[{'Name': 'validation-accuracy', 'Regex': 'validation-accuracy=(\d\.\d+)'}],
                                   max_jobs=100,
                                   max_parallel_jobs=10,
                                   early_stopping_type='Auto')

When early stopping is turned on, Amazon SageMaker will automatically stop a training job if it appears unlikely to produce a model of better quality than other jobs.
If not using built-in Amazon SageMaker algorithms, note that, for early stopping to be effective, the objective metric should be emitted at epoch level.

If you are using an Amazon SageMaker built-in algorithm, you don't need to pass in anything for ``metric_definitions``.
In addition, the ``fit()`` call uses a list of ``RecordSet`` objects instead of a dictionary:

.. code:: python

    # Create RecordSet object for each data channel
    train_records = RecordSet(...)
    test_records = RecordSet(...)

    # Start hyperparameter tuning job
    my_tuner.fit([train_records, test_records])

To help attach a previously-started hyperparameter tuning job to a ``HyperparameterTuner`` instance,
``fit()`` adds the module path of the class used to create the hyperparameter tuner to the list of static hyperparameters by default.
If you are using your own custom estimator class (i.e. not one provided in this SDK) and want that class to be used when attaching a hyperparamter tuning job,
set ``include_cls_metadata`` to ``True`` when you call ``fit`` to add the module path as static hyperparameters.

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

To train a model with the inter-container traffic encrypted, set the optional parameters ``subnets`` and ``security_group_ids`` and
the flag ``encrypt_inter_container_traffic`` as ``True`` on an Estimator (Note: This flag can be used only if you specify that the training
job runs in a VPC):

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator with subnets and security groups from your VPC
    mxnet_vpc_estimator = MXNet('train.py',
                                train_instance_type='ml.p2.xlarge',
                                train_instance_count=1,
                                framework_version='1.2.1',
                                subnets=['subnet-1', 'subnet-2'],
                                security_group_ids=['sg-1'],
                                encrypt_inter_container_traffic=True)

    # The SageMaker training job sets the VpcConfig, and training container instances run in your VPC with traffic between the containers encrypted
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


Inference Pipelines
-------------------
You can create a Pipeline for realtime or batch inference comprising of one or multiple model containers. This will help
you to deploy an ML pipeline behind a single endpoint and you can have one API call perform pre-processing, model-scoring
and post-processing on your data before returning it back as the response.

For this, you have to create a ``PipelineModel`` which will take a list of ``Model`` objects. Calling ``deploy()`` on the
``PipelineModel`` will provide you with an endpoint which can be invoked to perform the prediction on a data point against
the ML Pipeline.

.. code:: python

   xgb_image = get_image_uri(sess.boto_region_name, 'xgboost', repo_version="latest")
   xgb_model = Model(model_data='s3://path/to/model.tar.gz', image=xgb_image)
   sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})

   model_name = 'inference-pipeline-model'
   endpoint_name = 'inference-pipeline-endpoint'
   sm_model = PipelineModel(name=model_name, role=sagemaker_role, models=[sparkml_model, xgb_model])

This will define a ``PipelineModel`` consisting of SparkML model and an XGBoost model stacked sequentially. For more
information about how to train an XGBoost model, please refer to the XGBoost notebook here_.

.. _here: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html#xgboost-sample-notebooks

.. code:: python

   sm_model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge', endpoint_name=endpoint_name)

This returns a predictor the same way an ``Estimator`` does when ``deploy()`` is called. Whenever you make an inference
request using this predictor, you should pass the data that the first container expects and the predictor will return the
output from the last container.

For comprehensive examples on how to use Inference Pipelines please refer to the following notebooks:

- `inference_pipeline_sparkml_xgboost_abalone.ipynb <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/inference_pipeline_sparkml_xgboost_abalone/inference_pipeline_sparkml_xgboost_abalone.ipynb>`__
- `inference_pipeline_sparkml_blazingtext_dbpedia.ipynb <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/inference_pipeline_sparkml_blazingtext_dbpedia/inference_pipeline_sparkml_blazingtext_dbpedia.ipynb>`__


SageMaker Workflow
------------------

You can use Apache Airflow to author, schedule and monitor SageMaker workflow.

For more information, see `SageMaker Workflow in Apache Airflow`_.

.. _SageMaker Workflow in Apache Airflow: src/sagemaker/workflow/README.rst
