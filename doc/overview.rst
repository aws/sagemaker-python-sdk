##############################
Using the SageMaker Python SDK
##############################

SageMaker Python SDK provides several high-level abstractions for working with Amazon SageMaker. These are:

- **Estimators**: Encapsulate training on SageMaker.
- **Models**: Encapsulate built ML models.
- **Predictors**: Provide real-time inference and transformation using Python data-types against a SageMaker endpoint.
- **Session**: Provides a collection of methods for working with SageMaker resources.

``Estimator`` and ``Model`` implementations for MXNet, TensorFlow, Chainer, PyTorch, scikit-learn, Amazon SageMaker built-in algorithms, Reinforcement Learning,  are included.
There's also an ``Estimator`` that runs SageMaker compatible custom Docker containers, enabling you to run your own ML algorithms by using the SageMaker Python SDK.

.. contents::
   :depth: 2

*******************************************
Train a Model with the SageMaker Python SDK
*******************************************

To train a model by using the SageMaker Python SDK, you:

1. Prepare a training script
2. Create an estimator
3. Call the ``fit`` method of the estimator

After you train a model, you can save it, and then serve the model as an endpoint to get real-time inferences or get inferences for an entire dataset by using batch transform.

Prepare a Training script
=========================

Your training script must be a Python 2.7 or 3.6 compatible source file.

The training script is very similar to a training script you might run outside of SageMaker, but you can access useful properties about the training environment through various environment variables, including the following:

* ``SM_MODEL_DIR``: A string that represents the path where the training job writes the model artifacts to.
  After training, artifacts in this directory are uploaded to S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_CHANNEL_XXXX``: A string that represents the path to the directory that contains the input data for the specified channel.
  For example, if you specify two input channels in the MXNet estimator's ``fit`` call, named 'train' and 'test', the environment variables ``SM_CHANNEL_TRAIN`` and ``SM_CHANNEL_TEST`` are set.
* ``SM_HPS``: A json dump of the hyperparameters preserving json types (boolean, integer, etc.)

For the exhaustive list of available environment variables, see the `SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model, and saves a model to ``model_dir`` so that it can be deployed for inference later.
Hyperparameters are passed to your script as arguments and can be retrieved with an ``argparse.ArgumentParser`` instance.
For example, a training script might start with the following:

.. code:: python

    import argparse
    import os
    import json

    if __name__ =='__main__':

        parser = argparse.ArgumentParser()

        # hyperparameters sent by the client are passed as command-line arguments to the script.
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=100)
        parser.add_argument('--learning-rate', type=float, default=0.1)

        # an alternative way to load hyperparameters via SM_HPS environment variable.
        parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

        # input data and model directories
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
        parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

        args, _ = parser.parse_known_args()

        # ... load from args.train and args.test, train a model, write model to args.model_dir.

Because the SageMaker imports your training script, you should put your training code in a main guard (``if __name__=='__main__':``) if you are using the same script to host your model,
so that SageMaker does not inadvertently run your training code at the wrong point in execution.

Note that SageMaker doesn't support argparse actions.
If you want to use, for example, boolean hyperparameters, you need to specify ``type`` as ``bool`` in your script and provide an explicit ``True`` or ``False`` value for this hyperparameter when you create your estimator.

For more on training environment variables, please visit `SageMaker Containers <https://github.com/aws/sagemaker-containers>`_.


Using Estimators
================

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

    # Tears down the SageMaker endpoint and endpoint configuration
    mxnet_predictor.delete_endpoint()

    # Deletes the SageMaker model
    mxnet_predictor.delete_model()

The example above will eventually delete both the SageMaker endpoint and endpoint configuration through `delete_endpoint()`. If you want to keep your SageMaker endpoint configuration, use the value False for the `delete_endpoint_config` parameter, as shown below.

.. code:: python

    # Only delete the SageMaker endpoint, while keeping the corresponding endpoint configuration.
    mxnet_predictor.delete_endpoint(delete_endpoint_config=False)

Additionally, it is possible to deploy a different endpoint configuration, which links to your model, to an already existing SageMaker endpoint.
This can be done by specifying the existing endpoint name for the ``endpoint_name`` parameter along with the ``update_endpoint`` parameter as ``True`` within your ``deploy()`` call.
For more `information <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.update_endpoint>`__.

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

    # Deploys the model that was generated by fit() to an existing SageMaker endpoint
    mxnet_predictor = mxnet_estimator.deploy(initial_instance_count=1,
                                             instance_type='ml.p2.xlarge',
                                             update_endpoint=True,
                                             endpoint_name='existing-endpoint')

    # Serializes data and makes a prediction request to the SageMaker endpoint
    response = mxnet_predictor.predict(data)

    # Tears down the SageMaker endpoint and endpoint configuration
    mxnet_predictor.delete_endpoint()

    # Deletes the SageMaker model
    mxnet_predictor.delete_model()

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

Use Scripts Stored in a Git Repository
--------------------------------------
When you create an estimator, you can specify a training script that is stored in a GitHub (or other Git) or CodeCommit repository as the entry point for the estimator, so that you don't have to download the scripts locally.
If you do so, source directory and dependencies should be in the same repo if they are needed. Git support can be enabled simply by providing ``git_config`` parameter
when creating an ``Estimator`` object. If Git support is enabled, then ``entry_point``, ``source_dir`` and  ``dependencies``
should be relative paths in the Git repo if provided.

The ``git_config`` parameter includes fields ``repo``, ``branch``,  ``commit``, ``2FA_enabled``, ``username``,
``password`` and ``token``. The ``repo`` field is required. All other fields are optional. ``repo`` specifies the Git
repository where your training script is stored. If you don't provide ``branch``, the default value  'master' is used.
If you don't provide ``commit``, the latest commit in the specified branch is used.

``2FA_enabled``, ``username``, ``password`` and ``token`` are used for authentication. For GitHub
(or other Git) accounts, set ``2FA_enabled`` to 'True' if two-factor authentication is enabled for the
account, otherwise set it to 'False'. If you do not provide a value for ``2FA_enabled``, a default
value of 'False' is used. CodeCommit does not support two-factor authentication, so do not provide
"2FA_enabled" with CodeCommit repositories.

For GitHub or other Git repositories,
If ``repo`` is an SSH URL, you should either have no passphrase for the SSH key pairs, or have the ``ssh-agent`` configured
so that you are not prompted for the SSH passphrase when you run a ``git clone`` command with SSH URLs. For SSH URLs, it
does not matter whether two-factor authentication is enabled. If ``repo`` is an HTTPS URL, 2FA matters. When 2FA is disabled, either ``token`` or ``username``+``password`` will be
used for authentication if provided (``token`` prioritized). When 2FA is enabled, only token will be used for
authentication if provided. If required authentication info is not provided, python SDK will try to use local
credentials storage to authenticate. If that fails either, an error message will be thrown.

For CodeCommit repos, please make sure you have completed the authentication setup: https://docs.aws.amazon.com/codecommit/latest/userguide/setting-up.html.
2FA is not supported by CodeCommit, so ``2FA_enabled`` should not be provided. There is no token in CodeCommit, so
``token`` should not be provided either. If ``repo`` is an SSH URL, the requirements are the same as GitHub repos.
If ``repo`` is an HTTPS URL, ``username``+``password`` will be used for authentication if they are provided; otherwise,
Python SDK will try to use either CodeCommit credential helper or local credential storage for authentication.

Here are some examples of creating estimators with Git support:

.. code:: python

        # Specifies the git_config parameter. This example does not provide Git credentials, so python SDK will try
        # to use local credential storage.
        git_config = {'repo': 'https://github.com/username/repo-with-training-scripts.git',
                      'branch': 'branch1',
                      'commit': '4893e528afa4a790331e1b5286954f073b0f14a2'}

        # In this example, the source directory 'pytorch' contains the entry point 'mnist.py' and other source code.
        # and it is relative path inside the Git repo.
        pytorch_estimator = PyTorch(entry_point='mnist.py',
                                    role='SageMakerRole',
                                    source_dir='pytorch',
                                    git_config=git_config,
                                    train_instance_count=1,
                                    train_instance_type='ml.c4.xlarge')

.. code:: python

        # You can also specify git_config by providing only 'repo' and 'branch'.
        # If this is the case, the latest commit in that branch will be used.
        git_config = {'repo': 'git@github.com:username/repo-with-training-scripts.git',
                      'branch': 'branch1'}

        # In this example, the entry point 'mnist.py' is all we need for source code.
        # We need to specify the path to it in the Git repo.
        mx_estimator = MXNet(entry_point='mxnet/mnist.py',
                             role='SageMakerRole',
                             git_config=git_config,
                             train_instance_count=1,
                             train_instance_type='ml.c4.xlarge')

.. code:: python

        # Only providing 'repo' is also allowed. If this is the case, latest commit in 'master' branch will be used.
        # This example does not provide '2FA_enabled', so 2FA is treated as disabled by default. 'username' and
        # 'password' are provided for authentication
        git_config = {'repo': 'https://github.com/username/repo-with-training-scripts.git',
                      'username': 'username',
                      'password': 'passw0rd!'}

        # In this example, besides entry point and other source code in source directory, we still need some
        # dependencies for the training job. Dependencies should also be paths inside the Git repo.
        pytorch_estimator = PyTorch(entry_point='mnist.py',
                                    role='SageMakerRole',
                                    source_dir='pytorch',
                                    dependencies=['dep.py', 'foo/bar.py'],
                                    git_config=git_config,
                                    train_instance_count=1,
                                    train_instance_type='ml.c4.xlarge')

.. code:: python

        # This example specifies that 2FA is enabled, and token is provided for authentication
        git_config = {'repo': 'https://github.com/username/repo-with-training-scripts.git',
                      '2FA_enabled': True,
                      'token': 'your-token'}

        # In this exmaple, besides entry point, we also need some dependencies for the training job.
        pytorch_estimator = PyTorch(entry_point='pytorch/mnist.py',
                                    role='SageMakerRole',
                                    dependencies=['dep.py'],
                                    git_config=git_config,
                                    train_instance_count=1,
                                    train_instance_type='local')

.. code:: python

        # This example specifies a CodeCommit repository, and try to authenticate with provided username+password
        git_config = {'repo': 'https://git-codecommit.us-west-2.amazonaws.com/v1/repos/your_repo_name',
                      'username': 'username',
                      'password': 'passw0rd!'}

        mx_estimator = MXNet(entry_point='mxnet/mnist.py',
                             role='SageMakerRole',
                             git_config=git_config,
                             train_instance_count=1,
                             train_instance_type='ml.c4.xlarge')

Git support can be used not only for training jobs, but also for hosting models. The usage is the same as the above,
and ``git_config`` should be provided when creating model objects, e.g. ``TensorFlowModel``, ``MXNetModel``, ``PyTorchModel``.

Training Metrics
----------------
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

BYO Docker Containers with SageMaker Estimators
-----------------------------------------------

To use a Docker image that you created and use the SageMaker SDK for training, the easiest way is to use the dedicated ``Estimator`` class.
You can create an instance of the ``Estimator`` class with desired Docker image and use it as described in previous sections.

Please refer to the full example in the examples repo:

::

    git clone https://github.com/awslabs/amazon-sagemaker-examples.git


The example notebook is located here:
``advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb``

You can also find this notebook in the **Advanced Functionality** folder of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

Incremental Training
====================

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

************************************************
Using Models Trained Outside of Amazon SageMaker
************************************************

You can use models that you train outside of Amazon SageMaker, and model packages that you create or subscribe to in the AWS Marketplace to get inferences.

BYO Model
=========

You can create an endpoint from an existing model that you trained outside of Amazon Sagemaker.
That is, you can bring your own model:

First, package the files for the trained model into a ``.tar.gz`` file, and upload the archive to S3.

Next, create a ``Model`` object that corresponds to the framework that you are using: `MXNetModel <https://sagemaker.readthedocs.io/en/stable/sagemaker.mxnet.html#mxnet-model>`__ or `TensorFlowModel <https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html#tensorflow-model>`__.

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

You can also find this notebook in the **Advanced Functionality** section of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

Consuming SageMaker Model Packages
==================================

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

********************************
SageMaker Automatic Model Tuning
********************************

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

You can also find these notebooks in the **Hyperprameter Tuning** section of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

For more detailed explanations of the classes that this library provides for automatic model tuning, see:

- `API docs for HyperparameterTuner and parameter range classes <https://sagemaker.readthedocs.io/en/stable/tuner.html>`__
- `API docs for analytics classes <https://sagemaker.readthedocs.io/en/stable/analytics.html>`__

*************************
SageMaker Batch Transform
*************************

After you train a model, you can use Amazon SageMaker Batch Transform to perform inferences with the model.
Batch transform manages all necessary compute resources, including launching instances to deploy endpoints and deleting them afterward.
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

For a full list of the possible options to configure by using either of these methods, see the API docs for `Estimator <https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Estimator.transformer>`__ or `Transformer <https://sagemaker.readthedocs.io/en/stable/transformer.html#sagemaker.transformer.Transformer>`__.

After you create a ``Transformer`` object, you can invoke ``transform()`` to start a batch transform job with the S3 location of your data.
You can also specify other attributes of your data, such as the content type.

.. code:: python

    transformer.transform('s3://my-bucket/batch-transform-input')

For more details about what can be specified here, see `API docs <https://sagemaker.readthedocs.io/en/stable/transformer.html#sagemaker.transformer.Transformer.transform>`__.

**********
Local Mode
**********

The SageMaker Python SDK supports local mode, which allows you to create estimators and deploy them to your local environment.
This is a great way to test your deep learning scripts before running them in SageMaker's managed training or hosting environments.
Local Mode is supported for frameworks images (TensorFlow, MXNet, Chainer, PyTorch, and Scikit-Learn) and images you supply yourself.

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

    # Tears down the endpoint container and deletes the corresponding endpoint configuration
    mxnet_predictor.delete_endpoint()

    # Deletes the model
    mxnet_predictor.delete_model()


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

    # Tear down the endpoint container and delete the corresponding endpoint configuration
    predictor.delete_endpoint()

    # Deletes the model
    predictor.delete_model()


If you don't want to deploy your model locally, you can also choose to perform a Local Batch Transform Job. This is
useful if you want to test your container before creating a Sagemaker Batch Transform Job. Note that the performance
will not match Batch Transform Jobs hosted on SageMaker but it is still a useful tool to ensure you have everything
right or if you are not dealing with huge amounts of data.

Here is an end-to-end example:

.. code:: python

    from sagemaker.mxnet import MXNet

    mxnet_estimator = MXNet('train.py',
                            role='SageMakerRole',
                            train_instance_type='local',
                            train_instance_count=1,
                            framework_version='1.2.1')

    mxnet_estimator.fit('file:///tmp/my_training_data')
    transformer = mxnet_estimator.transformer(1, 'local', assemble_with='Line', max_payload=1)
    transformer.transform('s3://my/transform/data, content_type='text/csv', split_type='Line')
    transformer.wait()

    # Deletes the SageMaker model
    transformer.delete_model()


For detailed examples of running Docker in local mode, see:

- `TensorFlow local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb>`__.
- `MXNet local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_mnist/mnist_with_gluon_local_mode.ipynb>`__.

You can also find these notebooks in the **SageMaker Python SDK** section of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

A few important notes:

- Only one local mode endpoint can be running at a time.
- If you are using S3 data as input, it is pulled from S3 to your local environment. Ensure you have sufficient space to store the data locally.
- If you run into problems it often due to different Docker containers conflicting. Killing these containers and re-running often solves your problems.
- Local Mode requires Docker Compose and `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__ for ``local_gpu``.
- Distributed training is not yet supported for ``local_gpu``.

**************************************
Secure Training and Inference with VPC
**************************************

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

***********************************************************
Secure Training with Network Isolation (Internet-Free) Mode
***********************************************************

You can enable network isolation mode when running training and inference on Amazon SageMaker.

For more information about Amazon SageMaker network isolation mode, see the `SageMaker documentation on network isolation or internet-free mode <https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html>`__.

To train a model in network isolation mode, set the optional parameter ``enable_network_isolation`` to ``True`` in any network isolation supported Framework Estimator.

.. code:: python

    # set the enable_network_isolation parameter to True
    sklearn_estimator = SKLearn('sklearn-train.py',
                                train_instance_type='ml.m4.xlarge',
                                framework_version='0.20.0',
                                hyperparameters = {'epochs': 20, 'batch-size': 64, 'learning-rate': 0.1},
                                enable_network_isolation=True)

    # SageMaker Training Job will in the container without   any inbound or outbound network calls during runtime
    sklearn_estimator.fit({'train': 's3://my-data-bucket/path/to/my/training/data',
                            'test': 's3://my-data-bucket/path/to/my/test/data'})

When this training job is created, the SageMaker Python SDK will upload the files in ``entry_point``, ``source_dir``, and ``dependencies`` to S3 as a compressed ``sourcedir.tar.gz`` file (``'s3://mybucket/sourcedir.tar.gz'``).

A new training job channel, named ``code``, will be added with that S3 URI.  Before the training docker container is initialized, the ``sourcedir.tar.gz`` will be downloaded from S3 to the ML storage volume like any other offline input channel.

Once the training job begins, the training container will look at the offline input ``code`` channel to install dependencies and run the entry script. This isolates the training container, so no inbound or outbound network calls can be made.

*******************
Inference Pipelines
*******************

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

This defines a ``PipelineModel`` consisting of SparkML model and an XGBoost model stacked sequentially.
For more information about how to train an XGBoost model, please refer to the XGBoost notebook here_.

.. _here: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone.ipynb

You can also find this notebook in the **Introduction to Amazon Algorithms** section of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

.. code:: python

   sm_model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge', endpoint_name=endpoint_name)

This returns a predictor the same way an ``Estimator`` does when ``deploy()`` is called. Whenever you make an inference
request using this predictor, you should pass the data that the first container expects and the predictor will return the
output from the last container.

You can also use a ``PipelineModel`` to create Transform Jobs for batch transformations. Using the same ``PipelineModel`` ``sm_model`` as above:

.. code:: python

   # Only instance_type and instance_count are required.
   transformer = sm_model.transformer(instance_type='ml.c5.xlarge',
                                      instance_count=1,
                                      strategy='MultiRecord',
                                      max_payload=6,
                                      max_concurrent_transforms=8,
                                      accept='text/csv',
                                      assemble_with='Line',
                                      output_path='s3://my-output-bucket/path/to/my/output/data/')
   # Only data is required.
   transformer.transform(data='s3://my-input-bucket/path/to/my/csv/data',
                         content_type='text/csv',
                         split_type='Line')
   # Waits for the Pipeline Transform Job to finish.
   transformer.wait()

This runs a transform job against all the files under ``s3://mybucket/path/to/my/csv/data``, transforming the input
data in order with each model container in the pipeline. For each input file that was successfully transformed, one output file in ``s3://my-output-bucket/path/to/my/output/data/``
will be created with the same name, appended with '.out'.
This transform job will split CSV files by newline separators, which is especially useful if the input files are large.
The Transform Job assembles the outputs with line separators when writing each input file's corresponding output file.
Each payload entering the first model container will be up to six megabytes, and up to eight inference requests are sent at the
same time to the first model container. Because each payload consists of a mini-batch of multiple CSV records, the model
containers transform each mini-batch of records.

For comprehensive examples on how to use Inference Pipelines please refer to the following notebooks:

- `inference_pipeline_sparkml_xgboost_abalone.ipynb <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/inference_pipeline_sparkml_xgboost_abalone/inference_pipeline_sparkml_xgboost_abalone.ipynb>`__
- `inference_pipeline_sparkml_blazingtext_dbpedia.ipynb <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/inference_pipeline_sparkml_blazingtext_dbpedia/inference_pipeline_sparkml_blazingtext_dbpedia.ipynb>`__

You can also find these notebooks in the **Advanced Functionality** section of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

******************
SageMaker Workflow
******************

You can use Apache Airflow to author, schedule and monitor SageMaker workflow.

For more information, see `SageMaker Workflow in Apache Airflow`_.

.. _SageMaker Workflow in Apache Airflow: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/workflow/README.rst

***
FAQ
***

I want to train a SageMaker Estimator with local data, how do I do this?
========================================================================

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
=======================================================

Create a ``Predictor`` object and provide it with your endpoint name,
then call its ``predict()`` method with your input.

You can use either the generic ``RealTimePredictor`` class, which by default does not perform any serialization/deserialization transformations on your input,
but can be configured to do so through constructor arguments:
http://sagemaker.readthedocs.io/en/stable/predictors.html

Or you can use the TensorFlow / MXNet specific predictor classes, which have default serialization/deserialization logic:
http://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html#tensorflow-predictor
http://sagemaker.readthedocs.io/en/stable/sagemaker.mxnet.html#mxnet-predictor

Example code using the TensorFlow predictor:

::

    from sagemaker.tensorflow import TensorFlowPredictor

    predictor = TensorFlowPredictor('myexistingendpoint')
    result = predictor.predict(['my request body'])
