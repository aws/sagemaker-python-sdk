##############################
Using the SageMaker Python SDK
##############################

SageMaker Python SDK provides several high-level abstractions for working with Amazon SageMaker. These are:

- **Estimators**: Encapsulate training on SageMaker.
- **Models**: Encapsulate built ML models.
- **Predictors**: Provide real-time inference and transformation using Python data-types against a SageMaker endpoint.
- **Session**: Provides a collection of methods for working with SageMaker resources.
- **Transformers**: Encapsulate batch transform jobs for inference on SageMaker
- **Processors**: Encapsulate running processing jobs for data processing on SageMaker

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
                            instance_type='ml.p2.xlarge',
                            instance_count=1,
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

The example above will eventually delete both the SageMaker endpoint and endpoint configuration through ``delete_endpoint()``. If you want to keep your SageMaker endpoint configuration, use the value ``False`` for the ``delete_endpoint_config`` parameter, as shown below.

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
                            instance_type='ml.p2.xlarge',
                            instance_count=1,
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
            instance_count=1,
            instance_type='ml.c4.xlarge')

        train_input = algo.sagemaker_session.upload_data(path='/path/to/your/data')

        algo.fit({'training': train_input})
        predictor = algo.deploy(1, 'ml.m4.xlarge')

        # When you are done using your endpoint
        predictor.delete_endpoint()

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
                                    instance_count=1,
                                    instance_type='ml.c4.xlarge')

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
                             instance_count=1,
                             instance_type='ml.c4.xlarge')

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
                                    instance_count=1,
                                    instance_type='ml.c4.xlarge')

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
                                    instance_count=1,
                                    instance_type='local')

.. code:: python

        # This example specifies a CodeCommit repository, and try to authenticate with provided username+password
        git_config = {'repo': 'https://git-codecommit.us-west-2.amazonaws.com/v1/repos/your_repo_name',
                      'username': 'username',
                      'password': 'passw0rd!'}

        mx_estimator = MXNet(entry_point='mxnet/mnist.py',
                             role='SageMakerRole',
                             git_config=git_config,
                             instance_count=1,
                             instance_type='ml.c4.xlarge')

Git support can be used not only for training jobs, but also for hosting models. The usage is the same as the above,
and ``git_config`` should be provided when creating model objects, e.g. ``TensorFlowModel``, ``MXNetModel``, ``PyTorchModel``.

Use File Systems as Training Inputs
-------------------------------------
Amazon SageMaker supports using Amazon Elastic File System (EFS) and FSx for Lustre as data sources to use during training.
If you want use those data sources, create a file system (EFS/FSx) and mount the file system on an Amazon EC2 instance.
For more information about setting up EFS and FSx, see the following documentation:

- `Using File Systems in Amazon EFS <https://docs.aws.amazon.com/efs/latest/ug/using-fs.html>`__
- `Getting Started with Amazon FSx for Lustre <https://aws.amazon.com/fsx/lustre/getting-started/>`__

The general experience uses either the ``FileSystemInput`` or ``FileSystemRecordSet`` class, which encapsulates
all of the necessary arguments required by the service to use EFS or Lustre.

Here are examples of how to use Amazon EFS as input for training:

.. code:: python

        # This example shows how to use FileSystemInput class
        # Configure an estimator with subnets and security groups from your VPC. The EFS volume must be in
        # the same VPC as your Amazon EC2 instance
        estimator = TensorFlow(entry_point='tensorflow_mnist/mnist.py',
                               role='SageMakerRole',
                               instance_count=1,
                               instance_type='ml.c4.xlarge',
                               subnets=['subnet-1', 'subnet-2']
                               security_group_ids=['sg-1'])

        file_system_input = FileSystemInput(file_system_id='fs-1',
                                            file_system_type='EFS',
                                            directory_path='/tensorflow',
                                            file_system_access_mode='ro')

        # Start an Amazon SageMaker training job with EFS using the FileSystemInput class
        estimator.fit(file_system_input)

.. code:: python

        # This example shows how to use FileSystemRecordSet class
        # Configure an estimator with subnets and security groups from your VPC. The EFS volume must be in
        # the same VPC as your Amazon EC2 instance
        kmeans = KMeans(role='SageMakerRole',
                        instance_count=1,
                        instance_type='ml.c4.xlarge',
                        k=10,
                        subnets=['subnet-1', 'subnet-2'],
                        security_group_ids=['sg-1'])

        records = FileSystemRecordSet(file_system_id='fs-1,
                                      file_system_type='EFS',
                                      directory_path='/kmeans',
                                      num_records=784,
                                      feature_dim=784)

        # Start an Amazon SageMaker training job with EFS using the FileSystemRecordSet class
        kmeans.fit(records)

Here are examples of how to use Amazon FSx for Lustre as input for training:

.. code:: python

        # This example shows how to use FileSystemInput class
        # Configure an estimator with subnets and security groups from your VPC. The VPC should be the same as that
        # you chose for your Amazon EC2 instance

        estimator = TensorFlow(entry_point='tensorflow_mnist/mnist.py',
                               role='SageMakerRole',
                               instance_count=1,
                               instance_type='ml.c4.xlarge',
                               subnets=['subnet-1', 'subnet-2']
                               security_group_ids=['sg-1'])


        file_system_input = FileSystemInput(file_system_id='fs-2',
                                            file_system_type='FSxLustre',
                                            directory_path='/<mount-id>/tensorflow',
                                            file_system_access_mode='ro')

        # Start an Amazon SageMaker training job with FSx using the FileSystemInput class
        estimator.fit(file_system_input)

.. code:: python

        # This example shows how to use FileSystemRecordSet class
        # Configure an estimator with subnets and security groups from your VPC. The VPC should be the same as that
        # you chose for your Amazon EC2 instance
        kmeans = KMeans(role='SageMakerRole',
                        instance_count=1,
                        instance_type='ml.c4.xlarge',
                        k=10,
                        subnets=['subnet-1', 'subnet-2'],
                        security_group_ids=['sg-1'])

        records = FileSystemRecordSet(file_system_id='fs-=2,
                                      file_system_type='FSxLustre',
                                      directory_path='/<mount-id>/kmeans',
                                      num_records=784,
                                      feature_dim=784)

        # Start an Amazon SageMaker training job with FSx using the FileSystemRecordSet class
        kmeans.fit(records)

Data sources from EFS and FSx can also be used for hyperparameter tuning jobs. The usage is the same as above.

A few important notes:

- Local mode is not supported if using EFS and FSx as data sources

- Pipe mode is not supported if using EFS as data source

Training Metrics
----------------
The SageMaker Python SDK allows you to specify a name and a regular expression for metrics you want to track for training.
A regular expression (regex) matches what is in the training algorithm logs, like a search function.
Here is an example of how to define metrics:

.. code:: python

    # Configure an BYO Estimator with metric definitions (no training happens yet)
    byo_estimator = Estimator(image_uri=image_uri,
                              role='SageMakerRole', instance_count=1,
                              instance_type='ml.c4.xlarge',
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
                                              instance_count=1,
                                              instance_type='ml.p2.xlarge',
                                              volume_size=50,
                                              max_run=360000,
                                              input_mode='File',
                                              output_path=s3_output_location)

    # Start a SageMaker training job and waits until completion.
    estimator.fit('s3://my_bucket/my_training_data/')

    # Create a new estimator using the previous' model artifacts
    incr_estimator = sagemaker.estimator.Estimator(training_image,
                                                  role,
                                                  instance_count=1,
                                                  instance_type='ml.p2.xlarge',
                                                  volume_size=50,
                                                  max_run=360000,
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

.. _built-in-algos:

***********************************************************************
Use Built-in Algorithms with Pre-trained Models in SageMaker Python SDK
***********************************************************************

SageMaker Python SDK provides built-in algorithms with pre-trained models from popular open source model
hubs, such as TensorFlow Hub, Pytorch Hub, and HuggingFace. Customer can deploy these pre-trained models
as-is or first fine-tune them on a custom dataset and then deploy to a SageMaker endpoint for inference.


SageMaker SDK built-in algorithms allow customers access pre-trained models using model ids and model
versions. The ‘pre-trained model’ table below provides list of models with information useful in
selecting the correct model id and corresponding parameters. These models are also available through
the `JumpStart UI in SageMaker Studio <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html>`__.


.. toctree::
    :maxdepth: 2

    doc_utils/pretrainedmodels

Example notebooks
=================

SageMaker built-in algorithms with pre-trained models support 15 different machine learning problem types.
Below is a list of all the supported problem types with a link to a Jupyter notebook that provides example usage.

Vision
    - `Image Classification <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_image_classification/Amazon_JumpStart_Image_Classification.ipynb>`__
    - `Object Detection <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_object_detection/Amazon_JumpStart_Object_Detection.ipynb>`__
    - `Semantic Segmentation <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_semantic_segmentation/Amazon_JumpStart_Semantic_Segmentation.ipynb>`__
    - `Instance Segmentation <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_instance_segmentation/Amazon_JumpStart_Instance_Segmentation.ipynb>`__
    - `Image Embedding <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_image_embedding/Amazon_JumpStart_Image_Embedding.ipynb>`__

Text
    - `Text Classification <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_classification/Amazon_JumpStart_Text_Classification.ipynb>`__
    - `Sentence Pair Classification <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_sentence_pair_classification/Amazon_JumpStart_Sentence_Pair_Classification.ipynb>`__
    - `Question Answering <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_question_answering/Amazon_JumpStart_Question_Answering.ipynb>`__
    - `Named Entity Recognition <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_named_entity_recognition/Amazon_JumpStart_Named_Entity_Recognition.ipynb>`__
    - `Text Summarization <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_summarization/Amazon_JumpStart_Text_Summarization.ipynb>`__
    - `Text Generation <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_generation/Amazon_JumpStart_Text_Generation.ipynb>`__
    - `Machine Translation <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_machine_translation/Amazon_JumpStart_Machine_Translation.ipynb>`__
    - `Text Embedding <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_embedding/Amazon_JumpStart_Text_Embedding.ipynb>`__

Tabular
    - `Tabular Classification (LightGBM & Catboost) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/lightgbm_catboost_tabular/Amazon_Tabular_Classification_LightGBM_CatBoost.ipynb>`__
    - `Tabular Classification (XGBoost & Scikit-learn Linear Learner) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/xgboost_linear_learner_tabular/Amazon_Tabular_Classification_XGBoost_LinearLearner.ipynb>`__
    - `Tabular Classification (AutoGluon) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/autogluon_tabular/Amazon_Tabular_Classification_AutoGluon.ipynb>`__
    - `Tabular Classification (TabTransformer) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/tabtransformer_tabular/Amazon_Tabular_Classification_TabTransformer.ipynb>`__
    - `Tabular Regression (LightGBM & Catboost) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/lightgbm_catboost_tabular/Amazon_Tabular_Regression_LightGBM_CatBoost.ipynb>`__
    - `Tabular Regression (XGBoost & Scikit-learn Linear Learner) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/xgboost_linear_learner_tabular/Amazon_Tabular_Regression_XGBoost_LinearLearner.ipynb>`__
    - `Tabular Regression (AutoGluon) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/autogluon_tabular/Amazon_Tabular_Regression_AutoGluon.ipynb>`__
    - `Tabular Regression (TabTransformer) <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/tabtransformer_tabular/Amazon_Tabular_Regression_TabTransformer.ipynb>`__


The following topic give you information about JumpStart components,
as well as how to use the SageMaker Python SDK for these workflows.

Prerequisites
=============

.. container::

   -  You must set up AWS credentials following the steps
      in `Quick configuration with aws configure <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config>`__.
   -  Your IAM role must allow connection to Amazon SageMaker and
      Amazon S3. For more information about IAM role permissions,
      see `Policies and permissions in IAM <https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html>`__.

Built-in Components
===================

The following sections give information about the main built-in
components and their function.

Pre-trained models
------------------

SageMaker maintains a model zoo of over 300 models from popular open source model hubs, such as
TensorFlow Hub, Pytorch Hub, and HuggingFace. You can use the SageMaker Python SDK to fine-tune
a model on your own dataset or deploy it directly to a SageMaker endpoint for inference.

Model artifacts are stored as tarballs in a S3 bucket. Each model is versioned and contains a
unique ID which can be used to retrieve the model URI. The following information describes the
``model_id`` and ``model_version`` needed to retrieve the URI.

.. container::

   -  ``model_id``: A unique identifier for the JumpStart model.
   -  ``model_version``: The version of the specifications for the
      model. To use the latest version, enter ``"*"``. This is a
      required parameter.

To retrieve a model, first select a ``model ID`` and ``version`` from
the :doc:`available models <./doc_utils/pretrainedmodels>`.

.. code:: python

   model_id, model_version = "huggingface-spc-bert-base-cased", "1.0.0"
   scope = "training" # or "inference"

Then use those values to retrieve the model as follows.

.. code:: python

   from sagemaker import model_uris

   model_uri = model_uris.retrieve(
       model_id=model_id, model_version=model_version, model_scope=scope
   )

Model scripts
-------------

To adapt pre-trained models for SageMaker, a custom script is needed to perform training
or inference. SageMaker maintains a suite of scripts used for each of the models in the
S3 bucket, which can be accessed using the SageMaker Python SDK Use the ``model_id`` and
``version`` of the corresponding model to retrieve the related script as follows.

.. code:: python

   from sagemaker import script_uris

   script_uri = script_uris.retrieve(
       model_id=model_id, model_version=model_version, script_scope=scope
   )

Model images
-------------

A Docker image is required to perform training or inference on all
SageMaker models. SageMaker relies on Docker images from the
following repos https://github.com/aws/deep-learning-containers,
https://github.com/aws/sagemaker-xgboost-container,
and https://github.com/aws/sagemaker-scikit-learn-container. Use
the ``model_id`` and ``version`` of the corresponding model to
retrieve the related image as follows.

.. code:: python

   from sagemaker import image_uris

   image_uri = image_uris.retrieve(
       region=None,
       framework=None,
       image_scope=scope,
       model_id=model_id,
       model_version=model_version,
       instance_type="ml.m5.xlarge",
   )

Deploy a  Pre-Trained Model Directly to a SageMaker Endpoint
============================================================

In this section, you learn how to take a pre-trained model and deploy
it directly to a SageMaker Endpoint. This is the fastest way to start
machine learning with a pre-trained model. The following
assumes familiarity with `SageMaker
models <https://sagemaker.readthedocs.io/en/stable/api/inference/model.html>`__
and their deploy functions.

To begin, select a ``model_id`` and ``version`` from the pre-trained
models table, as well as a model scope of either “inference” or
“training”. For this example, you use a pre-trained model,
so select “inference”  for your model scope. Use the utility
functions to retrieve the URI of each of the three components you
need to continue.

.. code:: python

   from sagemaker import image_uris, model_uris, script_uris

   model_id, model_version = "tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2", "1.0.0"
   instance_type, instance_count = "ml.m5.xlarge", 1

   # Retrieve the URIs of the JumpStart resources
   base_model_uri = model_uris.retrieve(
       model_id=model_id, model_version=model_version, model_scope="inference"
   )
   script_uri = script_uris.retrieve(
       model_id=model_id, model_version=model_version, script_scope="inference"
   )
   image_uri = image_uris.retrieve(
       region=None,
       framework=None,
       image_scope="inference",
       model_id=model_id,
       model_version=model_version,
       instance_type=instance_type,
   )

Next, pass the URIs and other key parameters as part of a new
SageMaker Model class. The ``entry_point`` is a JumpStart script
named ``inference.py``. SageMaker handles the implementation of this
script. You must use this value for model inference to be successful.
For more information about the Model class and its parameters,
see `Model <https://sagemaker.readthedocs.io/en/stable/api/inference/model.html>`__.

.. code:: python

   from sagemaker.model import Model
   from sagemaker.predictor import Predictor
   from sagemaker.session import Session

   # Create the SageMaker model instance
   model = Model(
       image_uri=image_uri,
       model_data=base_model_uri,
       source_dir=script_uri,
       entry_point="inference.py",
       role=Session().get_caller_identity_arn(),
       predictor_cls=Predictor,
       enable_network_isolation=True,
   )

Save the output from deploying the model to a variable named
``predictor``. The predictor is used to make queries on the SageMaker
endpoint. Currently, the generic ``model.deploy`` call requires
the ``predictor_cls`` parameter to define the predictor class. Pass
in the default SageMaker Predictor class for this parameter.
Deployment may take about 5 minutes.

.. code:: python

   predictor = model.deploy(
       initial_instance_count=instance_count,
       instance_type=instance_type,
   )

Because the model and script URIs are distributed by SageMaker JumpStart,
the endpoint, endpoint config and model resources will be prefixed with
``sagemaker-jumpstart``. Refer to the model ``Tags`` to inspect the
model artifacts involved in the model creation.

Perform Inference
-----------------

Finally, use the ``predictor`` instance to query your endpoint. For
``catboost-classification-model``, for example, the predictor accepts
a csv. For more information about how to use the predictor, see
the
`Appendix <https://sagemaker.readthedocs.io/en/stable/overview.html#appendix>`__.

.. code:: python

   predictor.predict("this is the best day of my life", {"ContentType": "application/x-text"})

Fine-tune a Model and Deploy to a SageMaker Endpoint
====================================================

In this section, you initiate a training job to further train one of the pre-trained models
for your use case, then deploy it to a SageMaker Endpoint for inference. This lets you fine
tune the model for your use case with your custom dataset. The following assumes
familiarity with `SageMaker training jobs and their
architecture <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html>`__.

Fine-tune a Pre-trained Model on a Custom Dataset
-------------------------------------------------

To begin, select a ``model_id`` and ``version`` from the pre-trained
models table, as well as a model scope. In this case, you begin by
using “training” as the model scope. Use the utility functions to
retrieve the URI of each of the three components you need to
continue. The HuggingFace model in this example requires a GPU
instance, so use the ``ml.p3.2xlarge`` instance type. For a complete
list of available SageMaker instance types, see the `SageMaker On-Demand Pricing
Table <https://aws.amazon.com/sagemaker/pricing/#On-Demand_Pricing>`__ and select 'Training'.

.. code:: python

   from sagemaker import image_uris, model_uris, script_uris

   model_id, model_version = "huggingface-spc-bert-base-cased", "1.0.0"
   training_instance_type = "ml.p3.2xlarge"
   inference_instance_type = "ml.p3.2xlarge"
   instance_count = 1

   # Retrieve the JumpStart base model S3 URI
   base_model_uri = model_uris.retrieve(
       model_id=model_id, model_version=model_version, model_scope="training"
   )

   # Retrieve the training script and Docker image
   training_script_uri = script_uris.retrieve(
       model_id=model_id, model_version=model_version, script_scope="training"
   )
   training_image_uri = image_uris.retrieve(
       region=None,
       framework=None,
       image_scope="training",
       model_id=model_id,
       model_version=model_version,
       instance_type=training_instance_type,
   )

Next, use the model resource URIs to create an ``Estimator`` and
train it on a custom training dataset. You must specify the S3 path
of your custom training dataset. The Estimator class requires
an ``entry_point`` parameter. In this case, SageMaker uses
“transfer_learning.py”. The training job fails to execute if this
value is not set.

.. code:: python

   from sagemaker.estimator import Estimator
   from sagemaker.session import Session
   from sagemaker import hyperparameters

   # URI of your training dataset
   training_dataset_s3_path = "s3://jumpstart-cache-prod-us-west-2/training-datasets/spc/data.csv"

   # Get the default JumpStart hyperparameters
   default_hyperparameters = hyperparameters.retrieve_default(
       model_id=model_id,
       model_version=model_version,
   )
   # [Optional] Override default hyperparameters with custom values
   default_hyperparameters["epochs"] = "1"

   # Create your SageMaker Estimator instance
   estimator = Estimator(
       image_uri=training_image_uri,
       source_dir=training_script_uri,
       model_uri=base_model_uri,
       entry_point="transfer_learning.py",
       role=Session().get_caller_identity_arn(),
       hyperparameters=default_hyperparameters,
       instance_count=instance_count,
       instance_type=training_instance_type,
       enable_network_isolation=True,
   )

   # Specify the S3 location of training data for the training channel
   estimator.fit(
       {
           "training": training_dataset_s3_path,
       }
   )

While the model is fitting to your training dataset, you will see
console output that reflects the progress the training job is making.
This gives more context about the training job, including the
“transfer_learning.py” script. Model fitting takes a significant
amount of time. The time that it takes varies depending on the
hyperparameters, dataset, and model you use and can range from 15
minutes to 12 hours.

Deploy your Trained Model to a SageMaker Endpoint
-------------------------------------------------


Now that you’ve created your training job, use your
``estimator`` instance to create a SageMaker Endpoint that you can
query for prediction. For an in-depth explanation of this process,
see `Deploy a Pre-Trained Model Directly to a SageMaker
Endpoint <https://sagemaker.readthedocs.io/en/stable/overview.html#deploy-a-pre-trained-model-directly-to-a-sagemaker-endpoint>`__.

**Note:** If you do not pin the model version (i.e.
``_uris.retrieve(model_id="model_id" model_version="*")``), there is
a chance that you pick up a different version of the script or image
for deployment than you did for training. This edge case would arise
if there was a release of a new version of this model in the time it
took your model to train.

.. code:: python

   from sagemaker.utils import name_from_base

   # Retrieve the inference script and Docker image
   deploy_script_uri = script_uris.retrieve(
       model_id=model_id, model_version=model_version, script_scope="inference"
   )
   deploy_image_uri = image_uris.retrieve(
       region=None,
       framework=None,
       image_scope="inference",
       model_id=model_id,
       model_version=model_version,
       instance_type=training_instance_type,
   )

   # Use the estimator from the previous step to deploy to a SageMaker endpoint
   endpoint_name = name_from_base(f"{model_id}-transfer-learning")

   predictor = estimator.deploy(
       initial_instance_count=instance_count,
       instance_type=inference_instance_type,
       entry_point="inference.py",
       image_uri=deploy_image_uri,
       source_dir=deploy_script_uri,
       endpoint_name=endpoint_name,
       enable_network_isolation=True,
   )

Perform Inference on a SageMaker Endpoint
-----------------------------------------

Finally, use the ``predictor`` instance to query your endpoint. For
``huggingface-spc-bert-base-cased``, the predictor accepts an array
of strings. For more information about how to use the predictor, see
the
`Appendix <https://sagemaker.readthedocs.io/en/stable/overview.html#appendix>`__.

.. code:: python

   import json

   data = ["this is the best day of my life", "i am tired"]

   predictor.predict(json.dumps(data).encode("utf-8"), {"ContentType": "application/list-text"})

Appendix
========

To use the ``predictor`` class successfully, you must provide a
second parameter which contains options that the predictor uses to
query your endpoint. This argument must be a ``dict`` with a value
``ContentType`` that refers to the input type for this model. The
following is a list of available machine learning tasks and their
corresponding values.

The ``identifier`` column refers to the segment of the model ID that
corresponds to the model task. For example,
``huggingface-spc-bert-base-cased`` has a ``spc`` identifier, which
means that it is a Sentence Pair Classification model and requires a
ContentType of ``application/list-text``.

.. container::

   +-----------------------+-----------------------+-------------------------+
   | Task                  | Identifier            | ContentType             |
   +-----------------------+-----------------------+-------------------------+
   | Image Classification  | ic                    | "application/x-image"   |
   +-----------------------+-----------------------+-------------------------+
   | Object Detection      | od, od1               | "application/x-image"   |
   +-----------------------+-----------------------+-------------------------+
   | Semantic Segmentation | semseg                | "application/x-image"   |
   +-----------------------+-----------------------+-------------------------+
   | Instance Segmentation | is                    | "application/x-image"   |
   +-----------------------+-----------------------+-------------------------+
   | Text Classification   | tc                    | "application/x-text"    |
   +-----------------------+-----------------------+-------------------------+
   | Sentence Pair         | spc                   | "application/list-text" |
   | Classification        |                       |                         |
   +-----------------------+-----------------------+-------------------------+
   | Extractive Question   | eqa                   | "application/list-text" |
   | Answering             |                       |                         |
   +-----------------------+-----------------------+-------------------------+
   | Text Generation       | textgeneration        | "application/x-text"    |
   +-----------------------+-----------------------+-------------------------+
   | Image Classification  | icembedding           | "application/x-image"   |
   | Embedding             |                       |                         |
   +-----------------------+-----------------------+-------------------------+
   | Text Classification   | tcembedding           | "application/x-text"    |
   | Embedding             |                       |                         |
   +-----------------------+-----------------------+-------------------------+
   | Named-entity          | ner                   | "application/x-text"    |
   | Recognition           |                       |                         |
   +-----------------------+-----------------------+-------------------------+
   | Text Summarization    | summarization         | "application/x-text"    |
   +-----------------------+-----------------------+-------------------------+
   | Text Translation      | translation           | "application/x-text"    |
   +-----------------------+-----------------------+-------------------------+
   | Tabular Regression    | regression            | "text/csv"              |
   +-----------------------+-----------------------+-------------------------+
   | Tabular               | classification        | "text/csv"              |
   | Classification        |                       |                         |
   +-----------------------+-----------------------+-------------------------+

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
    my_predictor.delete_endpoint()

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

You can install all necessary for this feature dependencies using pip:

::

    pip install 'sagemaker[analytics]' --upgrade

For more detailed examples of running hyperparameter tuning jobs, see:

- `Using the TensorFlow estimator with hyperparameter tuning <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/tensorflow_mnist/hpo_tensorflow_mnist.ipynb>`__
- `Bringing your own estimator for hyperparameter tuning <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/r_bring_your_own/tune_r_bring_your_own.ipynb>`__
- `Analyzing results <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb>`__

You can also find these notebooks in the **Hyperprameter Tuning** section of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

For more detailed explanations of the classes that this library provides for automatic model tuning, see:

- `API docs for HyperparameterTuner and parameter range classes <https://sagemaker.readthedocs.io/en/stable/tuner.html>`__
- `API docs for analytics classes <https://sagemaker.readthedocs.io/en/stable/analytics.html>`__

**********************************
SageMaker Asynchronous Inference
**********************************
Amazon SageMaker Asynchronous Inference is a new capability in SageMaker that queues incoming requests and processes them asynchronously.
This option is ideal for requests with large payload sizes up to 1GB, long processing times, and near real-time latency requirements.
You can configure Asynchronous Inference scale the instance count to zero when there are no requests to process, thereby saving costs.
More information about SageMaker Asynchronous Inference can be found in the `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html>`__.

To deploy asynchronous inference endpoint, you will need to create a ``AsyncInferenceConfig`` object.
If you create ``AsyncInferenceConfig`` without specifying its arguments, the default ``S3OutputPath`` will
be ``s3://sagemaker-{REGION}-{ACCOUNTID}/async-endpoint-outputs/{UNIQUE-JOB-NAME}``. (example shown below):

.. code:: python

    from sagemaker.async_inference import AsyncInferenceConfig

    # Create an empty AsyncInferenceConfig object to use default values
    async_config = new AsyncInferenceConfig()

Or you can specify configurations in ``AsyncInferenceConfig`` as you like. All of those configuration parameters
are optional but if you don’t specify the ``output_path``, Amazon SageMaker will use the default ``S3OutputPath``
mentioned above (example shown below):

.. code:: python

    # Specify S3OutputPath, MaxConcurrentInvocationsPerInstance and NotificationConfig in the async config object
    async_config = new AsyncInferenceConfig(
        output_path="s3://{s3_bucket}/{bucket_prefix}/output",
        max_concurrent_invocations_per_instance=10,
        notification_config = {
            "SuccessTopic": "arn:aws:sns:aws-region:account-id:topic-name",
            "ErrorTopic": "arn:aws:sns:aws-region:account-id:topic-name",
        }
    )

Then use the ``AsyncInferenceConfig`` in the estimator's ``deploy()`` method to deploy an asynchronous inference endpoint:

.. code:: python

    # Deploys the model that was generated by fit() to a SageMaker asynchronous inference endpoint
    async_predictor = estimator.deploy(async_inference_config=async_config)

After deployment is complete, it will return an ``AsyncPredictor`` object. To perform asynchronous inference, you first
need to upload data to S3 and then use the ``predict_async()`` method with the s3 URI as the input. It will return an
``AsyncInferenceResponse`` object:

.. code:: python

    # Upload data to S3 bucket then use that as input
    async_response = async_predictor.predict_async(input_path=input_s3_path)

The Amazon SageMaker SDK also enables you to serialize the data and pass the payload data directly to the
``predict_async()`` method. For this pattern of invocation, the Amazon SageMaker SDK will upload the data to an Amazon
S3 bucket under ``s3://sagemaker-{REGION}-{ACCOUNTID}/async-endpoint-inputs/``.

.. code:: python

    # Serializes data and makes a prediction request to the SageMaker asynchronous endpoint
    async_response = async_predictor.predict_async(data=data)

Then you can switch to other stuff and wait the inference to complete. After it is completed, you can check
the result using ``AsyncInferenceResponse``:

.. code:: python

    # Switch back to check the result
    result = async_response.get_result()

Alternatively, if you would like to check for a result periodically and return it upon generation, use the
``predict()`` method

.. code:: python

    # Use predict() to wait for the result
    response = async_predictor.predict(data=data)

    # Or use Amazon S3 input path
    response = async_predictor.predict(input_path=input_s3_path)

Clean up the endpoint and model if needed after inference:

.. code:: python

    # Tears down the SageMaker endpoint and endpoint configuration
    async_predictor.delete_endpoint()

    # Deletes the SageMaker model
    async_predictor.delete_model()

For more details about Asynchronous Inference,
see the API docs for `Asynchronous Inference <https://sagemaker.readthedocs.io/en/stable/api/inference/async_inference.html>`__

*******************************
SageMaker Serverless Inference
*******************************
Amazon SageMaker Serverless Inference enables you to easily deploy machine learning models for inference without having
to configure or manage the underlying infrastructure. After you trained a model, you can deploy it to Amazon Sagemaker
Serverless endpoint and then invoke the endpoint with the model to get inference results back. More information about
SageMaker Serverless Inference can be found in the `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html>`__.

For using SageMaker Serverless Inference, you can either use SageMaker-provided container or Bring Your Own Container model.
A step by step example for using Serverless Inference with MXNet image :

Firstly, create MXNet model

.. code:: python

    from sagemaker.mxnet import MXNetModel
    from sagemaker.serverless import ServerlessInferenceConfig
    import sagemaker

    role = sagemaker.get_execution_role()

    # create MXNet Model Class
    model = MXNetModel(
        model_data="s3://my_bucket/pretrained_model/model.tar.gz", # path to your trained sagemaker model
        role=role, # iam role with permissions to create an Endpoint
        entry_point="inference.py",
        py_version="py3", # Python version
        framework_version="1.6.0", # MXNet framework version
    )

To deploy serverless endpoint, you will need to create a ``ServerlessInferenceConfig``.
If you create ``ServerlessInferenceConfig`` without specifying its arguments, the default ``MemorySizeInMB`` will be **2048** and
the default ``MaxConcurrency`` will be **5** :

.. code:: python

    from sagemaker.serverless import ServerlessInferenceConfig

    # Create an empty ServerlessInferenceConfig object to use default values
    serverless_config = ServerlessInferenceConfig()

Or you can specify ``MemorySizeInMB`` and ``MaxConcurrency`` in ``ServerlessInferenceConfig`` (example shown below):

.. code:: python

    # Specify MemorySizeInMB and MaxConcurrency in the serverless config object
    serverless_config = ServerlessInferenceConfig(
      memory_size_in_mb=4096,
      max_concurrency=10,
    )

Then use the ``ServerlessInferenceConfig`` in the estimator's ``deploy()`` method to deploy a serverless endpoint:

.. code:: python

    # Deploys the model that was generated by fit() to a SageMaker serverless endpoint
    serverless_predictor = estimator.deploy(serverless_inference_config=serverless_config)

Or directly using model's ``deploy()`` method to deploy a serverless endpoint:

.. code:: python

    # Deploys the model to a SageMaker serverless endpoint
    serverless_predictor = model.deploy(serverless_inference_config=serverless_config)

After deployment is complete, you can use predictor's ``predict()`` method to invoke the serverless endpoint just like
real-time endpoints:

.. code:: python

    # Serializes data and makes a prediction request to the SageMaker serverless endpoint
    response = serverless_predictor.predict(data)

Clean up the endpoint and model if needed after inference:

.. code:: python

    # Tears down the SageMaker endpoint and endpoint configuration
    serverless_predictor.delete_endpoint()

    # Deletes the SageMaker model
    serverless_predictor.delete_model()

For more details about ``ServerlessInferenceConfig``,
see the API docs for `Serverless Inference <https://sagemaker.readthedocs.io/en/stable/api/inference/serverless.html>`__

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

    from sagemaker.transformer import Transformer

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

The SageMaker Python SDK supports local mode, which allows you to create estimators, processors, and pipelines, and deploy
them to your local environment. This is a great way to test your deep learning scripts before running them in SageMaker's
managed training or hosting environments. Local Mode is supported for frameworks images (TensorFlow, MXNet, Chainer, PyTorch,
and Scikit-Learn) and images you supply yourself.

You can install all necessary for this feature dependencies using pip:

::

    pip install 'sagemaker[local]' --upgrade

If you want to keep everything local, and not use Amazon S3 either, you can enable "local code" in one of two ways:

- Create a file at ``~/.sagemaker/config.yaml`` that contains:

.. code:: yaml

    local:
      local_code: true

- Create a ``LocalSession`` or ``LocalPipelineSession`` (for local SageMaker pipelines) and configure it directly:

.. code:: python

    from sagemaker.local import LocalSession

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # pass sagemaker_session to your estimator or model

.. note::
    If you enable "local code," then you cannot use the ``dependencies`` parameter in your estimator or model.

We can take the example in  `Using Estimators <#using-estimators>`__ , and use either ``local`` or ``local_gpu`` as the instance type.

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator (no training happens yet)
    mxnet_estimator = MXNet('train.py',
                            role='SageMakerRole',
                            instance_type='local',
                            instance_count=1,
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
                            instance_type='local',
                            instance_count=1,
                            framework_version='1.2.1')

    mxnet_estimator.fit('file:///tmp/my_training_data')
    transformer = mxnet_estimator.transformer(1, 'local', assemble_with='Line', max_payload=1)
    transformer.transform('s3://my/transform/data, content_type='text/csv', split_type='Line')
    transformer.wait()

    # Deletes the SageMaker model
    transformer.delete_model()


Local pipelines
===============

To put everything together, you can use local pipelines to execute various SageMaker jobs in succession. Pipelines can be executed locally by providing a ``LocalPipelineSession`` object to the pipeline’s and pipeline steps’ initializer. ``LocalPipelineSession`` inherits from ``LocalSession``. The difference is ``LocalPipelineSession`` captures the job input step arguments and passes it to the pipeline object instead of executing the job. This behavior is similar to that of `PipelineSession <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_building_pipeline.html#pipeline-session>`__.

Here is an end-to-end example:

.. code:: python

    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.steps import TrainingStep, TransformStep
    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.workflow.pipeline_context import LocalPipelineSession
    from sagemaker.mxnet import MXNet
    from sagemaker.model import Model
    from sagemaker.inputs import TranformerInput
    from sagemaker.transformer import Transformer

    session = LocalPipelineSession()
    mxnet_estimator = MXNet('train.py',
                            role='SageMakerRole',
                            instance_type='local',
                            instance_count=1,
                            framework_version='1.2.1',
                            sagemaker_session=session)

    train_step_args = mxnet_estimator.fit('file:///tmp/my_training_data')

    # Define training step
    train_step = TrainingStep(name='local_mxnet_train', step_args=train_step_args)

    model = Model(
      image_uri=inference_image_uri,
      model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
      sagemaker_session=session,
      role='SageMakerRole'
    )

    # Define create model step
    model_step_args = model.create(instance_type="local", accelerator_type="local")
    model_step = ModelStep(
      name='local_mxnet_model',
      step_args=model_step_args
    )

    transformer =  Transformer(
      model_name=model_step.properties.ModelName,
      instance_type='local',
      instance_count=1,
      sagemaker_session=session
    )
    transform_args = transformer.transform('file:///tmp/my_transform_data')
    # Define transform step
    transform_step = TransformStep(name='local_mxnet_transform', step_args=transform_args)

    # Define the pipeline
    pipeline = Pipeline(name='local_pipeline',
                        steps=[train_step, model_step, transform_step],
                        sagemaker_session=session)

    # Create the pipeline
    pipeline.upsert(role_arn='SageMakerRole', description='local pipeline example')

    # Start a pipeline execution
    execution = pipeline.start()

.. note::
    Currently Pipelines Local Mode only supports the following step types: Training, Processing, Transform, Model (with Create Model arguments only), Condition, and Fail.


For detailed examples of running Docker in local mode, see:

- `TensorFlow local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_script_mode_using_shell_commands/tensorflow_script_mode_using_shell_commands.ipynb>`__.
- `MXNet local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_mnist/mxnet_mnist_with_gluon_local_mode.ipynb>`__.
- `PyTorch local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb>`__.
- `Pipelines local mode example notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/local-mode/sagemaker-pipelines-local-mode.ipynb>`__.


You can also find these notebooks in the **SageMaker Python SDK** section of the **SageMaker Examples** section in a notebook instance.
For information about using sample notebooks in a SageMaker notebook instance, see `Use Example Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html>`__
in the AWS documentation.

A few important notes:

- Only one local mode endpoint can be running at a time.
- If you are using S3 data as input, it is pulled from S3 to your local environment. Ensure you have sufficient space to store the data locally.
- If you run into problems it often due to different Docker containers conflicting. Killing these containers and re-running often solves your problems.
- Local Mode requires Docker Compose and `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__ for ``local_gpu``.

.. warning::

  Local Mode does not yet support the following:

  - Distributed training for ``local_gpu``
  - Gzip compression, Pipe Mode, or manifest files for inputs

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
                                instance_type='ml.p2.xlarge',
                                instance_count=1,
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
                                instance_type='ml.p2.xlarge',
                                instance_count=1,
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
                                instance_type='ml.m4.xlarge',
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

   from sagemaker import image_uris, session
   from sagemaker.model import Model
   from sagemaker.pipeline import PipelineModel
   from sagemaker.sparkml import SparkMLModel

   xgb_image = image_uris.retrieve("xgboost", session.Session().boto_region_name, repo_version="latest")
   xgb_model = Model(model_data="s3://path/to/model.tar.gz", image_uri=xgb_image)
   sparkml_model = SparkMLModel(model_data="s3://path/to/model.tar.gz", env={"SAGEMAKER_SPARKML_SCHEMA": schema})

   model_name = "inference-pipeline-model"
   endpoint_name = "inference-pipeline-endpoint"
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

You can use the following machine learning frameworks to author, schedule and monitor SageMaker workflow.

.. toctree::
    :maxdepth: 2

    workflows/airflow/index
    workflows/step_functions/index
    workflows/pipelines/index
    workflows/lineage/index

************************************
SageMaker Model Building Pipeline
************************************

You can use Amazon SageMaker Model Building Pipelines to orchestrate your machine learning workflow.

For more information, see `SageMaker Model Building Pipeline`_.

.. _SageMaker Model Building Pipeline: https://github.com/aws/sagemaker-python-sdk/blob/master/doc/amazon_sagemaker_model_building_pipeline.rst

**************************
SageMaker Model Monitoring
**************************
You can use Amazon SageMaker Model Monitoring to automatically detect concept drift by monitoring your machine learning models.

For more information, see `SageMaker Model Monitoring`_.

.. _SageMaker Model Monitoring: https://github.com/aws/sagemaker-python-sdk/blob/master/doc/amazon_sagemaker_model_monitoring.rst

******************
SageMaker Debugger
******************
You can use Amazon SageMaker Debugger to automatically detect anomalies while training your machine learning models.

For more information, see `SageMaker Debugger`_.

.. _SageMaker Debugger: https://github.com/aws/sagemaker-python-sdk/blob/master/doc/amazon_sagemaker_debugger.rst

********************
SageMaker Processing
********************
You can use Amazon SageMaker Processing with "Processors" to perform data processing tasks such as data pre- and post-processing, feature engineering, data validation, and model evaluation

.. toctree::
    :maxdepth: 2

    amazon_sagemaker_processing


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

You can use either the generic ``Predictor`` class, which by default does not perform any serialization/deserialization transformations on your input,
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
