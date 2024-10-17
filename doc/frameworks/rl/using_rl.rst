==========================================================
Using Reinforcement Learning with the SageMaker Python SDK
==========================================================

.. contents::

With Reinforcement Learning (RL) Estimators, you can train reinforcement learning models on Amazon SageMaker.

For supported RL toolkits and their versions, see https://github.com/aws/sagemaker-rl-container/#rl-images-provided-by-sagemaker

RL Training
-----------

Training RL models using ``RLEstimator`` is a two-step process:

1. Prepare a training script to run on SageMaker
2. Run this script on SageMaker via an ``RLEstimator``.

You should prepare your script in a separate source file than the notebook, terminal session, or source file you're
using to submit the script to SageMaker via an ``RLEstimator``. This will be discussed in further detail below.

Suppose that you already have a training script called ``coach-train.py`` and have an RL image in your ECR registry
called ``123123123123.dkr.ecr.us-west-2.amazonaws.com/your-rl-registry:your-cool-image-tag`` in ``us-west-2`` region.
You can then create an ``RLEstimator`` with keyword arguments to point to this script and define how SageMaker runs it:

.. code:: python

    from sagemaker.rl import RLEstimator, RLToolkit, RLFramework

    # Train my estimator
    rl_estimator = RLEstimator(
        entry_point='coach-train.py',
        image_uri='123123123123.dkr.ecr.us-west-2.amazonaws.com/your-rl-registry:your-cool-image-tag',
        role='SageMakerRole',
        instance_type='ml.c4.2xlarge',
        instance_count=1
    )


.. tip::
   Refer to `SageMaker RL Docker Containers <#sagemaker-rl-docker-containers>`_ for the more information on how to
   build your custom RL image.

After that, you simply tell the estimator to start a training job:

.. code:: python

    rl_estimator.fit()


In the following sections, we'll discuss how to prepare a training script for execution on SageMaker
and how to run that script on SageMaker using ``RLEstimator``.


Preparing the RL Training Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training script is very similar to a training script you might run outside of SageMaker, but you
can access useful properties about the training environment through various environment variables, such as

*  ``SM_MODEL_DIR``: A string representing the path to the directory to write model artifacts to.
   These artifacts are uploaded to S3 for model hosting.
*  ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
*  ``SM_OUTPUT_DATA_DIR``: A string representing the filesystem path to write output artifacts to. Output artifacts may
   include checkpoints, graphs, and other files to save, not including model artifacts. These artifacts are compressed
   and uploaded to S3 to the same S3 prefix as the model artifacts.

For the exhaustive list of available environment variables, see the
`SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

Note that your RL training script must have the Python version compatible with your custom RL Docker image.

RL Estimators
-------------

The ``RLEstimator`` constructor takes both required and optional arguments.

Required arguments
~~~~~~~~~~~~~~~~~~

The following are required arguments to the ``RLEstimator`` constructor. When you create an instance of ``RLEstimator``, you must include
these in the constructor, either positionally or as keyword arguments.

-  ``entry_point`` Path (absolute or relative) to the Python file which
   should be executed as the entry point to training.
-  ``role`` An AWS IAM role (either name or full ARN). The Amazon
   SageMaker training jobs and APIs that create Amazon SageMaker
   endpoints use this role to access training data and model artifacts.
   After the endpoint is created, the inference code might use the IAM
   role, if accessing AWS resource.
-  ``instance_count`` Number of Amazon EC2 instances to use for training.
-  ``instance_type`` Type of EC2 instance to use for training, for
   example, 'ml.m4.xlarge'.

You must also provide:

-  ``image_uri`` An alternative Docker image to use for training and serving.
   If specified, the estimator will use this image for training and
   hosting. Refer to: `SageMaker RL Docker Containers <#sagemaker-rl-docker-containers>`_
   for the source code to build your custom RL image.


Optional arguments
~~~~~~~~~~~~~~~~~~

When you create an ``RLEstimator`` object, you can specify a number of optional arguments.
For more information, see :class:`sagemaker.rl.estimator.RLEstimator`.

Calling fit
~~~~~~~~~~~

You start your training script by calling ``fit`` on an ``RLEstimator``.
For more information about what arguments can be passed to ``fit``, see :func:`sagemaker.estimator.EstimatorBase.fit`.

Distributed RL Training
-----------------------

Amazon SageMaker RL supports multi-core and multi-instance distributed training.
Depending on your use case, training and/or environment rollout can be distributed.

Please see the `Amazon SageMaker examples <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning>`_
on how it can be done using different RL toolkits.


Saving models
-------------

In order to save your trained PyTorch model for deployment on SageMaker, your training script should save your model
to a certain filesystem path ``/opt/ml/model``. This value is also accessible through the environment variable
``SM_MODEL_DIR``.

Deploying RL Models
-------------------

After an RL Estimator has been fit, you can host the newly created model in SageMaker.

After calling ``fit``, you can call ``deploy`` on an ``RLEstimator`` Estimator to create a SageMaker Endpoint.
The Endpoint runs provided image specified with ``image_uri`` and hosts the model produced by your
training script, which was run when you called ``fit``. This is the model you saved to ``model_dir``.

``deploy`` returns a ``sagemaker.mxnet.MXNetPredictor`` for MXNet or
``sagemaker.tensorflow.TensorFlowPredictor`` for TensorFlow.

``predict`` returns the result of inference against your model.

.. code:: python

    # Train my estimator
    region = 'us-west-2' # the AWS region of your training job
    rl_estimator = RLEstimator(
        entry_point='coach-train.py',
        image_uri=f'123123123123.dkr.ecr.{region}.amazonaws.com/your-rl-registry:your-cool-image-tag',
        role='SageMakerRole',
        instance_type='ml.c4.2xlarge',
        instance_count=1
    )

    rl_estimator.fit()

    # Deploy my estimator to a SageMaker Endpoint and get a MXNetPredictor
    predictor = rl_estimator.deploy(
        instance_type='ml.m4.xlarge',
        initial_instance_count=1
    )

    response = predictor.predict(data)

For more information please see `The SageMaker MXNet Model Server <https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#the-sagemaker-mxnet-model-server>`_
and `Deploying to TensorFlow Serving Endpoints <deploying_tensorflow_serving.html>`_ documentation.


Working with Existing Training Jobs
-----------------------------------

Attaching to existing training jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can attach an RL Estimator to an existing training job using the
``attach`` method.

.. code:: python

    my_training_job_name = 'MyAwesomeRLTrainingJob'
    rl_estimator = RLEstimator.attach(my_training_job_name)

After attaching, if the training job has finished with job status "Completed", it can be
``deploy``\ ed to create a SageMaker Endpoint and return a ``Predictor``. If the training job is in progress,
attach will block and display log messages from the training job, until the training job completes.

The ``attach`` method accepts the following arguments:

-  ``training_job_name:`` The name of the training job to attach to.
-  ``sagemaker_session:`` The Session used to interact with SageMaker

RL Training Examples
--------------------

Amazon provides several example Jupyter notebooks that demonstrate end-to-end training on Amazon SageMaker using RL.
Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the sample notebooks folder.


SageMaker RL Docker Containers
------------------------------

For more information about how build your own RL image and use script mode with your image, see
`Building your image section on SageMaker RL containers repository <https://github.com/aws/sagemaker-rl-container?tab=readme-ov-file#building-your-image/>`_
and `Bring your own model with Amazon SageMaker script mode <https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode/>`_.
