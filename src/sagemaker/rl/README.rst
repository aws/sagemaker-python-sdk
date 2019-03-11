===========================================
SageMaker Reinforcement Learning Estimators
===========================================

With Reinforcement Learning (RL) Estimators, you can train reinforcement learning models on Amazon SageMaker.

Supported versions of Coach: ``0.11.1``, ``0.10.1`` with TensorFlow, ``0.11.0`` with TensorFlow or MXNet.
For more information about Coach, see https://github.com/NervanaSystems/coach

Supported versions of Ray: ``0.5.3`` with TensorFlow.
For more information about Ray, see https://github.com/ray-project/ray

Table of Contents
-----------------

1. `RL Training <#rl-training>`__
2. `RL Estimators <#rl-estimators>`__
3. `Distributed RL Training <#distributed-rl-training>`__
4. `Saving models <#saving-models>`__
5. `Deploying RL Models <#deploying-rl-models>`__
6. `RL Training Examples <#rl-training-examples>`__
7. `SageMaker RL Docker Containers <#sagemaker-rl-docker-containers>`__


RL Training
-----------

Training RL models using ``RLEstimator`` is a two-step process:

1. Prepare a training script to run on SageMaker
2. Run this script on SageMaker via an ``RlEstimator``.

You should prepare your script in a separate source file than the notebook, terminal session, or source file you're
using to submit the script to SageMaker via an ``RlEstimator``. This will be discussed in further detail below.

Suppose that you already have a training script called ``coach-train.py``.
You can then create an ``RLEstimator`` with keyword arguments to point to this script and define how SageMaker runs it:

.. code:: python

    from sagemaker.rl import RLEstimator, RLToolkit, RLFramework

    rl_estimator = RLEstimator(entry_point='coach-train.py',
                               toolkit=RLToolkit.COACH,
                               toolkit_version='0.11.1',
                               framework=RLFramework.TENSORFLOW,
                               role='SageMakerRole',
                               train_instance_type='ml.p3.2xlarge',
                               train_instance_count=1)

After that, you simply tell the estimator to start a training job:

.. code:: python

    rl_estimator.fit()

In the following sections, we'll discuss how to prepare a training script for execution on SageMaker
and how to run that script on SageMaker using ``RLEstimator``.


Preparing the RL Training Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your RL training script must be a Python 3.5 compatible source file from MXNet framework or Python 3.6 for TensorFlow.

The training script is very similar to a training script you might run outside of SageMaker, but you
can access useful properties about the training environment through various environment variables, such as

* ``SM_MODEL_DIR``: A string representing the path to the directory to write model artifacts to.
  These artifacts are uploaded to S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_OUTPUT_DATA_DIR``: A string representing the filesystem path to write output artifacts to. Output artifacts may
  include checkpoints, graphs, and other files to save, not including model artifacts. These artifacts are compressed
  and uploaded to S3 to the same S3 prefix as the model artifacts.

For the exhaustive list of available environment variables, see the
`SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.


RL Estimators
-------------

The ``RLEstimator`` constructor takes both required and optional arguments.

Required arguments
~~~~~~~~~~~~~~~~~~

The following are required arguments to the ``RLEstimator`` constructor. When you create an instance of RLEstimator, you must include
these in the constructor, either positionally or as keyword arguments.

-  ``entry_point`` Path (absolute or relative) to the Python file which
   should be executed as the entry point to training.
-  ``role`` An AWS IAM role (either name or full ARN). The Amazon
   SageMaker training jobs and APIs that create Amazon SageMaker
   endpoints use this role to access training data and model artifacts.
   After the endpoint is created, the inference code might use the IAM
   role, if accessing AWS resource.
-  ``train_instance_count`` Number of Amazon EC2 instances to use for
   training.
-  ``train_instance_type`` Type of EC2 instance to use for training, for
   example, 'ml.m4.xlarge'.

You must as well include either:

-  ``toolkit`` RL toolkit (Ray RLlib or Coach) you want to use for executing your model training code.

-  ``toolkit_version`` RL toolkit version you want to be use for executing your model training code.

-  ``framework`` Framework (MXNet or TensorFlow) you want to be used as
   a toolkit backed for reinforcement learning training.

or provide:

-  ``image_name`` An alternative docker image to use for training and
   serving.  If specified, the estimator will use this image for training and
   hosting, instead of selecting the appropriate SageMaker official image based on
   framework_version and py_version. Refer to: `SageMaker RL Docker Containers
   <#sagemaker-rl-docker-containers>`_ for details on what the Official images support
   and where to find the source code to build your custom image.


Optional arguments
~~~~~~~~~~~~~~~~~~

The following are optional arguments. When you create an ``RlEstimator`` object, you can specify these as keyword arguments.

-  ``source_dir`` Path (absolute or relative) to a directory with any
   other training source code dependencies including the entry point
   file. Structure within this directory will be preserved when training
   on SageMaker.
-  ``dependencies (list[str])`` A list of paths to directories (absolute or relative) with
   any additional libraries that will be exported to the container (default: ``[]``).
   The library folders will be copied to SageMaker in the same folder where the entrypoint is copied.
   If the ``source_dir`` points to S3, code will be uploaded and the S3 location will be used
   instead. Example:

        The following call
        >>> RLEstimator(entry_point='train.py',
                        toolkit=RLToolkit.COACH,
                        toolkit_version='0.11.0',
                        framework=RLFramework.TENSORFLOW,
                        dependencies=['my/libs/common', 'virtual-env'])
        results in the following inside the container:

        >>> $ ls

        >>> opt/ml/code
        >>>     ├── train.py
        >>>     ├── common
        >>>     └── virtual-env

-  ``hyperparameters`` Hyperparameters that will be used for training.
   Will be made accessible as a ``dict[str, str]`` to the training code on
   SageMaker. For convenience, accepts other types besides strings, but
   ``str`` will be called on keys and values to convert them before
   training.
-  ``train_volume_size`` Size in GB of the EBS volume to use for storing
   input data during training. Must be large enough to store training
   data if ``input_mode='File'`` is used (which is the default).
-  ``train_max_run`` Timeout in seconds for training, after which Amazon
   SageMaker terminates the job regardless of its current status.
-  ``input_mode`` The input mode that the algorithm supports. Valid
   modes: 'File' - Amazon SageMaker copies the training dataset from the
   S3 location to a directory in the Docker container. 'Pipe' - Amazon
   SageMaker streams data directly from S3 to the container via a Unix
   named pipe.
-  ``output_path`` S3 location where you want the training result (model
   artifacts and optional output files) saved. If not specified, results
   are stored to a default bucket. If the bucket with the specific name
   does not exist, the estimator creates the bucket during the ``fit``
   method execution.
-  ``output_kms_key`` Optional KMS key ID to optionally encrypt training
   output with.
-  ``job_name`` Name to assign for the training job that the ``fit```
   method launches. If not specified, the estimator generates a default
   job name, based on the training image name and current timestamp

Calling fit
~~~~~~~~~~~

You start your training script by calling ``fit`` on an ``RLEstimator``. ``fit`` takes both a few optional
arguments.

Optional arguments
''''''''''''''''''

-  ``inputs``: This can take one of the following forms: A string
   S3 URI, for example ``s3://my-bucket/my-training-data``. In this
   case, the S3 objects rooted at the ``my-training-data`` prefix will
   be available in the default ``train`` channel. A dict from
   string channel names to S3 URIs. In this case, the objects rooted at
   each S3 prefix will available as files in each channel directory.
-  ``wait``: Defaults to True, whether to block and wait for the
   training script to complete before returning.
-  ``logs``: Defaults to True, whether to show logs produced by training
   job in the Python session. Only meaningful when wait is True.


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

After calling ``fit``, you can call ``deploy`` on an ``RlEstimator`` Estimator to create a SageMaker Endpoint.
The Endpoint runs one of the SageMaker-provided model server based on the ``framework`` parameter
specified in the ``RLEstimator`` constructor and hosts the model produced by your training script,
which was run when you called ``fit``. This was the model you saved to ``model_dir``.
In case if ``image_name`` was specified it would use provided image for the deployment.

``deploy`` returns a ``sagemaker.mxnet.MXNetPredictor`` for MXNet or
``sagemaker.tensorflow.serving.Predictor`` for TensorFlow.

``predict`` returns the result of inference against your model.

.. code:: python

    # Train my estimator
    rl_estimator = RLEstimator(entry_point='coach-train.py',
                               toolkit=RLToolkit.COACH,
                               toolkit_version='0.11.0',
                               framework=RLFramework.MXNET,
                               role='SageMakerRole',
                               train_instance_type='ml.c4.2xlarge',
                               train_instance_count=1)

    rl_estimator.fit()

    # Deploy my estimator to a SageMaker Endpoint and get a MXNetPredictor
    predictor = rl_estimator.deploy(instance_type='ml.m4.xlarge',
                                    initial_instance_count=1)

    response = predictor.predict(data)

For more information please see `The SageMaker MXNet Model Server <https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/mxnet#the-sagemaker-mxnet-model-server>`_
and `Deploying to TensorFlow Serving Endpoints <https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_tensorflow_serving.rst>`_ documentation.


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

-  ``training_job_name:`` The name of the training job to attach
   to.
-  ``sagemaker_session:`` The Session used
   to interact with SageMaker

RL Training Examples
--------------------

Amazon provides several example Jupyter notebooks that demonstrate end-to-end training on Amazon SageMaker using RL.
Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the sample notebooks folder.


SageMaker RL Docker Containers
------------------------------

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several
libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control
the environment your script runs in.

SageMaker runs RL Estimator scripts in either Python 3.5 for MXNet or Python 3.6 for TensorFlow.

The Docker images have the following dependencies installed:

+-------------------------+-------------------+-------------------+-------------------+
| Dependencies            |      Coach 0.10.1 |      Coach 0.11.0 |         Ray 0.5.3 |
+-------------------------+-------------------+-------------------+-------------------+
| Python                  |               3.6 |     3.5(MXNet) or |               3.6 |
|                         |                   |   3.6(TensorFlow) |                   |
+-------------------------+-------------------+-------------------+-------------------+
| CUDA (GPU image only)   |               9.0 |               9.0 |               9.0 |
+-------------------------+-------------------+-------------------+-------------------+
| DL Framework            | TensorFlow-1.11.0 |    MXNet-1.3.0 or | TensorFlow-1.11.0 |
|                         |                   | TensorFlow-1.11.0 |                   |
+-------------------------+-------------------+-------------------+-------------------+
| gym                     |            0.10.5 |            0.10.5 |            0.10.5 |
+-------------------------+-------------------+-------------------+-------------------+

The Docker images extend Ubuntu 16.04.

You can select version of  by passing a ``framework_version`` keyword arg to the RL Estimator constructor.
Currently supported versions are listed in the above table. You can also set ``framework_version`` to only specify major and
minor version, which will cause your training script to be run on the latest supported patch version of that minor
version.

Alternatively, you can build your own image by following the instructions in the SageMaker RL containers
repository, and passing ``image_name`` to the RL Estimator constructor.

You can visit `the SageMaker RL containers repository <https://github.com/aws/sagemaker-rl-container>`_.
