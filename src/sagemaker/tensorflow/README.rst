TensorFlow SageMaker Estimators and Models
==========================================

TensorFlow SageMaker Estimators allow you to run your own TensorFlow
training algorithms on SageMaker Learner, and to host your own TensorFlow
models on SageMaker Hosting.

Documentation of the previous Legacy Mode versions: `1.4.1 <https://github.com/aws/sagemaker-python-sdk/tree/v1.0.0#tensorflow-sagemaker-estimators>`_, `1.5.0 <https://github.com/aws/sagemaker-python-sdk/tree/v1.1.0#tensorflow-sagemaker-estimators>`_, `1.6.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.5.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.7.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.5.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.8.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.5.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.9.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.9.2/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_, `1.10.0 <https://github.com/aws/sagemaker-python-sdk/blob/v1.10.0/src/sagemaker/tensorflow/README.rst#tensorflow-sagemaker-estimators-and-models>`_

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| WARNING                                                                                                                                                                     |
+=============================================================================================================================================================================+
| We have added a new format of your TensorFlow training script with TensorFlow version 1.11.                                                                                 |
| This new way gives the user script more flexibility.                                                                                                                        |
| This new format is called Script Mode, as opposed to Legacy Mode, which is what we support with TensorFlow 1.11 and older versions.                                         |
| In addition we are adding Python 3 support with Script Mode.                                                                                                                |
| Last supported version of Legacy Mode will be TensorFlow 1.12.                                                                                                              |
| Script Mode is available with TensorFlow version 1.11 and newer.                                                                                                            |
| Make sure you refer to the correct version of this README when you prepare your script.                                                                                     |
| You can find the Legacy Mode README `here <https://github.com/aws/sagemaker-python-sdk/tree/v1.12.0/src/sagemaker/tensorflow#tensorflow-sagemaker-estimators-and-models>`_. |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Supported versions of TensorFlow for Elastic Inference: ``1.11.0``, ``1.12.0``.

Training with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~

Training TensorFlow models using ``sagemaker.tensorflow.TensorFlow`` is a two-step process.
First, you prepare your training script, then second, you run it on
SageMaker Learner via the ``sagemaker.tensorflow.TensorFlow`` estimator.

Preparing a Script Mode training script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your TensorFlow training script must be a Python 2.7- or 3.6-compatible source file.

The training script is very similar to a training script you might run outside of SageMaker, but you can access useful properties about the training environment through various environment variables, including the following:

* ``SM_MODEL_DIR``: A string that represents the local path where the training job can write the model artifacts to.
  After training, artifacts in this directory are uploaded to S3 for model hosting. This is different than the ``model_dir``
  argument passed in your training script which is a S3 location. ``SM_MODEL_DIR`` is always set to ``/opt/ml/model``.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_OUTPUT_DATA_DIR``: A string that represents the path to the directory to write output artifacts to.
  Output artifacts might include checkpoints, graphs, and other files to save, but do not include model artifacts.
  These artifacts are compressed and uploaded to S3 to an S3 bucket with the same prefix as the model artifacts.
* ``SM_CHANNEL_XXXX``: A string that represents the path to the directory that contains the input data for the specified channel.
  For example, if you specify two input channels in the TensorFlow estimator's ``fit`` call, named 'train' and 'test', the environment variables ``SM_CHANNEL_TRAIN`` and ``SM_CHANNEL_TEST`` are set.

For the exhaustive list of available environment variables, see the `SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model, and saves a model to ``SM_CHANNEL_TRAIN`` so that it can be deployed for inference later.
Hyperparameters are passed to your script as arguments and can be retrieved with an ``argparse.ArgumentParser`` instance.
For example, a training script might start with the following:

.. code:: python

    import argparse
    import os

    if __name__ =='__main__':

        parser = argparse.ArgumentParser()

        # hyperparameters sent by the client are passed as command-line arguments to the script.
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=0.1)

        # input data and model directories
        parser.add_argument('--model_dir', type=str)
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

        args, _ = parser.parse_known_args()

        # ... load from args.train and args.test, train a model, write model to args.model_dir.

Because the SageMaker imports your training script, putting your training launching code in a main guard (``if __name__=='__main__':``)
is good practice.

Note that SageMaker doesn't support argparse actions.
If you want to use, for example, boolean hyperparameters, you need to specify ``type`` as ``bool`` in your script and provide an explicit ``True`` or ``False`` value for this hyperparameter when instantiating your TensorFlow estimator.

Adapting your local TensorFlow script
'''''''''''''''''''''''''''''''''''''

If you have a TensorFlow training script that runs outside of SageMaker please follow the directions here:

1. Make sure your script can handle ``--model_dir`` as an additional command line argument. If you did not specify a
location when the TensorFlow estimator is constructed a S3 location under the default training job bucket will be passed
in here. Distributed training with parameter servers requires you use the ``tf.estimator.train_and_evaluate`` API and
a S3 location is needed as the model directory during training. Here is an example:

.. code:: python

    estimator = tf.estimator.Estimator(model_fn=my_model_fn, model_dir=args.model_dir)
    ...
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

2. Load input data from the input channels. The input channels are defined when ``fit`` is called. For example:

.. code:: python

    estimator.fit({'train':'s3://my-bucket/my-training-data',
                  'eval':'s3://my-bucket/my-evaluation-data'})

In your training script the channels will be stored in environment variables ``SM_CHANNEL_TRAIN`` and
``SM_CHANNEL_EVAL``. You can add them to your argument parsing logic like this:

.. code:: python

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))

3. Export your final model to path stored in environment variable ``SM_MODEL_DIR`` which should always be
   ``/opt/ml/model``. At end of training SageMaker will upload the model file under ``/opt/ml/model`` to
   ``output_path``.


Training with TensorFlow estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calling fit
'''''''''''

To use Script Mode, set at least one of these args

- ``py_version='py3'``
- ``script_mode=True``

Please note that when using Script Mode, your training script need to accept the following args:

- ``model_dir``

Please note that the following args are not permitted when using Script Mode:

- ``checkpoint_path``
- ``training_steps``
- ``evaluation_steps``
- ``requirements_file``

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                            train_instance_count=1, train_instance_type='ml.p2.xlarge',
                            framework_version='1.12', py_version='py3')
  tf_estimator.fit('s3://bucket/path/to/training/data')

Where the S3 url is a path to your training data, within Amazon S3. The
constructor keyword arguments define how SageMaker runs your training
script which we discussed earlier.

You start your training script by calling ``fit`` on a ``TensorFlow`` estimator. ``fit`` takes
both required and optional arguments.

Required argument
"""""""""""""""""

- ``inputs``: The S3 location(s) of datasets to be used for training. This can take one of two forms:

  - ``str``: An S3 URI, for example ``s3://my-bucket/my-training-data``, which indicates the dataset's location.
  - ``dict[str, str]``: A dictionary mapping channel names to S3 locations, for example ``{'train': 's3://my-bucket/my-training-data/train', 'test': 's3://my-bucket/my-training-data/test'}``
  - ``sagemaker.session.s3_input``: channel configuration for S3 data sources that can provide additional information as well as the path to the training dataset. See `the API docs <https://sagemaker.readthedocs.io/en/latest/session.html#sagemaker.session.s3_input>`_ for full details.

Optional arguments
""""""""""""""""""

- ``wait (bool)``: Defaults to True, whether to block and wait for the
  training script to complete before returning.
  If set to False, it will return immediately, and can later be attached to.
- ``logs (bool)``: Defaults to True, whether to show logs produced by training
  job in the Python session. Only meaningful when wait is True.
- ``run_tensorboard_locally (bool)``: Defaults to False. If set to True a Tensorboard command will be printed out.
- ``job_name (str)``: Training job name. If not specified, the estimator generates a default job name,
  based on the training image name and current timestamp.

What happens when fit is called
"""""""""""""""""""""""""""""""

Calling ``fit`` starts a SageMaker training job. The training job will execute the following.

- Starts ``train_instance_count`` EC2 instances of the type ``train_instance_type``.
- On each instance, it will do the following steps:

  - starts a Docker container optimized for TensorFlow.
  - downloads the dataset.
  - setup up training related environment varialbes
  - setup up distributed training environment if configured to use parameter server
  - starts asynchronous training

If the ``wait=False`` flag is passed to ``fit``, then it will return immediately. The training job will continue running
asynchronously. At a later time, a Tensorflow Estimator can be obtained by attaching to the existing training job. If
the training job is not finished it will start showing the standard output of training and wait until it completes.
After attaching, the estimator can be deployed as usual.

.. code:: python

    tf_estimator.fit(your_input_data, wait=False)
    training_job_name = tf_estimator.latest_training_job.name

    # after some time, or in a separate Python notebook, we can attach to it again.

    tf_estimator = TensorFlow.attach(training_job_name=training_job_name)

Distributed Training
''''''''''''''''''''

To run your training job with multiple instances in a distributed fashion, set ``train_instance_count``
to a number larger than 1. We support two different types of distributed training, parameter server and Horovod.
The ``distributions`` parameter is used to configure which distributed training strategy to use.

Training with parameter servers
"""""""""""""""""""""""""""""""

If you specify parameter_server as the value of the distributions parameter, the container launches a parameter server
thread on each instance in the training cluster, and then executes your training code. You can find more information on
TensorFlow distributed training at `TensorFlow docs <https://www.tensorflow.org/deploy/distributed>`__.
To enable parameter server training:

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                            train_instance_count=2, train_instance_type='ml.p2.xlarge',
                            framework_version='1.11', py_version='py3',
                            distributions={'parameter_server': {'enabled': True}})
  tf_estimator.fit('s3://bucket/path/to/training/data')

Training with Horovod
"""""""""""""""""""""

Horovod is a distributed training framework based on MPI. You can find more details at `Horovod README <https://github.com/uber/horovod>`__.

The container sets up the MPI environment and executes the ``mpirun`` command enabling you to run any Horovod
training script with Script Mode.

Training with ``MPI`` is configured by specifying following fields in ``distributions``:

- ``enabled (bool)``: If set to ``True``, the MPI setup is performed and ``mpirun`` command is executed.
- ``processes_per_host (int)``: Number of processes MPI should launch on each host. Note, this should not be
  greater than the available slots on the selected instance type. This flag should be set for the multi-cpu/gpu
  training.
- ``custom_mpi_options (str)``:  Any `mpirun` flag(s) can be passed in this field that will be added to the `mpirun`
  command executed by SageMaker to launch distributed horovod training.


In the below example we create an estimator to launch Horovod distributed training with 2 processes on one host:

.. code:: python

    from sagemaker.tensorflow import TensorFlow

    tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                              train_instance_count=1, train_instance_type='ml.p2.xlarge',
                              framework_version='1.12', py_version='py3',
                              distributions={
                                  'mpi': {
                                      'enabled': True,
                                      'processes_per_host': 2,
                                      'custom_mpi_options': '--NCCL_DEBUG INFO'
                                  }
                              })
    tf_estimator.fit('s3://bucket/path/to/training/data')

sagemaker.tensorflow.TensorFlow class
'''''''''''''''''''''''''''''''''''''

The ``TensorFlow`` constructor takes both required and optional arguments.

Required:

- ``entry_point (str)`` Path (absolute or relative) to the Python file which
  should be executed as the entry point to training.
- ``role (str)`` An AWS IAM role (either name or full ARN). The Amazon
  SageMaker training jobs and APIs that create Amazon SageMaker
  endpoints use this role to access training data and model artifacts.
  After the endpoint is created, the inference code might use the IAM
  role, if accessing AWS resource.
- ``train_instance_count (int)`` Number of Amazon EC2 instances to use for
  training.
- ``train_instance_type (str)`` Type of EC2 instance to use for training, for
  example, 'ml.c4.xlarge'.

Optional:

- ``source_dir (str)`` Path (absolute or relative) to a directory with any
  other training source code dependencies including the entry point
  file. Structure within this directory will be preserved when training
  on SageMaker.
- ``dependencies (list[str])`` A list of paths to directories (absolute or relative) with
  any additional libraries that will be exported to the container (default: ``[]``).
  The library folders will be copied to SageMaker in the same folder where the entrypoint is copied.
  If the ``source_dir`` points to S3, code will be uploaded and the S3 location will be used
  instead. Example:

  The following call

  >>> TensorFlow(entry_point='train.py', dependencies=['my/libs/common', 'virtual-env'])

  results in the following inside the container:

  >>> opt/ml/code
  >>>     ├── train.py
  >>>     ├── common
  >>>     └── virtual-env

- ``hyperparameters (dict[str, ANY])`` Hyperparameters that will be used for training.
  Will be made accessible as command line arguments.
- ``train_volume_size (int)`` Size in GB of the EBS volume to use for storing
  input data during training. Must be large enough to the store training
  data.
- ``train_max_run (int)`` Timeout in seconds for training, after which Amazon
  SageMaker terminates the job regardless of its current status.
- ``output_path (str)`` S3 location where you want the training result (model
  artifacts and optional output files) saved. If not specified, results
  are stored to a default bucket. If the bucket with the specific name
  does not exist, the estimator creates the bucket during the ``fit``
  method execution.
- ``output_kms_key`` Optional KMS key ID to optionally encrypt training
  output with.
- ``base_job_name`` Name to assign for the training job that the ``fit``
  method launches. If not specified, the estimator generates a default
  job name, based on the training image name and current timestamp.
- ``image_name`` An alternative docker image to use for training and
  serving.  If specified, the estimator will use this image for training and
  hosting, instead of selecting the appropriate SageMaker official image based on
  ``framework_version`` and ``py_version``. Refer to: `SageMaker TensorFlow Docker Containers
  <#sagemaker-tensorflow-docker-containers>`_ for details on what the official images support
  and where to find the source code to build your custom image.
- ``script_mode (bool)`` Whether to use Script Mode or not. Script mode is the only available training mode in Python 3,
  setting ``py_version`` to ``py3`` automatically sets ``script_mode`` to True.
- ``model_dir (str)`` Location where model data, checkpoint data, and TensorBoard checkpoints should be saved during training.
  If not specified a S3 location will be generated under the training job's default bucket. And ``model_dir`` will be
  passed in your training script as one of the command line arguments.
- ``distributions (dict)`` Configure your distribution strategy with this argument.

Training with Pipe Mode using PipeModeDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Amazon SageMaker allows users to create training jobs using Pipe input mode.
With Pipe input mode, your dataset is streamed directly to your training instances instead of being downloaded first.
This means that your training jobs start sooner, finish quicker, and need less disk space.

SageMaker TensorFlow provides an implementation of ``tf.data.Dataset`` that makes it easy to take advantage of Pipe
input mode in SageMaker. You can replace your ``tf.data.Dataset`` with a ``sagemaker_tensorflow.PipeModeDataset`` to
read TFRecords as they are streamed to your training instances.

In your ``entry_point`` script, you can use ``PipeModeDataset`` like a ``Dataset``. In this example, we create a
``PipeModeDataset`` to read TFRecords from the 'training' channel:


.. code:: python

    from sagemaker_tensorflow import PipeModeDataset

    features = {
        'data': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.int64),
    }

    def parse(record):
        parsed = tf.parse_single_example(record, features)
        return ({
            'data': tf.decode_raw(parsed['data'], tf.float64)
        }, parsed['labels'])

    def train_input_fn(training_dir, hyperparameters):
        ds = PipeModeDataset(channel='training', record_format='TFRecord')
        ds = ds.repeat(20)
        ds = ds.prefetch(10)
        ds = ds.map(parse, num_parallel_calls=10)
        ds = ds.batch(64)
        return ds


To run training job with Pipe input mode, pass in ``input_mode='Pipe'`` to your TensorFlow Estimator:


.. code:: python

    from sagemaker.tensorflow import TensorFlow

    tf_estimator = TensorFlow(entry_point='tf-train-with-pipemodedataset.py', role='SageMakerRole',
                              training_steps=10000, evaluation_steps=100,
                              train_instance_count=1, train_instance_type='ml.p2.xlarge',
                              framework_version='1.10.0', input_mode='Pipe')

    tf_estimator.fit('s3://bucket/path/to/training/data')


If your TFRecords are compressed, you can train on Gzipped TF Records by passing in ``compression='Gzip'`` to the call to
``fit()``, and SageMaker will automatically unzip the records as data is streamed to your training instances:

.. code:: python

    from sagemaker.session import s3_input

    train_s3_input = s3_input('s3://bucket/path/to/training/data', compression='Gzip')
    tf_estimator.fit(train_s3_input)


You can learn more about ``PipeModeDataset`` in the sagemaker-tensorflow-extensions repository: https://github.com/aws/sagemaker-tensorflow-extensions


Training with MKL-DNN disabled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker TensorFlow CPU images use TensorFlow built with Intel® MKL-DNN optimization.

In certain cases you might be able to get a better performance by disabling this optimization
(`for example when using small models <https://github.com/awslabs/amazon-sagemaker-examples/blob/d88d1c19861fb7733941969f5a68821d9da2982e/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/iris_dnn_classifier.py#L7-L9>`_)

You can disable MKL-DNN optimization for TensorFlow ``1.8.0`` and above by setting two following environment variables:

.. code:: python

    import os

    os.environ['TF_DISABLE_MKL'] = '1'
    os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'


Deploying TensorFlow Serving models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a TensorFlow estimator has been fit, it saves a TensorFlow SavedModel in
the S3 location defined by ``output_path``. You can call ``deploy`` on a TensorFlow
estimator to create a SageMaker Endpoint.

SageMaker provides two different options for deploying TensorFlow models to a SageMaker
Endpoint:

- The first option uses a Python-based server that allows you to specify your own custom
  input and output handling functions in a Python script. This is the default option.

  See `Deploying to Python-based Endpoints <deploying_python.rst>`_ to learn how to use this option.


- The second option uses a TensorFlow Serving-based server to provide a super-set of the
  `TensorFlow Serving REST API <https://www.tensorflow.org/serving/api_rest>`_. This option
  does not require (or allow) a custom python script.

  See `Deploying to TensorFlow Serving Endpoints <deploying_tensorflow_serving.rst>`_ to learn how to use this option.


SageMaker TensorFlow Docker containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The containers include the following Python packages:

+--------------------------------+---------------+-------------------+
| Dependencies                   | Script Mode   | Legacy Mode       |
+--------------------------------+---------------+-------------------+
| boto3                          | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| botocore                       | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| CUDA (GPU image only)          | 9.0           | 9.0               |
+--------------------------------+---------------+-------------------+
| numpy                          | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| Pillow                         | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| scipy                          | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| sklean                         | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| h5py                           | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| pip                            | 18.1          | 18.1              |
+--------------------------------+---------------+-------------------+
| curl                           | Latest        | Latest            |
+--------------------------------+---------------+-------------------+
| tensorflow                     | 1.12.0        | 1.12.0            |
+--------------------------------+---------------+-------------------+
| tensorflow-serving-api         | 1.12.0        | None              |
+--------------------------------+---------------+-------------------+
| sagemaker-containers           | >=2.3.5       | >=2.3.5           |
+--------------------------------+---------------+-------------------+
| sagemaker-tensorflow-container | 1.0           | 1.0               |
+--------------------------------+---------------+-------------------+
| Python                         | 2.7 or 3.6    | 2.7               |
+--------------------------------+---------------+-------------------+

Legacy Mode TensorFlow Docker images support Python 2.7. Script Mode TensorFlow Docker images support both Python 2.7
and Python 3.6. The Docker images extend Ubuntu 16.04.

You can select version of TensorFlow by passing a ``framework_version`` keyword arg to the TensorFlow Estimator constructor. Currently supported versions are listed in the table above. You can also set ``framework_version`` to only specify major and minor version, e.g ``'1.6'``, which will cause your training script to be run on the latest supported patch version of that minor version, which in this example would be 1.6.0.
Alternatively, you can build your own image by following the instructions in the SageMaker TensorFlow containers
repository, and passing ``image_name`` to the TensorFlow Estimator constructor.

For more information on the contents of the images, see the SageMaker TensorFlow containers repository here: https://github.com/aws/sagemaker-tensorflow-containers/
