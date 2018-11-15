TensorFlow SageMaker Estimators and Models
==========================================

TensorFlow SageMaker Estimators allow you to run your own TensorFlow
training algorithms on SageMaker Learner, and to host your own TensorFlow
models on SageMaker Hosting.

Supported versions of TensorFlow: ``1.4.1``, ``1.5.0``, ``1.6.0``, ``1.7.0``, ``1.8.0``, ``1.9.0``, ``1.10.0``, ``1.11.0``.

+----------------------------------------------------------------------------------------------------+
| WARNING                                                                                            |
+====================================================================================================+
| We are adding a new way of providing your TensorFlow training script with TensorFlow version 1.11. |
| This new way gives the user script more flexibility. We are calling this new feature Script Mode.  |
| In addition we are adding Python 3 support with Script Mode.                                       |
| Make sure you refer to the correct section of this README when you prepare your script.            |
| For clarification we will name the current training script configuration Legacy Mode.              |
| Legacy Mode is going to be deprecated early 2019.                                                  |
+----------------------------------------------------------------------------------------------------+

Training with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~

Training TensorFlow models using a ``sagemaker.tensorflow.TensorFlow``
is a two-step process.
First, you prepare your training script, then second, you run it on
SageMaker Learner via the ``sagemaker.tensorflow.TensorFlow`` estimator.

Suppose that you already have a TensorFlow training script called
``tf-train.py``. You can train this script in SageMaker Learner as
follows:

For Script Mode (only available with TensorFlow version 1.11 or higher)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Script Mode will be used if ``py_version`` is set to ``py3``. To use Python 2 with script mode set ``script_mode`` to
``py2``. ``checkpoint_path``, ``training_steps``, ``evaluation_steps`` and ``requirements_file`` are deprecated in
Script Mode. ``model_dir`` is added to specify where the checkpoint and saved model files will be export to during
training. ``model_dir`` will be passed to your training script as a command line argument. Please make sure your
training script can handle it.

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                            train_instance_count=1, train_instance_type='ml.p2.xlarge',
                            framework_version=1.11, py_version='py3')
  tf_estimator.fit('s3://bucket/path/to/training/data')

For Legacy Mode
^^^^^^^^^^^^^^^

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                            training_steps=10000, evaluation_steps=100,
                            train_instance_count=1, train_instance_type='ml.p2.xlarge',
                            framework_version=1.10.0)
  tf_estimator.fit('s3://bucket/path/to/training/data')

Where the S3 url is a path to your training data, within Amazon S3. The
constructor keyword arguments define how SageMaker runs your training
script and are discussed, in detail, in a later section.

In the following sections, we'll discuss how to prepare a training script for execution on
SageMaker, then how to run that script on SageMaker using a ``sagemaker.tensorflow.TensorFlow``
estimator.

To learn how to prepare training script and how training works in general see:

- Legacy Mode `Legacy Mode Training <legacy_training.rst>`_
- Script Mode `Script Mode Training <sm_training.rst>`_

sagemaker.tensorflow.TensorFlow class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``TensorFlow`` constructor takes both required and optional arguments.

Shared arguments
''''''''''''''''

Required:

-  ``entry_point (str)`` Path (absolute or relative) to the Python file which
   should be executed as the entry point to training.
-  ``role (str)`` An AWS IAM role (either name or full ARN). The Amazon
   SageMaker training jobs and APIs that create Amazon SageMaker
   endpoints use this role to access training data and model artifacts.
   After the endpoint is created, the inference code might use the IAM
   role, if accessing AWS resource.
-  ``train_instance_count (int)`` Number of Amazon EC2 instances to use for
   training.
-  ``train_instance_type (str)`` Type of EC2 instance to use for training, for
   example, 'ml.c4.xlarge'.

Optional:

-  ``source_dir (str)`` Path (absolute or relative) to a directory with any
   other training source code dependencies including the entry point
   file. Structure within this directory will be preserved when training
   on SageMaker.
- ``dependencies (list[str])`` A list of paths to directories (absolute or relative) with
        any additional libraries that will be exported to the container (default: []).
        The library folders will be copied to SageMaker in the same folder where the entrypoint is copied.
        If the ```source_dir``` points to S3, code will be uploaded and the S3 location will be used
        instead. Example:

            The following call
            >>> TensorFlow(entry_point='train.py', dependencies=['my/libs/common', 'virtual-env'])
            results in the following inside the container:

            >>> $ ls

            >>> opt/ml/code
            >>>     ├── train.py
            >>>     ├── common
            >>>     └── virtual-env

-  ``hyperparameters (dict[str,ANY])`` Hyperparameters that will be used for training.
   Will be made accessible as a dict[] to the training code on
   SageMaker. Some hyperparameters will be interpreted by TensorFlow and can be use to
   fine tune training. See `Optional Hyperparameters <#optional-hyperparameters>`_.
-  ``train_volume_size (int)`` Size in GB of the EBS volume to use for storing
   input data during training. Must be large enough to the store training
   data.
-  ``train_max_run (int)`` Timeout in seconds for training, after which Amazon
   SageMaker terminates the job regardless of its current status.
-  ``output_path (str)`` S3 location where you want the training result (model
   artifacts and optional output files) saved. If not specified, results
   are stored to a default bucket. If the bucket with the specific name
   does not exist, the estimator creates the bucket during the ``fit``
   method execution.
-  ``output_kms_key`` Optional KMS key ID to optionally encrypt training
   output with.
-  ``base_job_name`` Name to assign for the training job that the ``fit``
   method launches. If not specified, the estimator generates a default
   job name, based on the training image name and current timestamp.
-  ``image_name`` An alternative docker image to use for training and
   serving.  If specified, the estimator will use this image for training and
   hosting, instead of selecting the appropriate SageMaker official image based on
   framework_version and py_version. Refer to: `SageMaker TensorFlow Docker Containers
   <#sagemaker-tensorflow-docker-containers>`_ for details on what the Official images support
   and where to find the source code to build your custom image.

Script Mode Arguments
'''''''''''''''''''''

The following are Script Mode only arguments. They are both optional.

- ``script_mode (bool)`` Wether to use Script Mode or not. Setting ``py_version`` to ``py3`` overrides it.
- ``model_dir (str)`` S3 location where checkpoint data will saved and restored. If not specified a S3 location will
  be generated under the training job's default bucket. And ``model_dir`` will be passed in your training script as
  one of the command line arguments.
- ``distribution (dict)`` Configure your distrubtion strategy with this argument. For launching parameter server for
  for distributed training, you must set ``distribution`` to ``{'parameter_server': {'enabled': True}}``

Legacy Mode Arguments
'''''''''''''''''''''

The following are Legacy Mode only arguments. Specifying them when Script Mode is enabled will cause errors.

Required:

- ``training_steps (int)`` Perform this many steps of training. ``None``, means train forever.
- ``evaluation_steps (int)`` Perform this many steps of evaluation. ``None``, means
  that evaluation runs until input from ``eval_input_fn`` is exhausted (or another exception is raised).

Optional:

-  ``checkpoint_path`` S3 location where checkpoint data will saved and restored.
   The default location is *bucket_name/job_name/checkpoint*. If the location
   already has checkpoints before the training starts, the model will restore
   state from the last saved checkpoint. It is very useful to restart a training.
   See `Restoring from checkpoints <#restoring-from-checkpoints>`_.
-  ``requirements_file (str)`` Path to a ``requirements.txt`` file. The path should
   be within and relative to ``source_dir``. This is a file containing a list of items to be
   installed using pip install. Details on the format can be found in the
   `Pip User Guide <https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format>`_.


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


Training with Pipe Mode using PipeModeDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


SageMaker TensorFlow Docker containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Legacy Mode TensorFlow Docker images support Python 2.7. Script Mode TensorFlow Docker images support both Python 2.7
and Python 3.6. The Docker images extend Ubuntu 16.04.

You can select version of TensorFlow by passing a ``framework_version`` keyword arg to the TensorFlow Estimator constructor. Currently supported versions are listed in the table above. You can also set ``framework_version`` to only specify major and minor version, e.g ``1.6``, which will cause your training script to be run on the latest supported patch version of that minor version, which in this example would be 1.6.0.
Alternatively, you can build your own image by following the instructions in the SageMaker TensorFlow containers
repository, and passing ``image_name`` to the TensorFlow Estimator constructor.

For more information on the contents of the images, see the SageMaker TensorFlow containers repository here: https://github.com/aws/sagemaker-tensorflow-containers/
