############################################
Use TensorFlow with the SageMaker Python SDK
############################################

With the SageMaker Python SDK, you can train and host TensorFlow models on Amazon SageMaker.

For information about supported versions of TensorFlow, see the `AWS documentation <https://aws.amazon.com/releasenotes/available-deep-learning-containers-images>`_.
We recommend that you use the latest supported version because that's where we focus our development efforts.

For general information about using the SageMaker Python SDK, see :ref:`overview:Using the SageMaker Python SDK`.

.. warning::
    Support for TensorFlow versions 1.4-1.10 has been deprecated.
    For information on how to upgrade, see `Upgrade from Legacy TensorFlow Support <upgrade_from_legacy.html>`_.

.. contents::

*****************************
Train a Model with TensorFlow
*****************************

To train a TensorFlow model by using the SageMaker Python SDK:

.. |create tf estimator| replace:: Create a ``sagemaker.tensorflow.TensorFlow estimator``
.. _create tf estimator: #create-an-estimator

.. |call fit| replace:: Call the estimator's ``fit`` method
.. _call fit: #call-the-fit-method

1. `Prepare a training script <#prepare-a-training-script>`_
2. |create tf estimator|_
3. |call fit|_

Prepare a Training Script
=========================

The training script is very similar to a training script you might run outside of SageMaker, but you can access useful properties about the training environment through various environment variables, including the following:

* ``SM_MODEL_DIR``: A string that represents the local path where the training job writes the model artifacts to.
  After training, artifacts in this directory are uploaded to S3 for model hosting. This is different than the ``model_dir``
  argument passed in your training script, which can be an S3 location. ``SM_MODEL_DIR`` is always set to ``/opt/ml/model``.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_OUTPUT_DATA_DIR``: A string that represents the path to the directory to write output artifacts to.
  Output artifacts might include checkpoints, graphs, and other files to save, but do not include model artifacts.
  These artifacts are compressed and uploaded to S3 to an S3 bucket with the same prefix as the model artifacts.
* ``SM_CHANNEL_XXXX``: A string that represents the path to the directory that contains the input data for the specified channel.
  For example, if you specify two input channels in the TensorFlow estimator's ``fit`` call, named 'train' and 'test', the environment variables ``SM_CHANNEL_TRAIN`` and ``SM_CHANNEL_TEST`` are set.

For the exhaustive list of available environment variables, see the `SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`_.

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model, and saves a model to ``SM_MODEL_DIR`` so that it can be deployed for inference later.
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
For example, if you want to use a boolean hyperparameter, specify ``type`` as ``bool`` in your script and provide an explicit ``True`` or ``False`` value for this hyperparameter when you create the TensorFlow estimator.

For a complete example of a TensorFlow training script, see `mnist.py <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_script_mode_training_and_serving/mnist.py>`__.


Adapting your local TensorFlow script
-------------------------------------

If you have a TensorFlow training script that runs outside of SageMaker, do the following to adapt the script to run in SageMaker:

1. Make sure your script can handle ``--model_dir`` as an additional command line argument. If you did not specify a
location when you created the TensorFlow estimator, an S3 location under the default training job bucket is used.
Distributed training with parameter servers requires you to use the ``tf.estimator.train_and_evaluate`` API and
to provide an S3 location as the model directory during training. Here is an example:

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


Use third-party libraries
-------------------------

If there are other packages you want to use with your script, you can use a ``requirements.txt`` to install other dependencies at runtime.

For training, support for installing packages using ``requirements.txt`` varies by TensorFlow version as follows:

- For TensorFlow 1.15.2 with Python 3.7 or newer, and TensorFlow 2.2 or newer:
    - Include a ``requirements.txt`` file in the same directory as your training script.
    - You must specify this directory using the ``source_dir`` argument when creating a TensorFlow estimator.
- For TensorFlow versions 1.11-1.15.2, 2.0-2.1 with Python 2.7 or 3.6:
    - Write a shell script for your entry point that first calls ``pip install -r requirements.txt``, then runs your training script.
    - For an example of using shell scripts, see `this example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_script_mode_using_shell_commands/tensorflow_script_mode_using_shell_commands.ipynb>`__.
- For legacy versions of TensorFlow:
    - See `Upgrade from Legacy TensorFlow Support <upgrade_from_legacy.html>`_.

For serving, support for installing packages using ``requirements.txt`` varies by TensorFlow version as follows:

- For TensorFlow 1.11 or newer:
    - Include a ``requirements.txt`` file in the ``code`` directory.
- For legacy versions of TensorFlow:
    - See `Upgrade from Legacy TensorFlow Support <upgrade_from_legacy.html>`_.

A ``requirements.txt`` file is a text file that contains a list of items that are installed by using ``pip install``.
You can also specify the version of an item to install.
For information about the format of a ``requirements.txt`` file, see `Requirements Files <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`__ in the pip documentation.


Create an Estimator
===================

After you create your training script, create an instance of the :class:`sagemaker.tensorflow.TensorFlow` estimator.

To use Python 3.7, please specify both of the args:

- ``py_version='py37'``
- ``framework_version='1.15.2'``

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(
      entry_point="tf-train.py",
      role="SageMakerRole",
      instance_count=1,
      instance_type="ml.p2.xlarge",
      framework_version="2.2",
      py_version="py37",
  )
  tf_estimator.fit("s3://bucket/path/to/training/data")

Where the S3 url is a path to your training data within Amazon S3.
The constructor keyword arguments define how SageMaker runs your training script.

Specify a Docker image using an Estimator
-----------------------------------------

There are use cases, such as extending an existing pre-built Amazon SageMaker images, that require specifing a Docker image when creating an Estimator by directly specifying the ECR URI instead of the Python and framework version. For a full list of available container URIs, see `Available Deep Learning Containers Images <https://github.com/aws/deep-learning-containers/blob/master/available_images.md>`__ For more information on using Docker containers, see `Use Your Own Algorithms or Models with Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html>`__.

When specifying the image, you must use the ``image_name=''`` arg to replace the following arg:

- ``py_version=''``

You should still specify the ``framework_version=''`` arg because the SageMaker Python SDK accomodates for differences in the images based on the version.

The following example uses the ``image_name=''`` arg to specify the container image, Python version, and framework version.

.. code:: python

   tf_estimator = TensorFlow(entry_point='tf-train.py',
                             role='SageMakerRole',
                             train_instance_count=1,
                             train_instance_type='ml.p2.xlarge',
                             image_name='763104351884.dkr.ecr.<region>.amazonaws.com/<framework>-<job type>:<framework version>-<cpu/gpu>-<python version>-ubuntu18.04',
                             script_mode=True)

For more information about the sagemaker.tensorflow.TensorFlow estimator, see `SageMaker TensorFlow Classes`_.

Call the fit Method
===================

You start your training script by calling the ``fit`` method on a ``TensorFlow`` estimator.

Calling ``fit`` starts a SageMaker training job. The training job will execute the following.

- Starts ``instance_count`` EC2 instances of the type ``instance_type``.
- On each instance, it will do the following steps:

  - starts a Docker container optimized for TensorFlow.
  - downloads the dataset.
  - setup up training related environment varialbes
  - setup up distributed training environment if configured to use parameter server
  - starts asynchronous training

If the ``wait=False`` flag is passed to ``fit``, then it returns immediately. The training job continues running
asynchronously. Later, a TensorFlow estimator can be obtained by attaching to the existing training job.
If the training job is not finished, it starts showing the standard output of training and wait until it completes.
After attaching, the estimator can be deployed as usual.

.. code:: python

    tf_estimator.fit(your_input_data, wait=False)
    training_job_name = tf_estimator.latest_training_job.name

    # after some time, or in a separate Python notebook, we can attach to it again.

    tf_estimator = TensorFlow.attach(training_job_name=training_job_name)

For more information about the options available for ``fit``, see the `API documentation <https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase.fit>`_.

Distributed Training
====================

To run your training job with multiple instances in a distributed fashion, set ``instance_count``
to a number larger than 1. We support two different types of distributed training, parameter server and Horovod.
The ``distribution`` parameter is used to configure which distributed training strategy to use.

Training with parameter servers
-------------------------------

If you specify parameter_server as the value of the distribution parameter, the container launches a parameter server
thread on each instance in the training cluster, and then executes your training code. You can find more information on
TensorFlow distributed training at `TensorFlow docs <https://www.tensorflow.org/guide/distributed_training>`__.
To enable parameter server training:

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(
      entry_point="tf-train.py",
      role="SageMakerRole",
      instance_count=2,
      instance_type="ml.p2.xlarge",
      framework_version="2.2",
      py_version="py37",
      distribution={"parameter_server": {"enabled": True}},
  )
  tf_estimator.fit("s3://bucket/path/to/training/data")

Training with Horovod
---------------------

Horovod is a distributed training framework based on MPI. Horovod is only available with TensorFlow version ``1.12`` or newer.
You can find more details at `Horovod README <https://github.com/uber/horovod>`__.

The container sets up the MPI environment and executes the ``mpirun`` command, enabling you to run any Horovod
training script.

Training with ``MPI`` is configured by specifying following fields in ``distribution``:

- ``enabled (bool)``: If set to ``True``, the MPI setup is performed and ``mpirun`` command is executed.
- ``processes_per_host (int)``: Number of processes MPI should launch on each host. Note, this should not be
  greater than the available slots on the selected instance type. This flag should be set for the multi-cpu/gpu
  training.
- ``custom_mpi_options (str)``:  Any ``mpirun`` flag(s) can be passed in this field that will be added to the ``mpirun``
  command executed by SageMaker to launch distributed horovod training.


In the below example we create an estimator to launch Horovod distributed training with 4 processes on one host:

.. code:: python

    from sagemaker.tensorflow import TensorFlow

    tf_estimator = TensorFlow(
        entry_point="tf-train.py",
        role="SageMakerRole",
        instance_count=1,
        instance_type="ml.p3.8xlarge",
        framework_version="2.1.0",
        py_version="py3",
        distribution={
            "mpi": {
                "enabled": True,
                "processes_per_host": 4,
                "custom_mpi_options": "--NCCL_DEBUG INFO"
            },
        },
    )
    tf_estimator.fit("s3://bucket/path/to/training/data")


Training with Pipe Mode using PipeModeDataset
=============================================

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

    tf_estimator = TensorFlow(
        entry_point="tf-train-with-pipemodedataset.py",
        role="SageMakerRole",
        training_steps=10000,
        evaluation_steps=100,
        instance_count=1,
        instance_type="ml.p2.xlarge",
        framework_version="1.10.0",
        py_version="py3",
        input_mode="Pipe",
    )

    tf_estimator.fit("s3://bucket/path/to/training/data")


If your TFRecords are compressed, you can train on Gzipped TF Records by passing in ``compression='Gzip'`` to the call to
``fit()``, and SageMaker will automatically unzip the records as data is streamed to your training instances:

.. code:: python

    from sagemaker.inputs import TrainingInput

    train_s3_input = TrainingInput('s3://bucket/path/to/training/data', compression='Gzip')
    tf_estimator.fit(train_s3_input)


You can learn more about ``PipeModeDataset`` in the sagemaker-tensorflow-extensions repository: https://github.com/aws/sagemaker-tensorflow-extensions


Training with MKL-DNN disabled
==============================

SageMaker TensorFlow CPU images use TensorFlow built with Intel® MKL-DNN optimization.

In certain cases you might be able to get a better performance by disabling this optimization
(for example when using small models).

You can disable MKL-DNN optimization for TensorFlow ``1.8.0`` and above by setting two following environment variables:

.. code:: python

    import os

    os.environ['TF_DISABLE_MKL'] = '1'
    os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'

********************************
Deploy TensorFlow Serving models
********************************

After a TensorFlow estimator has been fit, it saves a TensorFlow SavedModel in
the S3 location defined by ``output_path``. You can call ``deploy`` on a TensorFlow
estimator to create a SageMaker Endpoint, or you can call ``transformer`` to create a ``Transformer`` that you can use to run a batch transform job.

Your model will be deployed to a TensorFlow Serving-based server. The server provides a super-set of the
`TensorFlow Serving REST API <https://www.tensorflow.org/serving/api_rest>`_.


Deploy to a SageMaker Endpoint
==============================

Deploying from an Estimator
---------------------------

After a TensorFlow estimator has been fit, it saves a TensorFlow
`SavedModel <https://www.tensorflow.org/guide/saved_model>`_ bundle in
the S3 location defined by ``output_path``. You can call ``deploy`` on a TensorFlow
estimator object to create a SageMaker Endpoint:

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  estimator = TensorFlow(
      entry_point="tf-train.py",
      ...,
      instance_count=1,
      instance_type="ml.c4.xlarge",
      framework_version="2.2",
      py_version="py37",
  )

  estimator.fit(inputs)

  predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.c5.xlarge")


The code block above deploys a SageMaker Endpoint with one instance of the type 'ml.c5.xlarge'.

What happens when deploy is called
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calling ``deploy`` starts the process of creating a SageMaker Endpoint. This process includes the following steps.

- Starts ``initial_instance_count`` EC2 instances of the type ``instance_type``.
- On each instance, it will do the following steps:

  - start a Docker container optimized for TensorFlow Serving, see `SageMaker TensorFlow Serving containers <https://github.com/aws/sagemaker-tensorflow-serving-container>`_.
  - start a `TensorFlow Serving` process configured to run your model.
  - start an HTTP server that provides access to TensorFlow Server through the SageMaker InvokeEndpoint API.


When the ``deploy`` call finishes, the created SageMaker Endpoint is ready for prediction requests. The
`Making predictions against a SageMaker Endpoint`_ section will explain how to make prediction requests
against the Endpoint.

Deploying directly from model artifacts
---------------------------------------

If you already have existing model artifacts in S3, you can skip training and deploy them directly to an endpoint:

.. code:: python

  from sagemaker.tensorflow import TensorFlowModel

  model = TensorFlowModel(model_data='s3://mybucket/model.tar.gz', role='MySageMakerRole')

  predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')

Python-based TensorFlow serving on SageMaker has support for `Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`__, which allows for inference acceleration to a hosted endpoint for a fraction of the cost of using a full GPU instance. In order to attach an Elastic Inference accelerator to your endpoint provide the accelerator type to accelerator_type to your deploy call.

.. code:: python

    from sagemaker.tensorflow import TensorFlowModel

    model = TensorFlowModel(model_data='s3://mybucket/model.tar.gz', role='MySageMakerRole')

    predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge', accelerator_type='ml.eia1.medium')

Making predictions against a SageMaker Endpoint
-----------------------------------------------

Once you have the ``Predictor`` instance returned by ``model.deploy(...)`` or ``estimator.deploy(...)``, you
can send prediction requests to your Endpoint.

The following code shows how to make a prediction request:

.. code:: python

  input = {
    'instances': [1.0, 2.0, 5.0]
  }
  result = predictor.predict(input)

The result object will contain a Python dict like this:

.. code:: python

  {
    'predictions': [3.5, 4.0, 5.5]
  }

The formats of the input and the output data correspond directly to the request and response formats
of the ``Predict`` method in the `TensorFlow Serving REST API <https://www.tensorflow.org/serving/api_rest>`_.

If your SavedModel includes the right ``signature_def``, you can also make Classify or Regress requests:

.. code:: python

  # input matches the Classify and Regress API
  input = {
    'signature_name': 'tensorflow/serving/regress',
    'examples': [{'x': 1.0}, {'x': 2.0}]
  }

  result = predictor.regress(input)  # or predictor.classify(...)

  # result contains:
  {
    'results': [3.5, 4.0]
  }

You can include multiple ``instances`` in your predict request (or multiple ``examples`` in
classify/regress requests) to get multiple prediction results in one request to your Endpoint:

.. code:: python

  input = {
    'instances': [
      [1.0, 2.0, 5.0],
      [1.0, 2.0, 5.0],
      [1.0, 2.0, 5.0]
    ]
  }
  result = predictor.predict(input)

  # result contains:
  {
    'predictions': [
      [3.5, 4.0, 5.5],
      [3.5, 4.0, 5.5],
      [3.5, 4.0, 5.5]
    ]
  }

If your application allows request grouping like this, it is **much** more efficient than making separate requests.

See `Deploying to TensorFlow Serving Endpoints <deploying_tensorflow_serving.html>`_ to learn how to deploy your model and make inference requests.

Run a Batch Transform Job
=========================

Batch transform allows you to get inferences for an entire dataset that is stored in an S3 bucket.

For general information about using batch transform with the SageMaker Python SDK, see :ref:`overview:SageMaker Batch Transform`.
For information about SageMaker batch transform, see `Get Inferences for an Entire Dataset with Batch Transform <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html>` in the AWS documentation.

To run a batch transform job, you first create a ``Transformer`` object, and then call that object's ``transform`` method.

Create a Transformer Object
---------------------------

If you used an estimator to train your model, you can call the ``transformer`` method of the estimator to create a ``Transformer`` object.

For example:

.. code:: python

  bucket = myBucket # The name of the S3 bucket where the results are stored
  prefix = 'batch-results' # The folder in the S3 bucket where the results are stored

  batch_output = 's3://{}/{}/results'.format(bucket, prefix) # The location to store the results

  tf_transformer = tf_estimator.transformer(instance_count=1, instance_type='ml.m4.xlarge', output_path=batch_output)

To use a model trained outside of SageMaker, you can package the model as a SageMaker model, and call the ``transformer`` method of the SageMaker model.

For example:

.. code:: python

  bucket = myBucket # The name of the S3 bucket where the results are stored
  prefix = 'batch-results' # The folder in the S3 bucket where the results are stored

  batch_output = 's3://{}/{}/results'.format(bucket, prefix) # The location to store the results

  tf_transformer = tensorflow_serving_model.transformer(instance_count=1, instance_type='ml.m4.xlarge', output_path=batch_output)

For information about how to package a model as a SageMaker model, see :ref:`overview:BYO Model`.
When you call the ``tranformer`` method, you specify the type and number of instances to use for the batch transform job, and the location where the results are stored in S3.



Call transform
--------------

After you create a ``Transformer`` object, you call that object's ``transform`` method to start a batch transform job.
For example:

.. code:: python

  batch_input = 's3://{}/{}/test/examples'.format(bucket, prefix) # The location of the input dataset

  tf_transformer.transform(data=batch_input, data_type='S3Prefix', content_type='text/csv', split_type='Line')

In the example, the content type is CSV, and each line in the dataset is treated as a record to get a predition for.

Batch Transform Supported Data Formats
--------------------------------------

When you call the ``tranform`` method to start a batch transform job,
you specify the data format by providing a MIME type as the value for the ``content_type`` parameter.

The following content formats are supported without custom intput and output handling:

* CSV - specify ``text/csv`` as the value of the ``content_type`` parameter.
* JSON - specify ``application/json`` as the value of the ``content_type`` parameter.
* JSON lines - specify ``application/jsonlines`` as the value of the ``content_type`` parameter.

For detailed information about how TensorFlow Serving formats these data types for input and output, see :ref:`tensorflow-serving-input-output`.

You can also accept any custom data format by writing input and output functions, and include them in the ``inference.py`` file in your model.
For information, see :ref:`custom-input-output`.


.. _tensorflow-serving-input-output:

TensorFlow Serving Input and Output
===================================

The following sections describe the data formats that TensorFlow Serving endpoints and batch transform jobs accept,
and how to write input and output functions to input and output custom data formats.

Supported Formats
-----------------

SageMaker's TensforFlow Serving endpoints can also accept some additional input formats that are not part of the
TensorFlow REST API, including a simplified json format, line-delimited json objects ("jsons" or "jsonlines"), and
CSV data.

Simplified JSON Input
^^^^^^^^^^^^^^^^^^^^^

The Endpoint will accept simplified JSON input that doesn't match the TensorFlow REST API's Predict request format.
When the Endpoint receives data like this, it will attempt to transform it into a valid
Predict request, using a few simple rules:

- python value, dict, or one-dimensional arrays are treated as the input value in a single 'instance' Predict request.
- multidimensional arrays are treated as a multiple values in a multi-instance Predict request.

Combined with the client-side ``Predictor`` object's JSON serialization, this allows you to make simple
requests like this:

.. code:: python

  input = [
    [1.0, 2.0, 5.0],
    [1.0, 2.0, 5.0]
  ]
  result = predictor.predict(input)

  # result contains:
  {
    'predictions': [
      [3.5, 4.0, 5.5],
      [3.5, 4.0, 5.5]
    ]
  }

Or this:

.. code:: python

  # 'x' must match name of input tensor in your SavedModel graph
  # for models with multiple named inputs, just include all the keys in the input dict
  input = {
    'x': [1.0, 2.0, 5.0]
  }

  # result contains:
  {
    'predictions': [
      [3.5, 4.0, 5.5]
    ]
  }


Line-delimited JSON
^^^^^^^^^^^^^^^^^^^

The Endpoint will accept line-delimited JSON objects (also known as "jsons" or "jsonlines" data).
The Endpoint treats each line as a separate instance in a multi-instance Predict request. To use
this feature from your python code, you need to create a ``Predictor`` instance that does not
try to serialize your input to JSON:

.. code:: python

  # create a Predictor without JSON serialization

  predictor = Predictor('endpoint-name', serializer=None, content_type='application/jsonlines')

  input = '''{'x': [1.0, 2.0, 5.0]}
  {'x': [1.0, 2.0, 5.0]}
  {'x': [1.0, 2.0, 5.0]}'''

  result = predictor.predict(input)

  # result contains:
  {
    'predictions': [
      [3.5, 4.0, 5.5],
      [3.5, 4.0, 5.5],
      [3.5, 4.0, 5.5]
    ]
  }

This feature is especially useful if you are reading data from a file containing jsonlines data.

**CSV (comma-separated values)**

The Endpoint will accept CSV data. Each line is treated as a separate instance. This is a
compact format for representing multiple instances of 1-d array data. To use this feature
from your python code, you need to create a ``Predictor`` instance that can serialize
your input data to CSV format:

.. code:: python

  # create a Predictor with JSON serialization

  predictor = Predictor('endpoint-name', serializer=sagemaker.serializers.CSVSerializer())

  # CSV-formatted string input
  input = '1.0,2.0,5.0\n1.0,2.0,5.0\n1.0,2.0,5.0'

  result = predictor.predict(input)

  # result contains:
  {
    'predictions': [
      [3.5, 4.0, 5.5],
      [3.5, 4.0, 5.5],
      [3.5, 4.0, 5.5]
    ]
  }

You can also use python arrays or numpy arrays as input and let the ``CSVSerializer`` object
convert them to CSV, but the client-size CSV conversion is more sophisticated than the
CSV parsing on the Endpoint, so if you encounter conversion problems, try using one of the
JSON options instead.

.. _custom-input-output:

Create Python Scripts for Custom Input and Output Formats
---------------------------------------------------------

You can add your customized Python code to process your input and output data.
This customized Python code must be named ``inference.py`` and is specified through the ``entry_point`` parameter:

.. code::

    from sagemaker.tensorflow import TensorFlowModel

    model = TensorFlowModel(entry_point='inference.py',
                            model_data='s3://mybucket/model.tar.gz',
                            role='MySageMakerRole')

In the example above, ``inference.py`` is assumed to be a file inside ``model.tar.gz``. If you want to use a local file instead, you must add the ``source_dir`` argument. See the documentation on `TensorFlowModel <https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/sagemaker.tensorflow.html#sagemaker.tensorflow.model.TensorFlowModel>`_.

How to implement the pre- and/or post-processing handler(s)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your entry point file must be named ``inference.py`` and should implement
either a pair of ``input_handler`` and ``output_handler`` functions or
a single ``handler`` function.
Note that if ``handler`` function is implemented, ``input_handler``
and ``output_handler`` are ignored.

To implement pre- and/or post-processing handler(s), use the Context
object that the Python service creates. The Context object is a namedtuple with the following attributes:

-  ``model_name (string)``: the name of the model to use for
   inference. For example, 'half-plus-three'

-  ``model_version (string)``: version of the model. For example, '5'

-  ``method (string)``: inference method. For example, 'predict',
   'classify' or 'regress', for more information on methods, please see
   `Classify and Regress
   API <https://www.tensorflow.org/tfx/serving/api_rest#classify_and_regress_api>`__
   and `Predict
   API <https://www.tensorflow.org/tfx/serving/api_rest#predict_api>`__

-  ``rest_uri (string)``: the TFS REST uri generated by the Python
   service. For example,
   'http://localhost:8501/v1/models/half_plus_three:predict'

-  ``grpc_uri (string)``: the GRPC port number generated by the Python
   service. For example, '9000'

-  ``custom_attributes (string)``: content of
   'X-Amzn-SageMaker-Custom-Attributes' header from the original
   request. For example,
   'tfs-model-name=half*plus*\ three,tfs-method=predict'

-  ``request_content_type (string)``: the original request content type,
   defaulted to 'application/json' if not provided

-  ``accept_header (string)``: the original request accept type,
   defaulted to 'application/json' if not provided

-  ``content_length (int)``: content length of the original request

The following code example implements ``input_handler`` and
``output_handler``. By providing these, the Python service posts the
request to the TFS REST URI with the data pre-processed by ``input_handler``
and passes the response to ``output_handler`` for post-processing.

.. code::

   import json

   def input_handler(data, context):
       """ Pre-process request input before it is sent to TensorFlow Serving REST API
       Args:
           data (obj): the request data, in format of dict or string
           context (Context): an object containing request and configuration details
       Returns:
           (dict): a JSON-serializable dict that contains request body and headers
       """
       if context.request_content_type == 'application/json':
           # pass through json (assumes it's correctly formed)
           d = data.read().decode('utf-8')
           return d if len(d) else ''

       if context.request_content_type == 'text/csv':
           # very simple csv handler
           return json.dumps({
               'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
           })

       raise ValueError('{{"error": "unsupported content type {}"}}'.format(
           context.request_content_type or "unknown"))


   def output_handler(data, context):
       """Post-process TensorFlow Serving output before it is returned to the client.
       Args:
           data (obj): the TensorFlow serving response
           context (Context): an object containing request and configuration details
       Returns:
           (bytes, string): data to return to client, response content type
       """
       if data.status_code != 200:
           raise ValueError(data.content.decode('utf-8'))

       response_content_type = context.accept_header
       prediction = data.content
       return prediction, response_content_type

You might want to have complete control over the request.
For example, you might want to make a TFS request (REST or GRPC) to the first model,
inspect the results, and then make a request to a second model. In this case, implement
the ``handler`` method instead of the ``input_handler`` and ``output_handler`` methods, as demonstrated
in the following code:

.. code::

   import json
   import requests


   def handler(data, context):
       """Handle request.
       Args:
           data (obj): the request data
           context (Context): an object containing request and configuration details
       Returns:
           (bytes, string): data to return to client, (optional) response content type
       """
       processed_input = _process_input(data, context)
       response = requests.post(context.rest_uri, data=processed_input)
       return _process_output(response, context)


   def _process_input(data, context):
       if context.request_content_type == 'application/json':
           # pass through json (assumes it's correctly formed)
           d = data.read().decode('utf-8')
           return d if len(d) else ''

       if context.request_content_type == 'text/csv':
           # very simple csv handler
           return json.dumps({
               'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
           })

       raise ValueError('{{"error": "unsupported content type {}"}}'.format(
           context.request_content_type or "unknown"))


   def _process_output(data, context):
       if data.status_code != 200:
           raise ValueError(data.content.decode('utf-8'))

       response_content_type = context.accept_header
       prediction = data.content
       return prediction, response_content_type

You can also bring in external dependencies to help with your data
processing. There are 2 ways to do this:

1. If your model archive contains ``code/requirements.txt``, the container will install the Python dependencies at runtime using ``pip install -r``.

.. code::

    from sagemaker.tensorflow import TensorFlowModel

    model = TensorFlowModel(entry_point='inference.py',
                            dependencies=['requirements.txt'],
                            model_data='s3://mybucket/model.tar.gz',
                            role='MySageMakerRole')


2. If you are working in a network-isolation situation or if you don't
   want to install dependencies at runtime every time your endpoint starts or a batch
   transform job runs, you might want to put
   pre-downloaded dependencies under a ``lib`` directory and this
   directory as dependency. The container adds the modules to the Python
   path. Note that if both ``lib`` and ``requirements.txt``
   are present in the model archive, the ``requirements.txt`` is ignored:

.. code::

    from sagemaker.tensorflow import TensorFlowModel

    model = TensorFlowModel(entry_point='inference.py',
                           dependencies=['/path/to/folder/named/lib'],
                           model_data='s3://mybucket/model.tar.gz',
                           role='MySageMakerRole')

For more information, see: https://github.com/aws/sagemaker-tensorflow-serving-container#prepost-processing

****************************
SageMaker TensorFlow Classes
****************************

For information about the different TensorFlow-related classes in the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html.

**************************************
SageMaker TensorFlow Docker containers
**************************************

For information about the SageMaker TensorFlow containers, see:

- `SageMaker TensorFlow training toolkit <https://github.com/aws/sagemaker-tensorflow-container>`_
- `SageMaker TensorFlow serving toolkit <https://github.com/aws/sagemaker-tensorflow-serving-container>`_
- `Deep Learning Container (DLC) Dockerfiles for TensorFlow <https://github.com/aws/deep-learning-containers/tree/master/tensorflow>`_
- `Deep Learning Container (DLC) Images <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`_ and `release notes <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html>`_
