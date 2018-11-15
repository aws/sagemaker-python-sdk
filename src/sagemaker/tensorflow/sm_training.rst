Preparing Script Mode training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your TensorFlow training script must be a Python 2.7 or 3.5 compatible source file.

The training script is very similar to a training script you might run outside of SageMaker, but you can access useful properties about the training environment through various environment variables, including the following:

* ``SM_MODEL_DIR``: A string that represents the path where the training job writes the model artifacts to.
  After training, artifacts in this directory are uploaded to S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_OUTPUT_DATA_DIR``: A string that represents the path to the directory to write output artifacts to.
  Output artifacts might include checkpoints, graphs, and other files to save, but do not include model artifacts.
  These artifacts are compressed and uploaded to S3 to an S3 bucket with the same prefix as the model artifacts.
* ``SM_CHANNEL_XXXX``: A string that represents the path to the directory that contains the input data for the specified channel.
  For example, if you specify two input channels in the MXNet estimator's ``fit`` call, named 'train' and 'test', the environment variables ``SM_CHANNEL_TRAIN`` and ``SM_CHANNEL_TEST`` are set.

For the exhaustive list of available environment variables, see the `SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model, and saves a model to ``model_dir`` so that it can be deployed for inference later.
Hyperparameters are passed to your script as arguments and can be retrieved with an ``argparse.ArgumentParser`` instance.
For example, a training script might start with the following:

.. code:: python

    import argparse
    import os

    if __name__ =='__main__':

        parser = argparse.ArgumentParser()

        # hyperparameters sent by the client are passed as command-line arguments to the script.
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=100)
        parser.add_argument('--learning-rate', type=float, default=0.1)

        # input data and model directories
        parser.add_argument('--model_dir', type=str)
        parser.add_argument('--sm-model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
        parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

        args, _ = parser.parse_known_args()

        # ... load from args.train and args.test, train a model, write model to args.model_dir.

Because the SageMaker imports your training script, you should put your training code in a main guard (``if __name__=='__main__':``) if you are using the same script to host your model,
so that SageMaker does not inadvertently run your training code at the wrong point in execution.

Note that SageMaker doesn't support argparse actions.
If you want to use, for example, boolean hyperparameters, you need to specify ``type`` as ``bool`` in your script and provide an explicit ``True`` or ``False`` value for this hyperparameter when instantiating your TensorFlow estimator.

For more on training environment variables, please visit `SageMaker Containers <https://github.com/aws/sagemaker-containers>`_.

Adapt your TensorFlow script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a TensorFlow training script with runs outside of SageaMaker please follow the directions here:

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
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_EVAL'])

3. Export your final model to path stored in environment variable ``SM_MODEL_DIR`` which should always be
   ``/opt/ml/model``. At end of training SageMaker will upload the model file under ``/opt/ml/model`` to
   ``output_path``.


Calling fit
^^^^^^^^^^^

You start your training script by calling ``fit`` on a ``TensorFlow`` estimator. ``fit`` takes
both required and optional arguments.

Required argument
'''''''''''''''''

- ``inputs``: The S3 location(s) of datasets to be used for training. This can take one of two forms:

  - ``str``: An S3 URI, for example ``s3://my-bucket/my-training-data``, which indicates the dataset's location.
  - ``dict[str, str]``: A dictionary mapping channel names to S3 locations, for example ``{'train': 's3://my-bucket/my-training-data/train', 'test': 's3://my-bucket/my-training-data/test'}``
  - ``sagemaker.session.s3_input``: channel configuration for S3 data sources that can provide additional information as well as the path to the training dataset. See `the API docs <https://sagemaker.readthedocs.io/en/latest/session.html#sagemaker.session.s3_input>`_ for full details.

Optional arguments
''''''''''''''''''

-  ``wait (bool)``: Defaults to True, whether to block and wait for the
   training script to complete before returning.
   If set to False, it will return immediately, and can later be attached to.
-  ``logs (bool)``: Defaults to True, whether to show logs produced by training
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

    # after some time, or in a separate python notebook, we can attach to it again.

    tf_estimator = TensorFlow.attach(training_job_name=training_job_name)
