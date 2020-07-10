###########################################
Using Chainer with the SageMaker Python SDK
###########################################

With Chainer Estimators, you can train and host Chainer models on Amazon SageMaker.

Supported versions of Chainer: ``4.0.0``, ``4.1.0``, ``5.0.0``.

We recommend that you use the latest supported version because that's where we focus most of our development efforts.

For more information about Chainer, see https://github.com/chainer/chainer.

For general information about using the SageMaker Python SDK, see :ref:`overview:Using the SageMaker Python SDK`.

.. contents::

**************************
Train a Model with Chainer
**************************

To train a Chainer model by using the SageMaker Python SDK:

.. |create chainer estimator| replace:: Create a ``sagemaker.chainer.Chainer`` Estimator
.. _create chainer estimator: #create-an-estimator

.. |call fit| replace:: Call the estimator's ``fit`` method
.. _call fit: #call-the-fit-method

1. `Prepare a training script <#prepare-a-chainer-training-script>`_
2. |create chainer estimator|_
3. |call fit|_


Prepare a Chainer training script
=================================

Your Chainer training script must be a Python 2.7 or 3.5 compatible source file.

The training script is similar to a training script you might run outside of SageMaker, but you
can access useful properties about the training environment through various environment variables,
such as the following:

* ``SM_MODEL_DIR``: A string representing the path to the directory to write model artifacts to.
  These artifacts are uploaded to S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_OUTPUT_DATA_DIR``: A string representing the filesystem path to write output artifacts to. Output artifacts may
  include checkpoints, graphs, and other files to save, not including model artifacts. These artifacts are compressed
  and uploaded to S3 to the same S3 prefix as the model artifacts.

Suppose you use two input channels, named 'train' and 'test', in the call to the Chainer estimator's ``fit()`` method.
The following environment variables are set, following the format "SM_CHANNEL_[channel_name]":

* ``SM_CHANNEL_TRAIN``: A string representing the path to the directory containing data in the 'train' channel
* ``SM_CHANNEL_TEST``: Same as above, but for the 'test' channel.

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model,
and saves a model to model_dir so that it can be hosted later. Hyperparameters are passed to your script as arguments
and can be retrieved with an argparse.ArgumentParser instance. For example, a training script might start
with the following:

.. code:: python

    import argparse
    import os

    if __name__ =='__main__':

        parser = argparse.ArgumentParser()

        # hyperparameters sent by the client are passed as command-line arguments to the script.
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--learning-rate', type=float, default=0.05)

        # Data, model, and output directories
        parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
        parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

        args, _ = parser.parse_known_args()

        # ... load from args.train and args.test, train a model, write model to args.model_dir.

Because the SageMaker imports your training script, you should put your training code in a main guard
(``if __name__=='__main__':``) if you are using the same script to host your model, so that SageMaker does not
inadvertently run your training code at the wrong point in execution.

For more on training environment variables, please visit https://github.com/aws/sagemaker-containers.

Save the Model
--------------

In order to save your trained Chainer model for deployment on SageMaker, your training script should save your model
to a certain filesystem path called `model_dir`. This value is accessible through the environment variable
``SM_MODEL_DIR``. The following code demonstrates how to save a trained Chainer model named ``model`` as
``model.npz`` at the end of training:

.. code:: python

    import chainer
    import argparse
    import os

    if __name__=='__main__':
        # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        args, _ = parser.parse_known_args()

        # ... train `model`, then save it to `model_dir` as file 'model.npz'
        chainer.serializers.save_npz(os.path.join(args.model_dir, 'model.npz'), model)

After your training job is complete, SageMaker will compress and upload the serialized model to S3, and your model data
will available in the s3 ``output_path`` you specified when you created the Chainer Estimator.

Using third-party libraries
---------------------------

When running your training script on SageMaker, it will have access to some pre-installed third-party libraries including ``chainer``, ``numpy``, and ``cupy``.
For more information on the runtime environment, including specific package versions, see `SageMaker Chainer Docker containers <#sagemaker-chainer-docker-containers>`__.

If there are other packages you want to use with your script, you can include a ``requirements.txt`` file in the same directory as your training script to install other dependencies at runtime.
Both ``requirements.txt`` and your training script should be put in the same folder.
You must specify this folder in ``source_dir`` argument when creating a Chainer estimator.
A ``requirements.txt`` file is a text file that contains a list of items that are installed by using ``pip install``.
You can also specify the version of an item to install.
For information about the format of a ``requirements.txt`` file, see `Requirements Files <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`__ in the pip documentation.

Create an Estimator
===================

You run Chainer training scripts on SageMaker by creating ``Chainer`` Estimators.
SageMaker training of your script is invoked when you call ``fit`` on a ``Chainer`` Estimator.
The following code sample shows how you train a custom Chainer script "chainer-train.py", passing
in three hyperparameters ('epochs', 'batch-size', and 'learning-rate'), and using two input channel
directories ('train' and 'test').

.. code:: python

    chainer_estimator = Chainer('chainer-train.py',
                                instance_type='ml.p3.2xlarge',
                                instance_count=1,
                                framework_version='5.0.0',
                                py_version='py3',
                                hyperparameters = {'epochs': 20, 'batch-size': 64, 'learning-rate': 0.1})
    chainer_estimator.fit({'train': 's3://my-data-bucket/path/to/my/training/data',
                           'test': 's3://my-data-bucket/path/to/my/test/data'})


Call the fit Method
===================

You start your training script by calling ``fit`` on a ``Chainer`` Estimator. ``fit`` takes both required and optional
arguments.

fit Required arguments
----------------------

-  ``inputs``: This can take one of the following forms: A string
   s3 URI, for example ``s3://my-bucket/my-training-data``. In this
   case, the s3 objects rooted at the ``my-training-data`` prefix will
   be available in the default ``train`` channel. A dict from
   string channel names to s3 URIs. In this case, the objects rooted at
   each s3 prefix will available as files in each channel directory.

For example:

.. code:: python

    {'train':'s3://my-bucket/my-training-data',
     'eval':'s3://my-bucket/my-evaluation-data'}

.. optional-arguments-1:

fit Optional arguments
----------------------

-  ``wait``: Defaults to True, whether to block and wait for the
   training script to complete before returning.
-  ``logs``: Defaults to True, whether to show logs produced by training
   job in the Python session. Only meaningful when wait is True.

Distributed Training
====================


Chainer allows you to train a model on multiple nodes using ChainerMN_, which distributes training with MPI.

.. _ChainerMN: https://github.com/chainer/chainermn

In order to run distributed Chainer training on SageMaker, your training script should use a ``chainermn`` Communicator
object to coordinate training between multiple hosts.

SageMaker runs your script with ``mpirun`` if ``instance_count`` is greater than two.
The following are optional arguments modify how MPI runs your distributed training script.

-  ``use_mpi`` Boolean that overrides whether to run your training script with MPI.
-  ``num_processes`` Integer that determines how many total processes to run with MPI. By default, this is equal to ``process_slots_per_host`` times the number of nodes.
-  ``process_slots_per_host`` Integer that determines how many processes can be run on each host. By default, this is equal to one process per host on CPU instances, or one process per GPU on GPU instances.
-  ``additional_mpi_options`` String of additional options to pass to the ``mpirun`` command.


*********************
Deploy Chainer models
*********************

After an Chainer Estimator has been fit, you can host the newly created model in SageMaker.

After calling ``fit``, you can call ``deploy`` on a ``Chainer`` Estimator to create a SageMaker Endpoint.
The Endpoint runs a SageMaker-provided Chainer model server and hosts the model produced by your training script,
which was run when you called ``fit``. This was the model you saved to ``model_dir``.

``deploy`` returns a ``Predictor`` object, which you can use to do inference on the Endpoint hosting your Chainer model.
Each ``Predictor`` provides a ``predict`` method which can do inference with numpy arrays or Python lists.
Inference arrays or lists are serialized and sent to the Chainer model server by an ``InvokeEndpoint`` SageMaker
operation.

``predict`` returns the result of inference against your model. By default, the inference result a NumPy array.

.. code:: python

    # Train my estimator
    chainer_estimator = Chainer(entry_point='train_and_deploy.py',
                                instance_type='ml.p3.2xlarge',
                                instance_count=1,
                                framework_version='5.0.0',
                                py_version='py3')
    chainer_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploy my estimator to a SageMaker Endpoint and get a Predictor
    predictor = chainer_estimator.deploy(instance_type='ml.m4.xlarge',
                                         initial_instance_count=1)

    # `data` is a NumPy array or a Python list.
    # `response` is a NumPy array.
    response = predictor.predict(data)

You use the SageMaker Chainer model server to host your Chainer model when you call ``deploy`` on an ``Chainer``
Estimator. The model server runs inside a SageMaker Endpoint, which your call to ``deploy`` creates.
You can access the name of the Endpoint by the ``name`` property on the returned ``Predictor``.


The SageMaker Chainer Model Server
==================================

The Chainer Endpoint you create with ``deploy`` runs a SageMaker Chainer model server.
The model server loads the model that was saved by your training script and performs inference on the model in response
to SageMaker InvokeEndpoint API calls.

You can configure two components of the SageMaker Chainer model server: Model loading and model serving.
Model loading is the process of deserializing your saved model back into an Chainer model.
Serving is the process of translating InvokeEndpoint requests to inference calls on the loaded model.

You configure the Chainer model server by defining functions in the Python source file you passed to the Chainer constructor.

Load a Model
------------

Before a model can be served, it must be loaded. The SageMaker Chainer model server loads your model by invoking a
``model_fn`` function that you must provide in your script. The ``model_fn`` should have the following signature:

.. code:: python

    def model_fn(model_dir)

SageMaker will inject the directory where your model files and sub-directories, saved by ``save``, have been mounted.
Your model function should return a model object that can be used for model serving.

SageMaker provides automated serving functions that work with Gluon API ``net`` objects and Module API ``Module`` objects. If you return either of these types of objects, then you will be able to use the default serving request handling functions.

The following code-snippet shows an example ``model_fn`` implementation.
This loads returns a Chainer Classifier from a multi-layer perceptron class ``MLP`` that extends ``chainer.Chain``.
It loads the model parameters from a ``model.npz`` file in the SageMaker model directory ``model_dir``.

.. code:: python

    import chainer
    import os

    def model_fn(model_dir):
        chainer.config.train = False
        model = chainer.links.Classifier(MLP(1000, 10))
        chainer.serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
        return model.predictor

Serve a Model
-------------

After the SageMaker model server has loaded your model by calling ``model_fn``, SageMaker will serve your model.
Model serving is the process of responding to inference requests, received by SageMaker InvokeEndpoint API calls.
The SageMaker Chainer model server breaks request handling into three steps:


-  input processing,
-  prediction, and
-  output processing.

In a similar way to model loading, you configure these steps by defining functions in your Python source file.

Each step involves invoking a python function, with information about the request and the return-value from the previous
function in the chain. Inside the SageMaker Chainer model server, the process looks like:

.. code:: python

    # Deserialize the Invoke request body into an object we can perform prediction on
    input_object = input_fn(request_body, request_content_type)

    # Perform prediction on the deserialized object, with the loaded model
    prediction = predict_fn(input_object, model)

    # Serialize the prediction result into the desired response content type
    output = output_fn(prediction, response_content_type)

The above code-sample shows the three function definitions:

-  ``input_fn``: Takes request data and deserializes the data into an
   object for prediction.
-  ``predict_fn``: Takes the deserialized request object and performs
   inference against the loaded model.
-  ``output_fn``: Takes the result of prediction and serializes this
   according to the response content type.

The SageMaker Chainer model server provides default implementations of these functions.
You can provide your own implementations for these functions in your hosting script.
If you omit any definition then the SageMaker Chainer model server will use its default implementation for that
function.

The ``Predictor`` used by Chainer in the SageMaker Python SDK serializes NumPy arrays to the `NPY <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ format
by default, with Content-Type ``application/x-npy``. The SageMaker Chainer model server can deserialize NPY-formatted
data (along with JSON and CSV data).

If you rely solely on the SageMaker Chainer model server defaults, you get the following functionality:

-  Prediction on models that implement the ``__call__`` method
-  Serialization and deserialization of NumPy arrays.

The default ``input_fn`` and ``output_fn`` are meant to make it easy to predict on NumPy arrays. If your model expects
a NumPy array and returns a NumPy array, then these functions do not have to be overridden when sending NPY-formatted
data.

In the following sections we describe the default implementations of input_fn, predict_fn, and output_fn.
We describe the input arguments and expected return types of each, so you can define your own implementations.

Process Input
^^^^^^^^^^^^^

When an InvokeEndpoint operation is made against an Endpoint running a SageMaker Chainer model server,
the model server receives two pieces of information:

-  The request Content-Type, for example "application/x-npy"
-  The request data body, a byte array

The SageMaker Chainer model server will invoke an "input_fn" function in your hosting script,
passing in this information. If you define an ``input_fn`` function definition,
it should return an object that can be passed to ``predict_fn`` and have the following signature:

.. code:: python

    def input_fn(request_body, request_content_type)

Where ``request_body`` is a byte buffer and ``request_content_type`` is a Python string

The SageMaker Chainer model server provides a default implementation of ``input_fn``.
This function deserializes JSON, CSV, or NPY encoded data into a NumPy array.

Default NPY deserialization requires ``request_body`` to follow the `NPY <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ format. For Chainer, the Python SDK
defaults to sending prediction requests with this format.

Default json deserialization requires ``request_body`` contain a single json list.
Sending multiple json objects within the same ``request_body`` is not supported.
The list must have a dimensionality compatible with the model loaded in ``model_fn``.
The list's shape must be identical to the model's input shape, for all dimensions after the first (which first
dimension is the batch size).

Default csv deserialization requires ``request_body`` contain one or more lines of CSV numerical data.
The data is loaded into a two-dimensional array, where each line break defines the boundaries of the first dimension.

The example below shows a custom ``input_fn`` for preparing pickled NumPy arrays.

.. code:: python

    import numpy as np

    def input_fn(request_body, request_content_type):
        """An input_fn that loads a pickled numpy array"""
        if request_content_type == "application/python-pickle":
            array = np.load(StringIO(request_body))
            return array
        else:
            # Handle other content-types here or raise an Exception
            # if the content type is not supported.
            pass



Get Predictions
---------------

After the inference request has been deserialized by ``input_fn``, the SageMaker Chainer model server invokes
``predict_fn`` on the return value of ``input_fn``.

As with ``input_fn``, you can define your own ``predict_fn`` or use the SageMaker Chainer model server default.

The ``predict_fn`` function has the following signature:

.. code:: python

    def predict_fn(input_object, model)

Where ``input_object`` is the object returned from ``input_fn`` and
``model`` is the model loaded by ``model_fn``.

The default implementation of ``predict_fn`` invokes the loaded model's ``__call__`` function on ``input_object``,
and returns the resulting value. The return-type should be a NumPy array to be compatible with the default
``output_fn``.

The example below shows an overridden ``predict_fn``. This model accepts a Python list and returns a tuple of
bounding boxes, labels, and scores from the model in a NumPy array. This ``predict_fn`` can rely on the default
``input_fn`` and ``output_fn`` because ``input_data`` is a NumPy array, and the return value of this function is
a NumPy array.

.. code:: python

    import chainer
    import numpy as np

    def predict_fn(input_data, model):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            bboxes, labels, scores = model.predict([input_data])
            bbox, label, score = bboxes[0], labels[0], scores[0]
            return np.array([bbox.tolist(), label, score])

If you implement your own prediction function, you should take care to ensure that:

-  The first argument is expected to be the return value from input_fn.
   If you use the default input_fn, this will be a NumPy array.
-  The second argument is the loaded model.
-  The return value should be of the correct type to be passed as the
   first argument to ``output_fn``. If you use the default
   ``output_fn``, this should be a NumPy array.

Process Output
^^^^^^^^^^^^^^

After invoking ``predict_fn``, the model server invokes ``output_fn``, passing in the return-value from ``predict_fn``
and the InvokeEndpoint requested response content-type.

The ``output_fn`` has the following signature:

.. code:: python

    def output_fn(prediction, content_type)

Where ``prediction`` is the result of invoking ``predict_fn`` and
``content_type`` is the InvokeEndpoint requested response content-type.
The function should return a byte array of data serialized to content_type.

The default implementation expects ``prediction`` to be an NumPy and can serialize the result to JSON, CSV, or NPY.
It accepts response content types of "application/json", "text/csv", and "application/x-npy".

Working with existing model data and training jobs
==================================================

Attach to Existing Training Jobs
--------------------------------

You can attach an Chainer Estimator to an existing training job using the
``attach`` method.

.. code:: python

    my_training_job_name = "MyAwesomeChainerTrainingJob"
    chainer_estimator = Chainer.attach(my_training_job_name)

After attaching, if the training job is in a Complete status, it can be
``deploy``\ ed to create a SageMaker Endpoint and return a
``Predictor``. If the training job is in progress,
attach will block and display log messages from the training job, until the training job completes.

The ``attach`` method accepts the following arguments:

-  ``training_job_name (str):`` The name of the training job to attach
   to.
-  ``sagemaker_session (sagemaker.Session or None):`` The Session used
   to interact with SageMaker

Deploy Endpoints from Model Data
--------------------------------

As well as attaching to existing training jobs, you can deploy models directly from model data in S3.
The following code sample shows how to do this, using the ``ChainerModel`` class.

.. code:: python

    chainer_model = ChainerModel(
        model_data="s3://bucket/model.tar.gz",
        role="SageMakerRole",
        entry_point="transform_script.py",
    )

    predictor = chainer_model.deploy(instance_type="ml.c4.xlarge", initial_instance_count=1)

To see what arguments are accepted by the ``ChainerModel`` constructor, see :class:`sagemaker.chainer.model.ChainerModel`.

Your model data must be a .tar.gz file in S3. SageMaker Training Job model data is saved to .tar.gz files in S3,
however if you have local data you want to deploy, you can prepare the data yourself.

Assuming you have a local directory containg your model data named "my_model" you can tar and gzip compress the file and
upload to S3 using the following commands:

::

    tar -czf model.tar.gz my_model
    aws s3 cp model.tar.gz s3://my-bucket/my-path/model.tar.gz

This uploads the contents of my_model to a gzip compressed tar file to S3 in the bucket "my-bucket", with the key
"my-path/model.tar.gz".

To run this command, you'll need the aws cli tool installed. Please refer to our `FAQ <#FAQ>`__ for more information on
installing this.

********
Examples
********

Amazon provides several example Jupyter notebooks that demonstrate end-to-end training on Amazon SageMaker using Chainer.
Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.

*************************
SageMaker Chainer Classes
*************************

For information about the different Chainer-related classes in the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/frameworks/chainer/sagemaker.chainer.html.

***********************************
SageMaker Chainer Docker containers
***********************************

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several
libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control
the environment your script runs in.

SageMaker runs Chainer Estimator scripts in either Python 2.7 or Python 3.5. You can select the Python version by
passing a py_version keyword arg to the Chainer Estimator constructor. Setting this to py3 (the default) will cause your
training script to be run on Python 3.5. Setting this to py2 will cause your training script to be run on Python 2.7
This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

The Chainer Docker images have the following dependencies installed:

+-----------------------------+-------------+-------------+-------------+
| Dependencies                | chainer 4.0 | chainer 4.1 | chainer 5.0 |
+-----------------------------+-------------+-------------+-------------+
| chainer                     | 4.0.0       | 4.1.0       | 5.0.0       |
+-----------------------------+-------------+-------------+-------------+
| chainercv                   | 0.9.0       | 0.10.0      | 0.10.0      |
+-----------------------------+-------------+-------------+-------------+
| chainermn                   | 1.2.0       | 1.3.0       | N/A         |
+-----------------------------+-------------+-------------+-------------+
| CUDA (GPU image only)       | 9.0         | 9.0         | 9.0         |
+-----------------------------+-------------+-------------+-------------+
| cupy                        | 4.0.0       | 4.1.0       | 5.0.0       |
+-----------------------------+-------------+-------------+-------------+
| matplotlib                  | 2.2.0       | 2.2.0       | 2.2.0       |
+-----------------------------+-------------+-------------+-------------+
| mpi4py                      | 3.0.0       | 3.0.0       | 3.0.0       |
+-----------------------------+-------------+-------------+-------------+
| numpy                       | 1.14.3      | 1.15.3      | 1.15.4      |
+-----------------------------+-------------+-------------+-------------+
| opencv-python               | 3.4.0.12    | 3.4.0.12    | 3.4.0.12    |
+-----------------------------+-------------+-------------+-------------+
| Pillow                      | 5.1.0       | 5.3.0       | 5.3.0       |
+-----------------------------+-------------+-------------+-------------+
| Python                      | 2.7 or 3.5  | 2.7 or 3.5  | 2.7 or 3.5  |
+-----------------------------+-------------+-------------+-------------+

The Docker images extend Ubuntu 16.04.

You must select a version of Chainer by passing a ``framework_version`` keyword arg to the Chainer Estimator
constructor. Currently supported versions are listed in the above table. You can also set framework_version to only
specify major and minor version, which will cause your training script to be run on the latest supported patch
version of that minor version.

Alternatively, you can build your own image by following the instructions in the SageMaker Chainer containers
repository, and passing ``image_uri`` to the Chainer Estimator constructor.

You can visit the SageMaker Chainer containers repository at https://github.com/aws/sagemaker-chainer-container
