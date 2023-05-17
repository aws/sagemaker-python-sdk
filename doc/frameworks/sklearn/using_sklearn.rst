################################################
Using Scikit-learn with the SageMaker Python SDK
################################################

With Scikit-learn Estimators, you can train and host Scikit-learn models on Amazon SageMaker.

For information about supported versions of Scikit-learn, see the `AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html>`__.
We recommend that you use the latest supported version because that's where we focus most of our development efforts.

For more information about the framework, see the `Scikit-Learn <https://github.com/scikit-learn/scikit-learn>`_ repository.
For general information about using the SageMaker Python SDK, see :ref:`overview:Using the SageMaker Python SDK`.

.. contents::

*******************************
Train a Model with Scikit-learn
*******************************

To train a Scikit-learn model by using the SageMaker Python SDK:

.. |create sklearn estimator| replace:: Create a ``sagemaker.sklearn.SKLearn`` Estimator
.. _create sklearn estimator: #create-an-estimator

.. |call fit| replace:: Call the estimator's ``fit`` method
.. _call fit: #call-the-fit-method

1. `Prepare a training script <#prepare-a-scikit-learn-training-script>`_
2. |create sklearn estimator|_
3. |call fit|_

Prepare a Scikit-learn Training Script
======================================

Your Scikit-learn training script must be a Python 3.7 compatible source file.

The training script is similar to a training script you might run outside of SageMaker, but you
can access useful properties about the training environment through various environment variables.
For example:

* ``SM_MODEL_DIR``: A string representing the path to the directory to write model artifacts to.
  These artifacts are uploaded to S3 for model hosting.
* ``SM_OUTPUT_DATA_DIR``: A string representing the filesystem path to write output artifacts to. Output artifacts may
  include checkpoints, graphs, and other files to save, not including model artifacts. These artifacts are compressed
  and uploaded to S3 to the same S3 prefix as the model artifacts.

Supposing two input channels, 'train' and 'test', were used in the call to the Scikit-learn estimator's ``fit()`` method,
the following will be set, following the format "SM_CHANNEL_[channel_name]":

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
        parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
        parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

        args, _ = parser.parse_known_args()

        # ... load from args.train and args.test, train a model, write model to args.model_dir.

Because the SageMaker imports your training script, you should put your training code in a main guard
(``if __name__=='__main__':``) if you are using the same script to host your model, so that SageMaker does not
inadvertently run your training code at the wrong point in execution.

For more on training environment variables, please visit
`SageMaker Training Toolkit <https://github.com/aws/sagemaker-training-toolkit>`_.

.. important::
    The sagemaker-containers repository has been deprecated,
    however it is still used to define Scikit-learn and XGBoost environment variables.

Save the Model
--------------

In order to save your trained Scikit-learn model for deployment on SageMaker, your training script should save your
model to a certain filesystem path called `model_dir`. This value is accessible through the environment variable
``SM_MODEL_DIR``. The following code demonstrates how to save a trained Scikit-learn model named ``model`` as
``model.joblib`` at the end of training:

.. code:: python

    from sklearn.externals import joblib
    import argparse
    import os

    if __name__=='__main__':
        # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        args, _ = parser.parse_known_args()

        # ... train classifier `clf`, then save it to `model_dir` as file 'model.joblib'
        joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

After your training job is complete, SageMaker will compress and upload the serialized model to S3, and your model data
will available in the s3 ``output_path`` you specified when you created the Scikit-learn Estimator.

Using third-party libraries
---------------------------

When running your training script on SageMaker, it has access to some pre-installed third-party libraries including ``scikit-learn``, ``numpy``, and ``pandas``.
For more information on the runtime environment, including specific package versions, see `SageMaker Scikit-learn Docker Container <https://github.com/aws/sagemaker-scikit-learn-container>`__.

If there are other packages you want to use with your script, you can include a ``requirements.txt`` file in the same directory as your training script to install other dependencies at runtime.
Both ``requirements.txt`` and your training script should be put in the same folder.
You must specify this folder in ``source_dir`` argument when creating a Scikit-learn estimator.
A ``requirements.txt`` file is a text file that contains a list of items that are installed by using ``pip install``.
You can also specify the version of an item to install.
For information about the format of a ``requirements.txt`` file, see `Requirements Files <https://pip.pypa.io/en/stable/user_guide#requirements-files>`__ in the pip documentation.

Create an Estimator
===================

You run Scikit-learn training scripts on SageMaker by creating ``SKLearn`` Estimators.
Call the ``fit`` method on a ``SKLearn`` Estimator to start a SageMaker training job.
The following code sample shows how you train a custom Scikit-learn script named "sklearn-train.py", passing
in three hyperparameters ('epochs', 'batch-size', and 'learning-rate'), and using two input channel
directories ('train' and 'test').

.. code:: python

    sklearn_estimator = SKLearn('sklearn-train.py',
                                instance_type='ml.m4.xlarge',
                                framework_version='1.0-1',
                                hyperparameters = {'epochs': 20, 'batch-size': 64, 'learning-rate': 0.1})
    sklearn_estimator.fit({'train': 's3://my-data-bucket/path/to/my/training/data',
                            'test': 's3://my-data-bucket/path/to/my/test/data'})





Call the fit Method
===================

You start your training script by calling ``fit`` on a ``SKLearn`` Estimator. ``fit`` takes both required and optional
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

***************************
Deploy a Scikit-learn Model
***************************

After you fit a Scikit-learn Estimator, you can host the newly created model in SageMaker.

After you call ``fit``, you can call ``deploy`` on an ``SKLearn`` estimator to create a SageMaker endpoint.
The endpoint runs a SageMaker-provided Scikit-learn model server and hosts the model produced by your training script,
which was run when you called ``fit``. This was the model you saved to ``model_dir``.

``deploy`` returns a ``Predictor`` object, which you can use to do inference on the Endpoint hosting your Scikit-learn
model. Each ``Predictor`` provides a ``predict`` method which can do inference with numpy arrays or Python lists.
Inference arrays or lists are serialized and sent to the Scikit-learn model server by an ``InvokeEndpoint`` SageMaker
operation.

``predict`` returns the result of inference against your model. By default, the inference result a NumPy array.

.. code:: python

    # Train my estimator
    sklearn_estimator = SKLearn(entry_point='train_and_deploy.py',
                                instance_type='ml.m4.xlarge',
                                framework_version='1.0-1')
    sklearn_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploy my estimator to a SageMaker Endpoint and get a Predictor
    predictor = sklearn_estimator.deploy(instance_type='ml.m4.xlarge',
                                         initial_instance_count=1)

    # `data` is a NumPy array or a Python list.
    # `response` is a NumPy array.
    response = predictor.predict(data)

You use the SageMaker Scikit-learn model server to host your Scikit-learn model when you call ``deploy``
on an ``SKLearn`` Estimator. The model server runs inside a SageMaker Endpoint, which your call to ``deploy`` creates.
You can access the name of the Endpoint by the ``name`` property on the returned ``Predictor``.


SageMaker Scikit-learn Model Server
===================================

The Scikit-learn Endpoint you create with ``deploy`` runs a SageMaker Scikit-learn model server.
The model server loads the model that was saved by your training script and performs inference on the model in response
to SageMaker InvokeEndpoint API calls.

You can configure two components of the SageMaker Scikit-learn model server: Model loading and model serving.
Model loading is the process of deserializing your saved model back into an Scikit-learn model.
Serving is the process of translating InvokeEndpoint requests to inference calls on the loaded model.

You configure the Scikit-learn model server by defining functions in the Python source file you passed to the
Scikit-learn constructor.

Load a Model
------------

Before a model can be served, it must be loaded. The SageMaker Scikit-learn model server loads your model by invoking a
``model_fn`` function that you must provide in your script. The ``model_fn`` should have the following signature:

.. code:: python

    def model_fn(model_dir):
        ...

SageMaker will inject the directory where your model files and sub-directories, saved by ``save``, have been mounted.
Your model function should return a model object that can be used for model serving.

SageMaker provides automated serving functions that work with Gluon API ``net`` objects and Module API ``Module`` objects. If you return either of these types of objects, then you will be able to use the default serving request handling functions.

The following code-snippet shows an example ``model_fn`` implementation.
This loads returns a Scikit-learn Classifier from a ``model.joblib`` file in the SageMaker model directory
``model_dir``.

.. code:: python

    from sklearn.externals import joblib
    import os

    def model_fn(model_dir):
        clf = joblib.load(os.path.join(model_dir, "model.joblib"))
        return clf

Serve a Model
-------------

After the SageMaker model server has loaded your model by calling ``model_fn``, SageMaker will serve your model.
Model serving is the process of responding to inference requests, received by SageMaker InvokeEndpoint API calls.
The SageMaker Scikit-learn model server breaks request handling into three steps:


-  input processing,
-  prediction, and
-  output processing.

In a similar way to model loading, you configure these steps by defining functions in your Python source file.

Each step involves invoking a python function, with information about the request and the return-value from the previous
function in the chain. Inside the SageMaker Scikit-learn model server, the process looks like:

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

The SageMaker Scikit-learn model server provides default implementations of these functions.
You can provide your own implementations for these functions in your hosting script.
If you omit any definition then the SageMaker Scikit-learn model server will use its default implementation for that
function.

The ``Predictor`` used by Scikit-learn in the SageMaker Python SDK serializes NumPy arrays to the `NPY <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ format
by default, with Content-Type ``application/x-npy``. The SageMaker Scikit-learn model server can deserialize NPY-formatted
data (along with JSON and CSV data).

If you rely solely on the SageMaker Scikit-learn model server defaults, you get the following functionality:

-  Prediction on models that implement the ``__call__`` method
-  Serialization and deserialization of NumPy arrays.

The default ``input_fn`` and ``output_fn`` are meant to make it easy to predict on NumPy arrays. If your model expects
a NumPy array and returns a NumPy array, then these functions do not have to be overridden when sending NPY-formatted
data.

In the following sections we describe the default implementations of input_fn, predict_fn, and output_fn.
We describe the input arguments and expected return types of each, so you can define your own implementations.

Process Input
^^^^^^^^^^^^^

When an InvokeEndpoint operation is made against an Endpoint running a SageMaker Scikit-learn model server,
the model server receives two pieces of information:

-  The request Content-Type, for example "application/x-npy"
-  The request data body, a byte array

The SageMaker Scikit-learn model server will invoke an "input_fn" function in your hosting script,
passing in this information. If you define an ``input_fn`` function definition,
it should return an object that can be passed to ``predict_fn`` and have the following signature:

.. code:: python

  def input_fn(request_body, request_content_type):
      ...

where ``request_body`` is a byte buffer and ``request_content_type`` is a Python string.

The SageMaker Scikit-learn model server provides a default implementation of ``input_fn``.
This function deserializes JSON, CSV, or NPY encoded data into a NumPy array.

Default NPY deserialization requires ``request_body`` to follow the
`NPY <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ format. For Scikit-learn, the Python SDK
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

After the inference request has been deserialized by ``input_fn``, the SageMaker Scikit-learn model server invokes
``predict_fn`` on the return value of ``input_fn``.

As with ``input_fn``, you can define your own ``predict_fn`` or use the SageMaker Scikit-learn model server default.

The ``predict_fn`` function has the following signature:

.. code:: python

    def predict_fn(input_object, model):
        ...

Where ``input_object`` is the object returned from ``input_fn`` and
``model`` is the model loaded by ``model_fn``.

The default implementation of ``predict_fn`` invokes the loaded model's ``predict`` function on ``input_object``,
and returns the resulting value. The return-type should be a NumPy array to be compatible with the default
``output_fn``.

The example below shows an overridden ``predict_fn`` for a Logistic Regression classifier. This model accepts a
Python list and returns a tuple of predictions and prediction probabilities from the model in a NumPy array.
This ``predict_fn`` can rely on the default ``input_fn`` and ``output_fn`` because ``input_data`` is a NumPy array,
and the return value of this function is a NumPy array.

.. code:: python

    import sklearn
    import numpy as np

    def predict_fn(input_data, model):
        prediction = model.predict(input_data)
        pred_prob = model.predict_proba(input_data)
        return np.array([prediction, pred_prob])

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

    def output_fn(prediction, content_type):
        ...

Where ``prediction`` is the result of invoking ``predict_fn`` and
``content_type`` is the InvokeEndpoint requested response content-type.
The function should return a byte array of data serialized to content_type.

The default implementation expects ``prediction`` to be an NumPy and can serialize the result to JSON, CSV, or NPY.
It accepts response content types of "application/json", "text/csv", and "application/x-npy".

Working with existing model data and training jobs
==================================================

Attach to Existing Training Jobs
--------------------------------

You can attach an Scikit-learn Estimator to an existing training job using the
``attach`` method.

.. code:: python

    my_training_job_name = "MyAwesomeSKLearnTrainingJob"
    sklearn_estimator = SKLearn.attach(my_training_job_name)

After attaching, if the training job is in a Complete status, it can be
``deploy``\ ed to create a SageMaker Endpoint and return a
``Predictor``. If the training job is in progress,
attach will block and display log messages from the training job, until the training job completes.

The ``attach`` method accepts the following arguments:

-  ``training_job_name (str):`` The name of the training job to attach
   to.
-  ``sagemaker_session (sagemaker.Session or None):`` The Session used
   to interact with SageMaker

Deploy an Endpoint from Model Data
----------------------------------

As well as attaching to existing training jobs, you can deploy models directly from model data in S3.
The following code sample shows how to do this, using the ``SKLearnModel`` class.

.. code:: python

    sklearn_model = SKLearnModel(model_data="s3://bucket/model.tar.gz",
                                 role="SageMakerRole",
                                 entry_point="transform_script.py",
                                 framework_version="1.0-1")

    predictor = sklearn_model.deploy(instance_type="ml.c4.xlarge", initial_instance_count=1)

To see what arguments are accepted by the ``SKLearnModel`` constructor, see :class:`sagemaker.sklearn.model.SKLearnModel`.

Your model data must be a .tar.gz file in S3. SageMaker Training Job model data is saved to .tar.gz files in S3,
however if you have local data you want to deploy, you can prepare the data yourself.

Assuming you have a local directory containing your model data named "my_model", you can tar and gzip compress the file and
upload to S3 using the following commands:

.. code::

    tar -czf model.tar.gz my_model
    aws s3 cp model.tar.gz s3://my-bucket/my-path/model.tar.gz

This uploads the contents of my_model to a gzip compressed tar file to S3 in the bucket "my-bucket", with the key
"my-path/model.tar.gz".

To run this command, you'll need the AWS CLI tool installed. Please refer to our `FAQ <#FAQ>`__ for more information on
installing this.

******************************
Scikit-learn Training Examples
******************************

To find example notebooks that demonstrate end-to-end training on Amazon SageMaker using Scikit-learn,
see the `Amazon SageMaker example notebooks repository
<https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk>`_.

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.

******************************
SageMaker Scikit-learn Classes
******************************

For information about the different Scikit-learn classes in the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html.

****************************************
SageMaker Scikit-learn Docker Containers
****************************************

To find the SageMaker-managed Scikit-learn containers,
visit the `SageMaker Scikit-Learn containers repository <https://github.com/aws/sagemaker-scikit-learn-container>`_.
