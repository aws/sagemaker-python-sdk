#########################################
Use XGBoost with the SageMaker Python SDK
#########################################

.. contents::

eXtreme Gradient Boosting (XGBoost) is a popular and efficient machine learning algorithm used for regression and classification tasks on tabular datasets.
It implements a technique known as gradient boosting on trees, which performs remarkably well in machine learning competitions.

Amazon SageMaker supports two ways to use the XGBoost algorithm:

 * XGBoost built-in algorithm
 * XGBoost open source algorithm

The XGBoost open source algorithm provides the following benefits over the built-in algorithm:

* Latest version - The open source XGBoost algorithm typically supports a more recent version of XGBoost.
  To see the XGBoost version that is currently supported,
  see `XGBoost SageMaker Estimators and Models <https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/xgboost#xgboost-sagemaker-estimators-and-models>`__.
* Flexibility - Take advantage of the full range of XGBoost functionality, such as cross-validation support.
  You can add custom pre- and post-processing logic and run additional code after training.
* Scalability - The XGBoost open source algorithm has a more efficient implementation of distributed training,
  which enables it to scale out to more instances and reduce out-of-memory errors.
* Extensibility - Because the open source XGBoost container is open source,
  you can extend the container to install additional libraries and change the version of XGBoost that the container uses.
  For an example notebook that shows how to extend SageMaker containers, see `Extending our PyTorch containers <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb>`__.


***********************************
Use XGBoost as a Built-in Algortihm
***********************************

Amazon SageMaker provides XGBoost as a built-in algorithm that you can use like other built-in algorithms.
Using the built-in algorithm version of XGBoost is simpler than using the open source version, because you don't have to write a training script.
If you don't need the features and flexibility of open source XGBoost, consider using the built-in version.
For information about using the Amazon SageMaker XGBoost built-in algorithm, see `XGBoost Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html>`__
in the *Amazon SageMaker Developer Guide*.

*************************************
Use the Open Source XGBoost Algorithm
*************************************

If you want the flexibility and additional features that it provides, use the SageMaker open source XGBoost algorithm.

For which XGBoost versions are supported, see `the AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html>`_.
We recommend that you use the latest supported version because that's where we focus most of our development efforts.

For a complete example of using the open source XGBoost algorithm, see the sample notebook at
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode.ipynb.

For more information about XGBoost, see `the XGBoost documentation <https://xgboost.readthedocs.io/en/latest>`_.

Train a Model with Open Source XGBoost
======================================

To train a model by using the Amazon SageMaker open source XGBoost algorithm:

.. |create xgboost estimator| replace:: Create a ``sagemaker.xgboost.XGBoost estimator``
.. _create xgboost estimator: #create-an-estimator

.. |call fit| replace:: Call the estimator's ``fit`` method
.. _call fit: #call-the-fit-method

1. `Prepare a training script <#prepare-a-training-script>`_
2. |create xgboost estimator|_
3. |call fit|_

Prepare a Training Script
-------------------------

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model,
and saves a model to ``model_dir`` so that it can be hosted later.
Hyperparameters are passed to your script as arguments and can be retrieved with an ``argparse.ArgumentParser`` instance.
For information about ``argparse.ArgumentParser``, see `argparse <https://docs.python.org/3/library/argparse.html>`__ in the Python documentation.


For a complete example of an XGBoost training script, see https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/abalone.py.

The training script is very similar to a training script you might run outside of Amazon SageMaker,
but you can access useful properties about the training environment through various environment variables, including the following:

* ``SM_MODEL_DIR``: A string that represents the path where the training job writes the model artifacts to.
  After training, artifacts in this directory are uploaded to Amazon S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_CHANNEL_XXXX``: A string that represents the path to the directory that contains the input data for the specified channel.
  For example, if you specify two input channels in the MXNet estimator's ``fit`` call, named 'train' and 'test', the environment variables ``SM_CHANNEL_TRAIN`` and ``SM_CHANNEL_TEST`` are set.
* ``SM_HPS``: A JSON dump of the hyperparameters preserving JSON types (boolean, integer, etc.)

For the exhaustive list of available environment variables, see the `SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

.. important::
    The sagemaker-containers repository has been deprecated,
    however it is still used to define Scikit-learn and XGBoost environment variables.

Let's look at the main elements of the script. Starting with the ``__main__`` guard,
use a parser to read the hyperparameters passed to the estimator when creating the training job.
These hyperparameters are made available as arguments to our input script.
We also parse a number of Amazon SageMaker-specific environment variables to get information about the training environment,
such as the location of input data and location where we want to save the model.

.. code:: python

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        # Hyperparameters are described here
        parser.add_argument('--num_round', type=int)
        parser.add_argument('--max_depth', type=int, default=5)
        parser.add_argument('--eta', type=float, default=0.2)
        parser.add_argument('--objective', type=str, default='reg:squarederror')

        # SageMaker specific arguments. Defaults are set in the environment variables.
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
        parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

        args = parser.parse_args()

        train_hp = {
            'max_depth': args.max_depth,
            'eta': args.eta,
            'gamma': args.gamma,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'silent': args.silent,
            'objective': args.objective
        }

        dtrain = xgb.DMatrix(args.train)
        dval = xgb.DMatrix(args.validation)
        watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

        callbacks = []
        prev_checkpoint, n_iterations_prev_run = add_checkpointing(callbacks)
        # If checkpoint is found then we reduce num_boost_round by previously run number of iterations

        bst = xgb.train(
            params=train_hp,
            dtrain=dtrain,
            evals=watchlist,
            num_boost_round=(args.num_round - n_iterations_prev_run),
            xgb_model=prev_checkpoint,
            callbacks=callbacks
        )

        # Save the model to the location specified by ``model_dir``
        model_location = args.model_dir + '/xgboost-model'
        pkl.dump(bst, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))

Create an Estimator
-------------------
After you create your training script, create an instance of the :class:`sagemaker.xgboost.estimator.XGBoost` estimator.
Pass an IAM role that has the permissions necessary to run an Amazon SageMaker training job,
the type and number of instances to use for the training job,
and a dictionary of the hyperparameters to pass to the training script.

.. code::

    from sagemaker.xgboost.estimator import XGBoost

    xgb_estimator = XGBoost(
        entry_point="abalone.py",
        hyperparameters=hyperparameters,
        role=role,
        instance_count=1,
        instance_type="ml.m5.2xlarge",
        framework_version="1.0-1",
    )


Call the fit Method
-------------------

After you create an estimator, call the ``fit`` method to run the training job.

.. code::

    xgb_script_mode_estimator.fit({"train": train_input})



Deploy Open Source XGBoost Models
=================================

After you fit an XGBoost Estimator, you can host the newly created model in SageMaker.

After you call ``fit``, you can call ``deploy`` on an ``XGBoost`` estimator to create a SageMaker endpoint.
The endpoint runs a SageMaker-provided XGBoost model server and hosts the model produced by your training script,
which was run when you called ``fit``. This was the model you saved to ``model_dir``.

``deploy`` returns a ``Predictor`` object, which you can use to do inference on the Endpoint hosting your XGBoost model.
Each ``Predictor`` provides a ``predict`` method which can do inference with numpy arrays, Python lists, or strings.
After inference arrays or lists are serialized and sent to the XGBoost model server, ``predict`` returns the result of
inference against your model.

.. code::

    serializer = StringSerializer(content_type="text/libsvm")

    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        serializer=serializer
    )

    with open("abalone") as f:
        payload = f.read()

    predictor.predict(payload)

SageMaker XGBoost Model Server
-----------------------------------

You can configure two components of the SageMaker XGBoost model server: Model loading and model serving.
Model loading is the process of deserializing your saved model back into an XGBoost model.
Model serving is the process of translating endpoint requests to inference calls on the loaded model.

You configure the XGBoost model server by defining functions in the Python source file you passed to the XGBoost constructor.

Load a Model
^^^^^^^^^^^^

Before a model can be served, it must be loaded. The SageMaker XGBoost model server loads your model by invoking a
``model_fn`` function that you must provide in your script. The ``model_fn`` should have the following signature:

.. code:: python

    def model_fn(model_dir)

SageMaker will inject the directory where your model files and sub-directories, saved by ``save``, have been mounted.
Your model function should return a ``xgboost.Booster`` object that can be used for model serving.

The following code-snippet shows an example ``model_fn`` implementation.
It loads and returns a pickled XGBoost model from a ``xgboost-model`` file in the SageMaker model directory ``model_dir``.

.. code:: python

    import pickle as pkl

    def model_fn(model_dir):
        with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
            booster = pkl.load(f)
        return booster

Serve a Model
^^^^^^^^^^^^^

After the SageMaker model server has loaded your model by calling ``model_fn``, SageMaker will serve your model.
The SageMaker Scikit-learn model server breaks request handling into three steps:

-  input processing,
-  prediction, and
-  output processing.

In a similar way to model loading, you can customize the inference behavior by defining functions in your inference
script, which can be either in the same file as your training script or in a separate file,

Each step involves invoking a python function, with information about the request and the return-value from the previous
function in the chain.
Inside the SageMaker XGBoost model server, the process looks like:

.. code:: python

    # Deserialize the Invoke request body into an object we can perform prediction on
    input_object = input_fn(request_body, request_content_type)

    # Perform prediction on the deserialized object, with the loaded model
    prediction = predict_fn(input_object, model)

    # Serialize the prediction result into the desired response content type
    output = output_fn(prediction, response_content_type)

The above code-sample shows the three function definitions:

-  ``input_fn``: Takes request data and deserializes the data into an object for prediction.
-  ``predict_fn``: Takes the deserialized request object and performs inference against the loaded model.
-  ``output_fn``: Takes the result of prediction and serializes this according to the response content type.

These functions are optional.
The SageMaker XGBoost model server provides default implementations of these functions.
You can provide your own implementations for these functions in your hosting script.
If you omit any definition then the SageMaker XGBoost model server will use its default implementation for that
function.

In the following sections we describe the default implementations of ``input_fn``, ``predict_fn``, and ``output_fn``.
We describe the input arguments and expected return types of each, so you can define your own implementations.

Process Input
"""""""""""""

When a request is made against an endpoint running a SageMaker XGBoost model server, the model server receives two
pieces of information:

-  The request Content-Type, for example "application/x-npy" or "text/libsvm"
-  The request data body, a byte array

The SageMaker XGBoost model server will invoke an ``input_fn`` function in your inference script, passing in this
information. If you define an ``input_fn`` function definition, it should return an object that can be passed
to ``predict_fn`` and have the following signature:

.. code:: python

    def input_fn(request_body, request_content_type)

where ``request_body`` is a byte buffer and ``request_content_type`` is a Python string.

The SageMaker XGBoost model server provides a default implementation of ``input_fn``.
This function deserializes CSV, LIBSVM, or protobuf recordIO into a ``xgboost.DMatrix``.

Default csv deserialization requires ``request_body`` contain one or more lines of CSV numerical data.
The data is first loaded into a two-dimensional array, where each line break defines the boundaries of the first
dimension, and then it is converted to an `xgboost.Dmatrix`. It assumes that CSV input does not have the
label column.

Default LIBSVM deserialization requires ``request_body`` to follow the `LIBSVM <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ format.

The example below shows a custom ``input_fn`` for preparing pickled NumPy arrays.

.. code:: python

    from io import BytesIO
    import numpy as np
    import xgboost as xgb

    def input_fn(request_body, request_content_type):
        """An input_fn that loads a numpy array"""
        if request_content_type == "application/npy":
            array = np.load(BytesIO(request_body))
            return xgb.DMatrix(array)
        else:
            # Handle other content-types here or raise an Exception
            # if the content type is not supported.
            pass

Get Predictions
"""""""""""""""

After the inference request has been deserialized by ``input_fn``, the SageMaker XGBoost model server invokes
``predict_fn`` on the return value of ``input_fn``.

As with ``input_fn``, you can define your own ``predict_fn`` or use the SageMaker XGBoost model server default.

The ``predict_fn`` function has the following signature:

.. code:: python

    def predict_fn(input_object, model)

Where ``input_object`` is the object returned from ``input_fn`` and ``model`` is the model loaded by ``model_fn``.

The default implementation of ``predict_fn`` invokes the loaded model's ``predict`` function on ``input_object``,
and returns the resulting value. The return-type should be a NumPy array to be compatible with the default
``output_fn``.

The example below shows an overriden ``predict_fn`` that returns a two-dimensional NumPy array where
the first columns are predictions and the remaining columns are the feature contributions
(`SHAP values <https://github.com/slundberg/shap>`_) for that prediction.
When ``pred_contribs`` is ``True`` in ``xgboost.Booster.predict()``, the output will be a matrix of size
(nsample, nfeats + 1) with each record indicating the feature contributions for that prediction.
Note the final column is the bias term.

.. code:: python

    import numpy as np

    def predict_fn(input_data, model):
        prediction = model.predict(input_data)
        feature_contribs = model.predict(input_data, pred_contribs=True)
        output = np.hstack((prediction[:, np.newaxis], feature_contribs))
        return output

If you implement your own prediction function, you should take care to ensure that:

-  The first argument is expected to be the return value from input_fn.
-  The second argument is the loaded model.
-  The return value should be of the correct type to be passed as the first argument to ``output_fn``.
   If you use the default ``output_fn``, this should be a NumPy array.

Process Output
""""""""""""""

After invoking ``predict_fn``, the model server invokes ``output_fn``, passing in the return value from
``predict_fn`` and the requested response content-type.

The ``output_fn`` has the following signature:

.. code:: python

    def output_fn(prediction, content_type)

``prediction`` is the result of invoking ``predict_fn`` and ``content_type`` is the requested response content-type.
The function should return a byte array of data serialized to ``content_type``.

The default implementation expects ``prediction`` to be a NumPy array and can serialize the result to JSON, CSV, or NPY.
It accepts response content types of "application/json", "text/csv", and "application/x-npy".

Bring Your Own Model
--------------------

You can deploy an XGBoost model that you trained outside of SageMaker by using the Amazon SageMaker XGBoost container.
Typically, you save an XGBoost model by pickling the ``Booster`` object or calling ``booster.save_model``.
The XGBoost `built-in algorithm mode <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html#xgboost-modes>`_
supports both a pickled ``Booster`` object and a model produced by ``booster.save_model``.
You can also deploy an XGBoost model by using XGBoost as a framework.
By using XGBoost as a framework, you have more flexibility.
To deploy an XGBoost model by using XGBoost as a framework, you need to:

- Write an inference script.
- Create the XGBoostModel object.

Write an Inference Script
^^^^^^^^^^^^^^^^^^^^^^^^^

You must create an inference script that implements (at least) the ``model_fn`` function that calls the loaded model to get a prediction.

Optionally, you can also implement ``input_fn`` and ``output_fn`` to process input and output,
and ``predict_fn`` to customize how the model server gets predictions from the loaded model.
For information about how to write an inference script, see `SageMaker XGBoost Model Server <#sagemaker-xgboost-model-server>`_.
Pass the filename of the inference script as the ``entry_point`` parameter when you create the `XGBoostModel` object.

Create an XGBoostModel Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a model object, call the ``sagemaker.xgboost.model.XGBoostModel`` constructor,
and then call its ``deploy()`` method to deploy your model for inference.

.. code:: python

    xgboost_model = XGBoostModel(
        model_data="s3://my-bucket/my-path/model.tar.gz",
        role="my-role",
        entry_point="inference.py",
        framework_version="1.0-1"
    )

    predictor = xgboost_model.deploy(
        instance_type='ml.c4.xlarge',
        initial_instance_count=1
    )

    # If payload is a string in LIBSVM format, we need to change serializer.
    predictor.serializer = str
    predictor.predict("<label> <index1>:<value1> <index2>:<value2>")

To get predictions from your deployed model, you can call the ``predict()`` method.

Host Multiple Models with Multi-Model Endpoints
-----------------------------------------------

To create an endpoint that can host multiple models, use multi-model endpoints.
Multi-model endpoints are supported in SageMaker XGBoost versions ``0.90-2``, ``1.0-1``, and later.
For information about using multiple XGBoost models with multi-model endpoints, see
`Host Multiple Models with Multi-Model Endpoints <https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html>`_
in the AWS documentation.
For a sample notebook that uses Amazon SageMaker to deploy multiple XGBoost models to an endpoint, see the
`Multi-Model Endpoint XGBoost Sample Notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_xgboost_home_value/xgboost_multi_model_endpoint_home_value.ipynb>`_.

*************************
SageMaker XGBoost Classes
*************************

For information about the SageMaker Python SDK XGBoost classes, see the following topics:

* :class:`sagemaker.xgboost.estimator.XGBoost`
* :class:`sagemaker.xgboost.model.XGBoostModel`
* :class:`sagemaker.xgboost.model.XGBoostPredictor`
* :class:`sagemaker.xgboost.processing.XGBoostProcessor`

***********************************
SageMaker XGBoost Docker Containers
***********************************

For information about SageMaker XGBoost Docker container and its dependencies, see `SageMaker XGBoost Container <https://github.com/aws/sagemaker-xgboost-container>`_.
