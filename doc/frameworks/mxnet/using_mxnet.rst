#######################################
Use MXNet with the SageMaker Python SDK
#######################################

With the SageMaker Python SDK, you can train and host MXNet models on Amazon SageMaker.

For information about supported versions of MXNet, see the `AWS documentation <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`__.
We recommend that you use the latest supported version because that's where we focus our development efforts.

For general information about using the SageMaker Python SDK, see :ref:`overview:Using the SageMaker Python SDK`.

.. contents::

************************
Train a Model with MXNet
************************

To train an MXNet model by using the SageMaker Python SDK:

.. |create mxnet estimator| replace:: Create a ``sagemaker.mxnet.MXNet`` Estimator
.. _create mxnet estimator: #create-an-estimator

.. |call fit| replace:: Call the estimator's ``fit`` method
.. _call fit: #call-the-fit-method

1. `Prepare a training script <#prepare-an-mxnet-training-script>`_
2. |create mxnet estimator|_
3. |call fit|_

Prepare an MXNet Training Script
================================

The training script is very similar to a training script you might run outside of Amazon SageMaker, but you can access useful properties about the training environment through various environment variables, including the following:

* ``SM_MODEL_DIR``: A string that represents the path where the training job writes the model artifacts to.
  After training, artifacts in this directory are uploaded to Amazon S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_CHANNEL_XXXX``: A string that represents the path to the directory that contains the input data for the specified channel.
  For example, if you specify two input channels in the MXNet estimator's ``fit`` call, named 'train' and 'test', the environment variables ``SM_CHANNEL_TRAIN`` and ``SM_CHANNEL_TEST`` are set.
* ``SM_HPS``: A JSON dump of the hyperparameters preserving JSON types (boolean, integer, etc.)

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

Because Amazon SageMaker imports your training script, you should put your training code in a main guard (``if __name__=='__main__':``) if you are using the same script to host your model,
so that Amazon SageMaker does not inadvertently run your training code at the wrong point in execution.

Note that Amazon SageMaker doesn't support argparse actions.
If you want to use, for example, boolean hyperparameters, you need to specify ``type`` as ``bool`` in your script and provide an explicit ``True`` or ``False`` value for this hyperparameter when instantiating your MXNet estimator.

For more on training environment variables, please visit `SageMaker Containers <https://github.com/aws/sagemaker-containers>`_.

.. note::
    If you want to use MXNet 1.2 or lower, see `an older version of this page <https://sagemaker.readthedocs.io/en/v1.61.0/frameworks/mxnet/using_mxnet.html>`_.

Save a Checkpoint
-----------------

It is good practice to save the best model after each training epoch,
so that you can resume a training job if it gets interrupted.
This is particularly important if you are using Managed Spot training.

To save MXNet model checkpoints, do the following in your training script:

* Set the ``CHECKPOINTS_DIR`` environment variable and enable checkpoints.

   .. code:: python

     CHECKPOINTS_DIR = '/opt/ml/checkpoints'
     checkpoints_enabled = os.path.exists(CHECKPOINTS_DIR)

* Make sure you are emitting a validation metric to test the model. For information, see `Evaluation Metric API <https://mxnet.incubator.apache.org/api/python/metric/metric.html>`_.
* After each training epoch, test whether the current model performs the best with respect to the validation metric, and if it does, save that model to ``CHECKPOINTS_DIR``.

   .. code:: python

     if checkpoints_enabled and current_host == hosts[0]:
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                logging.info('Saving the model, params and optimizer state')
                net.export(CHECKPOINTS_DIR + "/%.4f-cifar10"%(best_accuracy), epoch)
                trainer.save_states(CHECKPOINTS_DIR + '/%.4f-cifar10-%d.states'%(best_accuracy, epoch))

For a complete example of an MXNet training script that impelements checkpointing, see https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_cifar10/cifar10.py.

Save the Model
--------------

There is a default save method that can be imported when training on SageMaker:

.. code:: python

    from sagemaker_mxnet_training.training_utils import save

    if __name__ == '__main__':
        # arg parsing and training (shown above) goes here

        save(args.model_dir, model)

The default serialization system generates three files:

-  ``model-shapes.json``: A JSON list, containing a serialization of the
   ``Module`` ``data_shapes`` property. Each object in the list contains
   the serialization of one ``DataShape`` in the returned ``Module``.
   Each object has a ``name`` property, containing the ``DataShape``
   name and a ``shape`` property, which is a list of that dimensions for
   the shape of that ``DataShape``. For example:

.. code:: javascript

    [
        {"name":"images", "shape":[100, 1, 28, 28]},
        {"name":"labels", "shape":[100, 1]}
    ]

-  ``model-symbol.json``: The MXNet ``Module`` ``Symbol`` serialization,
   produced by invoking ``save`` on the ``symbol`` property of the
   ``Module`` being saved.
-  ``modle.params``: The MXNet ``Module`` parameters, produced by
   invoking ``save_params`` on the ``Module`` being saved.

Use third-party libraries
=========================

When running your training script on Amazon SageMaker, it has access to some pre-installed third-party libraries, including ``mxnet``, ``numpy``, ``onnx``, and ``keras-mxnet``.
For more information on the runtime environment, including specific package versions, see `SageMaker MXNet Containers <#sagemaker-mxnet-containers>`__.

If there are other packages you want to use with your script, you can include a ``requirements.txt`` file in the same directory as your training script to install other dependencies at runtime.
Both ``requirements.txt`` and your training script should be put in the same folder.
You must specify this folder in ``source_dir`` argument when creating an MXNet estimator.

The function of installing packages using ``requirements.txt`` is supported for MXNet versions 1.3.0 and higher during training.

When serving an MXNet model, support for this function varies with MXNet versions.
For MXNet 1.6.0 or newer, ``requirements.txt`` must be under folder ``code``.
The SageMaker MXNet Estimator automatically saves ``code`` in ``model.tar.gz`` after training (assuming you set up your script and ``requirements.txt`` correctly as stipulated in the previous paragraph).
In the case of bringing your own trained model for deployment, you must save ``requirements.txt`` under folder ``code`` in ``model.tar.gz`` yourself or specify it through ``dependencies``.
For MXNet 0.12.1-1.2.1, 1.4.0-1.4.1, ``requirements.txt`` is not supported for inference.
For MXNet 1.3.0, ``requirements.txt`` must be in ``source_dir``.

A ``requirements.txt`` file is a text file that contains a list of items that are installed by using ``pip install``.
You can also specify the version of an item to install.
For information about the format of a ``requirements.txt`` file, see `Requirements Files <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`__ in the pip documentation.

Create an Estimator
===================

You run MXNet training scripts on Amazon SageMaker by creating an ``MXNet`` estimator.
When you call ``fit`` on an ``MXNet`` estimator, Amazon SageMaker starts a training job using your script as training code.
The following code sample shows how you train a custom MXNet script "train.py".

.. code:: python

    mxnet_estimator = MXNet('train.py',
                            instance_type='ml.p2.xlarge',
                            instance_count=1,
                            framework_version='1.6.0',
                            py_version='py3',
                            hyperparameters={'batch-size': 100,
                                             'epochs': 10,
                                             'learning-rate': 0.1})
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

For more information about the sagemaker.mxnet.MXNet estimator, see `SageMaker MXNet Classes`_.


Distributed training
====================

If you want to use parameter servers for distributed training, set the following parameter in your ``MXNet`` constructor:

.. code:: python

    distribution={'parameter_server': {'enabled': True}}

Then, when writing a distributed training script, use an MXNet kvstore to store and share model parameters.
During training, Amazon SageMaker automatically starts an MXNet kvstore server and scheduler processes on hosts in your training job cluster.
Your script runs as an MXNet worker task, with one server process on each host in your cluster.
One host is selected arbitrarily to run the scheduler process.

To learn more about writing distributed MXNet programs, please see `Distributed Training <https://mxnet.incubator.apache.org/versions/master/faq/distributed_training.html>`__ in the MXNet docs.


Call the fit Method
===================

Start your training script by calling ``fit`` on an ``MXNet`` Estimator.
``fit`` takes both required and optional arguments.
For what arguments can be passed into ``fit``, see the `API reference <https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Framework>`_.

*******************
Deploy MXNet models
*******************

Once you have a trained MXNet model, you can host it in Amazon SageMaker by creating an Amazon SageMaker Endpoint.
The endpoint runs a SageMaker-provided MXNet model server and hosts the model produced by your training script.
This model can be one you trained in Amazon SageMaker or a pretrained one from somewhere else.

If you use the ``MXNet`` estimator to train the model, you can call ``deploy`` to create an Amazon SageMaker Endpoint:

.. code:: python

    # Train my estimator
    mxnet_estimator = MXNet('train.py',
                            framework_version='1.6.0',
                            py_version='py3',
                            instance_type='ml.p2.xlarge',
                            instance_count=1)
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploy my estimator to an Amazon SageMaker Endpoint and get a Predictor
    predictor = mxnet_estimator.deploy(instance_type='ml.m4.xlarge',
                                       initial_instance_count=1)

If using a pretrained model, create an ``MXNetModel`` object, and then call ``deploy`` to create the Amazon SageMaker Endpoint:

.. code:: python

    mxnet_model = MXNetModel(model_data='s3://my_bucket/pretrained_model/model.tar.gz',
                             role=role,
                             entry_point='inference.py',
                             framework_version='1.6.0',
                             py_version='py3')
    predictor = mxnet_model.deploy(instance_type='ml.m4.xlarge',
                                   initial_instance_count=1)

In both cases, ``deploy`` returns a ``Predictor`` object, which you can use to do inference on the endpoint hosting your MXNet model.

Each ``Predictor`` provides a ``predict`` method, which can do inference with numpy arrays or Python lists.
Inference arrays or lists are serialized and sent to the MXNet model server by an ``InvokeEndpoint`` SageMaker operation.
``predict`` returns the result of inference against your model.
By default, the inference result is either a Python list or dictionary.

Elastic Inference
=================

MXNet on Amazon SageMaker has support for `Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`_, which allows for inference acceleration to a hosted endpoint for a fraction of the cost of using a full GPU instance.
In order to attach an Elastic Inference accelerator to your endpoint provide the accelerator type to ``accelerator_type`` to your ``deploy`` call.

.. code:: python

  predictor = mxnet_estimator.deploy(instance_type='ml.m4.xlarge',
                                     initial_instance_count=1,
                                     accelerator_type='ml.eia1.medium')

Model Directory Structure
=========================

In general, if you use the same version of MXNet for both training and inference with the SageMaker Python SDK,
the SDK should take care of ensuring that the contents of your ``model.tar.gz`` file are organized correctly.

For versions 1.4 and higher
---------------------------

For MXNet versions 1.4 and higher, the contents of ``model.tar.gz`` should be organized as follows:

- Model files in the top-level directory
- Inference script (and any other source files) in a directory named ``code/`` (for more about the inference script, see `The SageMaker MXNet Model Server <#the-sagemaker-mxnet-model-server>`_)
- Optional requirements file located at ``code/requirements.txt`` (for more about requirements files, see `Use third-party libraries <#use-third-party-libraries>`_)

For example:

.. code::

  model.tar.gz/
  |- model-symbol.json
  |- model-shapes.json
  |- model-0000.params
  |- code/
    |- inference.py
    |- requirements.txt  # only for versions 1.6.0 and higher

In this example, ``model-symbol.json``, ``model-shapes.json``, and ``model-0000.params`` are the model files saved from training,
``inference.py`` is the inference script, and ``requirements.txt`` is a requirements file.

The ``MXNet`` and ``MXNetModel`` classes repack ``model.tar.gz`` to include the inference script (and related files),
as long as the ``framework_version`` is set to 1.4 or higher.

For versions 1.3 and lower
--------------------------

For MXNet versions 1.3 and lower, ``model.tar.gz`` should contain only the model files,
while your inference script and optional requirements file are packed in a separate tarball, named ``sourcedir.tar.gz`` by default.

For example:

.. code::

  model.tar.gz/
  |- model-symbol.json
  |- model-shapes.json
  |- model-0000.params

  sourcedir.tar.gz/
  |- script.py
  |- requirements.txt  # only for versions 0.12.1-1.3.0

In this example, ``model-symbol.json``, ``model-shapes.json``, and ``model-0000.params`` are the model files saved from training,
``script.py`` is the inference script, and ``requirements.txt`` is a requirements file.

The SageMaker MXNet Model Server
================================

The MXNet endpoint you create with ``deploy`` runs a SageMaker MXNet model server.
The model server loads the model provided and performs inference on the model in response to SageMaker ``InvokeEndpoint`` API calls.

You can configure two components of the model server: model loading and model serving.
Model loading is the process of deserializing your saved model back into an MXNet model.
Serving is the process of translating ``InvokeEndpoint`` requests to inference calls on the loaded model.
These are configured by defining functions in the Python source file you pass to the ``MXNet`` or ``MXNetModel`` constructor.

Load a Model
------------

Before a model can be served, it must be loaded.
The model server loads your model by invoking the ``model_fn`` function in your inference script.
If you don't provide a ``model_fn`` function, the model server uses a default ``model_fn`` function.
The default function works with MXNet Module model objects saved via the default ``save`` function.

If you wrote your own save logic, then you may need to write a custom ``model_fn`` function.
The ``model_fn`` function must have the following signature:

.. code:: python

    def model_fn(model_dir)

Amazon SageMaker injects the directory where your model files and sub-directories have been mounted.
Your model function should return a model object that can be used for model serving.

The following code snippet shows an example custom ``model_fn`` implementation.
This returns an MXNet Gluon net model for resnet-34 inference.
It loads the model parameters from a ``model.params`` file in the SageMaker model directory.

.. code:: python

    def model_fn(model_dir):
        """Load the Gluon model. Called when the hosting service starts.

        Args:
            model_dir (str): The directory where model files are stored.

        Returns:
            mxnet.gluon.nn.Block: a Gluon network (for this example)
        """
        net = models.get_model('resnet34_v2', ctx=mx.cpu(), pretrained=False, classes=10)
        net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
        return net

MXNet on Amazon SageMaker has support for `Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`__, which allows for inference acceleration to a hosted endpoint for a fraction of the cost of using a full GPU instance.
In order to load and serve your MXNet model through Amazon Elastic Inference, import the ``eimx`` Python package and make one change in the code to partition your model and optimize it for the ``EIA`` back end, as shown `here <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-mxnet-elastic-inference.html#ei-mxnet>`__.

Based on the example above, the following code-snippet shows an example custom ``model_fn`` implementation, which enables loading and serving our MXNet model through Amazon Elastic Inference.

.. code:: python

    def model_fn(model_dir):
        """Load the Gluon model. Called when the hosting service starts.

        Args:
            model_dir (str): The directory where model files are stored.

        Returns:
            mxnet.gluon.nn.Block: a Gluon network (for this example)
        """
        net = models.get_model('resnet34_v2', ctx=mx.cpu(), pretrained=False, classes=10)
        net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
        net.hybridize(backend='EIA', static_alloc=True, static_shape=True)
        return net

If you are using MXNet 1.5.1 and earlier, the `default_model_fn <https://github.com/aws/sagemaker-mxnet-container/pull/55/files#diff-aabf018d906ed282a3c738377d19a8deR71>`__ loads and serve your model through Elastic Inference, if applicable, within the Amazon SageMaker MXNet containers.

For more information on how to enable MXNet to interact with Amazon Elastic Inference, see `Use Elastic Inference with MXNet <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-mxnet-elastic-inference.html>`__.

Serve an MXNet Model
--------------------

After the MXNet model server loads your model by calling either the default ``model_fn`` or the implementation in your script, it serves your model.
Model serving is the process of responding to inference requests received by SageMaker ``InvokeEndpoint`` API calls.
Defining how to handle these requests can be done in one of two ways:

- using ``input_fn``, ``predict_fn``, and ``output_fn``, some of which may be your own implementations
- writing your own ``transform_fn`` for handling input processing, prediction, and output processing

Use ``input_fn``, ``predict_fn``, and ``output_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SageMaker MXNet model server breaks request handling into three steps:

-  input processing
-  prediction
-  output processing

Just like with ``model_fn``, you configure these steps by defining functions in your Python source file.

Each step has its own Python function, which takes in information about the request and the return value from the previous function in the chain.
Inside the MXNet model server, the process looks like:

.. code:: python

    # Deserialize the Invoke request body into an object we can perform prediction on
    input_object = input_fn(request_body, request_content_type)

    # Perform prediction on the deserialized object, with the loaded model
    prediction = predict_fn(input_object, model)

    # Serialize the prediction result into the desired response content type
    ouput = output_fn(prediction, response_content_type)

The above code sample shows the three function definitions that correlate to the three steps mentioned above:

-  ``input_fn``: Takes request data and deserializes the data into an
   object for prediction.
-  ``predict_fn``: Takes the deserialized request object and performs
   inference against the loaded model.
-  ``output_fn``: Takes the result of prediction and serializes this
   according to the response content type.

The MXNet model server provides default implementations of these functions.
These work with both Gluon API and Module API model objects.
The following content types are supported:

- Gluon API: 'application/json', 'application/x-npy'
- Module API: 'application/json', 'application/x-npy', 'text-csv'

You can also provide your own implementations for these functions in your training script.
If you omit any definition, the MXNet model server uses its default implementation for that function.

If you rely solely on the SageMaker MXNet model server defaults, you get the following functionality:

-  Prediction on MXNet Gluon API ``net`` and Module API ``Module`` objects.
-  Deserialization from CSV and JSON to NDArrayIters.
-  Serialization of NDArrayIters to CSV or JSON.

In the following sections, we describe the default implementations of ``input_fn``, ``predict_fn``, and ``output_fn``.
We describe the input arguments and expected return types of each, so you can define your own implementations.

Process Model Input
~~~~~~~~~~~~~~~~~~~

When an ``InvokeEndpoint`` operation is made against an endpoint running an MXNet model server, the model server receives two pieces of information:

-  The request's content type, e.g. 'application/json'
-  The request data body as a byte array

The MXNet model server invokes ``input_fn``, passing in this information.
If you define an ``input_fn`` function definition, it should return an object that can be passed to ``predict_fn`` and have the following signature:

.. code:: python

    def input_fn(request_body, request_content_type)

Where ``request_body`` is a byte buffer and ``request_content_type`` is the content type of the request.

The MXNet model server provides a default implementation of ``input_fn``. This function deserializes JSON or CSV encoded data into an MXNet ``NDArrayIter`` `(external API docs) <https://mxnet.incubator.apache.org/api/python/io.html#mxnet.io.NDArrayIter>`__ multi-dimensional array iterator. This works with the default ``predict_fn`` implementation, which expects an ``NDArrayIter`` as input.

Default JSON deserialization requires ``request_body`` contain a single JSON list. Sending multiple JSON objects within the same ``request_body`` is not supported. The list must have a dimensionality compatible with the MXNet ``net`` or ``Module`` object. Specifically, after the list is loaded, it's either padded or split to fit the first dimension of the model input shape. The list's shape must be identical to the model's input shape, for all dimensions after the first.

Default CSV deserialization requires ``request_body`` contain one or more lines of CSV numerical data. The data is loaded into a two-dimensional array, where each line break defines the boundaries of the first dimension. This two-dimensional array is then re-shaped to be compatible with the shape expected by the model object. Specifically, the first dimension is kept unchanged, but the second dimension is reshaped to be consistent with the shape of all dimensions in the model, following the first dimension.

If you provide your own implementation of input_fn, you should abide by the ``input_fn`` signature. If you want to use this with the default
``predict_fn``, then you should return an ``NDArrayIter``. The ``NDArrayIter`` should have a shape identical to the shape of the model being predicted on. The example below shows a custom ``input_fn`` for preparing pickled numpy arrays.

.. code:: python

    import numpy as np
    import mxnet as mx

    def input_fn(request_body, request_content_type):
        """An input_fn that loads a pickled numpy array"""
        if request_content_type == 'application/python-pickle':
            array = np.load(StringIO(request_body))
            array.reshape(model.data_shapes[0])
            return mx.io.NDArrayIter(mx.ndarray(array))
        else:
            # Handle other content-types here or raise an Exception
            # if the content type is not supported.
            pass

Predict from a Deployed Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the inference request has been deserialized by ``input_fn``, the MXNet model server invokes ``predict_fn``.
As with the other functions, you can define your own ``predict_fn`` or use the model server's default.

The ``predict_fn`` function has the following signature:

.. code:: python

    def predict_fn(input_object, model)

Where ``input_object`` is the object returned from ``input_fn`` and
``model`` is the model loaded by ``model_fn``.

The default implementation of ``predict_fn`` requires ``input_object`` be an ``NDArrayIter``, which is the return-type of the default
``input_fn``. It also requires that ``model`` be either an MXNet Gluon API ``net`` object or a Module API ``Module`` object.

The default implementation performs inference with the input
``NDArrayIter`` on the Gluon or Module object. If the model is a Gluon
``net`` it performs: ``net.forward(input_object)``. If the model is a Module object it performs ``module.predict(input_object)``. In both cases, it returns the result of that call.

If you implement your own prediction function, you should take care to ensure that:

-  The first argument is expected to be the return value from input_fn.
   If you use the default input_fn, this is an ``NDArrayIter``.
-  The second argument is the loaded model. If you use the default
   ``model_fn`` implementation, this is an MXNet Module object.
   Otherwise, it is the return value of your ``model_fn`` implementation.
-  The return value should be of the correct type to be passed as the
   first argument to ``output_fn``. If you use the default
   ``output_fn``, this should be an ``NDArrayIter``.

Process Model Output
~~~~~~~~~~~~~~~~~~~~

After invoking ``predict_fn``, the model server invokes ``output_fn``, passing in the return value from ``predict_fn`` and the ``InvokeEndpoint`` requested response content type.

The ``output_fn`` has the following signature:

.. code:: python

    def output_fn(prediction, content_type)

Where ``prediction`` is the result of invoking ``predict_fn`` and ``content_type`` is the requested response content type for ``InvokeEndpoint``.
The function should return an array of bytes serialized to the expected content type.

The default implementation expects ``prediction`` to be an ``NDArray`` and can serialize the result to either JSON or CSV. It accepts response content types of "application/json" and "text/csv".

Use ``transform_fn``
^^^^^^^^^^^^^^^^^^^^

If you would rather not structure your code around the three methods described above, you can instead define your own ``transform_fn`` to handle inference requests.
An error is thrown if a ``transform_fn`` is present in conjunction with any ``input_fn``, ``predict_fn``, and/or ``output_fn``.
``transform_fn`` has the following signature:

.. code:: python

    def transform_fn(model, request_body, content_type, accept_type)

Where ``model`` is the model objected loaded by ``model_fn``, ``request_body`` is the data from the inference request, ``content_type`` is the content type of the request, and ``accept_type`` is the request content type for the response.

This one function should handle processing the input, performing a prediction, and processing the output.
The return object should be one of the following:

For versions 1.4 and higher:

- a tuple with two items: the response data and ``accept_type`` (the content type of the response data), or
- the response data: (the content type of the response is set to either the accept header in the initial request or default to "application/json")

For versions 1.3 and lower:

- a tuple with two items: the response data and ``accept_type`` (the content type of the response data), or
- a Flask response object: http://flask.pocoo.org/docs/1.0/api/#response-objects

For an example inference script using this structure, see the `mxnet_gluon_sentiment <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_sentiment/sentiment.py#L344-L387>`__ notebook.

***********************************************
Work with Existing Model Data and Training Jobs
***********************************************

Attach to Existing Training Jobs
================================

You can attach an MXNet Estimator to an existing training job using the
``attach`` method.

.. code:: python

    my_training_job_name = 'MyAwesomeMXNetTrainingJob'
    mxnet_estimator = MXNet.attach(my_training_job_name)

After attaching, if the training job's status is "Complete", it can be ``deploy``\ ed to create an Amazon SageMaker Endpoint and return a ``Predictor``.
If the training job is in progress, ``attach`` blocks and displays log messages from the training job until the training job completes.

For information about arguments that ``attach`` accepts, see `the function's API reference <https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Framework.attach>`_.

Deploy Endpoints from Model Data
================================

As well as attaching to existing training jobs, you can deploy models directly from model data in Amazon S3. The following code sample shows how to do this, using the ``MXNetModel`` class.

.. code:: python

    mxnet_model = MXNetModel(model_data='s3://bucket/model.tar.gz', role='SageMakerRole', entry_point='trasform_script.py')

    predictor = mxnet_model.deploy(instance_type='ml.c4.xlarge', initial_instance_count=1)

For information about arguments that the ``MXNetModel`` constructor accepts, see `the class's API reference <https://sagemaker.readthedocs.io/en/stable/sagemaker.mxnet.html#sagemaker.mxnet.model.MXNetModel>`_.

Your model data must be a .tar.gz file in Amazon S3. Amazon SageMaker Training Job model data is saved to .tar.gz files in Amazon S3, however if you have local data you want to deploy, you can prepare the data yourself.

Assuming you have a local directory containing your model data named "my_model" you can tar and gzip compress the file and upload to Amazon S3 using the following commands:

::

    tar -czf model.tar.gz my_model
    aws s3 cp model.tar.gz s3://my-bucket/my-path/model.tar.gz

This uploads the contents of my_model to a gzip-compressed tar file to Amazon S3 in the bucket "my-bucket", with the key "my-path/model.tar.gz".

To run this command, you need the AWS CLI tool installed. Please refer to our `FAQ <#FAQ>`__ for more information on installing this.

********
Examples
********

Amazon provides several example Jupyter notebooks that demonstrate end-to-end training on Amazon SageMaker using MXNet. Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk

These are also available in Amazon SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.

***********************
SageMaker MXNet Classes
***********************

For information about the different MXNet-related classes in the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/sagemaker.mxnet.html.

**************************
SageMaker MXNet Containers
**************************

For information about the SageMaker MXNet containers, see:

- `SageMaker MXNet training toolkit <https://github.com/aws/sagemaker-mxnet-container>`_
- `SageMaker MXNet serving toolkit <https://github.com/aws/sagemaker-mxnet-serving-container>`_
- `Deep Learning Container (DLC) Dockerfiles for MXNet <https://github.com/aws/deep-learning-containers/tree/master/mxnet>`_
- `Deep Learning Container (DLC) Images <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`_ and `release notes <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html>`_
