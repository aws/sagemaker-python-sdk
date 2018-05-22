.. image:: branding/icon/sagemaker-banner.png
    :height: 100px
    :alt: SageMaker

====================
SageMaker Python SDK
====================

.. image:: https://travis-ci.org/aws/sagemaker-python-sdk.svg?branch=master
   :target: https://travis-ci.org/aws/sagemaker-python-sdk
   :alt: Build Status

.. image:: https://codecov.io/gh/aws/sagemaker-python-sdk/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/aws/sagemaker-python-sdk
   :alt: CodeCov

SageMaker Python SDK is an open source library for training and deploying machine learning models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks: **Apache MXNet** and **TensorFlow**. You can also train and deploy models with **Amazon algorithms**, these are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training. If you have **your own algorithms** built into SageMaker compatible Docker containers, you can train and host models using these as well.

For detailed API reference please go to: `Read the Docs <https://readthedocs.org/projects/sagemaker/>`_

Table of Contents
-----------------

1. `Getting SageMaker Python SDK <#getting-sagemaker-python-sdk>`__
2. `SageMaker Python SDK Overview <#sagemaker-python-sdk-overview>`__
3. `MXNet SageMaker Estimators <#mxnet-sagemaker-estimators>`__
4. `TensorFlow SageMaker Estimators <#tensorflow-sagemaker-estimators>`__
5. `AWS SageMaker Estimators <#aws-sagemaker-estimators>`__
6. `BYO Docker Containers with SageMaker Estimators <#byo-docker-containers-with-sagemaker-estimators>`__
7. `BYO Model <#byo-model>`__


Getting SageMaker Python SDK
----------------------------

SageMaker Python SDK is built to PyPI and can be installed with pip.

::

    pip install sagemaker

You can install from source by cloning this repository and issuing a pip install command in the root directory of the repository.

::

    git clone https://github.com/aws/sagemaker-python-sdk.git
    python setup.py sdist
    pip install dist/sagemaker-1.2.4.tar.gz

Supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Python SDK is tested on: \* Python 2.7 \* Python 3.5

Licensing
~~~~~~~~~
SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

Running tests
~~~~~~~~~~~~~

SageMaker Python SDK uses tox for running Python tests. You can run the tests by running tox:

::

    tox

Tests are defined in ``tests/`` and includes unit and integ tests. If you just want to run unit tests, then you can issue:

::

    tox tests/unit

To just run integ tests, issue the following command:

::

    pytest tests/integ

You can also filter by individual test function names (usable with any of the previous commands):

::

    pytest -k 'test_i_care_about'

Building Sphinx docs
~~~~~~~~~~~~~~~~~~~~

``cd`` into the ``doc`` directory and run:

::

    make html

You can edit the templates for any of the pages in the docs by editing the .rst files in the "doc" directory and then running "``make html``" again.


SageMaker Python SDK Overview
-----------------------------

SageMaker Python SDK provides several high-level abstractions for working with Amazon SageMaker. These are:

- **Estimators**: Encapsulate training on SageMaker. Can be ``fit()`` to run training, then the resulting model ``deploy()`` ed to a SageMaker Endpoint.
- **Models**: Encapsulate built ML models. Can be ``deploy()`` ed to a SageMaker Endpoint.
- **Predictors**: Provide real-time inference and transformation using Python data-types against a SageMaker Endpoint.
- **Session**: Provides a collection of convenience methods for working with SageMaker resources.

Estimator and Model implementations for MXNet, TensorFlow, and Amazon ML algorithms are included. There's also an Estimator that runs SageMaker compatible custom Docker containers, allowing you to run your own ML algorithms via SageMaker Python SDK.

Later sections of this document explain how to use the different Estimators and Models. These are:

* `MXNet SageMaker Estimators and Models <#mxnet-sagemaker-estimators>`__
* `TensorFlow SageMaker Estimators and Models <#tensorflow-sagemaker-estimators>`__
* `AWS SageMaker Estimators and Models <#aws-sagemaker-estimators>`__
* `Custom SageMaker Estimators and Models <#byo-docker-containers-with-sagemaker-estimators>`__


Estimator Usage
---------------

Here is an end to end example of how to use a SageMaker Estimator.

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator (no training happens yet)
    mxnet_estimator = MXNet('train.py',
                            train_instance_type='ml.p2.xlarge',
                            train_instance_count = 1)

    # Starts a SageMaker training job and waits until completion.
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploys the model that was generated by fit() to a SageMaker Endpoint
    mxnet_predictor = mxnet_estimator.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge')

    # Serializes data and makes a prediction request to the SageMaker Endpoint
    response = predictor.predict(data)

    # Tears down the SageMaker Endpoint
    mxnet_estimator.delete_endpoint()

Local Mode
~~~~~~~~~~

The SageMaker Python SDK now supports local mode, which allows you to create TensorFlow, MXNet and BYO estimators and
deploy to your local environment. This is a great way to test your deep learning script before running in
SageMaker's managed training or hosting environments.

We can take the example in  `Estimator Usage <#estimator-usage>`__ , and use either ``local`` or ``local_gpu`` as the
instance type.

.. code:: python

    from sagemaker.mxnet import MXNet

    # Configure an MXNet Estimator (no training happens yet)
    mxnet_estimator = MXNet('train.py',
                            train_instance_type='local',
                            train_instance_count=1)

    # In Local Mode, fit will pull the MXNet container docker image and run it locally
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Alternatively, you can train using data in your local file system. This is only supported in Local mode.
    mxnet_estimator.fit('file:///tmp/my_training_data')

    # Deploys the model that was generated by fit() to local endpoint in a container
    mxnet_predictor = mxnet_estimator.deploy(initial_instance_count=1, instance_type='local')

    # Serializes data and makes a prediction request to the local endpoint
    response = predictor.predict(data)

    # Tears down the endpoint container
    mxnet_estimator.delete_endpoint()


For detailed examples of running docker in local mode, see:

- `TensorFlow local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb>`__.
- `MXNet local mode example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_mnist/mnist_with_gluon_local_mode.ipynb>`__.

A few important notes:

- Only one local mode endpoint can be running at a time
- If you are using s3 data as input, it will be pulled from S3 to your local environment, please ensure you have sufficient space.
- If you run into problems, this is often due to different docker containers conflicting. Killing these containers and re-running often solves your problems.
- Local Mode requires docker-compose and `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__ for ``local_gpu``.
- Distributed training is not yet supported for ``local_gpu``.


MXNet SageMaker Estimators
--------------------------

With MXNet Estimators, you can train and host MXNet models on Amazon SageMaker.

Supported versions of MXNet: ``1.0.0``, ``0.12.1``.

Training with MXNet
~~~~~~~~~~~~~~~~~~~

Training MXNet models using ``MXNet`` Estimators is a two-step process. First, you prepare your training script, then second, you run this on SageMaker via an ``MXNet`` Estimator. You should prepare your script in a separate source file than the notebook, terminal session, or source file you're using to submit the script to SageMaker via an ``MXNet`` Estimator.

Suppose that you already have an MXNet training script called
``mxnet-train.py``. You can run this script in SageMaker as follows:

.. code:: python

    from sagemaker.mxnet import MXNet
    mxnet_estimator = MXNet("mxnet-train.py", role="SageMakerRole", train_instance_type="ml.p2.xlarge", )
    mxnet_estimator.fit("s3://bucket/path/to/training/data")

Where the s3 url is a path to your training data, within Amazon S3. The constructor keyword arguments define how SageMaker runs your training script and are discussed, in detail, in a later section.

In the following sections, we'll discuss how to prepare a training script for execution on SageMaker, then how to run that script on SageMaker using an ``MXNet`` Estimator.

Preparing the MXNet training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your MXNet training script must be a Python 2.7 or 3.5 compatible source file. The MXNet training script must contain a function ``train``, which SageMaker invokes to run training. You can include other functions as well, but it must contain a ``train`` function.

When you run your script on SageMaker via the ``MXNet`` Estimator, SageMaker injects information about the training environment into your training function via Python keyword arguments. You can choose to take advantage of these by including them as keyword arguments in your train function. The full list of arguments is:

-  ``hyperparameters (dict[string,string])``: The hyperparameters passed
   to SageMaker TrainingJob that runs your MXNet training script. You
   can use this to pass hyperparameters to your training script.
-  ``input_data_config (dict[string,dict])``: The SageMaker TrainingJob
   InputDataConfig object, that's set when the SageMaker TrainingJob is
   created. This is discussed in more detail below.
-  ``channel_input_dirs (dict[string,string])``: A collection of
   directories containing training data. When you run training, you can
   partition your training data into different logical "channels".
   Depending on your problem, some common channel ideas are: "train",
   "test", "evaluation" or "images',"labels".
-  ``output_data_dir (str)``: A directory where your training script can
   write data that will be moved to s3 after training is complete.
-  ``num_gpus (int)``: The number of GPU devices available on your
   training instance.
-  ``num_cpus (int)``: The number of CPU devices available on your training instance.
-  ``hosts (list[str])``: The list of host names running in the
   SageMaker Training Job cluster.
-  ``current_host (str)``: The name of the host executing the script.
   When you use SageMaker for MXNet training, the script is run on each
   host in the cluster.

A training script that takes advantage of all arguments would have the following definition:

.. code:: python

    def train(hyperparameters, input_data_config, channel_input_dirs, output_data_dir,
              num_gpus, num_cpus, hosts, current_host):
        pass

You don't have to use all the arguments, arguments you don't care about can be ignored by including ``**kwargs``.

.. code:: python

    # Only work with hyperparameters and num_gpus, ignore all other hyperparameters
    def train(hyperparameters, num_gpus, **kwargs):
        pass

**Note: Writing a training script that imports correctly**
When SageMaker runs your training script, it imports it as a Python module and then invokes ``train`` on the imported module. Consequently, you should not include any statements that won't execute successfully in SageMaker when your module is imported. For example, don't attempt to open any local files in top-level statements in your training script.

If you want to run your training script locally via the Python interpreter, look at using a ``___name__ == '__main__'`` guard, discussed in more detail here: https://stackoverflow.com/questions/419163/what-does-if-name-main-do .

Using MXNet and numpy
^^^^^^^^^^^^^^^^^^^^^

You can import both ``mxnet`` and ``numpy`` in your training script. When your script runs in SageMaker, it will run with access to MXNet version 1.0.0 and numpy version 1.13.3 by default. For more information on the environment your script runs in, please see `SageMaker MXNet Containers <#sagemaker-mxnet-containers>`__.

Running an MXNet training script in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You run MXNet training scripts on SageMaker by creating ``MXNet`` Estimators. SageMaker training of your script is invoked when you call ``fit`` on an ``MXNet`` Estimator. The following code sample shows how you train a custom MXNet script "train.py".

.. code:: python

    mxnet_estimator = MXNet("train.py",
                            train_instance_type="ml.p2.xlarge",
                            train_instance_count=1)
    mxnet_estimator.fit("s3://my_bucket/my_training_data/")

MXNet Estimators
^^^^^^^^^^^^^^^^

The ``MXNet`` constructor takes both required and optional arguments.

Required arguments
''''''''''''''''''

The following are required arguments to the ``MXNet`` constructor. When you create an MXNet object, you must include these in the constructor, either positionally or as keyword arguments.

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
   example, 'ml.c4.xlarge'.

Optional arguments
''''''''''''''''''

The following are optional arguments. When you create an ``MXNet`` object, you can specify these as keyword arguments.

-  ``source_dir`` Path (absolute or relative) to a directory with any
   other training source code dependencies aside from the entry point
   file. Structure within this directory will be preserved when training
   on SageMaker.
-  ``hyperparameters`` Hyperparameters that will be used for training.
   Will be made accessible as a dict[str, str] to the training code on
   SageMaker. For convenience, accepts other types besides str, but
   str() will be called on keys and values to convert them before
   training.
-  ``py_version`` Python version you want to use for executing your
   model training code.
-  ``train_volume_size`` Size in GB of the EBS volume to use for storing
   input data during training. Must be large enough to store training
   data if input_mode='File' is used (which is the default).
-  ``train_max_run`` Timeout in hours for training, after which Amazon
   SageMaker terminates the job regardless of its current status.
-  ``input_mode`` The input mode that the algorithm supports. Valid
   modes: 'File' - Amazon SageMaker copies the training dataset from the
   s3 location to a directory in the Docker container. 'Pipe' - Amazon
   SageMaker streams data directly from s3 to the container via a Unix
   named pipe.
-  ``output_path`` s3 location where you want the training result (model
   artifacts and optional output files) saved. If not specified, results
   are stored to a default bucket. If the bucket with the specific name
   does not exist, the estimator creates the bucket during the fit()
   method execution.
-  ``output_kms_key`` Optional KMS key ID to optionally encrypt training
   output with.
-  ``job_name`` Name to assign for the training job that the fit()
   method launches. If not specified, the estimator generates a default
   job name, based on the training image name and current timestamp

Calling fit
^^^^^^^^^^^

You start your training script by calling ``fit`` on an ``MXNet`` Estimator. ``fit`` takes both required and optional arguments.

Required argument
'''''''''''''''''

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

Optional arguments
''''''''''''''''''

-  ``wait``: Defaults to True, whether to block and wait for the
   training script to complete before returning.
-  ``logs``: Defaults to True, whether to show logs produced by training
   job in the Python session. Only meaningful when wait is True.

Saving models
~~~~~~~~~~~~~

When we run MXNet training, we often want to save or manipulate the models that MXNet produces. SageMaker Estimators provides several ways to save MXNet models. The method used is driven by functions you define on your training script, run via the ``MXNet`` Estimator in SageMaker in response to ``fit``.

Just as you enable training by defining a ``train`` function in your training script, you enable model saving by defining a ``save`` function in your script. If your script includes a ``save`` function, SageMaker will invoke it with the return-value of ``train``. Model saving is a two-step process, firstly you return the model you want to save from
``train``, then you define your model-serialization logic in ``save``.

SageMaker provides a default implementation of ``save`` that works with MXNet Module API ``Module`` objects. If your training script does not define a ``save`` function, then the default ``save`` function will be invoked on the return-value of your ``train`` function.

The following script demonstrates how to return a model from train, that's compatible with the default ``save`` function.

.. code:: python

    import mxnet as mx

    def create_graph():
        # Code to create graph omitted for brevity

    def train(num_gpus, channel_input_dirs, **kwargs):
        ctx = mx.cpu() if not num_gpus else [mx.gpu(i) for i in range(num_gpus)]
        sym = create_graph()
        mod = mx.mod.Module(symbol=sym, context=ctx)

        # Code to fit mod omitted for brevity
        # ...

        # Return the Module object. SageMaker will save this.
        return mod

If you define your own ``save`` function, it should have the following signature:

.. code:: python

    def save(model, model_dir)

Where ``model`` is the return-value from ``train`` and ``model_dir`` is the directory SageMaker requires you to save your model. If you write files into ``model_dir`` then they will be persisted to s3 after the SageMaker Training Job completes.

After your training job is complete, your model data will available in the s3 ``output_path`` you specified when you created the MXNet Estimator. Handling of s3 output is discussed in: `Accessing SageMaker output and model data in s3 <#accessing%20-sagemaker-output-and-model-data-in-s3>`__.

MXNet Module serialization in SageMaker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you train function returns a ``Module`` object, it will be serialized by the default Module serialization system, unless you've specified a custom ``save`` function.

The default serialization system generates three files:

-  ``model-shapes.json``: A json list, containing a serialization of the
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
-  ``modle.params``: The MXNet ``Module`` parameters. Produced by
   invoking ``save_params`` on the ``Module`` being saved.

Writing a custom save function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can provide your own save function. This is useful if you are not working with the ``Module`` API or you need special processing.

To provide your own save function, define a ``save`` function in your training script. The function should take two arguments:

-  model: This is the object that was returned from your ``train``
   function. If your ``train`` function does not return an object, it
   will be ``None``. You are free to return an object of any type from
   ``train``, you do not have to return ``Module`` or ``Gluon`` API
   specific objects.
-  model_dir: This is the string path on the SageMaker training host
   where you save your model. Files created in this directory will be
   accessible in S3 after your SageMaker Training Job completes.

After your ``train`` function completes, SageMaker will invoke ``save`` with the object returned from ``train``.

**Note: How to save Gluon models with SageMaker**

If your train function returns a Gluon API ``net`` object as its model, you'll need to write your own ``save`` function. You will want to serialize the ``net`` parameters. Saving ``net`` parameters is covered in the `Serialization section <http://gluon.mxnet.io/chapter03_deep-neural-networks/serialization.html>`__ of the collaborative Gluon deep-learning book `"The Straight Dope" <http://gluon.mxnet.io/index.html>`__.

Deploying MXNet models
~~~~~~~~~~~~~~~~~~~~~~

After an MXNet Estimator has been fit, you can host the newly created model in SageMaker.

After calling ``fit``, you can call ``deploy`` on an ``MXNet`` Estimator to create a SageMaker Endpoint. The Endpoint runs a SageMaker-provided MXNet model server and hosts the model produced by your training script, which was run when you called ``fit``. This was the model object you returned from ``train`` and saved with either a custom save function or the default save function.

``deploy`` returns a ``Predictor`` object, which you can use to do inference on the Endpoint hosting your MXNet model. Each ``Predictor`` provides a ``predict`` method which can do inference with numpy arrays or Python lists. Inference arrays or lists are serialized and sent to the MXNet model server by an ``InvokeEndpoint`` SageMaker operation.

``predict`` returns the result of inference against your model. By default, the inference result is either a Python list or dictionary.

.. code:: python

    # Train my estimator
    mxnet_estimator = MXNet("train.py",
                            train_instance_type="ml.p2.xlarge",
                            train_instance_count=1)
    mxnet_estimator.fit("s3://my_bucket/my_training_data/")

    # Deploy my estimator to a SageMaker Endpoint and get a Predictor
    predictor = mxnet_estimator.deploy(deploy_instance_type="ml.p2.xlarge",
                                       min_instances=1,

You use the SageMaker MXNet model server to host your MXNet model when you call ``deploy`` on an ``MXNet`` Estimator. The model server runs inside a SageMaker Endpoint, which your call to ``deploy`` creates. You can access the name of the Endpoint by the ``name`` property on the returned ``Predictor``.

The SageMaker MXNet Model Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MXNet Endpoint you create with ``deploy`` runs a SageMaker MXNet model server. The model server loads the model that was saved by your training script and performs inference on the model in response to SageMaker InvokeEndpoint API calls.

You can configure two components of the SageMaker MXNet model server: Model loading and model serving. Model loading is the process of deserializing your saved model back into an MXNet model. Serving is the process of translating InvokeEndpoint requests to inference calls on the loaded model.

As with MXNet training, you configure the MXNet model server by defining functions in the Python source file you passed to the MXNet constructor.

Model loading
^^^^^^^^^^^^^

Before a model can be served, it must be loaded. The SageMaker model server loads your model by invoking a ``model_fn`` function on your training script. If you don't provide a ``model_fn`` function, SageMaker will use a default ``model_fn`` function. The default function works with MXNet Module model objects, saved via the default ``save`` function.

If you wrote a custom ``save`` function then you may need to write a custom ``model_fn`` function. If your save function serializes ``Module`` objects under the same format as the default ``save`` function, then you won't need to write a custom model_fn function. If you do write a ``model_fn`` function must have the following signature:

.. code:: python

    def model_fn(model_dir)

SageMaker will inject the directory where your model files and sub-directories, saved by ``save``, have been mounted. Your model function should return a model object that can be used for model serving. SageMaker provides automated serving functions that work with Gluon API ``net`` objects and Module API ``Module`` objects. If you return either of these types of objects, then you will be able to use the default serving request handling functions.

The following code-snippet shows an example custom ``model_fn`` implementation. This loads returns an MXNet Gluon net model for resnet-34 inference. It loads the model parameters from a ``model.params`` file in the SageMaker model directory.

.. code:: python

    def model_fn(model_dir):
        """
        Load the gluon model. Called once when hosting service starts.
        :param: model_dir The directory where model files are stored.
        :return: a model (in this case a Gluon network)
        """
        net = models.get_model('resnet34_v2', ctx=mx.cpu(), pretrained=False, classes=10)
        net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
        return net

Model serving
^^^^^^^^^^^^^

After the SageMaker model server has loaded your model, by calling either the default ``model_fn`` or the implementation in your training script, SageMaker will serve your model. Model serving is the process of responding to inference requests, received by SageMaker InvokeEndpoint API calls. The SageMaker MXNet model server breaks request handling into three steps:


-  input processing,
-  prediction, and
-  output processing.

In a similar way to previous steps, you configure these steps by defining functions in your Python source file.

Each step involves invoking a python function, with information about the request and the return-value from the previous function in the chain. Inside the SageMaker MXNet model server, the process looks like:

.. code:: python

    # Deserialize the Invoke request body into an object we can perform prediction on
    input_object = input_fn(request_body, request_content_type, model)

    # Perform prediction on the deserialized object, with the loaded model
    prediction = predict_fn(input_object, model)

    # Serialize the prediction result into the desired response content type
    ouput = output_fn(prediction, response_content_type)

The above code-sample shows the three function definitions:

-  ``input_fn``: Takes request data and deserializes the data into an
   object for prediction.
-  ``predict_fn``: Takes the deserialized request object and performs
   inference against the loaded model.
-  ``output_fn``: Takes the result of prediction and serializes this
   according to the response content type.

The SageMaker MXNet model server provides default implementations of these functions. These work with common-content types, and Gluon API and Module API model objects. You can provide your own implementations for these functions in your training script. If you omit any definition then the SageMaker MXNet model server will use its default implementation for that function.

If you rely solely on the SageMaker MXNet model server defaults, you get the following functionality:

-  Prediction on MXNet Gluon API ``net`` and Module API ``Module``
   objects.
-  Deserialization from CSV and JSON to NDArrayIters.
-  Serialization of NDArrayIters to CSV or JSON.

In the following sections we describe the default implementations of input_fn, predict_fn, and output_fn. We describe the input arguments and expected return types of each, so you can define your own implementations.

Input processing
''''''''''''''''

When an InvokeEndpoint operation is made against an Endpoint running a SageMaker MXNet model server, the model server receives two pieces of information:

-  The request Content-Type, for example "application/json"
-  The request data body, a byte array which is at most 5 MB (5 \* 1024
   \* 1024 bytes) in size.

The SageMaker MXNet model server will invoke an "input_fn" function in your training script, passing in this information. If you define an ``input_fn`` function definition, it should return an object that can be passed to ``predict_fn`` and have the following signature:

.. code:: python

    def input_fn(request_body, request_content_type, model)

Where ``request_body`` is a byte buffer, ``request_content_type`` is a Python string, and model is the result of invoking ``model_fn``.

The SageMaker MXNet model server provides a default implementation of ``input_fn``. This function deserializes JSON or CSV encoded data into an MXNet ``NDArrayIter`` `(external API docs) <https://mxnet.incubator.apache.org/api/python/io.html#mxnet.io.NDArrayIter>`__ multi-dimensional array iterator. This works with the default ``predict_fn`` implementation, which expects an ``NDArrayIter`` as input.

Default json deserialization requires ``request_body`` contain a single json list. Sending multiple json objects within the same ``request_body`` is not supported. The list must have a dimensionality compatible with the MXNet ``net`` or ``Module`` object. Specifically, after the list is loaded, it's either padded or split to fit the first dimension of the model input shape. The list's shape must be identical to the model's input shape, for all dimensions after the first.

Default csv deserialization requires ``request_body`` contain one or more lines of CSV numerical data. The data is loaded into a two-dimensional array, where each line break defines the boundaries of the first dimension. This two-dimensional array is then re-shaped to be compatible with the shape expected by the model object. Specifically, the first dimension is kept unchanged, but the second dimension is reshaped to be consistent with the shape of all dimensions in the model, following the first dimension.

If you provide your own implementation of input_fn, you should abide by the ``input_fn`` signature. If you want to use this with the default
``predict_fn``, then you should return an NDArrayIter. The NDArrayIter should have a shape identical to the shape of the model being predicted on. The example below shows a custom ``input_fn`` for preparing pickled numpy arrays.

.. code:: python

    import numpy as np
    import mxnet as mx

    def input_fn(request_body, request_content_type, model):
        """An input_fn that loads a pickled numpy array"""
        if request_content_type == "application/python-pickle":
            array = np.load(StringIO(request_body))
            array.reshape(model.data_shpaes[0])
            return mx.io.NDArrayIter(mx.ndarray(array))
        else:
            # Handle other content-types here or raise an Exception
            # if the content type is not supported.
            pass

Prediction
''''''''''

After the inference request has been deserialized by ``input_fn``, the SageMaker MXNet model server invokes ``predict_fn``. As with ``input_fn``, you can define your own ``predict_fn`` or use the SageMaker Mxnet default.

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
   If you use the default input_fn, this will be an ``NDArrayIter``.
-  The second argument is the loaded model. If you use the default
   ``model_fn`` implementation, this will be an MXNet Module object.
   Otherwise, it will be the return value of your ``model_fn``
   implementation.
-  The return value should be of the correct type to be passed as the
   first argument to ``output_fn``. If you use the default
   ``output_fn``, this should be an ``NDArrayIter``.

Output processing
'''''''''''''''''

After invoking ``predict_fn``, the model server invokes ``output_fn``, passing in the return-value from ``predict_fn`` and the InvokeEndpoint requested response content-type.

The ``output_fn`` has the following signature:

.. code:: python

    def output_fn(prediction, content_type)

Where ``prediction`` is the result of invoking ``predict_fn`` and
``content_type`` is the InvokeEndpoint requested response content-type. The function should return a byte array of data serialized to content_type.

The default implementation expects ``prediction`` to be an ``NDArray`` and can serialize the result to either JSON or CSV. It accepts response content types of "application/json" and "text/csv".

Distributed MXNet training
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run a multi-machine, distributed MXNet training using the MXNet Estimator. By default, MXNet objects will submit single-machine training jobs to SageMaker. If you set ``train_instance_count`` to be greater than one, multi-machine training jobs will be launched when ``fit`` is called. When you run multi-machine training, SageMaker will import your training script and invoke ``train`` on each host in the cluster.

When you develop MXNet distributed learning algorithms, you often want to use an MXNet kvstore to store and share model parameters. To learn more about writing distributed MXNet programs, please see `Distributed Training <http://newdocs.readthedocs.io/en/latest/distributed_training.html>`__ in the MXNet docs.

When using an MXNet Estimator, SageMaker automatically starts MXNet kvstore server and scheduler processes on hosts in your training job cluster. Your script runs as an MXNet worker task. SageMaker runs one server process on each host in your cluster. One host is selected arbitrarily to run the scheduler process.

Working with existing model data and training jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attaching to existing training jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can attach an MXNet Estimator to an existing training job using the
``attach`` method.

.. code:: python

    my_training_job_name = "MyAwesomeMXNetTrainingJob"
    mxnet_estimator = MXNet.attach(my_training_job_name)

After attaching, if the training job is in a Complete status, it can be
``deploy``\ ed to create a SageMaker Endpoint and return a
``Predictor``. If the training job is in progress, attach will block and display log messages from the training job, until the training job completes.

The ``attach`` method accepts the following arguments:

-  ``training_job_name (str):`` The name of the training job to attach
   to.
-  ``sagemaker_session (sagemaker.Session or None):`` The Session used
   to interact with SageMaker

Deploying Endpoints from model data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As well as attaching to existing training jobs, you can deploy models directly from model data in S3. The following code sample shows how to do this, using the ``MXNetModel`` class.

.. code:: python

    mxnet_model = MXNetModel(model_data="s3://bucket/model.tar.gz", role="SageMakerRole", entry_point="trasform_script.py")

    predictor = mxnet_model.deploy(instance_type="ml.c4.xlarge", initial_instance_count=1)

The MXNetModel constructor takes the following arguments:

-  ``model_data (str):`` An S3 location of a SageMaker model data
   .tar.gz file
-  ``image (str):`` A Docker image URI
-  ``role (str):`` An IAM role name or Arn for SageMaker to access AWS
   resources on your behalf.
-  ``predictor_cls (callable[string,sagemaker.Session]):`` A function to
   call to create a predictor. If not None, ``deploy`` will return the
   result of invoking this function on the created endpoint name
-  ``env (dict[string,string]):`` Environment variables to run with
   ``image`` when hosted in SageMaker.
-  ``name (str):`` The model name. If None, a default model name will be
   selected on each ``deploy.``
-  ``entry_point (str):`` Path (absolute or relative) to the Python file
   which should be executed as the entry point to model hosting.
-  ``source_dir (str):`` Optional. Path (absolute or relative) to a
   directory with any other training source code dependencies aside from
   tne entry point file. Structure within this directory will be
   preserved when training on SageMaker.
-  ``enable_cloudwatch_metrics (boolean):`` Optional. If true, training
   and hosting containers will generate Cloudwatch metrics under the
   AWS/SageMakerContainer namespace.
-  ``container_log_level (int):`` Log level to use within the container.
   Valid values are defined in the Python logging module.
-  ``code_location (str):`` Optional. Name of the S3 bucket where your
   custom code will be uploaded to. If not specified, will use the
   SageMaker default bucket created by sagemaker.Session.
-  ``sagemaker_session (sagemaker.Session):`` The SageMaker Session
   object, used for SageMaker interaction"""

Your model data must be a .tar.gz file in S3. SageMaker Training Job model data is saved to .tar.gz files in S3, however if you have local data you want to deploy, you can prepare the data yourself.

Assuming you have a local directory containg your model data named "my_model" you can tar and gzip compress the file and upload to S3 using the following commands:

::

    tar -czf model.tar.gz my_model
    aws s3 cp model.tar.gz s3://my-bucket/my-path/model.tar.gz

This uploads the contents of my_model to a gzip compressed tar file to S3 in the bucket "my-bucket", with the key "my-path/model.tar.gz".

To run this command, you'll need the aws cli tool installed. Please refer to our `FAQ <#FAQ>`__ for more information on installing this.

MXNet Training Examples
~~~~~~~~~~~~~~~~~~~~~~~

Amazon provides several example Jupyter notebooks that demonstrate end-to-end training on Amazon SageMaker using MXNet. Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk


These are also availble in SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.

SageMaker MXNet Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control the environment your script runs in.

SageMaker runs MXNet Estimator scripts in either Python 2.7 or Python 3.5. You can select the Python version by passing a ``py_version`` keyword arg to the MXNet Estimator constructor. Setting this to ``py2`` (the default) will cause your training script to be run on Python 2.7. Setting this to ``py3`` will cause your training script to be run on Python 3.5. This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

Your MXNet training script will be run on version 1.1.0 by default. (See below for how to choose a different version, and currently supported versions.) The decision to use the GPU or CPU version of MXNet is made by the ``train_instance_type``, set on the MXNet constructor. If you choose a GPU instance type, your training job will be run on a GPU version of MXNet. If you choose a CPU instance type, your training job will be run on a CPU version of MXNet. Similarly, when you call deploy, specifying a GPU or CPU deploy_instance_type, will control which MXNet build your Endpoint runs.

The Docker images have the following dependencies installed:

+-------------------------+--------------+-------------+-------------+
| Dependencies            | MXNet 0.12.1 | MXNet 1.0.0 | MXNet 1.1.0 |
+-------------------------+--------------+-------------+-------------+
| Python                  |   2.7 or 3.5 |   2.7 or 3.5|   2.7 or 3.5|
+-------------------------+--------------+-------------+-------------+
| CUDA                    |          9.0 |         9.0 |         9.0 |
+-------------------------+--------------+-------------+-------------+
| numpy                   |       1.13.3 |      1.13.3 |      1.13.3 |
+-------------------------+--------------+-------------+-------------+

The Docker images extend Ubuntu 16.04.

You can select version of MXNet by passing a ``framework_version`` keyword arg to the MXNet Estimator constructor. Currently supported versions are listed in the above table. You can also set ``framework_version`` to only specify major and minor version, e.g ``1.1``, which will cause your training script to be run on the latest supported patch version of that minor version, which in this example would be 1.1.0.

TensorFlow SageMaker Estimators
-------------------------------

TensorFlow SageMaker Estimators allow you to run your own TensorFlow
training algorithms on SageMaker Learner, and to host your own TensorFlow
models on SageMaker Hosting.

Supported versions of TensorFlow: ``1.4.1``, ``1.5.0``, ``1.6.0``.

Training with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~

Training TensorFlow models using a ``sagemaker.tensorflow.TensorFlow``
is a two-step process.
First, you prepare your training script, then second, you run it on
SageMaker Learner via the ``sagemaker.tensorflow.TensorFlow`` estimator.

Suppose that you already have a TensorFlow training script called
``tf-train.py``. You can train this script in SageMaker Learner as
follows:

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                            training_steps=10000, evaluation_steps=100,
                            train_instance_count=1, train_instance_type='ml.p2.xlarge')
  tf_estimator.fit('s3://bucket/path/to/training/data')

Where the s3 url is a path to your training data, within Amazon S3. The
constructor keyword arguments define how SageMaker runs your training
script and are discussed, in detail, in a later section.

In the following sections, we'll discuss how to prepare a training script for execution on
SageMaker, then how to run that script on SageMaker using a ``sagemaker.tensorflow.TensorFlow``
estimator.

Preparing the TensorFlow training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your TensorFlow training script must be a **Python 2.7** source file. The current supported TensorFlow
versions are **1.6.0 (default)**, **1.5.0**, and **1.4.1**. The SageMaker TensorFlow docker image
uses this script by calling specifically-named functions from this script.

The training script **must contain** the following:

- Exactly one of the following:

  - ``model_fn``: defines the model that will be trained.
  - ``keras_model_fn``: defines the ``tf.keras`` model that will be trained.
  - ``estimator_fn``: defines the ``tf.estimator.Estimator`` that will train the model.

- ``train_input_fn``: preprocess and load training data.
- ``eval_input_fn``: preprocess and load evaluation data.

In addition, it may optionally contain:

- ``serving_input_fn``: Defines the features to be passed to the model during prediction. **Important:**
    this function is used only during training, but is required to deploy the model resulting from training
    in a SageMaker endpoint.

Creating a ``model_fn``
^^^^^^^^^^^^^^^^^^^^^^^

A ``model_fn`` is a function that contains all the logic to support training, evaluation,
and prediction. The basic skeleton for a ``model_fn`` looks like this:

.. code:: python

  def model_fn(features, labels, mode, hyperparameters):
    # Logic to do the following:
    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)

The ``model_fn`` must accept four positional arguments:

- ``features``: A dict containing the features passed to the model via ``train_input_fn``
  in **training** mode, via ``eval_input_fn`` in **evaluation** mode, and via ``serving_input_fn``
  in **predict** mode.
- ``labels``: A ``Tensor`` containing the labels passed to the model via ``train_input_fn``
  in **training** mode and ``eval_input_fn`` in **evaluation** mode. It will be empty for
  **predict** mode.
- ``mode``: One of the following ``tf.estimator.ModeKeys`` string values indicating the
  context in which the ``model_fn`` was invoked:
  - ``TRAIN``: the ``model_fn`` was invoked in **training** mode.
  - ``EVAL``: the ``model_fn`` was invoked in **evaluation** mode.
  - ``PREDICT``: the ``model_fn`` was invoked in **predict** mode.
- ``hyperparameters``: The hyperparameters passed to SageMaker TrainingJob that runs
  your TensorFlow training script. You can use this to pass hyperparameters to your
  training script.

The ``model_fn`` must return a ``tf.estimator.EstimatorSpec``.

Example of a complete ``model_fn``
''''''''''''''''''''''''''''''''''

.. code:: python

  def model_fn(features, labels, mode, hyperparameters):
    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation
    first_hidden_layer = Dense(10, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = Dense(20, activation='relu')(first_hidden_layer)

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = Dense(1, activation='linear')(second_hidden_layer)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions={"ages": predictions})

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float64), predictions)
    }

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=hyperparameters["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

Distributed training
''''''''''''''''''''

When distributed training happens, a copy of the same neural network will be sent to
multiple training instances. Each instance will train with a batch of the dataset,
calculate loss and minimize the optimizer. One entire loop of this process is called training step.

A `global step <https://www.tensorflow.org/api_docs/python/tf/train/global_step>`_ is a global
counter shared between the instances. It is necessary for distributed training, so the optimizer
can keep track of the number of training steps across instances. The only change in the
previous complete ``model_fn`` to enable distributed training is to pass in the global
step into the ``optimizer.minimize`` function:

.. code:: python

  train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

More information about distributed training can be find in talk from the TensorFlow Dev Summit 2017
`Distributed TensorFlow <https://www.youtube.com/watch?time_continue=1&v=la_M6bCV91M>`_.


More details on how to create a ``model_fn`` can be find in `Constructing the model_fn <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/extend/estimators.md#constructing-the-model_fn-constructing-modelfn>`_.


Creating ``train_input_fn`` and ``eval_input_fn`` functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``train_input_fn`` is used to pass ``features`` and ``labels`` to the ``model_fn``
in **training** mode. The ``eval_input_fn`` is used to ``features`` and ``labels`` to the
``model_fn`` in **evaluation** mode.

The basic skeleton for the ``train_input_fn`` looks like this:

.. code:: python

  def train_input_fn(training_dir, hyperparameters):
    # Logic to the following:
    # 1. Reads the **training** dataset files located in training_dir
    # 2. Preprocess the dataset
    # 3. Return 1)  a dict of feature names to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return features, labels

An ``eval_input_fn`` follows the same format:

.. code:: python

  def eval_input_fn(training_dir, hyperparameters):
    # Logic to the following:
    # 1. Reads the **evaluation** dataset files located in training_dir
    # 2. Preprocess the dataset
    # 3. Return 1)  a dict of feature names to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return features, labels

**Note:** For TensorFlow 1.4 and 1.5, ``train_input_fn`` and ``eval_input_fn`` may also return a no-argument
function which returns the tuple ``features, labels``. This is no longer supported for TensorFlow 1.6 and up.

Example of a complete ``train_input_fn`` and ``eval_input_fn``
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: python

  def train_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with training dataset
    return _input_fn(training_dir, 'training_dataset.csv')

  def eval_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with evaluation dataset
    return _input_fn(training_dir, 'evaluation_dataset.csv')

  def _input_fn(training_dir, training_filename):
      # reads the dataset using tf.dataset API
      training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
          filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)

      # returns features x and labels y
      return tf.estimator.inputs.numpy_input_fn(
          x={INPUT_TENSOR_NAME: np.array(training_set.data)},
          y=np.array(training_set.target),
          num_epochs=None,
          shuffle=True)()


More details on how to create input functions can be find in `Building Input Functions with tf.estimator <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/get_started/input_fn.md#building-input-functions-with-tfestimator>`_.

Creating a ``serving_input_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``serving_input_fn`` is used to define the shapes and types of the inputs the model accepts when the model is exported for Tensorflow Serving. It is optional, but required for deploying the trained model to a SageMaker endpoint.

``serving_input_fn`` is called at the end of model training and is **not** called during inference. (If you'd like to preprocess inference data, please see **Overriding input preprocessing with an input_fn**).

The basic skeleton for the ``serving_input_fn`` looks like this:

.. code:: python

  def serving_input_fn(hyperparameters):
    # Logic to the following:
    # 1. Defines placeholders that TensorFlow serving will feed with inference requests
    # 2. Preprocess input data
    # 3. Returns a tf.estimator.export.ServingInputReceiver or tf.estimator.export.TensorServingInputReceiver,
    # which packages the placeholders and the resulting feature Tensors together.

**Note:** For TensorFlow 1.4 and 1.5, ``serving_input_fn`` may also return a no-argument function which returns a ``tf.estimator.export.ServingInputReceiver`` or``tf.estimator.export.TensorServingInputReceiver``. This is no longer supported for TensorFlow 1.6 and up.

Example of a complete ``serving_input_fn``
''''''''''''''''''''''''''''''''''''''''''

.. code:: python

  def serving_input_fn(hyperparameters):
      # defines the input placeholder
      tensor = tf.placeholder(tf.float32, shape=[1, 7])
      # returns the ServingInputReceiver object.
      return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()

More details on how to create a `serving_input_fn` can be find in `Preparing serving inputs <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/programmers_guide/saved_model.md#preparing-serving-inputs>`_.

The complete example described above can find in `Abalone age predictor using layers notebook example <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_layers/tensorflow_abalone_age_predictor_using_layers.ipynb>`_.

More examples on how to create a TensorFlow training script can be find in the `Amazon SageMaker examples repository <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk>`_.

Support for pre-made ``tf.estimator`` and ``Keras`` models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to ``model_fn``, ``sagemaker.tensorflow.TensorFlow`` supports pre-canned ``tf.estimator``
and ``Keras`` models.

Using a pre-made ``tensorflow.estimator`` instead of a ``model_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-canned estimators are machine learning estimators premade for general purpose problems.
``tf.estimator`` provides the following pre-canned estimators:

- `tf.estimator.LinearClassifier <https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier>`_: Constructs
  a linear classification model.
- `tf.estimator.LinearRegressor <https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor>`_: Constructs
  a linear regression model.
- `tf.estimator.DNNClassifier <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier>`_: Constructs
  a neural network classification model.
- `tf.estimator.DNNRegressor <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor>`_: Construct
  a neural network regression model.
- `tf.estimator.DNNLinearCombinedClassifier <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier>`_: Constructs
  a neural network and linear combined classification model.
- `tf.estimator.DNNLinearCombinedRegressor <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedRegressor>`_: Constructs
  a neural network and linear combined regression model.

To use a pre-canned ``tensorflow.estimator`` instead of creating a ``model_fn``, you need to write a ``estimator_fn``.
The base skeleton for the ``estimator_fn`` looks like this:

.. code:: python

  def estimator_fn(run_config, hyperparameters):
    # Logic to the following:
    # 1. Defines the features columns that will be the input of the estimator
    # 2. Returns an instance of a ``tensorflow.estimator`` passing in, the input run_config in the
    #    constructor.

Example of a complete ``estimator_fn``
''''''''''''''''''''''''''''''''''''''

.. code:: python

  def estimator_fn(run_config, hyperparameters):
      # Defines the features columns that will be the input of the estimator
      feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])]
      # Returns the instance of estimator.
      return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        config=run_config)

More details on how to create a ``tensorflow.estimator`` can be find in `Creating Estimators in tf.estimator <https://www.tensorflow.org/extend/estimators>`_.

An example on how to create a TensorFlow training script with an ``estimator_fn`` can find in this `example <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators>`_.


Using a ``Keras`` model instead of a ``model_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``tf.keras`` is an full implementation inside TensorFlow of the Keras API. To use a ``tf.keras``
model for training instead of ``model_fn``, you need to write a ``keras_model_fn``. The base skeleton of
a ``keras_model_fn`` looks like this:

.. code:: python

  def keras_model_fn(hyperparameters):
      # Logic to do the following:
      # 1. Instantiate the Keras model
      # 2. Compile the Keras model
      return compiled_model


Example of a complete ``keras_model_fn``
''''''''''''''''''''''''''''''''''''''''

.. code:: python

  def keras_model_fn(hyperparameters):
    # Instantiate a Keras inception v3 model.
    keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
    # Compile model with the optimizer, loss, and metrics you'd like to train with.
    keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy', metric='accuracy')
    return keras_inception_v3


TensorFlow 1.4.0 support for ``Keras`` models is limited only for **non-distributed** training;
i.e. set the ``train_instance_count`` parameter in the ``TensorFlow`` estimator equal to 1.

More details on how to create a ``Keras`` model can be find in the `Keras documentation <https://keras.io/>`_.

Running a TensorFlow training script in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You run TensorFlow training scripts on SageMaker by creating a ``sagemaker.tensorflow.TensorFlow`` estimator.
When you call ``fit`` on the ``TensorFlow`` estimator, a training job is created in SageMaker.
The following code sample shows how to train a custom TensorFlow script 'tf-train.py'.

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                            training_steps=10000, evaluation_steps=100,
                            train_instance_count=1, train_instance_type='ml.p2.xlarge')
  tf_estimator.fit('s3://bucket/path/to/training/data')

sagemaker.tensorflow.TensorFlow class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``TensorFlow`` constructor takes both required and optional arguments.

Required arguments
''''''''''''''''''

The following are required arguments to the TensorFlow constructor.

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
- ``training_steps (int)`` Perform this many steps of training. ``None``, means train forever.
- ``evaluation_steps (int)`` Perform this many steps of evaluation. ``None``, means
  that evaluation runs until input from ``eval_input_fn`` is exhausted (or another exception is raised).

Optional Arguments
''''''''''''''''''

The following are optional arguments. When you create a ``TensorFlow`` object,
you can specify these as keyword arguments.

-  ``source_dir (str)`` Path (absolute or relative) to a directory with any
   other training source code dependencies aside from the entry point
   file. Structure within this directory will be preserved when training
   on SageMaker.
-  ``requirements_file (str)`` Path to a ``requirements.txt`` file. The path should
   be within and relative to ``source_dir``. This is a file containing a list of items to be
   installed using pip install. Details on the format can be found in the
   `Pip User Guide <https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format>`_.
-  ``hyperparameters (dict[str,ANY])`` Hyperparameters that will be used for training.
   Will be made accessible as a dict[] to the training code on
   SageMaker. Some hyperparameters will be interpreted by TensorFlow and can be use to
   fine tune training. See `Optional Hyperparameters <#optional-hyperparameters>`_.
-  ``train_volume_size (int)`` Size in GB of the EBS volume to use for storing
   input data during training. Must be large enough to the store training
   data.
-  ``train_max_run (int)`` Timeout in hours for training, after which Amazon
   SageMaker terminates the job regardless of its current status.
-  ``output_path (str)`` S3 location where you want the training result (model
   artifacts and optional output files) saved. If not specified, results
   are stored to a default bucket. If the bucket with the specific name
   does not exist, the estimator creates the bucket during the ``fit``
   method execution.
-  ``checkpoint_path`` S3 location where checkpoint data will saved and restored.
   The default location is *bucket_name/job_name/checkpoint*. If the location
   already has checkpoints before the training starts, the model will restore
   state from the last saved checkpoint. It is very useful to restart a training.
   See `Restoring from checkpoints <#restoring-from-checkpoints>`_.
-  ``output_kms_key`` Optional KMS key ID to optionally encrypt training
   output with.
-  ``base_job_name`` Name to assign for the training job that the ``fit``
   method launches. If not specified, the estimator generates a default
   job name, based on the training image name and current timestamp.


Optional Hyperparameters
''''''''''''''''''''''''

These hyperparameters are used by TensorFlow to fine tune the training.
You need to add them inside the hyperparameters dictionary in the
``TensorFlow`` estimator constructor.

**All versions**

-  ``save_summary_steps (int)`` Save summaries every this many steps.
-  ``save_checkpoints_secs (int)`` Save checkpoints every this many seconds. Can not be specified with ``save_checkpoints_steps``.
-  ``save_checkpoints_steps (int)`` Save checkpoints every this many steps. Can not be specified with ``save_checkpoints_secs``.
-  ``keep_checkpoint_max (int)`` The maximum number of recent checkpoint files to keep. As new files are created, older files are deleted. If None or 0, all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
-  ``keep_checkpoint_every_n_hours (int)`` Number of hours between each checkpoint to be saved. The default value of 10,000 hours effectively disables the feature.
-  ``log_step_count_steps (int)`` The frequency, in number of global steps, that the global step/sec will be logged during training.

**TensorFlow 1.6 and up**

- ``start_delay_secs (int)`` See docs for this parameter in `tf.estimator.EvalSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec>`_.
- ``throttle_secs (int)`` See docs for this parameter in `tf.estimator.EvalSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec>`_.

**TensorFlow 1.4 and 1.5**

-  ``eval_metrics (dict)`` ``dict`` of string, metric function. If `None`, default set is used. This should be ``None`` if the ``estimator`` is `tf.estimator.Estimator <https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator>`_. If metrics are provided they will be *appended* to the default set.
-  ``eval_delay_secs (int)`` Start evaluating after waiting for this many seconds.
-  ``continuous_eval_throttle_secs (int)`` Do not re-evaluate unless the last evaluation was started at least this many seconds ago.
-  ``min_eval_frequency (int)`` The minimum number of steps between evaluations. Of course, evaluation does not occur if no new snapshot is available, hence, this is the minimum. If 0, the evaluation will only happen after training. If None, defaults to 1000.
-  ``delay_workers_by_global_step (bool)`` if ``True`` delays training workers based on global step instead of time.
- ``train_steps_per_iteration (int)`` Perform this many (integer) number of train steps for each training-evaluation iteration. With a small value, the model will be evaluated more frequently with more checkpoints saved.

Calling fit
^^^^^^^^^^^

You start your training script by calling ``fit`` on a ``TensorFlow`` estimator. ``fit`` takes
both required and optional arguments.

Required argument
'''''''''''''''''

-  ``inputs (str)``: A S3 URI, for example ``s3://my-bucket/my-training-data``, which contains
   the dataset that will be used for training. When the training job starts in SageMaker the
   container will download the dataset. Both ``train_input_fn`` and ``eval_input_fn`` functions
   have a parameter called ``training_dir`` which contains the directory inside the container
   where the dataset was saved into. See `Creating train_input_fn and eval_input_fn functions`_.

Optional arguments
''''''''''''''''''

-  ``wait (bool)``: Defaults to True, whether to block and wait for the
   training script to complete before returning.
   If set to False, it will return immediately, and can later be attached to.
-  ``logs (bool)``: Defaults to True, whether to show logs produced by training
   job in the Python session. Only meaningful when wait is True.
- ``run_tensorboard_locally (bool)``: Defaults to False. Executes TensorBoard in a different
  process with downloaded checkpoint information. Requires modules TensorBoard and AWS CLI.
  installed. Terminates TensorBoard when the execution ends. See `Running TensorBoard`_.
- ``job_name (str)``: Training job name. If not specified, the estimator generates a default job name,
  based on the training image name and current timestamp.

What happens when fit is called
"""""""""""""""""""""""""""""""

Calling ``fit`` starts a SageMaker training job. The training job will execute the following.

- Starts ``train_instance_count`` EC2 instances of the type ``train_instance_type``.
- On each instance, it will do the following steps:

  - starts a Docker container optimized for TensorFlow, see `SageMaker TensorFlow Docker containers`_.
  - downloads the dataset.
  - setup up distributed training.
  - starts asynchronous training, executing the ``model_fn`` function defined in your script
    in **training** mode; i.e., ``features`` and ``labels`` are fed by a batch of the
    training dataset defined by ``train_input_fn``. See `Creating train_input_fn and eval_input_fn functions`_.

The training job finishes after the number of training steps reaches the value defined by
the ``TensorFlow`` estimator parameter ``training_steps`` is finished or when the training
job execution time reaches the ``TensorFlow`` estimator parameter ``train_max_run``.

When the training job finishes, a `TensorFlow serving <https://www.tensorflow.org/serving/serving_basic>`_
with the result of the training is generated and saved to the S3 location defined by
the ``TensorFlow`` estimator parameter ``output_path``.


If the ``wait=False`` flag is passed to ``fit``, then it will return immediately. The training job will continue running
asynchronously. At a later time, a Tensorflow Estimator can be obtained by attaching to the existing training job. If
the training job is not finished it will start showing the standard output of training and wait until it completes.
After attaching, the estimator can be deployed as usual.

.. code:: python

    tf_estimator.fit(your_input_data, wait=False)
    training_job_name = tf_estimator.latest_training_job.name

    # after some time, or in a separate python notebook, we can attach to it again.

    tf_estimator = TensorFlow.attach(training_job_name=training_job_name)


The evaluation process
""""""""""""""""""""""

During the training job, the first EC2 instance that is executing the training is named ``master``. All the other instances are called ``workers``.

All instances execute the training loop, feeding the ``model_fn`` with ``train_input_fn``.
Every ``min_eval_frequency`` steps (see `Optional Hyperparameters`_), the ``master`` instance
will execute the ``model_fn`` in **evaluation** mode; i.e., ``features`` and ``labels`` are
fed with the evaluation dataset defined by ``eval_input_fn``. See `Creating train_input_fn and eval_input_fn functions`_.

For more information on training and evaluation process, see `tf.estimator.train_and_evaluate <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/estimator/training.py#L256>`_.

For more information on fit, see `SageMaker Python SDK Overview <#sagemaker-python-sdk-overview>`_.

TensorFlow serving models
^^^^^^^^^^^^^^^^^^^^^^^^^

After your training job is complete in SageMaker and the ``fit`` call ends, the training job
will generate a `TensorFlow serving <https://www.tensorflow.org/serving/serving_basic>`_
model ready for deployment. Your TensorFlow serving model will be available in the S3 location
``output_path`` that you specified when you created your `sagemaker.tensorflow.TensorFlow`
estimator.

Restoring from checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^

While your training job is executing, TensorFlow will generate checkpoints and save them in the S3
location defined by ``checkpoint_path`` parameter in the ``TensorFlow`` constructor.
These checkpoints can be used to restore a previous session or to evaluate the current training using ``TensorBoard``.

To restore a previous session, you just need to create a new ``sagemaker.tensorflow.TensorFlow``
estimator pointing to the previous checkpoint path:

.. code:: python

  previous_checkpoint_path = 's3://location/of/my/previous/generated/checkpoints'

  tf_estimator = TensorFlow('tf-train.py', role='SageMakerRole',
                            checkpoint_path=previous_checkpoint_path
                            training_steps=10000, evaluation_steps=100,
                            train_instance_count=1, train_instance_type='ml.p2.xlarge')
  tf_estimator.fit('s3://bucket/path/to/training/data')


Running TensorBoard
^^^^^^^^^^^^^^^^^^^

When the ``fit`` parameter ``run_tensorboard_locally`` is set ``True``, all the checkpoint data
located in ``checkpoint_path`` will be downloaded to a local temporary folder and a local
``TensorBoard`` application will be watching that temporary folder.
Every time a new checkpoint is created by the training job in the S3 bucket, ``fit`` will download that checkpoint to the same temporary folder and update ``TensorBoard``.

When the ``fit`` method starts the training, it will log the port that ``TensorBoard`` is using
to display metrics. The default port is **6006**, but another port can be chosen depending on
availability. The port number will increase until finds an available port. After that, the port
number will be printed in stdout.

It takes a few minutes to provision containers and start the training job. TensorBoard will start to display metrics shortly after that.

You can access TensorBoard locally at http://localhost:6006 or using your SakeMaker workspace at
`https*workspace_base_url*proxy/6006/ <proxy/6006/>`_ (TensorBoard will not work if you forget to put the slash,
'/', in end of the url). If TensorBoard started on a different port, adjust these URLs to match.

Note that TensorBoard is not supported when passing wait=False to ``fit``.


Deploying TensorFlow Serving models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a ``TensorFlow`` Estimator has been fit, it saves a ``TensorFlow Serving`` model in
the S3 location defined by ``output_path``. You can call ``deploy`` on a ``TensorFlow``
estimator to create a SageMaker Endpoint.

A common usage of the ``deploy`` method, after the ``TensorFlow`` estimator has been fit look
like this:

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  estimator = TensorFlow(entry_point='tf-train.py', ..., train_instance_count=1, train_instance_type='ml.c4.xlarge')

  estimator.fit(inputs)

  predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


The code block above deploys a SageMaker Endpoint with one instance of the type 'ml.c4.xlarge'.

What happens when deploy is called
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calling ``deploy`` starts the process of creating a SageMaker Endpoint. This process includes the following steps.

- Starts ``initial_instance_count`` EC2 instances of the type ``instance_type``.
- On each instance, it will do the following steps:

  - start a Docker container optimized for TensorFlow Serving, see `SageMaker TensorFlow Docker containers`_.
  - start a production ready HTTP Server which supports protobuf, JSON and CSV content types, see `Making predictions against a SageMaker Endpoint`_.
  - start a `TensorFlow Serving` process

When the ``deploy`` call finishes, the created SageMaker Endpoint is ready for prediction requests. The next chapter will explain
how to make predictions against the Endpoint, how to use different content-types in your requests, and how to extend the Web server
functionality.


Making predictions against a SageMaker Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code adds a prediction request to the previous code example:

.. code:: python

  estimator = TensorFlow(entry_point='tf-train.py', ..., train_instance_count=1, train_instance_type='ml.c4.xlarge')

  estimator.fit(inputs)

  predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

  result = predictor.predict([6.4, 3.2, 4.5, 1.5])

The ``predictor.predict`` method call takes one parameter, the input ``data`` for which you want the ``SageMaker Endpoint``
to provide inference. ``predict`` will serialize the input data, and send it in as request to the ``SageMaker Endpoint`` by
an ``InvokeEndpoint`` SageMaker operation. ``InvokeEndpoint`` operation requests can be made by ``predictor.predict``, by
boto3 ``SageMaker.runtime`` client or by AWS CLI.

The ``SageMaker Endpoint`` web server will process the request, make an inference using the deployed model, and return a response.
The ``result`` returned by ``predict`` is
a Python dictionary with the model prediction. In the code example above, the prediction ``result`` looks like this:

.. code:: python

  {'result':
    {'classifications': [
      {'classes': [
        {'label': '0', 'score': 0.0012890376383438706},
        {'label': '1', 'score': 0.9814321994781494},
        {'label': '2', 'score': 0.017278732731938362}
      ]}
    ]}
  }

Specifying the output of a prediction request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The format of the prediction ``result`` is determined by the parameter ``export_outputs`` of the `tf.estimator.EstimatorSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec>`_ that you returned when you created your ``model_fn``, see
`Example of a complete model_fn`_ for an example of ``export_outputs``.

More information on how to create ``export_outputs`` can find in `specifying the outputs of a custom model <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/programmers_guide/saved_model.md#specifying-the-outputs-of-a-custom-model>`_.

Endpoint prediction request handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever a prediction request is made to a SageMaker Endpoint via a ``InvokeEndpoint`` SageMaker operation, the request will
be deserialized by the web server, sent to TensorFlow Serving, and serialized back to the client as response.

The TensorFlow Web server breaks request handling into three steps:

-  input processing,
-  TensorFlow Serving prediction, and
-  output processing.

The SageMaker Endpoint provides default input and output processing, which support by default JSON, CSV, and protobuf requests.
This process looks like this:

.. code:: python

    # Deserialize the Invoke request body into an object we can perform prediction on
    deserialized_input = input_fn(serialized_input, request_content_type)

    # Perform prediction on the deserialized object, with the loaded model
    prediction_result = make_tensorflow_serving_prediction(deserialized_input)

    # Serialize the prediction result into the desired response content type
    serialized_output = output_fn(prediction_result, accepts)

The common functionality can be extended by the addiction of the following two functions to your training script:

Overriding input preprocessing with an ``input_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of ``input_fn`` for the content-type "application/python-pickle" can be seen below:

.. code:: python

    import numpy as np

    def input_fn(serialized_input, content_type):
        """An input_fn that loads a pickled object"""
        if request_content_type == "application/python-pickle":
            deserialized_input = pickle.loads(serialized_input)
            return deserialized_input
        else:
            # Handle other content-types here or raise an Exception
            # if the content type is not supported.
            pass

Overriding output postprocessing with an ``output_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of ``output_fn`` for the accept type "application/python-pickle" can be seen below:

.. code:: python

    import numpy as np

    def output_fn(prediction_result, accepts):
        """An output_fn that dumps a pickled object as response"""
        if request_content_type == "application/python-pickle":
            return np.dumps(prediction_result)
        else:
            # Handle other content-types here or raise an Exception
            # if the content type is not supported.
            pass

A example with ``input_fn`` and ``output_fn`` above can be found in
`here <https://github.com/aws/sagemaker-python-sdk/blob/master/tests/data/cifar_10/source/resnet_cifar_10.py#L143>`_.

SageMaker TensorFlow Docker containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TensorFlow Docker images support Python 2.7 and have the following Python modules installed:

+------------------------+------------------+------------------+------------------+
| Dependencies           | tensorflow 1.4.1 | tensorflow 1.5.0 | tensorflow 1.6.0 |
+------------------------+------------------+------------------+------------------+
| boto3                  |            1.4.7 |           1.5.22 |          1.6.21  |
+------------------------+------------------+------------------+------------------+
| botocore               |           1.5.92 |           1.8.36 |          1.9.21  |
+------------------------+------------------+------------------+------------------+
| grpcio                 |            1.7.0 |            1.9.0 |          1.10.0  |
+------------------------+------------------+------------------+------------------+
| numpy                  |           1.13.3 |           1.14.0 |          1.14.2  |
+------------------------+------------------+------------------+------------------+
| pandas                 |           0.21.0 |           0.22.0 |          0.22.0  |
+------------------------+------------------+------------------+------------------+
| protobuf               |            3.4.0 |            3.5.1 |          3.5.2   |
+------------------------+------------------+------------------+------------------+
| scikit-learn           |           0.19.1 |           0.19.1 |          0.19.1  |
+------------------------+------------------+------------------+------------------+
| scipy                  |            1.0.0 |            1.0.0 |          1.0.1   |
+------------------------+------------------+------------------+------------------+
| sklearn                |              0.0 |              0.0 |          0.0     |
+------------------------+------------------+------------------+------------------+
| tensorflow             |            1.4.1 |            1.5.0 |          1.6.0   |
+------------------------+------------------+------------------+------------------+
| tensorflow-serving-api |            1.4.0 |            1.5.0 |          1.5.0   |
+------------------------+------------------+------------------+------------------+

The Docker images extend Ubuntu 16.04.

You can select version of TensorFlow by passing a ``framework_version`` keyword arg to the TensorFlow Estimator constructor. Currently supported versions are listed in the table above. You can also set ``framework_version`` to only specify major and minor version, e.g ``1.6``, which will cause your training script to be run on the latest supported patch version of that minor version, which in this example would be 1.6.0.

AWS SageMaker Estimators
------------------------
Amazon SageMaker provides several built-in machine learning algorithms that you can use for a variety of problem types.

The full list of algorithms is available on the AWS website: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

SageMaker Python SDK includes Estimator wrappers for the AWS K-means, Principal Components Analysis(PCA), Linear Learner, Factorization Machines, Latent Dirichlet Allocation(LDA), Neural Topic Model(NTM) and Random Cut Forest algorithms.

Definition and usage
~~~~~~~~~~~~~~~~~~~~
Estimators that wrap Amazon's built-in algorithms define algorithm's hyperparameters with defaults. When a default is not possible you need to provide the value during construction, e.g.:

- ``KMeans`` Estimator requires parameter ``k`` to define number of clusters
- ``PCA`` Estimator requires parameter ``num_components`` to define number of principal components

Interaction is identical as any other Estimators. There are additional details about how data is specified.

Input data format
^^^^^^^^^^^^^^^^^
Please note that Amazon's built-in algorithms are working best with protobuf ``recordIO`` format.
The data is expected to be available in S3 location and depending on algorithm it can handle dat in multiple data channels.

This package offers support to prepare data into required fomrat and upload data to S3.
Provided class ``RecordSet`` captures necessary details like S3 location, number of records, data channel and is expected as input parameter when calling ``fit()``.

Function ``record_set`` is available on algorithms objects to make it simple to achieve the above.
It takes 2D numpy array as input, uploads data to S3 and returns ``RecordSet`` objects. By default it uses ``train`` data channel and no labels but can be specified when called.

Please find an example code snippet for illustration:

.. code:: python

    from sagemaker import PCA
    pca_estimator = PCA(role='SageMakerRole', train_instance_count=1, train_instance_type='ml.m4.xlarge', num_components=3)

    import numpy as np
    records = pca_estimator.record_set(np.arange(10).reshape(2,5))

    pca_estimator.fit(records)


Predictions support
~~~~~~~~~~~~~~~~~~~
Calling inference on deployed Amazon's built-in algorithms requires specific input format. By default, this library creates a predictor that allows to use just numpy data.
Data is converted so that ``application/x-recordio-protobuf`` input format is used. Received response is deserialized from the protobuf and provided as result from the ``predict`` call.


BYO Docker Containers with SageMaker Estimators
-----------------------------------------------

When you want to use a Docker image prepared earlier and use SageMaker SDK for training the easiest way is to use dedicated ``Estimator`` class. You will be able to instantiate it with desired image and use it in same way as described in previous sections.

Please refer to the full example in the examples repo:

::

    git clone https://github.com/awslabs/amazon-sagemaker-examples.git


The example notebook is is located here:
``advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb``

FAQ
---

I want to train a SageMaker Estimator with local data, how do I do this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You'll need to upload the data to S3 before training. You can use the AWS Command Line Tool (the aws cli) to achieve this.

If you don't have the aws cli, you can install it using pip:

::

    pip install awscli --upgrade --user

If you don't have pip or want to learn more about installing the aws cli, please refer to the official `Amazon aws cli installation guide <http://docs.aws.amazon.com/cli/latest/userguide/installing.html>`__.

Once you have the aws cli installed, you can upload a directory of files to S3 with the following command:

::

    aws s3 cp /tmp/foo/ s3://bucket/path

You can read more about using the aws cli for manipulating S3 resources in the `AWS cli command reference <http://docs.aws.amazon.com/cli/latest/reference/s3/index.html>`__.


How do I make predictions against an existing endpoint?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a Predictor object and provide it your endpoint name. Then, simply call its predict() method with your input.

You can either use the generic RealTimePredictor class, which by default does not perform any serialization/deserialization transformations on your input, but can be configured to do so through constructor arguments:
http://sagemaker.readthedocs.io/en/latest/predictors.html

Or you can use the TensorFlow / MXNet specific predictor classes, which have default serialization/deserialization logic:
http://sagemaker.readthedocs.io/en/latest/sagemaker.tensorflow.html#tensorflow-predictor
http://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-predictor

Example code using the TensorFlow predictor:

::

    from sagemaker.tensorflow import TensorFlowPredictor

    predictor = TensorFlowPredictor('myexistingendpoint')
    result = predictor.predict(['my request body'])


BYO Model
-----------------------------------------------
You can also create an endpoint from an existing model rather than training one - i.e. bring your own model.

First, package the files for the trained model into a ``.tar.gz`` file, and upload the archive to S3.

Next, create a ``Model`` object that corresponds to the framework that you are using: `MXNetModel <https://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-model>`__ or `TensorFlowModel <https://sagemaker.readthedocs.io/en/latest/sagemaker.tensorflow.html#tensorflow-model>`__.

Example code using ``MXNetModel``:

.. code:: python

   from sagemaker.mxnet.model import MXNetModel

   sagemaker_model = MXNetModel(model_data='s3://path/to/model.tar.gz',
                                role='arn:aws:iam::accid:sagemaker-role',
                                entry_point='entry_point.py')

After that, invoke the ``deploy()`` method on the ``Model``:

.. code:: python

   predictor = sagemaker_model.deploy(initial_instance_count=1,
                                      instance_type='ml.m4.xlarge')

This returns a predictor the same way an ``Estimator`` does when ``deploy()`` is called. You can now get inferences just like with any other model deployed on Amazon SageMaker.

A full example is available in the `Amazon SageMaker examples repository <https://github.com/ragavvenkatesan/amazon-sagemaker-examples/tree/3c8394f21ee357da0b553b0ab024c5c5e425182a/advanced_functionality/mxnet_mnist_byom>`__.
