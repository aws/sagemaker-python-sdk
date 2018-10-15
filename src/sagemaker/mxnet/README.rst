=====================================
MXNet SageMaker Estimators and Models
=====================================

With MXNet Estimators, you can train and host MXNet models on Amazon SageMaker.

Supported versions of MXNet: ``1.2.1``, ``1.1.0``, ``1.0.0``, ``0.12.1``.

Training with MXNet
~~~~~~~~~~~~~~~~~~~

Training MXNet models using ``MXNet`` Estimators is a two-step process. First, you prepare your training script, then second, you run this on SageMaker via an ``MXNet`` Estimator. You should prepare your script in a separate source file than the notebook, terminal session, or source file you're using to submit the script to SageMaker via an ``MXNet`` Estimator.

Suppose that you already have an MXNet training script called
``mxnet-train.py``. You can run this script in SageMaker as follows:

.. code:: python

    from sagemaker.mxnet import MXNet
    mxnet_estimator = MXNet('mxnet-train.py',
                            role='SageMakerRole',
                            train_instance_type='ml.p3.2xlarge',
                            train_instance_count=1,
                            framework_version='1.2.1')
    mxnet_estimator.fit('s3://bucket/path/to/training/data')

Where the s3 url is a path to your training data, within Amazon S3. The constructor keyword arguments define how SageMaker runs your training script and are discussed, in detail, in a later section.

In the following sections, we'll discuss how to prepare a training script for execution on SageMaker, then how to run that script on SageMaker using an ``MXNet`` Estimator.

Preparing the MXNet training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------------------------------------------------------------------------------------------------------+
| WARNING                                                                                                                       |
+===============================================================================================================================+
| This required structure for training scripts will be deprecated with the next major release of MXNet images.                  |
| The ``train`` function will no longer be required; instead the training script must be able to be run as a standalone script. |
| For more information, see `"Updating your MXNet training script" <#updating-your-mxnet-training-script>`__.                   |
+-------------------------------------------------------------------------------------------------------------------------------+

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

You can import both ``mxnet`` and ``numpy`` in your training script. When your script runs in SageMaker, it will run with access to MXNet version 1.2.1 and numpy version 1.14.5 by default. For more information on the environment your script runs in, please see `SageMaker MXNet Containers <#sagemaker-mxnet-containers>`__.

Running an MXNet training script in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You run MXNet training scripts on SageMaker by creating ``MXNet`` Estimators. SageMaker training of your script is invoked when you call ``fit`` on an ``MXNet`` Estimator. The following code sample shows how you train a custom MXNet script "train.py".

.. code:: python

    mxnet_estimator = MXNet('train.py',
                            train_instance_type='ml.p2.xlarge',
                            train_instance_count=1,
                            framework_version='1.2.1')
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

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
   other training source code dependencies including the entry point
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
-  ``train_max_run`` Timeout in seconds for training, after which Amazon
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
-  ``image_name`` An alternative docker image to use for training and
   serving.  If specified, the estimator will use this image for training and
   hosting, instead of selecting the appropriate SageMaker official image based on
   framework_version and py_version. Refer to: `SageMaker MXNet Docker Containers
   <#sagemaker-mxnet-docker-containers>`_ for details on what the Official images support
   and where to find the source code to build your custom image.

Calling fit
^^^^^^^^^^^

You start your training script by calling ``fit`` on an ``MXNet`` Estimator. ``fit`` takes both required and optional arguments.

Required argument
'''''''''''''''''

-  ``inputs``: This can take one of the following forms: A string
   s3 URI, for example ``s3://my-bucket/my-training-data``. In this
   case, the s3 objects rooted at the ``my-training-data`` prefix will
   be available in the default ``training`` channel. A dict from
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
    mxnet_estimator = MXNet('train.py',
                            train_instance_type='ml.p2.xlarge',
                            train_instance_count=1,
                            framework_version='1.2.1')
    mxnet_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploy my estimator to a SageMaker Endpoint and get a Predictor
    predictor = mxnet_estimator.deploy(instance_type='ml.m4.xlarge',
                                       initial_instance_count=1)

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
   directory with any other training source code dependencies including
   tne entry point file. Structure within this directory will be
   preserved when training on SageMaker.
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

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the "sample notebooks" folder.


Updating your MXNet training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The required structure for training scripts will be deprecated with the next major release of MXNet images.
The ``train`` function will no longer be required; instead the training script must be able to be run as a standalone script.
In this way, the training script will become similar to a training script you might run outside of SageMaker.

There are a few steps needed to make a training script with the old format compatible with the new format.
You don't need to do this yet, but it's documented here for future reference, as this change is coming soon.

First, add a `main guard <https://docs.python.org/3/library/__main__.html>`__ (``if __name__ == '__main__':``).
The code executed from your main guard needs to:

1. Set hyperparameters and directory locations
2. Initiate training
3. Save the model

Hyperparameters will be passed as command-line arguments to your training script.
In addition, the container will define the locations of input data and where to save the model artifacts and output data as environment variables rather than passing that information as arguments to the ``train`` function.
You can find the full list of available environment variables in the `SageMaker Containers README <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

We recommend using `an argument parser <https://docs.python.org/3.5/howto/argparse.html>`__ for this part.
Using the ``argparse`` library as an example, the code would look something like this:

.. code:: python

    import argparse
    import os

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        # hyperparameters sent by the client are passed as command-line arguments to the script.
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=100)
        parser.add_argument('--learning-rate', type=float, default=0.1)

        # input data and model directories
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
        parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

        args, _ = parser.parse_known_args()

The code in the main guard should also take care of training and saving the model.
This can be as simple as just calling the ``train`` and ``save`` methods used in the previous training script format:

.. code:: python

    if __name__ == '__main__':
        # arg parsing (shown above) goes here

        model = train(args.batch_size, args.epochs, args.learning_rate, args.train, args.test)
        save(args.model_dir, model)

Note that saving the model will no longer be done by default; this must be done by the training script.
If you were previously relying on the default save method, here is one you can copy into your code:

.. code:: python

    import json
    import os

    def save(model_dir, model):
        model.symbol.save(os.path.join(model_dir, 'model-symbol.json'))
        model.save_params(os.path.join(model_dir, 'model-0000.params'))

        signature = [{'name': data_desc.name, 'shape': [dim for dim in data_desc.shape]}
                     for data_desc in model.data_shapes]
        with open(os.path.join(model_dir, 'model-shapes.json'), 'w') as f:
            json.dump(signature, f)

These changes will make training with MXNet similar to training with Chainer or PyTorch on SageMaker.
For more information about those experiences, see `"Preparing the Chainer training script" <https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/chainer#preparing-the-chainer-training-script>`__ and `"Preparing the PyTorch Training Script" <https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/pytorch#preparing-the-pytorch-training-script>`__.


SageMaker MXNet Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control the environment your script runs in.

SageMaker runs MXNet Estimator scripts in either Python 2.7 or Python 3.5. You can select the Python version by passing a ``py_version`` keyword arg to the MXNet Estimator constructor. Setting this to ``py2`` (the default) will cause your training script to be run on Python 2.7. Setting this to ``py3`` will cause your training script to be run on Python 3.5. This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

Your MXNet training script will be run on version 1.2.1 by default. (See below for how to choose a different version, and currently supported versions.) The decision to use the GPU or CPU version of MXNet is made by the ``train_instance_type``, set on the MXNet constructor. If you choose a GPU instance type, your training job will be run on a GPU version of MXNet. If you choose a CPU instance type, your training job will be run on a CPU version of MXNet. Similarly, when you call deploy, specifying a GPU or CPU deploy_instance_type, will control which MXNet build your Endpoint runs.

The Docker images have the following dependencies installed:

+-------------------------+--------------+-------------+-------------+-------------+
| Dependencies            | MXNet 0.12.1 | MXNet 1.0.0 | MXNet 1.1.0 | MXNet 1.2.1 |
+-------------------------+--------------+-------------+-------------+-------------+
| Python                  |   2.7 or 3.5 |   2.7 or 3.5|   2.7 or 3.5|   2.7 or 3.5|
+-------------------------+--------------+-------------+-------------+-------------+
| CUDA (GPU image only)   |          9.0 |         9.0 |         9.0 |         9.0 |
+-------------------------+--------------+-------------+-------------+-------------+
| numpy                   |       1.13.3 |      1.13.3 |      1.13.3 |      1.14.5 |
+-------------------------+--------------+-------------+-------------+-------------+

The Docker images extend Ubuntu 16.04.

You can select version of MXNet by passing a ``framework_version`` keyword arg to the MXNet Estimator constructor. Currently supported versions are listed in the above table. You can also set ``framework_version`` to only specify major and minor version, e.g ``1.2``, which will cause your training script to be run on the latest supported patch version of that minor version, which in this example would be 1.2.1.
Alternatively, you can build your own image by following the instructions in the SageMaker MXNet containers repository, and passing ``image_name`` to the MXNet Estimator constructor.

You can visit the SageMaker MXNet containers repository here: https://github.com/aws/sagemaker-mxnet-containers/
