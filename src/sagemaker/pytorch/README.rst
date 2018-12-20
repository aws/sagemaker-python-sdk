=======================================
SageMaker PyTorch Estimators and Models
=======================================

With PyTorch Estimators and Models, you can train and host PyTorch models on Amazon SageMaker.

Supported versions of PyTorch: ``0.4.0``, ``1.0.0``.

We recommend that you use the latest supported version, because that's where we focus most of our development efforts.

You can visit the PyTorch repository at https://github.com/pytorch/pytorch.

Table of Contents
-----------------

1. `Training with PyTorch <#training-with-pytorch>`__
2. `PyTorch Estimators <#pytorch-estimators>`__
3. `Distributed PyTorch Training <#distributed-pytorch-training>`__
4. `Saving models <#saving-models>`__
5. `Deploying PyTorch Models <#deploying-pytorch-models>`__
6. `SageMaker PyTorch Model Server <#sagemaker-pytorch-model-server>`__
7. `Working with Existing Model Data and Training Jobs <#working-with-existing-model-data-and-training-jobs>`__
8. `PyTorch Training Examples <#pytorch-training-examples>`__
9. `SageMaker PyTorch Docker Containers <#sagemaker-pytorch-docker-containers>`__


Training with PyTorch
------------------------

Training PyTorch models using ``PyTorch`` Estimators is a two-step process:

1. Prepare a PyTorch script to run on SageMaker
2. Run this script on SageMaker via a ``PyTorch`` Estimator.


First, you prepare your training script, then second, you run this on SageMaker via a ``PyTorch`` Estimator.
You should prepare your script in a separate source file than the notebook, terminal session, or source file you're
using to submit the script to SageMaker via a ``PyTorch`` Estimator. This will be discussed in further detail below.

Suppose that you already have a PyTorch training script called `pytorch-train.py`.
You can then setup a ``PyTorch`` Estimator with keyword arguments to point to this script and define how SageMaker runs it:

.. code:: python

    from sagemaker.pytorch import PyTorch

    pytorch_estimator = PyTorch(entry_point='pytorch-train.py',
                                role='SageMakerRole',
                                train_instance_type='ml.p3.2xlarge',
                                train_instance_count=1,
                                framework_version='1.0.0')

After that, you simply tell the estimator to start a training job and provide an S3 URL
that is the path to your training data within Amazon S3:

.. code:: python

    pytorch_estimator.fit('s3://bucket/path/to/training/data')

In the following sections, we'll discuss how to prepare a training script for execution on SageMaker,
then how to run that script on SageMaker using a ``PyTorch`` Estimator.


Preparing the PyTorch Training Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your PyTorch training script must be a Python 2.7 or 3.5 compatible source file.

The training script is very similar to a training script you might run outside of SageMaker, but you
can access useful properties about the training environment through various environment variables, such as

* ``SM_MODEL_DIR``: A string representing the path to the directory to write model artifacts to.
  These artifacts are uploaded to S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_OUTPUT_DATA_DIR``: A string representing the filesystem path to write output artifacts to. Output artifacts may
  include checkpoints, graphs, and other files to save, not including model artifacts. These artifacts are compressed
  and uploaded to S3 to the same S3 prefix as the model artifacts.

Supposing two input channels, 'train' and 'test', were used in the call to the PyTorch estimator's ``fit`` method,
the following will be set, following the format "SM_CHANNEL_[channel_name]":

* ``SM_CHANNEL_TRAIN``: A string representing the path to the directory containing data in the 'train' channel
* ``SM_CHANNEL_TEST``: Same as above, but for the 'test' channel.

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model,
and saves a model to `model_dir` so that it can be hosted later. Hyperparameters are passed to your script as arguments
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
        parser.add_argument('--use-cuda', type=bool, default=False)

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

Note that SageMaker doesn't support argparse actions. If you want to use, for example, boolean hyperparameters,
you need to specify `type` as `bool` in your script and provide an explicit `True` or `False` value for this hyperparameter
when instantiating PyTorch Estimator.

For more on training environment variables, please visit `SageMaker Containers <https://github.com/aws/sagemaker-containers>`_.

Running a PyTorch training script in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You run PyTorch training scripts on SageMaker by creating ``PyTorch`` Estimators.
SageMaker training of your script is invoked when you call ``fit`` on a ``PyTorch`` Estimator.
The following code sample shows how you train a custom PyTorch script "pytorch-train.py", passing
in three hyperparameters ('epochs', 'batch-size', and 'learning-rate'), and using two input channel
directories ('train' and 'test').

.. code:: python

    pytorch_estimator = PyTorch('pytorch-train.py',
                                train_instance_type='ml.p3.2xlarge',
                                train_instance_count=1,
                                framework_version='1.0.0',
                                hyperparameters = {'epochs': 20, 'batch-size': 64, 'learning-rate': 0.1})
    pytorch_estimator.fit({'train': 's3://my-data-bucket/path/to/my/training/data',
                           'test': 's3://my-data-bucket/path/to/my/test/data'})


PyTorch Estimators
------------------

The `PyTorch` constructor takes both required and optional arguments.

Required arguments
~~~~~~~~~~~~~~~~~~

The following are required arguments to the ``PyTorch`` constructor. When you create a PyTorch object, you must include
these in the constructor, either positionally or as keyword arguments.

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
   example, 'ml.m4.xlarge'.

Optional arguments
~~~~~~~~~~~~~~~~~~

The following are optional arguments. When you create a ``PyTorch`` object, you can specify these as keyword arguments.

-  ``source_dir`` Path (absolute or relative) to a directory with any
   other training source code dependencies including the entry point
   file. Structure within this directory will be preserved when training
   on SageMaker.
- ``dependencies (list[str])`` A list of paths to directories (absolute or relative) with
        any additional libraries that will be exported to the container (default: []).
        The library folders will be copied to SageMaker in the same folder where the entrypoint is copied.
        If the ```source_dir``` points to S3, code will be uploaded and the S3 location will be used
        instead. Example:

            The following call
            >>> PyTorch(entry_point='train.py', dependencies=['my/libs/common', 'virtual-env'])
            results in the following inside the container:

            >>> $ ls

            >>> opt/ml/code
            >>>     ├── train.py
            >>>     ├── common
            >>>     └── virtual-env

-  ``hyperparameters`` Hyperparameters that will be used for training.
   Will be made accessible as a dict[str, str] to the training code on
   SageMaker. For convenience, accepts other types besides strings, but
   ``str`` will be called on keys and values to convert them before
   training.
-  ``py_version`` Python version you want to use for executing your
   model training code.
-  ``framework_version`` PyTorch version you want to use for executing
   your model training code. You can find the list of supported versions
   in `the section below <#sagemaker-pytorch-docker-containers>`__.
-  ``train_volume_size`` Size in GB of the EBS volume to use for storing
   input data during training. Must be large enough to store training
   data if input_mode='File' is used (which is the default).
-  ``train_max_run`` Timeout in seconds for training, after which Amazon
   SageMaker terminates the job regardless of its current status.
-  ``input_mode`` The input mode that the algorithm supports. Valid
   modes: 'File' - Amazon SageMaker copies the training dataset from the
   S3 location to a directory in the Docker container. 'Pipe' - Amazon
   SageMaker streams data directly from S3 to the container via a Unix
   named pipe.
-  ``output_path`` S3 location where you want the training result (model
   artifacts and optional output files) saved. If not specified, results
   are stored to a default bucket. If the bucket with the specific name
   does not exist, the estimator creates the bucket during the ``fit``
   method execution.
-  ``output_kms_key`` Optional KMS key ID to optionally encrypt training
   output with.
-  ``job_name`` Name to assign for the training job that the ``fit```
   method launches. If not specified, the estimator generates a default
   job name, based on the training image name and current timestamp
-  ``image_name`` An alternative docker image to use for training and
   serving.  If specified, the estimator will use this image for training and
   hosting, instead of selecting the appropriate SageMaker official image based on
   framework_version and py_version. Refer to: `SageMaker PyTorch Docker Containers
   <#sagemaker-pytorch-docker-containers>`_ for details on what the Official images support
   and where to find the source code to build your custom image.

Calling fit
~~~~~~~~~~~

You start your training script by calling ``fit`` on a ``PyTorch`` Estimator. ``fit`` takes both required and optional
arguments.

Required arguments
''''''''''''''''''

-  ``inputs``: This can take one of the following forms: A string
   S3 URI, for example ``s3://my-bucket/my-training-data``. In this
   case, the S3 objects rooted at the ``my-training-data`` prefix will
   be available in the default ``train`` channel. A dict from
   string channel names to S3 URIs. In this case, the objects rooted at
   each S3 prefix will available as files in each channel directory.

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


Distributed PyTorch Training
----------------------------

You can run a multi-machine, distributed PyTorch training using the PyTorch Estimator. By default, PyTorch objects will
submit single-machine training jobs to SageMaker. If you set ``train_instance_count`` to be greater than one, multi-machine
training jobs will be launched when ``fit`` is called. When you run multi-machine training, SageMaker will import your
training script and run it on each host in the cluster.

To initialize distributed training in your script you would call ``dist.init_process_group`` providing desired backend
and rank and setting 'WORLD_SIZE' environment variable similar to how you would do it outside of SageMaker using
environment variable initialization:

.. code:: python

    if args.distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank)

SageMaker sets 'MASTER_ADDR' and 'MASTER_PORT' environment variables for you, but you can overwrite them.

Supported backends:
-  `gloo` and `tcp` for cpu instances
-  `gloo` and `nccl` for gpu instances

Saving models
-------------

In order to save your trained PyTorch model for deployment on SageMaker, your training script should save your model
to a certain filesystem path called ``model_dir``. This value is accessible through the environment variable
``SM_MODEL_DIR``. The following code demonstrates how to save a trained PyTorch model named ``model`` as
``model.pth`` at the :

.. code:: python

    import argparse
    import os
    import torch

    if __name__=='__main__':
        # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        args, _ = parser.parse_known_args()

        # ... train `model`, then save it to `model_dir`
        with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
            torch.save(model.state_dict(), f)

After your training job is complete, SageMaker will compress and upload the serialized model to S3, and your model data
will be available in the S3 ``output_path`` you specified when you created the PyTorch Estimator.

Deploying PyTorch Models
------------------------

After an PyTorch Estimator has been fit, you can host the newly created model in SageMaker.

After calling ``fit``, you can call ``deploy`` on a ``PyTorch`` Estimator to create a SageMaker Endpoint.
The Endpoint runs a SageMaker-provided PyTorch model server and hosts the model produced by your training script,
which was run when you called ``fit``. This was the model you saved to ``model_dir``.

``deploy`` returns a ``Predictor`` object, which you can use to do inference on the Endpoint hosting your PyTorch model.
Each ``Predictor`` provides a ``predict`` method which can do inference with numpy arrays or Python lists.
Inference arrays or lists are serialized and sent to the PyTorch model server by an ``InvokeEndpoint`` SageMaker
operation.

``predict`` returns the result of inference against your model. By default, the inference result a NumPy array.

.. code:: python

    # Train my estimator
    pytorch_estimator = PyTorch(entry_point='train_and_deploy.py',
                                train_instance_type='ml.p3.2xlarge',
                                train_instance_count=1,
                                framework_version='1.0.0')
    pytorch_estimator.fit('s3://my_bucket/my_training_data/')

    # Deploy my estimator to a SageMaker Endpoint and get a Predictor
    predictor = pytorch_estimator.deploy(instance_type='ml.m4.xlarge',
                                         initial_instance_count=1)

    # `data` is a NumPy array or a Python list.
    # `response` is a NumPy array.
    response = predictor.predict(data)

You use the SageMaker PyTorch model server to host your PyTorch model when you call ``deploy`` on an ``PyTorch``
Estimator. The model server runs inside a SageMaker Endpoint, which your call to ``deploy`` creates.
You can access the name of the Endpoint by the ``name`` property on the returned ``Predictor``.


The SageMaker PyTorch Model Server
----------------------------------

The PyTorch Endpoint you create with ``deploy`` runs a SageMaker PyTorch model server.
The model server loads the model that was saved by your training script and performs inference on the model in response
to SageMaker InvokeEndpoint API calls.

You can configure two components of the SageMaker PyTorch model server: Model loading and model serving.
Model loading is the process of deserializing your saved model back into an PyTorch model.
Serving is the process of translating InvokeEndpoint requests to inference calls on the loaded model.

You configure the PyTorch model server by defining functions in the Python source file you passed to the PyTorch constructor.

Model loading
~~~~~~~~~~~~~

Before a model can be served, it must be loaded. The SageMaker PyTorch model server loads your model by invoking a
``model_fn`` function that you must provide in your script. The ``model_fn`` should have the following signature:

.. code:: python

    def model_fn(model_dir)

SageMaker will inject the directory where your model files and sub-directories, saved by ``save``, have been mounted.
Your model function should return a model object that can be used for model serving.

The following code-snippet shows an example ``model_fn`` implementation.
It loads the model parameters from a ``model.pth`` file in the SageMaker model directory ``model_dir``.

.. code:: python

    import torch
    import os

    def model_fn(model_dir):
        model = Your_Model()
        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f))
        return model

Model serving
~~~~~~~~~~~~~

After the SageMaker model server has loaded your model by calling ``model_fn``, SageMaker will serve your model.
Model serving is the process of responding to inference requests, received by SageMaker InvokeEndpoint API calls.
The SageMaker PyTorch model server breaks request handling into three steps:


-  input processing,
-  prediction, and
-  output processing.

In a similar way to model loading, you configure these steps by defining functions in your Python source file.

Each step involves invoking a python function, with information about the request and the return value from the previous
function in the chain. Inside the SageMaker PyTorch model server, the process looks like:

.. code:: python

    # Deserialize the Invoke request body into an object we can perform prediction on
    input_object = input_fn(request_body, request_content_type)

    # Perform prediction on the deserialized object, with the loaded model
    prediction = predict_fn(input_object, model)

    # Serialize the prediction result into the desired response content type
    output = output_fn(prediction, response_content_type)

The above code sample shows the three function definitions:

-  ``input_fn``: Takes request data and deserializes the data into an
   object for prediction.
-  ``predict_fn``: Takes the deserialized request object and performs
   inference against the loaded model.
-  ``output_fn``: Takes the result of prediction and serializes this
   according to the response content type.

The SageMaker PyTorch model server provides default implementations of these functions.
You can provide your own implementations for these functions in your hosting script.
If you omit any definition then the SageMaker PyTorch model server will use its default implementation for that
function.

The ``RealTimePredictor`` used by PyTorch in the SageMaker Python SDK serializes NumPy arrays to the `NPY <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ format
by default, with Content-Type ``application/x-npy``. The SageMaker PyTorch model server can deserialize NPY-formatted
data (along with JSON and CSV data).

If you rely solely on the SageMaker PyTorch model server defaults, you get the following functionality:

-  Prediction on models that implement the ``__call__`` method
-  Serialization and deserialization of torch.Tensor.

The default ``input_fn`` and ``output_fn`` are meant to make it easy to predict on torch.Tensors. If your model expects
a torch.Tensor and returns a torch.Tensor, then these functions do not have to be overridden when sending NPY-formatted
data.

In the following sections we describe the default implementations of input_fn, predict_fn, and output_fn.
We describe the input arguments and expected return types of each, so you can define your own implementations.

Input processing
''''''''''''''''

When an InvokeEndpoint operation is made against an Endpoint running a SageMaker PyTorch model server,
the model server receives two pieces of information:

-  The request Content-Type, for example "application/x-npy"
-  The request data body, a byte array

The SageMaker PyTorch model server will invoke an ``input_fn`` function in your hosting script,
passing in this information. If you define an ``input_fn`` function definition,
it should return an object that can be passed to ``predict_fn`` and have the following signature:

.. code:: python

    def input_fn(request_body, request_content_type)

Where ``request_body`` is a byte buffer and ``request_content_type`` is a Python string

The SageMaker PyTorch model server provides a default implementation of ``input_fn``.
This function deserializes JSON, CSV, or NPY encoded data into a torch.Tensor.

Default NPY deserialization requires ``request_body`` to follow the `NPY <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ format. For PyTorch, the Python SDK
defaults to sending prediction requests with this format.

Default JSON deserialization requires ``request_body`` contain a single json list.
Sending multiple JSON objects within the same ``request_body`` is not supported.
The list must have a dimensionality compatible with the model loaded in ``model_fn``.
The list's shape must be identical to the model's input shape, for all dimensions after the first (which first
dimension is the batch size).

Default csv deserialization requires ``request_body`` contain one or more lines of CSV numerical data.
The data is loaded into a two-dimensional array, where each line break defines the boundaries of the first dimension.

The example below shows a custom ``input_fn`` for preparing pickled torch.Tensor.

.. code:: python

    import numpy as np
    import torch
    from six import BytesIO

    def input_fn(request_body, request_content_type):
        """An input_fn that loads a pickled tensor"""
        if request_content_type == 'application/python-pickle':
            return torch.load(BytesIO(request_body))
        else:
            # Handle other content-types here or raise an Exception
            # if the content type is not supported.
            pass



Prediction
''''''''''

After the inference request has been deserialized by ``input_fn``, the SageMaker PyTorch model server invokes
``predict_fn`` on the return value of ``input_fn``.

As with ``input_fn``, you can define your own ``predict_fn`` or use the SageMaker PyTorch model server default.

The ``predict_fn`` function has the following signature:

.. code:: python

    def predict_fn(input_object, model)

Where ``input_object`` is the object returned from ``input_fn`` and
``model`` is the model loaded by ``model_fn``.

The default implementation of ``predict_fn`` invokes the loaded model's ``__call__`` function on ``input_object``,
and returns the resulting value. The return-type should be a torch.Tensor to be compatible with the default
``output_fn``.

The example below shows an overridden ``predict_fn``:

.. code:: python

    import torch
    import numpy as np

    def predict_fn(input_data, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            return model(input_data.to(device))

If you implement your own prediction function, you should take care to ensure that:

-  The first argument is expected to be the return value from input_fn.
   If you use the default input_fn, this will be a torch.Tensor.
-  The second argument is the loaded model.
-  The return value should be of the correct type to be passed as the
   first argument to ``output_fn``. If you use the default
   ``output_fn``, this should be a torch.Tensor.

Output processing
'''''''''''''''''

After invoking ``predict_fn``, the model server invokes ``output_fn``, passing in the return value from ``predict_fn``
and the content type for the response, as specified by the InvokeEndpoint request.

The ``output_fn`` has the following signature:

.. code:: python

    def output_fn(prediction, content_type)

Where ``prediction`` is the result of invoking ``predict_fn`` and
the content type for the response, as specified by the InvokeEndpoint request.
The function should return a byte array of data serialized to content_type.

The default implementation expects ``prediction`` to be a torch.Tensor and can serialize the result to JSON, CSV, or NPY.
It accepts response content types of "application/json", "text/csv", and "application/x-npy".

Working with Existing Model Data and Training Jobs
--------------------------------------------------

Attaching to existing training jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can attach an PyTorch Estimator to an existing training job using the
``attach`` method.

.. code:: python

    my_training_job_name = 'MyAwesomePyTorchTrainingJob'
    pytorch_estimator = PyTorch.attach(my_training_job_name)

After attaching, if the training job has finished with job status "Completed", it can be
``deploy``\ ed to create a SageMaker Endpoint and return a
``Predictor``. If the training job is in progress,
attach will block and display log messages from the training job, until the training job completes.

The ``attach`` method accepts the following arguments:

-  ``training_job_name:`` The name of the training job to attach
   to.
-  ``sagemaker_session:`` The Session used
   to interact with SageMaker

Deploying Endpoints from model data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As well as attaching to existing training jobs, you can deploy models directly from model data in S3.
The following code sample shows how to do this, using the ``PyTorchModel`` class.

.. code:: python

    pytorch_model = PyTorchModel(model_data='s3://bucket/model.tar.gz', role='SageMakerRole',
                                 entry_point='transform_script.py')

    predictor = pytorch_model.deploy(instance_type='ml.c4.xlarge', initial_instance_count=1)

The PyTorchModel constructor takes the following arguments:

-  ``model_dat:`` An S3 location of a SageMaker model data
   .tar.gz file
-  ``image:`` A Docker image URI
-  ``role:`` An IAM role name or Arn for SageMaker to access AWS
   resources on your behalf.
-  ``predictor_cls:`` A function to
   call to create a predictor. If not None, ``deploy`` will return the
   result of invoking this function on the created endpoint name
-  ``env:`` Environment variables to run with
   ``image`` when hosted in SageMaker.
-  ``name:`` The model name. If None, a default model name will be
   selected on each ``deploy.``
-  ``entry_point:`` Path (absolute or relative) to the Python file
   which should be executed as the entry point to model hosting.
-  ``source_dir:`` Optional. Path (absolute or relative) to a
   directory with any other training source code dependencies including
   tne entry point file. Structure within this directory will be
   preserved when training on SageMaker.
-  ``enable_cloudwatch_metrics:`` Optional. If true, training
   and hosting containers will generate Cloudwatch metrics under the
   AWS/SageMakerContainer namespace.
-  ``container_log_level:`` Log level to use within the container.
   Valid values are defined in the Python logging module.
-  ``code_location:`` Optional. Name of the S3 bucket where your
   custom code will be uploaded to. If not specified, will use the
   SageMaker default bucket created by sagemaker.Session.
-  ``sagemaker_session:`` The SageMaker Session
   object, used for SageMaker interaction

Your model data must be a .tar.gz file in S3. SageMaker Training Job model data is saved to .tar.gz files in S3,
however if you have local data you want to deploy, you can prepare the data yourself.

Assuming you have a local directory containg your model data named "my_model" you can tar and gzip compress the file and
upload to S3 using the following commands:

::

    tar -czf model.tar.gz my_model
    aws s3 cp model.tar.gz s3://my-bucket/my-path/model.tar.gz

This uploads the contents of my_model to a gzip compressed tar file to S3 in the bucket "my-bucket", with the key
"my-path/model.tar.gz".

To run this command, you'll need the AWS CLI tool installed. Please refer to our `FAQ`_ for more information on
installing this.

.. _FAQ: ../../../README.rst#faq

PyTorch Training Examples
-------------------------

Amazon provides several example Jupyter notebooks that demonstrate end-to-end training on Amazon SageMaker using PyTorch.
Please refer to:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk

These are also available in SageMaker Notebook Instance hosted Jupyter notebooks under the sample notebooks folder.


SageMaker PyTorch Docker Containers
-----------------------------------

When training and deploying training scripts, SageMaker runs your Python script in a Docker container with several
libraries installed. When creating the Estimator and calling deploy to create the SageMaker Endpoint, you can control
the environment your script runs in.

SageMaker runs PyTorch Estimator scripts in either Python 2 or Python 3. You can select the Python version by
passing a ``py_version`` keyword arg to the PyTorch Estimator constructor. Setting this to `py3` (the default) will cause your
training script to be run on Python 3.5. Setting this to `py2` will cause your training script to be run on Python 2.7
This Python version applies to both the Training Job, created by fit, and the Endpoint, created by deploy.

The PyTorch Docker images have the following dependencies installed:

+-----------------------------+---------------+-------------------+
| Dependencies                | pytorch 0.4.0 | pytorch 1.0.0     |
+-----------------------------+---------------+-------------------+
| boto3                       | >=1.7.35      | >=1.9.11          |
+-----------------------------+---------------+-------------------+
| botocore                    | >=1.10.35     | >=1.12.11         |
+-----------------------------+---------------+-------------------+
| CUDA (GPU image only)       | 9.0           | 9.0               |
+-----------------------------+---------------+-------------------+
| numpy                       | >=1.14.3      | >=1.15.2          |
+-----------------------------+---------------+-------------------+
| Pillow                      | >=5.1.0       | >=5.2.0           |
+-----------------------------+---------------+-------------------+
| pip                         | >=10.0.1      | >=18.0            |
+-----------------------------+---------------+-------------------+
| python-dateutil             | >=2.7.3       | >=2.7.3           |
+-----------------------------+---------------+-------------------+
| retrying                    | >=1.3.3       | >=1.3.3           |
+-----------------------------+---------------+-------------------+
| s3transfer                  | >=0.1.13      | >=0.1.13          |
+-----------------------------+---------------+-------------------+
| sagemaker-containers        | >=2.1.0       | >=2.1.0           |
+-----------------------------+---------------+-------------------+
| sagemaker-pytorch-container | 1.0           | 1.0               |
+-----------------------------+---------------+-------------------+
| setuptools                  | >=39.2.0      | >=40.4.3          |
+-----------------------------+---------------+-------------------+
| six                         | >=1.11.0      | >=1.11.0          |
+-----------------------------+---------------+-------------------+
| torch                       | 0.4.0         | 1.0.0             |
+-----------------------------+---------------+-------------------+
| torchvision                 | 0.2.1         | 0.2.1             |
+-----------------------------+---------------+-------------------+
| Python                      | 2.7 or 3.5    | 2.7 or 3.6        |
+-----------------------------+---------------+-------------------+

The Docker images extend Ubuntu 16.04.

If you need to install other dependencies you can put them into `requirements.txt` file and put it in the source directory
(``source_dir``) you provide to the `PyTorch Estimator <#pytorch-estimators>`__.

You can select version of PyTorch by passing a ``framework_version`` keyword arg to the PyTorch Estimator constructor.
Currently supported versions are listed in the above table. You can also set ``framework_version`` to only specify major and
minor version, which will cause your training script to be run on the latest supported patch version of that minor
version.

Alternatively, you can build your own image by following the instructions in the SageMaker Chainer containers
repository, and passing ``image_name`` to the Chainer Estimator constructor.

You can visit `the SageMaker PyTorch containers repository <https://github.com/aws/sagemaker-pytorch-containers>`_.
