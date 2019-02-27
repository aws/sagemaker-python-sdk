Deploying to Python-based Endpoints
===================================

Deploying from an Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a TensorFlow estimator has been fit, it saves a TensorFlow ``SavedModel`` in
the S3 location defined by ``output_path``. You can call ``deploy`` on a TensorFlow
estimator to create a SageMaker Endpoint.

A common usage of the ``deploy`` method, after the TensorFlow estimator has been fit look
like this:

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  estimator = TensorFlow(entry_point='tf-train.py', ..., train_instance_count=1,
                         train_instance_type='ml.c4.xlarge', framework_version='1.10.0')

  estimator.fit(inputs)

  predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


The code block above deploys a SageMaker Endpoint with one instance of the type 'ml.c4.xlarge'.

Python-based TensorFlow serving on SageMaker has support for `Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`_, which allows for inference acceleration to a hosted endpoint for a fraction of the cost of using a full GPU instance. In order to attach an Elastic Inference accelerator to your endpoint provide the accelerator type to ``accelerator_type`` to your ``deploy`` call.

.. code:: python

  predictor = estimator.deploy(initial_instance_count=1,
                               instance_type='ml.c5.xlarge',
                               accelerator_type='ml.eia1.medium')

What happens when deploy is called
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calling ``deploy`` starts the process of creating a SageMaker Endpoint. This process includes the following steps.

- Starts ``initial_instance_count`` EC2 instances of the type ``instance_type``.
- On each instance, it will do the following steps:

  - start a Docker container optimized for TensorFlow Serving, see `SageMaker TensorFlow Docker containers`_.
  - start a `TensorFlow Serving` process configured to run your model.
  - start a Python-based HTTP server which supports protobuf, JSON and CSV content types, and can run your custom
    input and output python functions. See `Making predictions against a SageMaker Endpoint`_.


When the ``deploy`` call finishes, the created SageMaker Endpoint is ready for prediction requests. The next chapter will explain
how to make predictions against the Endpoint, how to use different content-types in your requests, and how to extend the Web server
functionality.

Deploying directly from model artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have existing model artifacts, you can skip training and deploy them directly to an endpoint:

.. code:: python

  from sagemaker.tensorflow import TensorFlowModel

  tf_model = TensorFlowModel(model_data='s3://mybucket/model.tar.gz',
                             role='MySageMakerRole',
                             entry_point='entry.py',
                             name='model_name')

  predictor = tf_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

You can also optionally specify a pip `requirements file <https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format>`_ if you need to install additional packages into the deployed
runtime environment by including it in your source_dir and specifying it in the ``'SAGEMAKER_REQUIREMENTS'`` env variable:

.. code:: python

  from sagemaker.tensorflow import TensorFlowModel

  tf_model = TensorFlowModel(model_data='s3://mybucket/model.tar.gz',
                             role='MySageMakerRole',
                             entry_point='entry.py',
                             source_dir='my_src', # directory which contains entry_point script and requirements file
                             name='model_name',
                             env={'SAGEMAKER_REQUIREMENTS': 'requirements.txt'}) # path relative to source_dir

  predictor = tf_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


Making predictions against a SageMaker Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code adds a prediction request to the previous code example:

.. code:: python

  estimator = TensorFlow(entry_point='tf-train.py', ..., train_instance_count=1,
                         train_instance_type='ml.c4.xlarge', framework_version='1.10.0')

  estimator.fit(inputs)

  predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

  result = predictor.predict([6.4, 3.2, 4.5, 1.5])

The ``predictor.predict`` method call takes one parameter, the input ``data`` for which you want the SageMaker Endpoint
to provide inference. ``predict`` will serialize the input data, and send it in as request to the SageMaker Endpoint by
an ``InvokeEndpoint`` SageMaker operation. ``InvokeEndpoint`` operation requests can be made by ``predictor.predict``,
by boto3 `SageMakerRuntime <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html>`_
client or by AWS CLI.

The SageMaker Endpoint web server will process the request, make an inference using the deployed model, and return a response.
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
