Deploying to TensorFlow Serving Endpoints
=========================================

Table of Contents
~~~~~~~~~~~~~~~~~

- `Deploying from an Estimator`_
- `Deploying directly from model artifacts`_
- `Making predictions against a SageMaker Endpoint`_
- `Deploying more than one model to your Endpoint`_
- `Making predictions with the AWS CLI`_

Deploying from an Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a TensorFlow estimator has been fit, it saves a TensorFlow
`SavedModel <https://www.tensorflow.org/guide/saved_model>`_ bundle in
the S3 location defined by ``output_path``. You can call ``deploy`` on a TensorFlow
estimator object to create a SageMaker Endpoint:

.. code:: python

  from sagemaker.tensorflow import TensorFlow

  estimator = TensorFlow(entry_point='tf-train.py', ..., train_instance_count=1,
                         train_instance_type='ml.c4.xlarge', framework_version='1.11')

  estimator.fit(inputs)

  predictor = estimator.deploy(initial_instance_count=1,
                               instance_type='ml.c5.xlarge',
                               endpoint_type='tensorflow-serving')


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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have existing model artifacts in S3, you can skip training and deploy them directly to an endpoint:

.. code:: python

  from sagemaker.tensorflow.serving import Model

  model = Model(model_data='s3://mybucket/model.tar.gz', role='MySageMakerRole')

  predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')

Python-based TensorFlow serving on SageMaker has support for `Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`__, which allows for inference acceleration to a hosted endpoint for a fraction of the cost of using a full GPU instance. In order to attach an Elastic Inference accelerator to your endpoint provide the accelerator type to accelerator_type to your deploy call.

.. code:: python

    from sagemaker.tensorflow.serving import Model

    model = Model(model_data='s3://mybucket/model.tar.gz', role='MySageMakerRole')

    predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge', accelerator_type='ml.eia1.medium')

Making predictions against a SageMaker Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Other input formats
^^^^^^^^^^^^^^^^^^^

SageMaker's TensforFlow Serving endpoints can also accept some additional input formats that are not part of the
TensorFlow REST API, including a simplified json format, line-delimited json objects ("jsons" or "jsonlines"), and
CSV data.

**Simplified JSON Input**

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


**Line-delimited JSON**

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

  predictor = Predictor('endpoint-name', serializer=sagemaker.predictor.csv_serializer)

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

You can also use python arrays or numpy arrays as input and let the `csv_serializer` object
convert them to CSV, but the client-size CSV conversion is more sophisticated than the
CSV parsing on the Endpoint, so if you encounter conversion problems, try using one of the
JSON options instead.


Specifying the output of a prediction request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The structure of the prediction ``result`` is determined at the end of the training process before SavedModel is created. For example, if
you are using TensorFlow's Estimator API for training, you control inference outputs using the ``export_outputs`` parameter of the `tf.estimator.EstimatorSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec>`_ that you return from
your ``model_fn`` (see `Example of a complete model_fn`_ for an example of ``export_outputs``).

More information on how to create ``export_outputs`` can be found in `specifying the outputs of a custom model <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/programmers_guide/saved_model.md#specifying-the-outputs-of-a-custom-model>`_. You can also
refer to TensorFlow's `Save and Restore <https://www.tensorflow.org/guide/saved_model>`_ documentation for other ways to control the
inference-time behavior of your SavedModels.

Providing Python scripts for pre/pos-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can add your customized Python code to process your input and output data:

.. code::

    from sagemaker.tensorflow.serving import Model

    model = Model(entry_point='inference.py',
                  model_data='s3://mybucket/model.tar.gz',
                  role='MySageMakerRole')

How to implement the pre- and/or post-processing handler(s)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your entry point file should implement either a pair of ``input_handler``
   and ``output_handler`` functions or a single ``handler`` function.
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

1. If you included ``requirements.txt`` in your ``source_dir`` or in
   your dependencies, the container installs the Python dependencies at runtime using ``pip install -r``:

.. code::

    from sagemaker.tensorflow.serving import Model

    model = Model(entry_point='inference.py',
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

    from sagemaker.tensorflow.serving import Model

    model = Model(entry_point='inference.py',
                  dependencies=['/path/to/folder/named/lib'],
                  model_data='s3://mybucket/model.tar.gz',
                  role='MySageMakerRole')


Deploying more than one model to your Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow Serving Endpoints allow you to deploy multiple models to the same Endpoint when you create the endpoint.

To use this feature, you will need to:

#. create a multi-model archive file
#. create a SageMaker Model and deploy it to an Endpoint
#. create Predictor instances that direct requests to a specific model

Creating a multi-model archive file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating an archive file that contains multiple SavedModels is simple, but involves a few
steps:

- obtaining some models
- repackaging the models into a new archive file
- uploading the new archive to S3

**Obtaining model files**

Let's imagine you have already run two Tensorflow training jobs in SageMaker, and they exported
SavedModels to ``s3://mybucket/models/model1.tar.gz`` and ``s3://mybucket/models/model2.tar.gz``.

First, download the models and extract them:

.. code:: bash

  aws s3 cp s3://mybucket/models/model1/model.tar.gz model1.tar.gz
  aws s3 cp s3://mybucket/models/model2/model.tar.gz model2.tar.gz
  mkdir -p multi/model1
  mkdir -p multi/model2

  tar xvf model1.tar.gz -C ./multi/model1
  tar xvf model2.tar.gz -C ./multi/model2

**Repackaging the models**

Next, examine the directories in ``multi``. If you trained the models using SageMaker's TensorFlow containers,
you are likely to have ``./multi/model1/export/Servo/...`` and ``./multi/model2/export/Servo/...``. In both cases,
"Servo" is the base name for the SaveModel files. When serving multiple models, each model needs a unique
basename, so one or both of these will need to be changed. The ``/export/`` part of the path isn't needed
either, so you can simplify the layout at the same time:

.. code:: bash

  mv multi/model1/export/Servo/* multi/model1/
  mv multi/model2/export/Servo/* multi/model2/
  rm -fr multi/model1/export
  rm -fr multi/model2/export

You should now have a directory structure like this:

::

  └── multi
    ├── model1
    │   └── <version number>
    │       ├── saved_model.pb
    │       └── variables
    │           └── ...
    └── model2
        └── <version number>
            ├── saved_model.pb
            └── variables
                └── ...

To repackage the files into a new archive, use ``tar`` again:

.. code:: bash

  tar -C "$PWD/multi/" -czvf multi.tar.gz multi/

The ``multi.tar.gz`` file is now ready to use.

**Uploading the new archive to S3**

.. code:: bash

  aws s3 cp multi.tar.gz s3://mybucket/models/multi.tar.gz

Creating and Deploying a SageMaker Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the remaining steps, let's return to python code using the SageMaker Python SDK.

.. code:: python

  from sagemaker.tensorflow.serving import Model, Predictor

  # change this to the name or ARN of your SageMaker execution role
  role = 'SageMakerRole'

  model_data = 's3://mybucket/models/multi.tar.gz'

  # For multi-model endpoints, you should set the default model name in
  # an environment variable. If it isn't set, the endpoint will work,
  # but the model it will select as default is unpredictable.
  env = {
    'SAGEMAKER_TFS_DEFAULT_MODEL_NAME': 'model1'
  }

  model = Model(model_data=model_data, role=role, framework_version='1.11', env=env)
  predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')

The ``predictor`` object returned by the deploy function is ready to use to make predictions
using the default model (``model1`` in this example).

Creating Predictor instances for different models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``predictor`` returned by the ``model.deploy(...)`` function can only send requests to
the default model. To use other models deployed to the same Endpoint, you need to create
additional ``Predictor`` instances. Here's how:

.. code:: python

  # ... continuing from the previous example

  # get the endpoint name from the default predictor
  endpoint = predictor.endpoint

  # get a predictor for 'model2'
  model2_predictor = Predictor(endpoint, model_name='model2')

  # note: that will for actual SageMaker endpoints, but if you are using
  # local mode you need to create the new Predictor this way:
  #
  # model2_predictor = Predictor(endpoint, model_name='model2'
  #                              sagemaker_session=predictor.sagemaker_session)


  # result is prediction from 'model2'
  result = model2_predictor.predict(...)

Making predictions with the AWS CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker Python SDK is not the only way to access your Endpoint. The AWS CLI is simple to
use and a convenient way to test your endpoint. Here are a few examples that show how to use
different features of SageMaker TensorFlow Serving Endpoints using the CLI.

Note: The ``invoke-endpoint`` command usually writes prediction results to a file.  In the examples
below, the ``>(cat) 1>/dev/null`` part is a shell trick to redirect the result to stdout so it
can be seen.

.. code:: bash

  # TensorFlow Serving REST API - predict request
  aws sagemaker-runtime invoke-endpoint \
      --endpoint-name my-endpoint \
      --content-type 'application/json' \
      --body '{"instances": [1.0, 2.0, 5.0]}' \
      >(cat) 1>/dev/null

  # Predict request for specific model name
  aws sagemaker-runtime invoke-endpoint \
      --endpoint-name my-endpoint \
      --content-type 'application/json' \
      --body '{"instances": [1.0, 2.0, 5.0]}' \
      --custom-attributes 'tfs-model-name=other_model' \
      >(cat) 1>/dev/null

  # TensorFlow Serving REST API - regress request
  aws sagemaker-runtime invoke-endpoint \
      --endpoint-name my-endpoint \
      --content-type 'application/json' \
      --body '{"signature_name": "tensorflow/serving/regress","examples": [{"x": 1.0}]}' \
      --custom-attributes 'tfs-method=regress' \
      >(cat) 1>/dev/null

  # Simple json request (2 instances)
  aws sagemaker-runtime invoke-endpoint \
      --endpoint-name my-endpoint \
      --content-type 'application/json' \
      --body '[[1.0, 2.0, 5.0],[2.0, 3.0, 4.0]]' \
      >(cat) 1>/dev/null

  # CSV request (2 rows)
  aws sagemaker-runtime invoke-endpoint \
      --endpoint-name my-endpoint \
      --content-type 'text/csv' \
      --body "1.0,2.0,5.0"$'\n'"2.0,3.0,4.0" \
      >(cat) 1>/dev/null

  # Line delimited JSON from an input file
  aws sagemaker-runtime invoke-endpoint \
      --endpoint-name my-endpoint \
      --content-type 'application/jsons' \
      --body "$(cat input.jsons)" \
      results.json
