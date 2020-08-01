######################################
Upgrade from Legacy TensorFlow Support
######################################

With version 2.0 and later of the SageMaker Python SDK, support for legacy SageMaker TensorFlow images has been deprecated.
This guide explains how to upgrade your SageMaker Python SDK usage.

For more information about using TensorFlow with the SageMaker Python SDK, see `Use TensorFlow with the SageMaker Python SDK <using_tf.html>`_.

.. contents::

********************************************
What Constitutes "Legacy TensorFlow Support"
********************************************

This guide is relevant if one of the following applies:

#. You are using TensorFlow versions 1.4-1.10
#. You are using TensorFlow versions 1.11-1.12 with Python 2, and

   - you do *not* have ``script_mode=True`` when creating your estimator
   - you are using ``sagemaker.tensorflow.model.TensorFlowModel`` and/or ``sagemaker.tensorflow.model.TensorFlowPredictor``

#. You are using a pre-built SageMaker image whose URI looks like ``520713654638.dkr.ecr.<region>.amazonaws.com/sagemaker-tensorflow:<tag>``

If one of the above applies, then keep reading.

**************
How to Upgrade
**************

We recommend that you use the latest supported version of TensorFlow because that's where we focus our development efforts.
For information about supported versions of TensorFlow, see the `AWS documentation <https://aws.amazon.com/releasenotes/available-deep-learning-containers-images>`_.

For general information about using TensorFlow with the SageMaker Python SDK, see `Use TensorFlow with the SageMaker Python SDK <using_tf.html>`_.

Training Script
===============

Newer versions of TensorFlow require your training script to be runnable as a command-line script, similar to what you might run outside of SageMaker. For more information, including how to adapt a locally-runnable script, see `Prepare a Training Script <using_tf.html#id1>`_.

In addition, your training script needs to save your model. If you have your own ``serving_input_fn`` implementation, then that can be passed to an exporter:

.. code:: python

    import tensorflow as tf

    exporter = tf.estimator.LatestExporter("Servo", serving_input_receiver_fn=serving_input_fn)

For an example of how to repackage your legacy TensorFlow training script for use with a newer version of TensorFlow,
see `this example notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_moving_from_framework_mode_to_script_mode/tensorflow_moving_from_framework_mode_to_script_mode.ipynb>`_.

Inference Script
================

Newer versions of TensorFlow Serving require a different format for the inference script. Some key differences:

- The script must be named ``inference.py``.
- ``input_fn`` has been replaced by ``input_handler``.
- ``output_fn`` has been replaced by ``output_handler``.

Like with the legacy versions, the pre-built SageMaker TensorFlow Serving images do have default implementations for pre- and post-processing.

For more information about implementing your own handlers, see `How to implement the pre- and/or post-processing handler(s) <using_tf.html#how-to-implement-the-pre-and-or-post-processing-handler-s>`_.

*****************************
Continue with Legacy Versions
*****************************

While not recommended, you can still use a legacy TensorFlow version with version 2.0 and later of the SageMaker Python SDK.
In order to do so, you need to change how a few parameters are defined.

Training
========

When creating an estimator, the Python SDK version 2.0 and later requires the following changes:

#. Explicitly specify the ECR image URI via ``image_uri``.
   To determine the URI, you can use :func:`sagemaker.fw_utils.create_image_uri`.
#. Specify ``model_dir=False``.
#. Use hyperparameters for ``training_steps``, ``evaluation_steps``, ``checkpoint_path``, and ``requirements_file``.

For example, if using TF 1.10.0 with an ml.m4.xlarge instance in us-west-2,
the difference in code would be as follows:

.. code:: python

    from sagemaker.tensorflow import TensorFlow

    # v1.x
    estimator = TensorFlow(
        ...
        source_dir="code",
        framework_version="1.10.0",
        train_instance_type="ml.m4.xlarge",
        training_steps=100,
        evaluation_steps=10,
        checkpoint_path="s3://bucket/path",
        requirements_file="requirements.txt",
    )

    # v2.0 and later
    estimator = TensorFlow(
        ...
        source_dir="code",
        framework_version="1.10.0",
        py_version="py2",
        instance_type="ml.m4.xlarge",
        image_uri="520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.10.0-cpu-py2",
        hyperparameters={
            "training_steps": 100,
            "evaluation_steps": 10,
            "checkpoint_path": "s3://bucket/path",
            "sagemaker_requirements": "requirements.txt",
        },
        model_dir=False,
    )

Requirements File with Training
-------------------------------

To provide a requirements file, define a hyperparameter named "sagemaker_requirements" that contains the relative path to the requirements file from ``source_dir``.

Inference
=========

Using a legacy TensorFlow version for endpoints and batch transform can be achieved with version 2.0 and later of the SageMaker Python SDK with some minor changes to your code.

From an Estimator
-----------------

If you are starting with a training job, you can call :func:`sagemaker.estimator.EstimatorBase.deploy` or :func:`sagemaker.tensorflow.estimator.Estimator.transformer` from your estimator for inference.

To specify the number of model server workers, you need to set it through an environment variable named ``MODEL_SERVER_WORKERS``:

.. code:: python

    # v1.x
    estimator.deploy(..., model_server_workers=4)

    # v2.0 and later
    estimator.deploy(..., env={"MODEL_SERVER_WORKERS": 4})

From a Trained Model
--------------------

If you are starting with a trained model, the Python SDK version 2.0 and later requires the following changes:

#. Use the the :class:`sagemaker.model.FrameworkModel` class.
#. Explicitly specify the ECR image URI via ``image_uri``.
   To determine the URI, you can use :func:`sagemaker.fw_utils.create_image_uri`.
#. Use an environment variable for ``model_server_workers``.

For example, if using TF 1.10.0 with a CPU instance in us-west-2,
the difference in code would be as follows:

.. code:: python

    # v1.x
    from sagemaker.tensorflow import TensorFlowModel

    model = TensorFlowModel(
        ...
        py_version="py2",
        framework_version="1.10.0",
        model_server_workers=4,
    )

    # v2.0 and later
    from sagemaker.model import FrameworkModel

    model = FrameworkModel(
        ...
        image_uri="520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.10.0-cpu-py2",
        env={"MODEL_SERVER_WORKERS": 4},
    )

Requirements File with Inference
--------------------------------

To provide a requirements file, define an environment variable named ``SAGEMAKER_REQUIREMENTS`` that contains the relative path to the requirements file from ``source_dir``.

From an estimator:

.. code:: python

    # for an endpoint
    estimator.deploy(..., env={"SAGEMAKER_REQUIREMENTS": "requirements.txt"})

    # for batch transform
    estimator.transformer(..., env={"SAGEMAKER_REQUIREMENTS": "requirements.txt"})

From a model:

.. code:: python

    from sagemaker.model import FrameworkModel

    model = FrameworkModel(
        ...
        source_dir="code",
        env={"SAGEMAKER_REQUIREMENTS": "requirements.txt"},
    )


Predictors
----------

If you want to use your model for endpoints, then you can use the :class:`sagemaker.predictor.Predictor` class instead of the legacy ``sagemaker.tensorflow.TensorFlowPredictor`` class:

.. code:: python

    from sagemaker.model import FrameworkModel
    from sagemaker.predictor import Predictor

    model = FrameworkModel(
        ...
        predictor_cls=Predictor,
    )

    predictor = model.deploy(...)

If you are using protobuf prediction data, then you need to serialize and deserialize the data yourself.

For example:

.. code:: python

    from google.protobuf import json_format
    from protobuf_to_dict import protobuf_to_dict
    from tensorflow.core.framework import tensor_pb2

    # Serialize the prediction data
    json_format.MessageToJson(data)

    # Get the prediction result
    result = predictor.predict(data)

    # Deserialize the prediction result
    protobuf_to_dict(json_format.Parse(result, tensor_pb2.TensorProto()))

Otherwise, you can use the serializers and deserialzers available in the SageMaker Python SDK or write your own.

For example, if you want to use JSON serialization and deserialization:

.. code:: python

    from sagemaker.deserializers import JSONDeserializer
    from sagemaker.serializers import JSONSerializer

    predictor = model.deploy(..., serializer=JSONSerializer(), deserializer=JSONDeserializer())

    predictor.predict(data)
