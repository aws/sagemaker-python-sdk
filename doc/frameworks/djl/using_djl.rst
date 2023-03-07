#######################################
Use DJL with the SageMaker Python SDK
#######################################

With the SageMaker Python SDK, you can use Deep Java Library to host models on Amazon SageMaker.

`Deep Java Library (DJL) Serving <https://docs.djl.ai/docs/serving/index.html>`_ is a high performance universal stand-alone model serving solution powered by `DJL <https://docs.djl.ai/index.html>`_.
DJL Serving supports loading models trained with a variety of different frameworks. With the SageMaker Python SDK you can
use DJL Serving to host large models using backends like DeepSpeed and HuggingFace Accelerate.

For information about supported versions of DJL Serving, see the `AWS documentation <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`_.
We recommend that you use the latest supported version because that's where we focus our development efforts.

For general information about using the SageMaker Python SDK, see :ref:`overview:Using the SageMaker Python SDK`.

.. contents::

*******************
Deploy DJL models
*******************

With the SageMaker Python SDK, you can use DJL Serving to host models that have been saved in the HuggingFace pretrained format.
These can either be models you have trained/fine-tuned yourself, or models available publicly from the HuggingFace Hub.
DJL Serving in the SageMaker Python SDK supports hosting models for the popular HuggingFace NLP tasks, as well as Stable Diffusion.

You can either deploy your model using DeepSpeed or HuggingFace Accelerate, or let DJL Serving determine the best backend based on your model architecture and configuration.

.. code:: python

    # Create a DJL Model, backend is chosen automatically
    djl_model = DJLModel(
        "s3://my_bucket/my_saved_model_artifacts/",
        "my_sagemaker_role",
        data_type="fp16",
        task="text-generation",
        number_of_partitions=2 # number of gpus to partition the model across
    )

    # Deploy the model to an Amazon SageMaker Endpoint and get a Predictor
    predictor = djl_model.deploy("ml.g5.12xlarge",
                                 initial_instance_count=1)

If you want to use a specific backend, then you can create an instance of the corresponding model directly.

.. code:: python

    # Create a model using the DeepSpeed backend
    deepspeed_model = DeepSpeedModel(
        "s3://my_bucket/my_saved_model_artifacts/",
        "my_sagemaker_role",
        data_type="bf16",
        task="text-generation",
        tensor_parallel_degree=2, # number of gpus to partition the model across using tensor parallelism
    )

    # Create a model using the HuggingFace Accelerate backend

    hf_accelerate_model = HuggingFaceAccelerateModel(
        "s3://my_bucket/my_saved_model_artifacts/",
        "my_sagemaker_role",
        data_type="fp16",
        task="text-generation",
        number_of_partitions=2, # number of gpus to partition the model across
    )

    # Deploy the model to an Amazon SageMaker Endpoint and get a Predictor
    deepspeed_predictor = deepspeed_model.deploy("ml.g5.12xlarge",
                                                 initial_instance_count=1)
    hf_accelerate_predictor = hf_accelerate_model.deploy("ml.g5.12xlarge",
                                                         initial_instance_count=1)

Regardless of which way you choose to create your model, a ``Predictor`` object is returned. You can use this ``Predictor``
to do inference on the endpoint hosting your DJLModel.

Each ``Predictor`` provides a ``predict`` method, which can do inference with json data, numpy arrays, or Python lists.
Inference data are serialized and sent to the DJL Serving model server by an ``InvokeEndpoint`` SageMaker operation. The
``predict`` method returns the result of inference against your model.

By default, the inference data is serialized to a json string, and the inference result is a Python dictionary.

Model Directory Structure
=========================

There are two components that are needed to deploy DJL Serving Models on Sagemaker.
1. Model Artifacts (required)
2. Inference code and Model Server Properties (optional)

These are stored and handled separately. Model artifacts should not be stored with the custom inference code and
model server configuration.

Model Artifacts
---------------

DJL Serving Models expect a different model structure than most of the other frameworks in the SageMaker Python SDK.
Specifically, DJLModels do not support loading models stored in tar.gz format.
You must provide an Amazon S3 url pointing to uncompressed model artifacts (bucket and prefix).
This is because DJL Serving is optimized for large models, and it implements a fast downloading mechanism for large models that require the artifacts be uncompressed.

For example, lets say you want to deploy the EleutherAI/gpt-j-6B model available on the HuggingFace Hub.
You can download the model and upload to S3 like this:

.. code::

    # Requires Git LFS
    git clone https://huggingface.co/EleutherAI/gpt-j-6B

    # Upload to S3
    aws s3 sync gpt-j-6B s3://my_bucket/gpt-j-6B

You would then pass "s3://my_bucket/gpt-j-6B" as ``model_s3_uri`` to the ``DJLModel``.

For language models we expect that the model weights, model config, and tokenizer config are provided in S3. The model
should be loadable from the HuggingFace Transformers AutoModelFor<Task>.from_pretrained API, where task
is the NLP task you want to host the model for. The weights must be stored as PyTorch compatible checkpoints.

Example:

.. code::

    my_bucket/my_model/
    |- config.json
    |- added_tokens.json
    |- config.json
    |- pytorch_model-*-of-*.bin # model weights can be partitioned into multiple checkpoints
    |- tokenizer.json
    |- tokenizer_config.json
    |- vocab.json

For Stable Diffusion models, the model should be loadable from the HuggingFace Diffusers DiffusionPipeline.from_pretrained API.

Inference code and Model Server Properties
------------------------------------------

You can provide custom inference code and model server configuration by specifying the ``source_dir`` and
``entry_point`` arguments of the ``DJLModel``. These are not required. The model server configuration can be generated
based on the arguments passed to the constructor, and we provide default inference handler code for DeepSpeed,
HuggingFaceAccelerate, and Stable Diffusion. You can find these handler implementations in the `DJL Serving Github repository. <https://github.com/deepjavalibrary/djl-serving/tree/master/engines/python/setup/djl_python>`_

You can find documentation for the model server configurations on the `DJL Serving Docs website <https://docs.djl.ai/docs/serving/serving/docs/configurations.html>`_.

The code and configuration you want to deploy can either be stored locally or in S3. These files will be bundled into
a tar.gz file that will be uploaded to SageMaker.

For example:

.. code::

    sourcedir/
    |- script.py # Inference handler code
    |- serving.properties # Model Server configuration file
    |- requirements.txt # Additional Python requirements that will be installed at runtime via PyPi

In the above example, sourcedir will be bundled and compressed into a tar.gz file and uploaded as part of creating the Inference Endpoint.

The DJL Serving Model Server
============================

The endpoint you create with ``deploy`` runs the DJL Serving model server.
The model server loads the model from S3 and performs inference on the model in response to SageMaker ``InvokeEndpoint`` API calls.

DJL Serving is highly customizable. You can control aspects of both model loading and model serving. Most of the model server
configuration are exposed through the ``DJLModel`` API. The SageMaker Python SDK will use the values it is passed to
create the proper configuration file used when creating the inference endpoint. You can optionally provide your own
``serving.properties`` file via the ``source_dir`` argument. You can find documentation about serving.properties in the
`DJL Serving Documentation for model specific settings. <https://docs.djl.ai/docs/serving/serving/docs/configurations.html#model-specific-settings>`_

Within the SageMaker Python SDK, DJL Serving is used in Python mode. This allows users to provide their inference script,
and data processing scripts in python. For details on how to write custom inference and data processing code, please
see the `DJL Serving Documentation on Python Mode. <https://docs.djl.ai/docs/serving/serving/docs/modes.html#python-mode>`_

For more information about DJL Serving, see the `DJL Serving documentation. <https://docs.djl.ai/docs/serving/index.html>`_

***********************
SageMaker DJL Classes
***********************

For information about the different DJL Serving related classes in the SageMaker Python SDK, see https://sagemaker.readthedocs.io/en/stable/sagemaker.djl_inference.html.

********************************
SageMaker DJL Serving Containers
********************************

For information about the SageMaker DJL Serving containers, see:

- `Deep Learning Container (DLC) Images <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`_ and `release notes <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html>`_
