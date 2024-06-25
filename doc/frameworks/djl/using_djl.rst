#######################################
Use DJL with the SageMaker Python SDK
#######################################

`Deep Java Library (DJL) Serving <https://docs.djl.ai/docs/serving/index.html>`_ is a high performance universal stand-alone model serving solution powered by `DJL <https://docs.djl.ai/index.html>`_.
DJL Serving supports loading models trained with a variety of different frameworks. With the SageMaker Python SDK you can
use DJL Serving to host large language models for text-generation and text-embedding use-cases.

You can learn more about Large Model Inference using DJLServing on the `docs site <https://docs.djl.ai/docs/serving/serving/docs/lmi/index.html>`_.

For general information about using the SageMaker Python SDK, see :ref:`overview:Using the SageMaker Python SDK`.

.. contents::

*******************
Deploy DJL models
*******************

With the SageMaker Python SDK, you can use DJL Serving to host text-generation and text-embedding models that have been saved in the HuggingFace pretrained format.
These can either be models you have trained/fine-tuned yourself, or models available publicly from the HuggingFace Hub.

.. code:: python

    # DJLModel will infer which container to use, and apply some starter configuration
    djl_model = DJLModel(
        model_id="<hf hub id | s3 uri>",
        role="my_sagemaker_role",
        task="text-generation",
    )

    # Deploy the model to an Amazon SageMaker Endpoint and get a Predictor
    predictor = djl_model.deploy("ml.g5.12xlarge",
                                 initial_instance_count=1)

Alternatively, you can provide full specifications to the DJLModel to have full control over the model configuration:

.. code:: python

    djl_model = DJLModel(
        model_id="<hf hub id | s3 uri>",
        role="my_sagemaker_role",
        task="text-generation",
        engine="Python",
        env={
            "OPTION_ROLLING_BATCH": "lmi-dist",
            "TENSOR_PARALLEL_DEGREE": "2",
            "OPTION_DTYPE": "bf16",
            "OPTION_MAX_ROLLING_BATCH_SIZE": "64",
        },
        image_uri=<djl lmi image uri>,
    )
    # Deploy the model to an Amazon SageMaker Endpoint and get a Predictor
    predictor = djl_model.deploy("ml.g5.12xlarge",
                                 initial_instance_count=1)

Regardless of how you create your model, a ``Predictor`` object is returned.
Each ``Predictor`` provides a ``predict`` method, which can do inference with json data, numpy arrays, or Python lists.
Inference data are serialized and sent to the DJL Serving model server by an ``InvokeEndpoint`` SageMaker operation. The
``predict`` method returns the result of inference against your model.

By default, the inference data is serialized to a json string, and the inference result is a Python dictionary.

**************************************
DJL Serving for Large Model Inference
**************************************

You can learn more about using DJL Serving for Large Model Inference use-cases on our `documentation site <https://docs.djl.ai/docs/serving/serving/docs/lmi/index.html>`_.



********************************
SageMaker DJL Serving Containers
********************************

For information about the SageMaker DJL Serving containers, see:

- `Deep Learning Container (DLC) Images <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`_ and `release notes <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html>`_
