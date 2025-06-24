.. _inference-with-hyperpod:

Inference with SageMaker HyperPod
=================================

This guide explains how to deploy, invoke, and manage inference endpoints on SageMaker HyperPod clusters using both the CLI and Python SDK.

.. note::

   This guide applies to HyperPod CLI v0.5+ and SDK v0.3+.
   Run ``hp version`` or ``pip show sagemaker`` to check your versions.

Cluster Setup
-------------

Before deploying any endpoints, ensure your cluster context is properly configured.

.. tabs::

   .. tab:: CLI

      .. code-block:: bash

         # List available clusters
         hp cluster list

         # Set the active cluster context
         hp cluster set-context <cluster-name>

         # View current context
         hp cluster get-context

   .. tab:: Python SDK

      .. code-block:: python

         from hyperpod.hyperpod_manager import HyperPodManager

         hyperpod_manager = HyperPodManager()
         hyperpod_manager.list_clusters()
         hyperpod_manager.set_cluster(cluster_name="<cluster-name>")
         hyperpod_manager.get_context()

Deploying JumpStart Models
--------------------------

Use JumpStart models to quickly deploy popular open-source models without managing Docker images or source locations.

.. tabs::

   .. tab:: Create JumpStart Endpoint (CLI)

      .. code-block:: bash

         # Create with minimal inputs
         hp hp-jumpstart-endpoint create \
           --model-id huggingface-llm-falcon-7b \
           --instance-type ml.g5.2xlarge

         # Interactive config editing
         hp hp-jumpstart-endpoint create \
           --model-id huggingface-llm-falcon-7b \
           --instance-type ml.g5.2xlarge -i

      .. note::

         The interactive mode opens a YAML template with editable fields like `model_id`, `instance_type`, `namespace`, and more.

      .. code-block:: bash

         # List, describe, or delete endpoints
         hp hp-jumpstart-endpoint list
         hp hp-jumpstart-endpoint describe <endpoint-name>
         hp hp-jumpstart-endpoint delete <endpoint-name>

   .. tab:: Create JumpStart Endpoint (SDK)

      .. code-block:: python

         from hyperpod.inference.config.jumpstart_model_endpoint_config import (
             Model, Server, SageMakerEndpoint, JumpStartModelSpec
         )
         from hyperpod.inference.hp_jumpstart_endpoint import HPJumpStartEndpoint

         model = Model(model_id="sklearn-regression-linear")
         server = Server(instance_type="ml.t3.medium")
         endpoint_name = SageMakerEndpoint(name="my-endpoint")

         spec = JumpStartModelSpec(model=model, server=server, sage_maker_endpoint=endpoint_name)

         endpoint = HPJumpStartEndpoint()
         endpoint.create_from_spec(spec)

      .. code-block:: python

         # Quick job creation from inputs
         endpoint = HPJumpStartEndpoint()
         endpoint.create(
             namespace="default",
             model_id="sklearn-regression-linear",
             instance_type="ml.t3.medium"
         )

      .. code-block:: python

         # List, describe, or delete
         endpoint.list_endpoints(namespace="default")
         endpoint.describe_endpoint(name="my-endpoint", namespace="default")
         endpoint.delete_endpoint(name="my-endpoint", namespace="default")

Deploying Custom Models
-----------------------

Use this approach when hosting your own models packaged in a custom Docker image and stored in S3 or FSx.

.. tabs::

   .. tab:: Create Custom Endpoint (CLI)

      .. code-block:: bash

         hp hp-endpoint create \
           --model-name custom-bert \
           --image <image-uri> \
           --container-port 8080 \
           --instance-type ml.g5.xlarge \
           --model-source-type s3 \
           --bucket-name my-bucket \
           --bucket-region us-west-2

   .. tab:: Create Custom Endpoint (SDK)

      .. code-block:: python

         from hyperpod.inference.hp_endpoint import HPEndpoint
         from hyperpod.inference.config.inference_endpoint_config import (
             InferenceEndpointConfigSpec, ModelSourceConfig, S3Storage
         )

         model_source = ModelSourceConfig(
             model_source_type='s3',
             s3_storage=S3Storage(bucket_name='my-bucket', region='us-west-2')
         )

         spec = InferenceEndpointConfigSpec(
             endpoint_name='my-endpoint',
             instance_type='ml.t3.medium',
             model_name='custom-bert',
             image='image-uri',
             container_port=8080,
             model_source_config=model_source
         )

         endpoint = HPEndpoint()
         endpoint.create_from_spec(spec)

      .. code-block:: python

         # Simpler version using raw inputs
         endpoint = HPEndpoint()
         endpoint.create(
             namespace="default",
             model_name="custom-bert",
             instance_type="ml.t3.medium",
             image="image-uri",
             container_port=8080,
             model_source_type="s3",
             bucket_name="my-bucket",
             bucket_region="us-west-2"
         )

Invoking Endpoints
------------------

Send inference requests once the endpoint is active.

.. tabs::

   .. tab:: Invoke via CLI

      .. code-block:: bash

         hp hp-jumpstart-endpoint invoke <endpoint-name> \
           --body '{"inputs": ["hello world"]}'

   .. tab:: Invoke via SDK

      .. code-block:: python

         import json

         payload = json.dumps({"inputs": ["Hello", "Goodbye"]})
         response = endpoint.invoke(body=payload)
         print(response)

CLI Configuration Reference
---------------------------

The following options apply across CLI commands for inference endpoints.

**Identification & Namespace**

- ``--namespace`` *(Optional)*: Kubernetes namespace
- ``--model-name`` *(Required)*: Identifier for custom model
- ``--model-id`` *(Required for JumpStart models)*

**Infrastructure**

- ``--instance-type`` *(Required)*: e.g., ml.g5.xlarge
- ``--container-port`` *(Required)*: Container port exposed
- ``--image`` *(Required for custom models)*: Inference image URI

**Model Source**

- ``--model-source-type`` *(Required)*: s3 \| fsx
- ``--config-file`` *(Optional)*: YAML spec file path

**S3 Storage**

- ``--bucket-name`` *(Required if using s3)*
- ``--bucket-region`` *(Required if using s3)*

**FSx Storage**

- ``--fsx-dns-name`` *(Required if using fsx)*
- ``--fsx-file-system-id`` *(Required if using fsx)*
- ``--fsx-mount-name`` *(Required if using fsx)*

