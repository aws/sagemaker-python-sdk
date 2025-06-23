Inference with SageMaker Hyperpod
=================================

This guide covers how to deploy and manage inference endpoints using both the CLI and Python SDK.

Setup
-----

.. tabs::

   .. tab:: CLI

      .. code-block:: bash

         # List available clusters
         hp cluster list

         # Set the cluster context
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


JumpStart Model
---------------

.. tabs::

   .. tab:: CLI

      **Create from Command**

      .. code-block:: bash

         hp hp-jumpstart-endpoint create \
           --model-id huggingface-llm-falcon-7b \
           --instance-type ml.g5.2xlarge

      **Create Interactively**

      .. code-block:: bash

         hp hp-jumpstart-endpoint create \
           --model-id huggingface-llm-falcon-7b \
           --instance-type ml.g5.2xlarge -i

      User will be prompted to edit a YAML config file with fields like model_id, instance_type, namespace, etc.

      **Other Commands**

      .. code-block:: bash

         hp hp-jumpstart-endpoint list
         hp hp-jumpstart-endpoint describe <endpoint-name>
         hp hp-jumpstart-endpoint delete <endpoint-name>

   .. tab:: Python SDK

      **Create from Spec**

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

      **Create from Inputs**

      .. code-block:: python

         endpoint = HPJumpStartEndpoint()
         endpoint.create(
             namespace="default",
             model_id="sklearn-regression-linear",
             instance_type="ml.t3.medium"
         )

      **Other Operations**

      .. code-block:: python

         endpoint.list_endpoints(namespace="default")
         endpoint.describe_endpoint(name="my-endpoint", namespace="default")
         endpoint.delete_endpoint(name="my-endpoint", namespace="default")


Custom Model
------------

.. tabs::

   .. tab:: CLI

      .. code-block:: bash

         hp hp-endpoint create \
           --model-name custom-bert \
           --image <image-uri> \
           --container-port 8080 \
           --instance-type ml.g5.xlarge \
           --model-source-type s3 \
           --bucket-name my-bucket \
           --bucket-region us-west-2

   .. tab:: Python SDK

      **Create from Spec**

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

      **Create from Inputs**

      .. code-block:: python

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


Invoke Endpoint
---------------

.. tabs::

   .. tab:: CLI

      .. code-block:: bash

         hp hp-jumpstart-endpoint invoke <endpoint-name> --body '{"inputs": ["hello world"]}'

   .. tab:: Python SDK

      .. code-block:: python

         import json

         payload = json.dumps({"inputs": ["Hello", "Goodbye"]})
         response = endpoint.invoke(body=payload)
         print(response)


CLI Configuration Options
-------------------------

**Identification & Namespace**

- --namespace: (Optional) Kubernetes namespace

- --model-name: (Required) Model identifier

- --model-id: (Required if no config file)

**Infrastructure**

- --instance-type: (Required) Instance type (e.g., ml.g5.xlarge)

- --container-port: (Required) Container port

- --image: (Required) Inference container image

**Model Source**

- --model-source-type: (Required) s3 or fsx

- --config-file: (Optional) Path to deployment YAML config

**S3 Configuration**

- --bucket-name: (Required if source is s3)

- --bucket-region: (Required if source is s3)

**FSX Configuration**

- --fsx-dns-name: (Required if source is fsx)

- --fsx-file-system-id: (Required if source is fsx)

- --fsx-mount-name: (Required if source is fsx)
