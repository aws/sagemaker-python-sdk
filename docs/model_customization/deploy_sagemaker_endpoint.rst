Deploy Models to SageMaker Endpoint
======================================

Deploy your fine-tuned model to a SageMaker real-time endpoint for low-latency inference.


Deploy from an S3 Checkpoint
------------------------------

Manually specify the S3 prefix when you have a raw checkpoint path (e.g., from an escrow bucket).
This gives you explicit control over the inference image URI and environment variables.

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder

   model_builder = ModelBuilder(
       s3_model_data_url={
           "S3DataSource": {
               "S3Uri": "s3://customer-escrow-.../checkpoints/step_10/",
               "S3DataType": "S3Prefix",
               "CompressionType": "None",
           }
       },
       image_uri="<your-inference-image-uri>",
       instance_type="ml.g5.48xlarge",
       role_arn="arn:aws:iam::123456789012:role/MySageMakerRole",
       env_vars={
           "CONTEXT_LENGTH": "8192",
           "MAX_CONCURRENCY": "4",
       },
   )

   model_builder.model_name = "my-finetuned-model"
   model = model_builder.build()
   endpoint = model_builder.deploy(endpoint_name="my-endpoint", wait=False)

When deploying Nova models, set ``CONTEXT_LENGTH`` and ``MAX_CONCURRENCY`` via ``env_vars``
to control the maximum input context window and concurrent request capacity. Values are
validated at build time against per-(model, instance) tier bounds.


Deploy from a TrainingJob
---------------------------

Pass a ``TrainingJob`` object directly — the SDK extracts the S3 model path automatically.

.. code-block:: python

   from sagemaker.core.resources import TrainingJob
   from sagemaker.serve import ModelBuilder

   training_job = TrainingJob.get(training_job_name="my-sft-job")

   model_builder = ModelBuilder(model=training_job)
   model = model_builder.build(model_name="my-finetuned-model")
   endpoint = model_builder.deploy(endpoint_name="my-endpoint")


Deploy from a ModelPackage
----------------------------

Pass a versioned ``ModelPackage`` from the SageMaker Model Registry for governed,
production deployments.

.. code-block:: python

   from sagemaker.core.resources import ModelPackage
   from sagemaker.serve import ModelBuilder

   model_package = ModelPackage.get(
       model_package_name="arn:aws:sagemaker:us-east-1:123456789012:model-package/my-models/3"
   )

   model_builder = ModelBuilder(model=model_package)
   model = model_builder.build(model_name="my-registered-model")
   endpoint = model_builder.deploy(endpoint_name="my-endpoint")


Reuse Deployed Resources
--------------------------

Pass ``reuse_resources=True`` to ``build()`` and ``deploy()`` to avoid creating duplicate
endpoints when deploying the same model source multiple times.

On first deploy, the SDK tags the endpoint with ``sagemaker.amazonaws.com/model-source``
derived from the model's stable source identifier (model package ARN, escrow URI, S3 path,
or JumpStart model ID). On subsequent deploys with ``reuse_resources=True``, the SDK discovers
the existing endpoint by that tag and returns it instead of creating a new one.

.. list-table::
   :header-rows: 1

   * - Model input style
     - Source ID used for tag
   * - ``ModelPackage``
     - Model package ARN
   * - ``BaseTrainer`` (with completed training job)
     - Model package ARN if available, otherwise escrow resolution
   * - ``TrainingJob``
     - Via model package ARN or escrow resolution
   * - Raw S3 URI string
     - The S3 path itself
   * - JumpStart model ID string
     - The model ID string

.. code-block:: python

   from sagemaker.serve import ModelBuilder

   builder = ModelBuilder(
       model=my_trainer,  # or ModelPackage, TrainingJob, S3 URI, etc.
       role_arn="arn:aws:iam::123456789012:role/MySageMakerRole",
       instance_type="ml.p4d.24xlarge",
       image_uri="my-inference-image:latest",
   )

   # build() checks for an existing Model with matching source tag
   builder.build(region="us-east-1", reuse_resources=True)

   # deploy() checks for an existing Endpoint with matching source tag
   endpoint = builder.deploy(
       endpoint_name="my-endpoint",
       instance_type="ml.p4d.24xlarge",
       reuse_resources=True,
   )
   # If a match is found: returns the existing endpoint
   # If no match: creates a new endpoint as normal

Without ``reuse_resources=True`` (the default), every deploy creates a new endpoint. The
model-source tag is still applied so that future deploys with reuse enabled can discover it.

.. note::

   The ``reuse_resources`` flag must be passed to each call independently — it is not
   inherited between ``build()`` and ``deploy()``.

   Inference component builds (``modelbuilder_list``) manage their own reuse by component
   name and bypass the endpoint-return reuse gate.
