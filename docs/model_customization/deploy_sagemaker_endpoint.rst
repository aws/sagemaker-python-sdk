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
