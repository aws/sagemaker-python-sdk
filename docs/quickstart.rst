Quickstart
===========

Get started with SageMaker Python SDK V3 in minutes. This guide walks you through the essential steps to train and deploy your first model.

Prerequisites
-------------

* Python 3.9+ installed
* AWS account with appropriate permissions
* AWS credentials configured

Installation
------------

Install SageMaker Python SDK V3:

.. code-block:: bash

   pip install sagemaker

Basic Setup
-----------

Import the SDK and create a session:

.. code-block:: python

   import sagemaker
   from sagemaker.train import ModelTrainer
   from sagemaker.serve import ModelBuilder
   
   # Create a SageMaker session
   session = sagemaker.Session()
   role = sagemaker.get_execution_role()  # Or specify your IAM role ARN
   
   print(f"Using role: {role}")
   print(f"Default bucket: {session.default_bucket()}")

Training Your First Model
-------------------------

Train a custom PyTorch model using the unified ModelTrainer:

.. code-block:: python

   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import SourceCode
   
   # Create ModelTrainer with custom code
   trainer = ModelTrainer(
       training_image="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-cpu-py39",
       source_code=SourceCode(
           source_dir="./training_code",
           entry_script="train.py",
           requirements="requirements.txt"
       ),
       role=role
   )
   
   # Start training
   trainer.train()
   print(f"Training completed: {training_job.name}")

Deploying Your Model
--------------------

Deploy the trained model using the V3 workflow: build() → deploy() → invoke():

.. code-block:: python

   from sagemaker.serve import ModelBuilder
   from sagemaker.serve.builder.schema_builder import SchemaBuilder
   from sagemaker.serve.utils.types import ModelServer
   
   # Create schema for model input/output
   sample_input = [[0.1, 0.2, 0.3, 0.4]]
   sample_output = [[0.8, 0.2]]
   schema_builder = SchemaBuilder(sample_input, sample_output)
   
   # Create ModelBuilder from training job
   model_builder = ModelBuilder(
       model=trainer,
       schema_builder=schema_builder,
       model_server=ModelServer.TORCHSERVE,
       role=role
   )
   
   # Build the model
   model = model_builder.build(model_name="my-pytorch-model")
   
   # Deploy to endpoint
   endpoint = model_builder.deploy(
       endpoint_name="my-endpoint",
       instance_type="ml.m5.large",
       initial_instance_count=1
   )
   
   print(f"Endpoint deployed: {endpoint.endpoint_name}")

Making Predictions
------------------

Use your deployed model to make predictions:

.. code-block:: python

   import json
   
   # Sample tensor data for prediction
   test_data = [[0.5, 0.3, 0.2, 0.1]]
   
   # Make a prediction
   result = endpoint.invoke(
       body=json.dumps(test_data),
       content_type="application/json"
   )
   
   # Parse the result
   prediction = json.loads(result.body.read().decode('utf-8'))
   print(f"Prediction: {prediction}")

Cleanup
-------

Don't forget to clean up resources to avoid charges:

.. code-block:: python

   # Delete the endpoint
   endpoint.delete()
   
   print("Endpoint deleted")

Foundation Model Fine-Tuning
----------------------------

Try V3's new foundation model fine-tuning capabilities:

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.common import TrainingType
   
   # Fine-tune a foundation model
   sft_trainer = SFTTrainer(
       model="huggingface-textgeneration-gpt2",
       training_type=TrainingType.LORA,
       training_dataset="s3://your-bucket/training-data.jsonl",
       role=role
   )
   
   # Start fine-tuning
   fine_tuning_job = sft_trainer.train()
   print(f"Fine-tuning job: {fine_tuning_job.name}")

Next Steps
----------

Now that you've completed the quickstart:

1. **Explore Training**: Learn more about :doc:`training/index` capabilities
2. **Try Inference**: Discover advanced :doc:`inference/index` features  
3. **Model Customization**: Experiment with :doc:`model_customization/index`
4. **Build Pipelines**: Create workflows with :doc:`ml_ops/index`
5. **Use SageMaker Core**: Access low-level resources with :doc:`sagemaker_core/index`

Common Issues
-------------

**ImportError**: Ensure you have the latest version installed
**Credential errors**: Run ``aws configure`` to set up credentials
**Permission denied**: Check your IAM role has SageMaker permissions

For detailed troubleshooting, see the :doc:`installation` guide.
