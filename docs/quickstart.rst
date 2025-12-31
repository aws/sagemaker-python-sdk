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

   pip install sagemaker>=3.0.0

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

Train a simple model using the unified ModelTrainer:

.. code-block:: python

   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import InputData
   
   # Create a ModelTrainer
   trainer = ModelTrainer(
       training_image="382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest",
       role=role
   )
   
   # Configure training data
   train_data = InputData(
       channel_name="training",
       data_source="s3://sagemaker-sample-data-us-east-1/xgboost/census-income/train.csv"
   )
   
   # Start training
   training_job = trainer.train(
       input_data_config=[train_data],
       hyperparameters={
           "objective": "binary:logistic",
           "num_round": "100"
       }
   )
   
   print(f"Training job: {training_job.name}")

Deploying Your Model
--------------------

Deploy the trained model for inference:

.. code-block:: python

   from sagemaker.serve import ModelBuilder
   
   # Create a ModelBuilder from the training job
   model_builder = ModelBuilder(
       model_data=training_job.model_artifacts,
       image_uri="382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest",
       role=role
   )
   
   # Deploy to an endpoint
   endpoint = model_builder.build(
       instance_type="ml.m5.large",
       initial_instance_count=1
   )
   
   print(f"Endpoint: {endpoint.endpoint_name}")

Making Predictions
------------------

Use your deployed model to make predictions:

.. code-block:: python

   # Sample data for prediction
   test_data = "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States"
   
   # Make a prediction
   result = endpoint.invoke(test_data, content_type="text/csv")
   print(f"Prediction: {result}")

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
