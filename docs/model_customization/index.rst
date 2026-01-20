Customizing Models
===================

.. raw:: html

   <div class="v3-exclusive-feature">
   <strong>🆕 V3 EXCLUSIVE FEATURE</strong><br>
   Model customization with specialized trainers is available only in SageMaker Python SDK V3, built from the ground up for foundation model fine-tuning.
   It streamlines the complex process of customizing AI models from months to days with a guided UI and serverless infrastructure that removes operational overhead. Whether you are building legal research applications, customer service chatbots, or domain-specific AI agents, this feature accelerates your path from proof-of-concept to production deployment.
   </div>

Key Benefits
-------------

* **Serverless Training**: Fully managed compute infrastructure that abstracts away all infrastructure complexity, allowing you to focus purely on model development
* **Advanced Customization Techniques**: Comprehensive set of methods including supervised fine-tuning (SFT), direct preference optimization (DPO), reinforcement learning with verifiable rewards (RLVR), and reinforcement learning with AI feedback (RLAIF)
* **AI Model Customization Assets**: Integrated datasets and evaluators for training, refining, and evaluating custom models
* **Production Ready**: Built-in evaluation, monitoring, and deployment capabilities with automatic resource management

Key Concepts
------------

**Serverless Training**
  A fully managed compute infrastructure that abstracts away all infrastructure complexity, allowing you to focus purely on model development. This includes automatic provisioning of GPU instances (P5, P4de, P4d, G5) based on model size and training requirements, pre-optimized training recipes that incorporate best practices for each customization technique, real-time monitoring with live metrics and logs accessible through the UI, and automatic cleanup of resources after training completion to optimize costs.

**Model Customization Techniques**
  Comprehensive set of advanced methods including supervised fine-tuning (SFT), direct preference optimization (DPO), reinforcement learning with verifiable rewards (RLVR), and reinforcement learning with AI feedback (RLAIF).

**Logged Model**
  A specialized version of a base foundation model that has been adapted to a specific use case by training it on your own data, resulting in an AI model that retains the general capabilities of the original foundation model while adding domain-specific knowledge, terminology, style, or behavior tailored to your requirements.

**AI Model Customization Assets**
  Resources and artifacts used to train, refine, and evaluate custom models during the model customization process. These assets include:
  
  * **Datasets**: Collections of training examples (prompt-response pairs, domain-specific text, or labeled data) used to fine-tune a foundation model to learn specific behaviors, knowledge, or styles
  * **Evaluators**: Mechanisms for assessing and improving model performance through either reward functions (code-based logic that scores model outputs based on specific criteria, used in RLVR training and custom scorer evaluation) or reward prompts (natural language instructions that guide an LLM to judge the quality of model responses, used in RLAIF training and LLM-as-a-judge evaluation)

Getting Started
---------------

Prerequisites and Setup
~~~~~~~~~~~~~~~~~~~~~~~

Before you begin, complete the following prerequisites:

1. **SageMaker AI Domain Setup**: Onboard to a SageMaker AI domain with Studio access. If you don't have permissions to set Studio as the default experience for your domain, contact your administrator.

2. **AWS CLI Configuration**: Update the AWS CLI and configure your credentials:

   .. code-block:: bash

      # Update AWS CLI
      pip install --upgrade awscli
      
      # Configure credentials
      aws configure

3. **IAM Permissions**: Attach the following AWS managed policies to your execution role:

   * ``AmazonSageMakerFullAccess`` - Full access to SageMaker resources
   * ``AmazonSageMakerPipelinesIntegrations`` - For pipeline operations
   * ``AmazonSageMakerModelRegistryFullAccess`` - For model registry features

4. **Additional IAM Permissions**: Add the following inline policy to your SageMaker domain execution role:

   .. code-block:: json

      {
          "Version": "2012-10-17",
          "Statement": [
              {
                  "Sid": "LambdaCreateDeletePermission",
                  "Effect": "Allow",
                  "Action": [
                      "lambda:CreateFunction",
                      "lambda:DeleteFunction",
                      "lambda:InvokeFunction"
                  ],
                  "Resource": [
                      "arn:aws:lambda:*:*:function:*SageMaker*",
                      "arn:aws:lambda:*:*:function:*sagemaker*",
                      "arn:aws:lambda:*:*:function:*Sagemaker*"
                  ]
              },
              {
                  "Sid": "BedrockDeploy",
                  "Effect": "Allow",
                  "Action": [
                      "bedrock:CreateModelImportJob",
                      "bedrock:GetModelImportJob",
                      "bedrock:GetImportedModel"
                  ],
                  "Resource": ["*"]
              },
              {
                  "Sid": "AIRegistry",
                  "Effect": "Allow",
                  "Action": [
                      "sagemaker:CreateHub",
                      "sagemaker:DeleteHub",
                      "sagemaker:DescribeHub",
                      "sagemaker:ListHubs",
                      "sagemaker:ImportHubContent",
                      "sagemaker:DeleteHubContent",
                      "sagemaker:UpdateHubContent",
                      "sagemaker:ListHubContents",
                      "sagemaker:ListHubContentVersions",
                      "sagemaker:DescribeHubContent"
                  ],
                  "Resource": "*"
              }
          ]
      }

Creating Assets for Model Customization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Using the SageMaker Python SDK:**

.. code-block:: python

   from sagemaker.train.common import CustomizationTechnique
   from sagemaker.assets import DataSet

   # Create a dataset asset
   dataset = DataSet.create(
       name="demo-sft-dataset",
       data_location="s3://your-bucket/dataset/training_dataset.jsonl",
       customization_technique=CustomizationTechnique.SFT,
       wait=True
   )
   
   print(f"Dataset ARN: {dataset.arn}")

Quick Start Example
-------------------

**Model Customization via SDK:**

.. code-block:: python

   from sagemaker.train import DPOTrainer
   from sagemaker.train.common import TrainingType

   # Submit a DPO model customization job
   trainer = DPOTrainer(
       model="meta-llama/Llama-2-7b-hf",
       training_type=TrainingType.LORA,
       model_package_group_name="my-custom-models",
       training_dataset="s3://my-bucket/preference-data.jsonl",
       s3_output_path="s3://my-bucket/output/",
       sagemaker_session=sagemaker_session,
       role=role_arn
   )
   
   # Start training
   training_job = trainer.train()


Supported Model Types and Use Cases
-----------------------------------

**Foundation Models**
  * Large Language Models (LLaMA, GPT, BERT, T5)
  * Conversational AI models and dialogue systems
  * Domain-specific models (legal, medical, financial, technical)
  * Multimodal models for vision-language understanding

**Customization Scenarios**
  * Task-specific adaptation (summarization, QA, classification)
  * Instruction following and multi-step reasoning
  * Safety and alignment improvements
  * Style and persona customization

**Advanced Techniques**
  * **LoRA (Low-Rank Adaptation)** - Parameter-efficient fine-tuning with minimal memory requirements
  * **Full Fine-Tuning** - Complete model parameter updates for maximum customization
  * **Preference Learning** - Train models using human feedback and preference data
  * **Reinforcement Learning** - Advanced alignment techniques for improved model behavior

.. toctree::
   :maxdepth: 2
   :hidden:

   open_weight_model_customization
   nova


