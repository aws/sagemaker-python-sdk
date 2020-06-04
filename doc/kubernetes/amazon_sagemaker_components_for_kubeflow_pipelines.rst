Amazon SageMaker Components for Kubeflow Pipelines
==================================================

This document outlines how to use Amazon SageMaker Components
for Kubeflow Pipelines (KFP). With these pipeline components, you can
create and monitor training, tuning, endpoint deployment, and batch
transform jobs in Amazon SageMaker. By running Kubeflow Pipeline jobs on
Amazon SageMaker, you move data processing and training jobs from the
Kubernetes cluster to Amazon SageMaker’s machine learning-optimized
managed service. This document assumes prior knowledge of Kubernetes and
Kubeflow.

What is Kubeflow Pipelines?
---------------------------

Kubeflow Pipelines (KFP) is a platform for building and deploying
portable, scalable machine learning (ML) workflows based on Docker
containers. The Kubeflow Pipelines platform consists of the following:

-  A user interface (UI) for managing and tracking experiments, jobs,
   and runs.

-  An engine (Argo) for scheduling multi-step ML workflows.

-  A Python SDK for defining and manipulating pipelines and components.

-  Notebooks for interacting with the system using the SDK.

A pipeline is a description of an ML workflow expressed as a directed
acyclic \ `graph <https://www.kubeflow.org/docs/pipelines/concepts/graph/>`__
as shown in the following diagram.  Every step in the workflow is
expressed as a Kubeflow Pipeline
`component <https://www.kubeflow.org/docs/pipelines/overview/concepts/component/>`__,
which is a Python module.

If your data has been preprocessed, the standard pipeline takes a subset
of the data and runs hyperparameter optimization of the model. The
pipeline then trains a model with the full dataset using the optimal
hyperparameters. This model is used for both batch inference and
endpoint creation.

For more information on Kubeflow Pipelines, see the \ `Kubeflow
Pipelines documentation <https://www.kubeflow.org/docs/pipelines/>`__.

Kubeflow Pipeline components
----------------------------

A Kubeflow Pipeline component is a set of code used to execute one step
in a Kubeflow pipeline. Components are represented by a Python module
that is converted into a Docker image. These components make it fast and
easy to write pipelines for experimentation and production environments
without having to interact with the underlying Kubernetes
infrastructure.

What do Amazon SageMaker Components for Kubeflow Pipelines provide?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Amazon SageMaker Components for Kubeflow Pipelines offer an alternative
to launching compute-intensive jobs in Amazon SageMaker. These
components integrate Amazon SageMaker with the portability and
orchestration of Kubeflow Pipelines. Using the Amazon SageMaker
components, each of the jobs in the pipeline workflow runs on Amazon
SageMaker instead of the local Kubernetes cluster. The job parameters,
status, logs, and outputs from Amazon SageMaker are still accessible
from the Kubeflow Pipelines UI. The following Amazon SageMaker
components have been created to integrate 6 key Amazon SageMaker
features into your ML workflows. You can create a Kubeflow Pipeline
built entirely using these components, or integrate individual
components into your workflow as needed.

There is no additional charge for using Amazon SageMaker Components for
Kubeflow Pipelines. You incur charges for any Amazon SageMaker resources
you use through these components.

Training components
^^^^^^^^^^^^^^^^^^^

**Training**

The Training component allows you to submit Amazon SageMaker Training
jobs directly from a Kubeflow Pipelines workflow. For more information,
see \ `SageMaker Training Kubeflow Pipelines
component <https://github.com/kubeflow/pipelines/tree/master/components/aws/sagemaker/train>`__.

**Hyperparameter Optimization**

The Hyperparameter Optimization component enables you to submit
hyperparameter tuning jobs to Amazon SageMaker directly from a Kubeflow
Pipelines workflow. For more information, see \ `SageMaker
hyperparameter optimization Kubeflow Pipeline
component <https://github.com/kubeflow/pipelines/tree/master/components/aws/sagemaker/hyperparameter_tuning>`__.

Inference components
^^^^^^^^^^^^^^^^^^^^

**Hosting Deploy**

The Deploy component enables you to deploy a model in Amazon SageMaker
Hosting from a Kubeflow Pipelines workflow. For more information,
see \ `SageMaker Hosting Services - Create Endpoint Kubeflow Pipeline
component <https://github.com/kubeflow/pipelines/tree/master/components/aws/sagemaker/deploy>`__.

**Batch Transform component**

The Batch Transform component enables you to run inference jobs for an
entire dataset in Amazon SageMaker from a Kubeflow Pipelines workflow.
For more information, see \ `SageMaker Batch Transform Kubeflow Pipeline
component <https://github.com/kubeflow/pipelines/tree/master/components/aws/sagemaker/batch_transform>`__.

Ground Truth components
^^^^^^^^^^^^^^^^^^^^^^^

**Ground Truth**\

The Ground Truth component enables you to to submit Amazon SageMaker
Ground Truth labeling jobs directly from a Kubeflow Pipelines workflow.
For more information, see \ `SageMaker Ground Truth Kubeflow Pipelines
component <https://github.com/kubeflow/pipelines/tree/master/components/aws/sagemaker/ground_truth>`__.

**Workteam**

The Workteam component enables you to create Amazon SageMaker private
workteam jobs directly from a Kubeflow Pipelines workflow. For more
information, see \ `SageMaker create private workteam Kubeflow Pipelines
component <https://github.com/kubeflow/pipelines/tree/master/components/aws/sagemaker/workteam>`__.

IAM permissions
---------------

Deploying Kubeflow Pipelines with Amazon SageMaker components requires
the following three levels of IAM permissions:

-  An IAM user/role to access your AWS account (**your\_credentials**).
   Note: You don’t need this at all if you already have access to KFP
   web UI and have your input data in Amazon S3, or if you already have
   an Amazon Elastic Kubernetes Service (Amazon EKS) cluster with KFP.

   You use this user/role from your gateway node, which can be your
   local machine or a remote instance, to:

   -  Create an Amazon EKS cluster and install KFP

   -  Create IAM roles/users

   -  Create S3 buckets for your sample input data

   The IAM user/role needs the following permissions:

   -  CloudWatchLogsFullAccess

   -  `AWSCloudFormationFullAccess <https://console.aws.amazon.com/iam/home?region=us-east-1#/policies/arn%3Aaws%3Aiam%3A%3Aaws%3Apolicy%2FAWSCloudFormationFullAccess>`__

   -  IAMFullAccess

   -  AmazonS3FullAccess

   -  AmazonEC2FullAccess

   -  AmazonEKSAdminPolicy - Create this policy using the schema
      from \ `Amazon EKS Identity-Based Policy
      Examples <https://docs.aws.amazon.com/eks/latest/userguide/security_iam_id-based-policy-examples.html>`__

-  An IAM role used by KFP pods to access Amazon Sagemaker
   (**kfp-example-pod-role**) The KFP pods use this permission to create
   Amazon SageMaker jobs from KFP components. Note: If you want to limit
   permissions to the KFP pods, create your own custom policy and attach
   it.

   The role needs the following permission:

   -  AmazonSageMakerFullAccess

-  An IAM role used by SageMaker jobs to access resources such as Amazon
   S3, ECR etc. (**kfp-example-sagemaker-execution-role**).

   Your Amazon SageMaker jobs use this role to:

   -  Access Amazon Sagemaker resources

   -  Input Data from S3

   -  Store your output model to S3

   The role needs the following permissions:

   -  AmazonSageMakerFullAccess

   -  AmazonS3FullAccess

These are all the IAM users/roles you need to run KFP components for
Amazon SageMaker.

When you have run the components and have created the Amazon SageMaker
endpoint, you also need a role with the ``sagemaker:InvokeEndpoint``
permission to query inference endpoints.

Converting Pipelines to use Amazon SageMaker
--------------------------------------------

You can convert an existing pipeline to use Amazon SageMaker by porting
your generic Python `processing
containers <https://docs.aws.amazon.com/sagemaker/latest/dg/amazon-sagemaker-containers.html>`__
and \ `training
containers <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html>`__.
If you are using Amazon SageMaker for inference, you also need to attach
IAM permissions to your cluster and convert an artifact to a model.
