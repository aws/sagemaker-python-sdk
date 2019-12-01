#########################################
Amazon SageMaker Operators for Kubernetes
#########################################



Amazon SageMaker Operators for Kubernetes are operators that can be used
to train machine learning models, optimize hyperparameters for a given
model, run batch transform jobs over existing models, and set up
inference endpoints. With these operators, users can manage their jobs
in Amazon SageMaker from their Kubernetes cluster. This document assumes
prior knowledge of Kubernetes and standard commands.

There is no additional charge to use these operators. You incur charges
for any Amazon SageMaker resources used through these operators. 


.. contents::

What is an Operator?
--------------------

Kubernetes is built on top of what is called the controller pattern.
This pattern allows applications and tools to listen to a central state
manager (ETCD) and take action when something happens. Examples of such
applications
include \ ``cloud-controller-manager``, \ ``controller-manager``, etc.
The controller pattern allows us to create decoupled experiences and not
have to worry about how other components are integrated. An operator is
a purpose-built application that will manage a specific type of
component using this same pattern.

Prerequisites
~~~~~~~~~~~~~

The Amazon SageMaker Operators for Kubernetes guide assumes that you’ve
completed the following prerequisites:

-  Installed the following tools:

   -  `kubectl <https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html>`__
      Version 1.13 or later. Use a \ ``kubectl`` version that is within
      one minor version of your Amazon Elastic Kubernetes Service
      (Amazon EKS) cluster control plane. For example, a
      1.13 \ ``kubectl`` client works with Kubernetes 1.13 and 1.14
      clusters. OpenID Connect (OIDC) is not supported in versions earlier than 1.13.

   -  `eksctl <https://github.com/weaveworks/eksctl>`__ Version 0.7.0 or
      later

   -  `AWS
      CLI <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv1.html>`__ Version
      1.16.232 or later

   -  (optional) `Helm <https://helm.sh/docs/intro/install/>`__ Version
      3.0 or later

   -  `aws-iam-authenticator <https://docs.aws.amazon.com/eks/latest/userguide/install-aws-iam-authenticator.html>`__ 

-  Have IAM permissions to create roles and attach policies to roles.

-  Created a Kubernetes cluster to run the operators on. It should either be
   Kubernetes version 1.13 or 1.14. For automated cluster
   creation using \ ``eksctl``, see `Getting Started with eksctl <https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html>`__.
   It takes 20 to 30 minutes to provision a cluster.

Permissions Overview
~~~~~~~~~~~~~~~~~~~~

The Amazon SageMaker Operators for Kubernetes allow you to manage jobs
in Amazon SageMaker from your Kubernetes cluster. Thus the operators
will access Amazon SageMaker resources on your behalf. Additionally, the
IAM role that the operator assumes to interact with AWS resources differs
from the credentials you use to access the Kubernetes cluster and the
role that Amazon SageMaker assumes when running your machine learning
jobs. The following image explains this design and flow.

.. image:: ./amazon_sagemaker_operators_for_kubernetes_authentication.png

Setup and Operator Deployment
-----------------------------

The following sections describe the steps to setup and deploy the
operator.

IAM Role-Based Operator Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you can deploy your operator using an IAM Role, you need to
associate an OpenID Connect (OIDC) provider with your IAM role to
authenticate with the IAM service.

Associate an OpenID Connect Provider to Your Instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to create an OIDC identity provider for your cluster. If your
cluster is managed by EKS, then your cluster will already have an OIDC
attached to it. 

Set the local ``CLUSTER_NAME`` and \ ``AWS_REGION`` environment
variables as follows:

::

    # Set the region and cluster
    export CLUSTER_NAME="<your cluster name>"
    export AWS_REGION="<your region>"

Use the following command to associate the OIDC provider with your
cluster. For more information, see \ `Enabling IAM Roles for Service
Accounts on your
Cluster. <https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html>`__

::

    eksctl utils associate-iam-oidc-provider --cluster ${CLUSTER_NAME} \
        --region ${AWS_REGION} --approve

Your output should look like the following:

::

    [_]  eksctl version 0.10.1
    [_]  using region us-east-1
    [_]  IAM OpenID Connect provider is associated with cluster "my-cluster" in "us-east-1"

Now that the cluster has an OIDC identity provider, you can create a
role and give a Kubernetes ServiceAccount permission to assume the role.

Get the OIDC ID
^^^^^^^^^^^^^^^

To set up the ServiceAccount, first obtain the OpenID Connect issuer URL
using the following command:

::

    aws eks describe-cluster --name ${CLUSTER_NAME} --region ${AWS_REGION} \
        --query cluster.identity.oidc.issuer --output text

The command will return a URL like the following:

::

    https://oidc.eks.${AWS_REGION}.amazonaws.com/id/D48675832CA65BD10A532F597OIDCID

In this URL, the value D48675832CA65BD10A532F597OIDCID is the OIDC ID.
The OIDC ID for your cluster will be different. You’ll need this OIDC ID
value to create an IAM role.

If your output is \ ``None``, it means that your client version is old.
To work around this, run the following command: 

::

    aws eks describe-cluster --query cluster --name ${CLUSTER_NAME} --output text | grep OIDC

The OIDC URL will be returned as follows:

::

    OIDC https://oidc.eks.us-east-1.amazonaws.com/id/D48675832CA65BD10A532F597OIDCID

Create an IAM Role 
^^^^^^^^^^^^^^^^^^^

Create a file named \ ``trust.json``  and insert the following trust
relationship code block into it. Be sure to replace all \ ``<OIDC ID>``, \ ``<AWS account number>``, and \ ``<EKS Cluster region>`` placeholders with values corresponding to your cluster.

::

    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "Federated": "arn:aws:iam::<AWS account number>:oidc-provider/oidc.eks.<EKS Cluster region>.amazonaws.com/id/<OIDC ID>"
          },
          "Action": "sts:AssumeRoleWithWebIdentity",
          "Condition": {
            "StringEquals": {
              "oidc.eks.<EKS Cluster region>.amazonaws.com/id/<OIDC ID>:aud": "sts.amazonaws.com",
              "oidc.eks.<EKS Cluster region>.amazonaws.com/id/<OIDC ID>:sub": "system:serviceaccount:sagemaker-k8s-operator-system:sagemaker-k8s-operator-default"
            }
          }
        }
      ]
    }

Run the following command to create an IAM role with the trust
relationship defined in \ ``trust.json``. This role enables the
Amazon EKS cluster to get and refresh credentials from IAM.

::

    aws iam create-role --role-name <role name> --assume-role-policy-document file://trust.json --output=text

Your output should look like the following:

::

    ROLE    arn:aws:iam::123456789012:role/my-role 2019-11-22T21:46:10Z    /       ABCDEFSFODNN7EXAMPLE   my-role
    ASSUMEROLEPOLICYDOCUMENT        2012-10-17
    STATEMENT       sts:AssumeRoleWithWebIdentity   Allow
    STRINGEQUALS    sts.amazonaws.com       system:serviceaccount:sagemaker-k8s-operator-system:sagemaker-k8s-operator-default
    PRINCIPAL       arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/

Take note of \ ``ROLE ARN``, you need to pass this value to your
operator. 

Attach the AmazonSageMakerFullAccess Policy to the Role
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To give the IAM role access to Amazon SageMaker, attach
the \ `AmazonSageMakerFullAccess <https://console.aws.amazon.com/iam/home?#/policies/arn:aws:iam::aws:policy/AmazonSageMakerFullAccess>`__ policy.
If you want to limit permissions to the operator, you can create your
own custom policy and attach it.

To attach AmazonSageMakerFullAccess, run the following command:

::

    aws iam attach-role-policy --role-name <role name>  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

The Kubernetes
ServiceAccount \ ``sagemaker-k8s-operator-default`` should
have \ ``AmazonSageMakerFullAccess`` permissions. Confirm this when you
install the operator.

Deploy the Operator
^^^^^^^^^^^^^^^^^^^

When deploying your operator, you can use either a YAML file or Helm
charts. 

Deploy the Operator Using YAML
''''''''''''''''''''''''''''''

This is the simplest way to deploy your operators. The process is as
follows: 

-  Download the installer script using the following command:

   ::

       wget https://raw.githubusercontent.com/aws/amazon-sagemaker-operator-for-k8s/master/release/rolebased/installer.yaml

-  Edit the \ ``installer.yaml`` file to
   replace \ ``eks.amazonaws.com/role-arn``. Replace the ARN here with
   the ARN for the OIDC-based role you’ve created. 

-  Use the following command to deploy the cluster:  

   ::

       kubectl apply -f installer.yaml

Deploy the Operator Using Helm Charts
'''''''''''''''''''''''''''''''''''''

Alternatively, we have prepared a Helm Chart that you can use to install
the operator.

Get the Helm Installer Directory 


Clone the Helm installer directory using the following command:

::

    git clone https://github.com/aws/amazon-sagemaker-operator-for-k8s.git

Navigate to the
``amazon-sagemaker-operator-for-k8s/hack/charts/installer`` folder. Edit
the \ ``values.yaml`` file, which includes high-level parameters for the
Chart. Replace the ARN here with the ARN for the OIDC-based role you’ve
created. 

Install the Helm Chart using the following command:

::

    helm install rolebased/ --generate-name


After a moment, the chart will be installed with a randomly-generated
name. Verify that the installation succeeded by running the following
command:

::

    helm ls

Your output should look like the following:

::

    NAME                    NAMESPACE       REVISION        UPDATED                                 STATUS          CHART                           APP VERSION
    rolebased-1234567    default         1               2019-11-20 23:14:59.6777082 +0000 UTC   deployed        sagemaker-k8s-operator-0.1.0


Verify the Operator Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You should be able to see the Amazon SageMaker Custom Resource
Definitions (CRDs) for each operator deployed to your cluster by running
the following command: 

::

    kubectl get crd | grep sagemaker

Your output should look like the following:

::

    batchtransformjobs.sagemaker.aws.amazon.com         2019-11-20T17:12:34Z
    endpointconfigs.sagemaker.aws.amazon.com            2019-11-20T17:12:34Z
    hostingdeployments.sagemaker.aws.amazon.com         2019-11-20T17:12:34Z
    hyperparametertuningjobs.sagemaker.aws.amazon.com   2019-11-20T17:12:34Z
    models.sagemaker.aws.amazon.com                     2019-11-20T17:12:34Z
    trainingjobs.sagemaker.aws.amazon.com               2019-11-20T17:12:34Z

Ensure that the operator pod is running successfully. Use the following
command to list all pods:

::

    kubectl -n sagemaker-k8s-operator-system get pods

You should see a pod
named \ ``sagemaker-k8s-operator-controller-manager-*****`` in the
namespace \ ``sagemaker-k8s-operator-system``  as follows:

::

    NAME                                                         READY   STATUS    RESTARTS   AGE
    sagemaker-k8s-operator-controller-manager-12345678-r8abc   2/2     Running   0          23s

​

Install the Amazon SageMaker Logs \ ``kubectl`` Plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As part of the Amazon SageMaker Operators for Kubernetes, you can use
the \ ``smlogs`` `plugin <https://kubernetes.io/docs/tasks/extend-kubectl/kubectl-plugins/>`__ for ``kubectl`` .
This enables Amazon SageMaker CloudWatch logs to be streamed
with \ ``kubectl``. \ ``kubectl``\ must be installed onto
your `PATH <http://www.linfo.org/path_env_var.html>`__. The
following commands place the binary in
the \ ``sagemaker-k8s-bin`` directory in your home directory, and add
that directory to your \ ``PATH``.

::

    export os="linux"

    wget https://amazon-sagemaker-operator-for-k8s-us-east-1.s3.amazonaws.com/kubectl-smlogs-plugin/latest/${os}.amd64.tar.gz
    tar xvzf ${os}.amd64.tar.gz

    # Move binaries to a directory in your homedir.
    mkdir ~/sagemaker-k8s-bin
    cp ./kubectl-smlogs.${os}.amd64/kubectl-smlogs ~/sagemaker-k8s-bin/.

    # This line will add the binaries to your PATH in your .bashrc. 

    echo 'export PATH=$PATH:~/sagemaker-k8s-bin' >> ~/.bashrc

    # Source your .bashrc to update environment variables:
    source ~/.bashrc

Use the following command to verify that the \ ``kubectl`` plugin is
installed correctly:

::

    kubectl smlogs

If the \ ``kubectl`` plugin is installed correctly, your output should
look like the following:

::

    View Amazon SageMaker logs via Kubernetes

    Usage:
      smlogs [command]

    Aliases:
      smlogs, SMLogs, Smlogs

    Available Commands:
      BatchTransformJob       View BatchTransformJob logs via Kubernetes
      TrainingJob             View TrainingJob logs via Kubernetes
      help                    Help about any command

    Flags:
      -h, --help   help for smlogs

    Use "smlogs [command] --help" for more information about a command.


Delete Operators from the Cluster 
----------------------------------

Operators Installed Using YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To uninstall the operator from your cluster, make sure that all
Amazon SageMaker resources have been deleted from the cluster. Failure
to do so will cause the operator delete operation to hang. Once you have
deleted all Amazon SageMaker kubernetes jobs, use \ ``kubectl`` to
delete the operator from the cluster. Run the following commands to stop
all jobs and delete the operator from the cluster:

::

    # Delete all Amazon SageMaker jobs from Kubernetes
    kubectl delete --all --all-namespaces hyperparametertuningjob.sagemaker.aws.amazon.com
    kubectl delete --all --all-namespaces trainingjobs.sagemaker.aws.amazon.com
    kubectl delete --all --all-namespaces batchtransformjob.sagemaker.aws.amazon.com
    kubectl delete --all --all-namespaces hostingdeployment.sagemaker.aws.amazon.com

    # Delete the operator and its resources
    kubectl delete -f /installer.yaml

You should see output like the following:

::

    $ kubectl delete --all --all-namespaces trainingjobs.sagemaker.aws.amazon.com
    trainingjobs.sagemaker.aws.amazon.com "xgboost-mnist-from-for-s3" deleted

    $ kubectl delete --all --all-namespaces hyperparametertuningjob.sagemaker.aws.amazon.com
    hyperparametertuningjob.sagemaker.aws.amazon.com "xgboost-mnist-hpo" deleted

    $ kubectl delete --all --all-namespaces batchtransformjob.sagemaker.aws.amazon.com
    batchtransformjob.sagemaker.aws.amazon.com "xgboost-mnist" deleted

    $ kubectl delete --all --all-namespaces hostingdeployment.sagemaker.aws.amazon.com
    hostingdeployment.sagemaker.aws.amazon.com "host-xgboost" deleted

    $ kubectl delete -f raw-yaml/installer.yaml
    namespace "sagemaker-k8s-operator-system" deleted
    customresourcedefinition.apiextensions.k8s.io "batchtransformjobs.sagemaker.aws.amazon.com" deleted
    customresourcedefinition.apiextensions.k8s.io "endpointconfigs.sagemaker.aws.amazon.com" deleted
    customresourcedefinition.apiextensions.k8s.io "hostingdeployments.sagemaker.aws.amazon.com" deleted
    customresourcedefinition.apiextensions.k8s.io "hyperparametertuningjobs.sagemaker.aws.amazon.com" deleted
    customresourcedefinition.apiextensions.k8s.io "models.sagemaker.aws.amazon.com" deleted
    customresourcedefinition.apiextensions.k8s.io "trainingjobs.sagemaker.aws.amazon.com" deleted
    role.rbac.authorization.k8s.io "sagemaker-k8s-operator-leader-election-role" deleted
    clusterrole.rbac.authorization.k8s.io "sagemaker-k8s-operator-manager-role" deleted
    clusterrole.rbac.authorization.k8s.io "sagemaker-k8s-operator-proxy-role" deleted
    rolebinding.rbac.authorization.k8s.io "sagemaker-k8s-operator-leader-election-rolebinding" deleted
    clusterrolebinding.rbac.authorization.k8s.io "sagemaker-k8s-operator-manager-rolebinding" deleted
    clusterrolebinding.rbac.authorization.k8s.io "sagemaker-k8s-operator-proxy-rolebinding" deleted
    service "sagemaker-k8s-operator-controller-manager-metrics-service" deleted
    deployment.apps "sagemaker-k8s-operator-controller-manager" deleted
    secrets "sagemaker-k8s-operator-abcde" deleted

Operators Installed Using Helm Charts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To delete the operator CRDs, first delete all the running jobs. Then
delete the helm chart that was used to deploy the operators using the
following commands: 

::

    # get the helm charts 
    $ helm ls

    # delete the charts
    $ helm delete <chart name>

​

Troubleshooting
---------------

Debugging a Failed Job
~~~~~~~~~~~~~~~~~~~~~~

Check the job status by running:

::

    kubectl get <CRD Type> <job name>

If the job was created in Amazon SageMaker, you can use the following
command to see the \ ``STATUS`` and the ``SageMaker Job Name``: 

::

    kubectl get <crd type> <job name>

-  You can use \ ``smlogs`` to find the cause of the issue using the
   following command: 

   ::

       kubectl smlogs <crd type> <job name>

-  You can also use the \ ``describe`` command to get more details about
   the job using the following command.The output will have
   an \ ``additional`` field that will have more information about the
   status of the job.

   ::

       kubectl describe <crd type> <job name>

If the job was not created in Amazon SageMaker, then use the logs of the
operator’s pod to find the cause of the issue as follows:

::

    $ kubectl get pods -A | grep sagemaker
    # Output: 
    sagemaker-k8s-operator-system   sagemaker-k8s-operator-controller-manager-5cd7df4d74-wh22z   2/2     Running   0          3h33m

    $ kubectl logs -p <pod name> -c manager -n sagemaker-k8s-operator-system

Deleting an Operator CRD
~~~~~~~~~~~~~~~~~~~~~~~~

If deleting a job is stuck, check if the operator is running. If the
operator is not running, then you will have to delete the finalizer
using the following steps:

-  In a new terminal, open the job in an editor using ``kubectl edit``
   as follows: 

   ::

       $ kubectl edit <crd type> <job name>

       # for example for the batchtransformjob xgboost-mnist
       $ kubectl edit batchtransformjobs xgboost-mnist 

-  Edit the job to delete the finalizer by removing the following two
   lines from the file. Save the file and the job should immediately get
   deleted/updated. 

   ::

         finalizers:
         - sagemaker-operator-finalizer

Appendix
--------

The following table lists the available operator images and SMLogs in
each region.

+-------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| Region      | Controller Image                                                                            | Linux SMLogs                                                                                                           |
+=============+=============================================================================================+========================================================================================================================+
| us-east-1   | ``957583890962.dkr.ecr.us-east-1.amazonaws.com/amazon-sagemaker-operator-for-k8s:latest``   | https://amazon-sagemaker-operator-for-k8s-us-east-1.s3.amazonaws.com/kubectl-smlogs-plugin/latest/linux.amd64.tar.gz   |
+-------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| us-east-2   | ``922499468684.dkr.ecr.us-east-2.amazonaws.com/amazon-sagemaker-operator-for-k8s:latest``   | https://amazon-sagemaker-operator-for-k8s-us-east-2.s3.amazonaws.com/kubectl-smlogs-plugin/latest/linux.amd64.tar.gz   |
+-------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| us-west-2   | ``640106867763.dkr.ecr.us-west-2.amazonaws.com/amazon-sagemaker-operator-for-k8s:latest``   | https://amazon-sagemaker-operator-for-k8s-us-west-2.s3.amazonaws.com/kubectl-smlogs-plugin/latest/linux.amd64.tar.gz   |
+-------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| eu-west-1   | ``613661167059.dkr.ecr.eu-west-1.amazonaws.com/amazon-sagemaker-operator-for-k8s:latest``   | https://amazon-sagemaker-operator-for-k8s-eu-west-1.s3.amazonaws.com/kubectl-smlogs-plugin/latest/linux.amd64.tar.gz   |
+-------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
