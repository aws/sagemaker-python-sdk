Using Amazon SageMaker Components
=================================

In this tutorial, you run a pipeline using Amazon SageMaker Components
for Kubeflow Pipelines to train a classification model using Kmeans with
the MNIST dataset. This workflow uses Kubeflow pipelines as the
orchestrator and Amazon SageMaker as the backend to run the steps in the
workflow. For the full code for this and other pipeline examples, see
the `Sample AWS SageMaker Kubeflow
Pipelines <https://github.com/kubeflow/pipelines/tree/master/samples/contrib/aws-samples>`__.
For information on the components used, see the `KubeFlow Pipelines
GitHub
repository <https://github.com/kubeflow/pipelines/tree/master/components/aws/sagemaker>`__.

Setup
-----

To use Kubeflow Pipelines (KFP), you need an Amazon Elastic Kubernetes
Service (Amazon EKS) cluster and a gateway node to interact with that
cluster. The following sections show the steps needed to set up these
resources.

Set up a gateway node
~~~~~~~~~~~~~~~~~~~~~

A gateway node is used to create an Amazon EKS cluster and access the
Kubeflow Pipelines UI. Use your local machine or an Amazon EC2 instance
as your gateway node. If you want to use a new EC2 instance, create one
with the latest Ubuntu 18.04 DLAMI version from the AWS console using
the steps in `Launching and Configuring a
DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html>`__.

Complete the following steps to set up your gateway node. Depending on
your environment, you may have certain requirements already configured.

If you don’t have an existing Amazon EKS cluster, create a user named ``your_credentials`` using the steps in `Creating an IAM User in Your
AWS
Account <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html>`__. If
you have an existing Amazon EKS cluster, use the credentials of the IAM
role or user that has access to it.

Add the following permissions to your user using the steps in `Changing
Permissions for an IAM
User: <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_change-permissions.html#users_change_permissions-add-console>`__

-  CloudWatchLogsFullAccess

-  `AWSCloudFormationFullAccess <https://console.aws.amazon.com/iam/home?region=us-east-1#/policies/arn%3Aaws%3Aiam%3A%3Aaws%3Apolicy%2FAWSCloudFormationFullAccess>`__

-  IAMFullAccess

-  AmazonS3FullAccess

-  AmazonEC2FullAccess

-  AmazonEKSAdminPolicy - Create this policy using the schema
   from `Amazon EKS Identity-Based Policy
   Examples <https://docs.aws.amazon.com/eks/latest/userguide/security_iam_id-based-policy-examples.html>`__

Install the following on your gateway node to access the Amazon EKS
cluster and KFP UI.

-  `AWS
   CLI <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html>`__.
   If you are using an IAM user, configure your `Access Key ID, Secret
   Access
   Key <https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html#access-keys-and-secret-access-keys>`__ and
   preferred AWS Region by running: ``aws configure``

-  `aws-iam-authenticator <https://docs.aws.amazon.com/eks/latest/userguide/install-aws-iam-authenticator.html>`__
   version 0.1.31 and above.

-  ``eksctl`` version above 0.15.

-  `kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl/#install-kubectl>`__.
   The version needs to match your Kubernetes version within 1 minor
   version.

Install \ ``boto3``.

::

    pip install boto3

Set up an Amazon EKS cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following steps from the command line of your gateway node to
set up an Amazon EKS cluster:

If you do not have an existing Amazon EKS cluster, complete the
following substeps. If you already have an Amazon EKS cluster, skip this
step.

Run the following from your command line to create an Amazon EKS Cluster
with version 1.14 or above. Replace ``<your-cluster-name>`` with any
name for your cluster.

::

    eksctl create cluster --name <your-cluster-name> --region us-east-1 --auto-kubeconfig --timeout=50m --managed --nodes=1

When cluster creation is complete, verify that you have access to the
cluster using the following command.

::

    kubectl get nodes

Verify that the current kubectl context is the cluster you want to use
with the following command. The current context is marked with an
asterisk (\*) in the output.

::

    kubectl config get-contexts

    CURRENT NAME     CLUSTER
    *   <username>@<clustername>.us-east-1.eksctl.io   <clustername>.us-east-1.eksctl.io

If the desired cluster is not configured as your current default, update
the default with the following command.

::

    aws eks update-kubeconfig --name <clustername> --region us-east-1

Install Kubeflow Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following steps from the command line of your gateway node to
install Kubeflow Pipelines on your cluster.

Install Kubeflow Pipelines on your cluster by following step 1
of `Deploying Kubeflow Pipelines
documentation <https://www.kubeflow.org/docs/pipelines/installation/standalone-deployment/#deploying-kubeflow-pipelines>`__.
Your KFP version must be 0.5.0 or above.

Verify that the Kubeflow Pipelines service and other related resources
are running.

::

    kubectl -n kubeflow get all | grep pipeline

Your output should look like the following.

::

    pod/ml-pipeline-6b88c67994-kdtjv                      1/1     Running            0          2d
    pod/ml-pipeline-persistenceagent-64d74dfdbf-66stk     1/1     Running            0          2d
    pod/ml-pipeline-scheduledworkflow-65bdf46db7-5x9qj    1/1     Running            0          2d
    pod/ml-pipeline-ui-66cc4cffb6-cmsdb                   1/1     Running            0          2d
    pod/ml-pipeline-viewer-crd-6db65ccc4-wqlzj            1/1     Running            0          2d
    pod/ml-pipeline-visualizationserver-9c47576f4-bqmx4   1/1     Running            0          2d
    service/ml-pipeline                       ClusterIP   10.100.170.170   <none>        8888/TCP,8887/TCP   2d
    service/ml-pipeline-ui                    ClusterIP   10.100.38.71     <none>        80/TCP              2d
    service/ml-pipeline-visualizationserver   ClusterIP   10.100.61.47     <none>        8888/TCP            2d
    deployment.apps/ml-pipeline                       1/1     1            1           2d
    deployment.apps/ml-pipeline-persistenceagent      1/1     1            1           2d
    deployment.apps/ml-pipeline-scheduledworkflow     1/1     1            1           2d
    deployment.apps/ml-pipeline-ui                    1/1     1            1           2d
    deployment.apps/ml-pipeline-viewer-crd            1/1     1            1           2d
    deployment.apps/ml-pipeline-visualizationserver   1/1     1            1           2d
    replicaset.apps/ml-pipeline-6b88c67994                      1         1         1       2d
    replicaset.apps/ml-pipeline-persistenceagent-64d74dfdbf     1         1         1       2d
    replicaset.apps/ml-pipeline-scheduledworkflow-65bdf46db7    1         1         1       2d
    replicaset.apps/ml-pipeline-ui-66cc4cffb6                   1         1         1       2d
    replicaset.apps/ml-pipeline-viewer-crd-6db65ccc4            1         1         1       2d
    replicaset.apps/ml-pipeline-visualizationserver-9c47576f4   1         1         1       2d

Access the KFP UI
~~~~~~~~~~~~~~~~~

The Kubeflow Pipelines UI is used for managing and tracking experiments,
jobs, and runs on your cluster. You can use port forwarding to access
the Kubeflow Pipelines UI from your gateway node.

Set up port forwarding to the KFP UI service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following from the command line of your gateway node:

Verify that the KFP UI service is running using the following command:

::

    kubectl -n kubeflow get service ml-pipeline-ui

    NAME             TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
    ml-pipeline-ui   ClusterIP   10.100.38.71   <none>        80/TCP    2d22h

Run the following command to setup port forwarding to the KFP UI
service. This forwards the KFP UI to port 8080 on your gateway node and
allows you to access the KFP UI from your browser.

    **Note**

    The port-forward from your remote machine drops if there is no
    activity. Run this command again if your dashboard is unable to get
    logs or updates. If the commands return an error, ensure that there
    is no process already running on the port you are trying to use.

::

    kubectl port-forward -n kubeflow service/ml-pipeline-ui 8080:80

Your method of accessing the KFP UI depends on your gateway node type.

Local machine as the gateway node

Access the dashboard in your browser as follows:

::

    http://localhost:8080

Click **Pipelines** to access the pipelines UI.

EC2 instance as the gateway node

You need to setup an SSH tunnel on your EC2 instance to access the
Kubeflow dashboard from your local machine’s browser.

From a new terminal session in your local machine, run the following.
Replace ``<public-DNS-of-gateway-node>`` with the IP address of your
instance found on the EC2 console. You can also use the public DNS.
Replace ``<path_to_key>`` with the path to the pem key used to access
the gateway node.

::

    public_DNS_address=<public-DNS-of-gateway-node>
    key=<path_to_key>

    on Ubuntu:
    ssh -i ${key} -L 9000:localhost:8080 ubuntu@${public_DNS_address}

    or on Amazon Linux:
    ssh -i ${key} -L 9000:localhost:8080 ec2-user@${public_DNS_address}

Access the dashboard in your browser.

::

    http://localhost:9000

Click **Pipelines** to access the KFP UI.

Create IAM Users/Roles for KFP pods and the Amazon SageMaker service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You now have a Kubernetes cluster with Kubeflow set up. To run Amazon
SageMaker Components for Kubeflow Pipelines, the Kubeflow Pipeline pods
need access to SageMaker. In this section, you create IAM Users/Roles to
be used by Kubeflow Pipeline pods and Amazon SageMaker.

Create a KFP execution role
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following from the command line of your gateway node:

Enable OIDC support on the Amazon EKS cluster with the following
command. Replace ``<cluster_name>`` with the name of your cluster
and ``<cluster_region>`` with the region your cluster is in.

::

    eksctl utils associate-iam-oidc-provider --cluster <cluster-name> \
            --region <cluster-region> --approve

Run the following to get the `OIDC <https://openid.net/connect/>`__
issuer URL. This URL is in the
form ``https://oidc.eks.<region>.amazonaws.com/id/<OIDC_ID>`` .

::

    aws eks describe-cluster --region <cluster-region> --name <cluster-name> --query "cluster.identity.oidc.issuer" --output text

Run the following to create a file named ``trust.json``.
Replace ``<OIDC_URL>`` with your OIDC issuer URL. Don’t
include ``https://`` when in your OIDC issuer URL.
Replace ``<AWS_account_number>`` with your AWS account number.

::

    OIDC_URL="<OIDC-URL>"
    AWS_ACC_NUM="<AWS-account-number>"

    # Run this to create trust.json file
    cat <<EOF > trust.json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "Federated": "arn:aws:iam::${AWS_ACC_NUM}:oidc-provider/${OIDC_URL}"
          },
          "Action": "sts:AssumeRoleWithWebIdentity",
          "Condition": {
            "StringEquals": {
              "${OIDC_URL}:aud": "sts.amazonaws.com",
              "${OIDC_URL}:sub": "system:serviceaccount:kubeflow:pipeline-runner"
            }
          }
        }
      ]
    }
    EOF

Create an IAM role named ``kfp-example-pod-role`` using ``trust.json``
using the following command. This role is used by KFP pods to create
Amazon SageMaker jobs from KFP components. Note the ARN returned in the
output.

::

    aws iam create-role --role-name kfp-example-pod-role --assume-role-policy-document file://trust.json
    aws iam attach-role-policy --role-name kfp-example-pod-role --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
    aws iam get-role --role-name kfp-example-pod-role --output text --query 'Role.Arn'

Edit your pipeline-runner service account with the following command.

::

    kubectl edit -n kubeflow serviceaccount pipeline-runner

In the file, add the following Amazon EKS role annotation and
replace ``<role_arn>`` with your role ARN.

::

    eks.amazonaws.com/role-arn: <role-arn>

Your file should look like the following when you’ve added the Amazon
EKS role annotation. Save the file.

::

    apiVersion: v1
    kind: ServiceAccount
    metadata:
      annotations:
        eks.amazonaws.com/role-arn: <role-arn>
        kubectl.kubernetes.io/last-applied-configuration: |
          {"apiVersion":"v1","kind":"ServiceAccount","metadata":{"annotations":{},"labels":{"app":"pipeline-runner","app.kubernetes.io/component":"pipelines-runner","app.kubernetes.io/instance":"pipelines-runner-0.2.0","app.kubernetes.io/managed-by":"kfctl","app.kubernetes.io/name":"pipelines-runner","app.kubernetes.io/part-of":"kubeflow","app.kubernetes.io/version":"0.2.0"},"name":"pipeline-runner","namespace":"kubeflow"}}
      creationTimestamp: "2020-04-16T05:48:06Z"
      labels:
        app: pipeline-runner
        app.kubernetes.io/component: pipelines-runner
        app.kubernetes.io/instance: pipelines-runner-0.2.0
        app.kubernetes.io/managed-by: kfctl
        app.kubernetes.io/name: pipelines-runner
        app.kubernetes.io/part-of: kubeflow
        app.kubernetes.io/version: 0.2.0
      name: pipeline-runner
      namespace: kubeflow
      resourceVersion: "11787"
      selfLink: /api/v1/namespaces/kubeflow/serviceaccounts/pipeline-runner
      uid: d86234bd-7fa5-11ea-a8f2-02934be6dc88
    secrets:
    - name: pipeline-runner-token-dkjrk

Create an Amazon SageMaker execution role
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``kfp-example-sagemaker-execution-role`` IAM role is used
by Amazon SageMaker jobs to access AWS resources. For more information,
see the IAM Permissions section. You provide this role as an input
parameter when running the pipeline.

Run the following to create the role. Note the ARN that is returned in
your output.

::

    SAGEMAKER_EXECUTION_ROLE_NAME=kfp-example-sagemaker-execution-role

    TRUST="{ \"Version\": \"2012-10-17\", \"Statement\": [ { \"Effect\": \"Allow\", \"Principal\": { \"Service\": \"sagemaker.amazonaws.com\" }, \"Action\": \"sts:AssumeRole\" } ] }"
    aws iam create-role --role-name ${SAGEMAKER_EXECUTION_ROLE_NAME} --assume-role-policy-document "$TRUST"
    aws iam attach-role-policy --role-name ${SAGEMAKER_EXECUTION_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
    aws iam attach-role-policy --role-name ${SAGEMAKER_EXECUTION_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

    aws iam get-role --role-name ${SAGEMAKER_EXECUTION_ROLE_NAME} --output text --query 'Role.Arn'

Add access to additional IAM users or roles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use an intuitive IDE like Jupyter or want other people in your
organization to use the cluster you set up, you can also give them
access. The following steps run through this workflow using Amazon
SageMaker notebooks. An Amazon SageMaker notebook instance is a fully
managed Amazon EC2 compute instance that runs the Jupyter Notebook App.
You use the notebook instance to create and manage Jupyter notebooks to
create ML workflows. You can define, compile, deploy, and run your
pipeline using the KFP Python SDK or CLI. If you’re not using an Amazon
SageMaker notebook to run Jupyter, you need to install the `AWS
CLI  <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html>`__\ and
the latest version
of `kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl/#install-kubectl>`__.

Follow the steps in `Create an Amazon SageMaker Notebook
Instance <https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html>`__
to create a Amazon SageMaker notebook instance if you do not already
have one. Give the IAM role for this instance the ``S3FullAccess``
permission.

Amazon EKS clusters use IAM users and roles to control access to the
cluster. The rules are implemented in a config map named ``aws-auth``.
Only the user/role that has access to the cluster will be able to edit
this config map. Run the following from the command line of your gateway
node to get the IAM role of the notebook instance you created.
Replace ``<instance-name>`` with the name of your instance.

::

    aws sagemaker describe-notebook-instance --notebook-instance-name <instance-name> --region <region> --output text --query 'RoleArn'

This command outputs the IAM role ARN in
the ``arn:aws:iam::<account-id>:role/<role-name>`` format. Take note
of this ARN.

Run the following to attach the policies the IAM role.
Replace ``<role-name>`` with the ``<role-name>`` in your ARN.

::

    aws iam attach-role-policy --role-name <role-name> --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
    aws iam attach-role-policy --role-name <role-name> --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
    aws iam attach-role-policy --role-name <role-name> --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

``eksctl`` provides commands to read and edit the ``aws-auth`` config
map. ``system:masters`` is one of the default user groups. You add the
user to this group. The "system:masters" group has super user
permissions to the cluster. You can also create a group with more
restrictive permissions or you can bind permissions directly to users.
Replace ``<IAM-Role-arn>`` with the ARN of the IAM
role. ``<your_username>`` can be any unique username.

::

    eksctl create iamidentitymapping \
        --cluster <cluster-name> \
        --arn <IAM-Role-arn> \
        --group system:masters \
        --username <your-username> \
        --region <region>

Open the Jupyter notebook on your Amazon SageMaker instance and run the
following to verify that it has access to the cluster.

::

    aws eks --region <region> update-kubeconfig --name <cluster-name>
    kubectl -n kubeflow get all | grep pipeline

Running the Kubeflow Pipeline
-----------------------------

Now that setup of your gateway node and Amazon EKS cluster is complete,
you can create your classification pipeline. To create your pipeline,
you need to define and compile it. You then deploy it and use it to run
workflows. You can define your pipeline in Python and use the KFP
dashboard, KFP CLI, or Python SDK to compile, deploy, and run your
workflows.

Prepare datasets
~~~~~~~~~~~~~~~~

To run the pipelines, you need to have the datasets in an S3 bucket in
your account. This bucket must be located in the region where you want
to run Amazon SageMaker jobs. If you don’t have a bucket, create one
using the steps in `Creating a
bucket <https://docs.aws.amazon.com/AmazonS3/latest/gsg/CreatingABucket.html>`__.

From your gateway node, run the `sample dataset
creation <https://github.com/kubeflow/pipelines/tree/34615cb19edfacf9f4d9f2417e9254d52dd53474/samples/contrib/aws-samples/mnist-kmeans-sagemaker#the-sample-dataset>`__
script to copy the datasets into your bucket. Change the bucket name in
the script to the one you created.

Create a Kubeflow Pipeline using Amazon SageMaker Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full code for the MNIST classification pipeline is available in the
`Kubeflow Github
repository <https://github.com/kubeflow/pipelines/blob/master/samples/contrib/aws-samples/mnist-kmeans-sagemaker>`__.
To use it, clone the example Python files to your gateway node.

Input Parameters
^^^^^^^^^^^^^^^^

The full MNIST classification pipeline has run-specific parameters that
you must provide values for when creating a run. You must provide these
parameters for each component of your pipeline. These parameters can
also be updated when using other pipelines. We have provided default
values for all parameters in the sample classification pipeline file.

The following are the only parameters you may need to modify to run the
sample pipelines. To modify these parameters, update their entries in
the sample classification pipeline file.

-  **Role-ARN:** This must be the ARN of an IAM role that has full
   Amazon SageMaker access in your AWS account. Use the ARN
   of  ``kfp-example-pod-role``.

-  **The Dataset Buckets**: You must change the S3 bucket with the input
   data for each of the components. Replace the following with the link
   to your S3 bucket:

   -  **Train channel:** ``"S3Uri": "s3://<your-s3-bucket-name>/data"``

   -  **HPO channels for test/HPO channel for
      train:** ``"S3Uri": "s3://<your-s3-bucket-name>/data"``

   -  **Batch
      transform:** ``"batch-input": "s3://<your-s3-bucket-name>/data"``

-  **Output buckets:** Replace the output buckets with S3 buckets you
   have write permission to. Replace the following with the link to your
   S3 bucket:

   -  **Training/HPO**:
      ``output_location='s3://<your-s3-bucket-name>/output'``

   -  **Batch Transform**:
      ``batch_transform_ouput='s3://<your-s3-bucket-name>/output'``

-  **Region:**\ The default pipelines work in us-east-1. If your
   cluster is in a different region, update the following:

   -  The ``region='us-east-1'`` Parameter in the input list.

   -  The algorithm images for Amazon SageMaker. If you use one of
      the Amazon SageMaker built-in algorithm images, select the image
      for your region. Construct the image name using the information
      in `Common parameters for built-in
      algorithms <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html>`__.
      For Example:

      ::

          382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:1

   -  The S3 buckets with the dataset. Use the steps in Prepare datasets
      to copy the data to a bucket in the same region as the cluster.

You can adjust any of the input parameters using the KFP UI and trigger
your run again.

Compile and deploy your pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After defining the pipeline in Python, you must compile the pipeline to
an intermediate representation before you can submit it to the Kubeflow
Pipelines service. The intermediate representation is a workflow
specification in the form of a YAML file compressed into a tar.gz
file. You need the KFP SDK to compile your pipeline.

Install KFP SDK
^^^^^^^^^^^^^^^

Run the following from the command line of your gateway node:

Install the KFP SDK following the instructions in the \ `Kubeflow
pipelines
documentation <https://www.kubeflow.org/docs/pipelines/sdk/install-sdk/>`__.

Verify that the KFP SDK is installed with the following command:

::

    pip show kfp

Verify that ``dsl-compile`` has been installed correctly as follows:

::

    which dsl-compile

Compile your pipeline
^^^^^^^^^^^^^^^^^^^^^

You have three options to interact with Kubeflow Pipelines: KFP UI, KFP
CLI, or the KFP SDK. The following sections illustrate the workflow
using the KFP UI and CLI.

Complete the following from your gateway node to compile your pipeline.

Modify your Python file with your S3 bucket name and IAM role ARN.

Use the ``dsl-compile`` command from the command line to compile your
pipeline as follows. Replace ``<path-to-python-file>`` with the path
to your pipeline and ``<path-to-output>`` with the location where you
want your tar.gz file to be.

::

    dsl-compile --py <path-to-python-file> --output <path-to-output>

Upload and run the pipeline using the KFP CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Complete the following steps from the command line of your gateway node.
KFP organizes runs of your pipeline as experiments. You have the option
to specify an experiment name. If you do not specify one, the run will
be listed under ‘Default’ experiment.

Upload your pipeline as follows:

::

    kfp pipeline upload --pipeline-name <pipeline-name> <path-to-output-tar.gz>

Your output should look like the following. Take note of the \ ``ID``.

::

    Pipeline 29c3ff21-49f5-4dfe-94f6-618c0e2420fe has been submitted

    Pipeline Details
    ------------------
    ID           29c3ff21-49f5-4dfe-94f6-618c0e2420fe
    Name         sm-pipeline
    Description
    Uploaded at  2020-04-30T20:22:39+00:00
    ...
    ...

Create a run using the following command. The KFP CLI run command
currently does not support specifying input parameters while creating
the run. You need to update your parameters in the Python pipeline file
before compiling. Replace ``<experiment-name>`` and ``<job-name>``
with any names. Replace ``<pipeline-id>`` with the ID of your submitted
pipeline.

::

    kfp run submit --experiment-name <experiment-name> --run-name <job-name> --pipeline-id <pipeline-id>

You can also directly submit a run using the compiled pipeline package
created as the output of the ``dsl-compile`` command.

::

    kfp run submit --experiment-name <experiment-name> --run-name <job-name> --package-file <path-to-output>

Your output should look like the following:

::

    Creating experiment aws.
    Run 95084a2c-f18d-4b77-a9da-eba00bf01e63 is submitted
    +--------------------------------------+--------+----------+---------------------------+
    | run id                               | name   | status   | created at                |
    +======================================+========+==========+===========================+
    | 95084a2c-f18d-4b77-a9da-eba00bf01e63 | sm-job |          | 2020-04-30T20:36:41+00:00 |
    +--------------------------------------+--------+----------+---------------------------+

Navigate to the UI to check the progress of the job

Upload and run the pipeline using the KFP UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  On the left panel, choose the **Pipelines** tab.

-  In the upper-right corner, choose ``+UploadPipeline``.

-  Enter the pipeline name and description.

-  Choose ``Upload a file`` and enter the path to the tar.gz file you
   created using the CLI or with the Python SDK.

-  On the left panel, choose the **Pipelines** tab.

-  Find the pipeline you created.

-  Choose ``+CreateRun``.

-  Enter your input parameters.

-  Choose ``Run``.

Running predictions
~~~~~~~~~~~~~~~~~~~

Once your classification pipeline is deployed, you can run
classification predictions against the endpoint that was created by the
Deploy component. Use the KFP UI to check the output artifacts
for ``sagemaker-deploy-model-endpoint_name``. Download the .tgz
file to extract the endpoint name or check the Amazon SageMaker console
in the region you used.

Configure permissions to run predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to run predictions from your gateway node, skip this
section.

If you want to use any other machine to run predictions, assign
the ``sagemaker:InvokeEndpoint`` permission to the IAM role or IAM
user used by the client machine. This permission is used to run
predictions.

On your gateway node, run the following to create a policy file:

::

    cat <<EoF > ./sagemaker-invoke.json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:InvokeEndpoint"
                ],
                "Resource": "*"
            }
        ]
    }
    EoF

Attach the policy to the client node’s IAM role or IAM user.

If your client machine has an IAM role attached, run the following.
Replace ``<your-instance-IAM-role>`` with the name of the client
node’s IAM role. Replace ``<path-to-sagemaker-invoke-json>`` with the
path to the policy file you created.

::

    aws iam put-role-policy --role-name <your-instance-IAM-role> --policy-name sagemaker-invoke-for-worker --policy-document file://<path-to-sagemaker-invoke-json>

If your client machine has IAM user credentials configured, run the
following. Replace ``<your_IAM_user_name>`` with the name of the client
node’s IAM user. Replace ``<path-to-sagemaker-invoke-json>`` with the
path to the policy file you created.

::

    aws iam put-user-policy --user-name <your-IAM-user-name> --policy-name sagemaker-invoke-for-worker --policy-document file://<path-to-sagemaker-invoke-json>

Run predictions
^^^^^^^^^^^^^^^

Create a Python file from your client machine
named ``mnist-predictions.py`` with the following content . Replace
the ``ENDPOINT_NAME`` and ``REGION`` variables. This script loads the
MNIST dataset, then creates a CSV from those digits and sends it to the
endpoint for prediction. It then outputs the results.

::

    import pickle, gzip, numpy, urllib.request, json
    from urllib.parse import urlparse
    import json
    import io
    import boto3

    ENDPOINT_NAME='<endpoint-name>'
    REGION = '<region>'

    # Load the dataset
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    # Simple function to create a csv from our numpy array
    def np2csv(arr):
        csv = io.BytesIO()
        numpy.savetxt(csv, arr, delimiter=',', fmt='%g')
        return csv.getvalue().decode().rstrip()

    runtime = boto3.Session(region_name=REGION).client('sagemaker-runtime')

    payload = np2csv(train_set[0][30:31])

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
    result = json.loads(response['Body'].read().decode())
    print(result)

Run the Python file as follows:

::

    python mnist-predictions.py

View results and logs
~~~~~~~~~~~~~~~~~~~~~

When the pipeline is running, you can click on any component to check
execution details, such as inputs and outputs. This will list the names
of created resources.

If the KFP request is successfully processed and an Amazon SageMaker job
is created, the component logs in the KFP UI will provide a link to the
job created in Amazon SageMaker. The CloudWatch logs will also be
provided if the job is successfully created.

If you run too many pipeline jobs on the same cluster, you may see an
error message that indicates you do not have enough pods available. To
fix this, log in to your gateway node and delete the pods created by the
pipelines you are not using as follows:

::

    kubectl get pods -n kubeflow
    kubectl delete pods -n kubeflow <name-of-pipeline-pod>

Cleanup
~~~~~~~

When you’re finished with your pipeline, you need to cleanup your
resources.

From the KFP dashboard, terminate your pipeline runs if they do not exit
properly by clicking ``Terminate``.

If the ``Terminate`` option doesn’t work, log in to your gateway node
and terminate all the pods created by your pipeline run manually as
follows:

::

    kubectl get pods -n kubeflow
    kubectl delete pods -n kubeflow <name-of-pipeline-pod>

Using your AWS account, log in to the Amazon SageMaker service. Manually
stop all training, batch transform, and HPO jobs. Delete models, data
buckets and endpoints to avoid incurring any additional
costs. Terminating the pipeline runs does not stop the jobs in Amazon
SageMaker.
