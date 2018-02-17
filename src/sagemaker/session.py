# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import print_function, absolute_import

import logging
import re

import os
import sys
import time

import boto3
import json
import six
from botocore.exceptions import ClientError

from sagemaker.user_agent import prepend_user_agent
from sagemaker.utils import name_from_image
import sagemaker.logs


logging.basicConfig()
LOGGER = logging.getLogger('sagemaker')
LOGGER.setLevel(logging.INFO)


class LogState(object):
    STARTING = 1
    WAIT_IN_PROGRESS = 2
    TAILING = 3
    JOB_COMPLETE = 4
    COMPLETE = 5


class Session(object):
    """Manage interactions with the Amazon SageMaker APIs and any other AWS services needed.

    This class provides convenient methods for manipulating entities and resources that Amazon SageMaker uses,
    such as training jobs, endpoints, and input datasets in S3.

    AWS service calls are delegated to an underlying Boto3 session, which by default
    is initialized using the AWS configuration chain. When you make an Amazon SageMaker API call that
    accesses an S3 bucket location and one is not specified, the ``Session`` creates a default bucket based on
    a naming convention which includes the current AWS account ID.
    """

    def __init__(self, boto_session=None, sagemaker_client=None, sagemaker_runtime_client=None):
        """Initialize a SageMaker ``Session``.

        Args:
            boto_session (boto3.session.Session): The underlying Boto3 session which AWS service calls
                are delegated to (default: None). If not provided, one is created with default AWS configuration chain.
            sagemaker_client (boto3.SageMaker.Client): Client which makes Amazon SageMaker service calls other
                than ``InvokeEndpoint`` (default: None). Estimators created using this ``Session`` use this client.
                If not provided, one will be created using this instance's ``boto_session``.
            sagemaker_runtime_client (boto3.SageMakerRuntime.Client): Client which makes ``InvokeEndpoint``
                calls to Amazon SageMaker (default: None). Predictors created using this ``Session`` use this client.
                If not provided, one will be created using this instance's ``boto_session``.
        """
        self._default_bucket = None
        self.boto_session = boto_session or boto3.Session()

        region = self.boto_session.region_name
        if region is None:
            raise ValueError('Must setup local AWS configuration with a region supported by SageMaker.')

        self.sagemaker_client = sagemaker_client or self.boto_session.client('sagemaker')
        prepend_user_agent(self.sagemaker_client)

        self.sagemaker_runtime_client = sagemaker_runtime_client or self.boto_session.client('runtime.sagemaker')
        prepend_user_agent(self.sagemaker_runtime_client)

    @property
    def boto_region_name(self):
        return self.boto_session.region_name

    def upload_data(self, path, bucket=None, key_prefix='data'):
        """Upload local file or directory to S3.

        If a single file is specified for upload, the resulting S3 object key is ``{key_prefix}/{filename}``
        (filename does not include the local path, if any specified).

        If a directory is specified for upload, the API uploads all content, recursively,
        preserving relative structure of subdirectories. The resulting object key names are:
        ``{key_prefix}/{relative_subdirectory_path}/filename``.

        Args:
            path (str): Path (absolute or relative) of local file or directory to upload.
            bucket (str): Name of the S3 Bucket to upload to (default: None). If not specified, the
                default bucket of the ``Session`` is used. If the bucket does not exist, the ``Session``
                creates the bucket.
            key_prefix (str): Optional S3 object key name prefix (default: 'data'). S3 uses the prefix to
                create a directory structure for the bucket content that it display in the S3 console.

        Returns:
            str: The S3 URI of the uploaded file(s). If a file is specified in the path argument, the URI format is:
                ``s3://{bucket name}/{key_prefix}/{original_file_name}``.
                If a directory is specified in the path argument, the URI format is ``s3://{bucket name}/{key_prefix}``.
        """
        # Generate a tuple for each file that we want to upload of the form (local_path, s3_key).
        files = []
        key_suffix = None
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for name in filenames:
                    local_path = os.path.join(dirpath, name)
                    s3_relative_prefix = '' if path == dirpath else os.path.relpath(dirpath, start=path) + '/'
                    s3_key = '{}/{}{}'.format(key_prefix, s3_relative_prefix, name)
                    files.append((local_path, s3_key))
        else:
            _, name = os.path.split(path)
            s3_key = '{}/{}'.format(key_prefix, name)
            files.append((path, s3_key))
            key_suffix = name

        bucket = bucket or self.default_bucket()
        s3 = self.boto_session.resource('s3')

        for local_path, s3_key in files:
            s3.Object(bucket, s3_key).upload_file(local_path)

        s3_uri = 's3://{}/{}'.format(bucket, key_prefix)
        # If a specific file was used as input (instead of a directory), we return the full S3 key
        # of the uploaded object. This prevents unintentionally using other files under the same prefix
        # during training.
        if key_suffix:
            s3_uri = '{}/{}'.format(s3_uri, key_suffix)
        return s3_uri

    def default_bucket(self):
        """Return the name of the default bucket to use in relevant Amazon SageMaker interactions.

        Returns:
            str: The name of the default bucket, which is of the form: ``sagemaker-{region}-{AWS account ID}``.
        """
        if self._default_bucket:
            return self._default_bucket

        s3 = self.boto_session.resource('s3')
        account = self.boto_session.client('sts').get_caller_identity()['Account']
        region = self.boto_session.region_name
        default_bucket = 'sagemaker-{}-{}'.format(region, account)

        try:
            # 'us-east-1' cannot be specified because it is the default region:
            # https://github.com/boto/boto3/issues/125
            if region == 'us-east-1':
                s3.create_bucket(Bucket=default_bucket)
            else:
                s3.create_bucket(Bucket=default_bucket, CreateBucketConfiguration={'LocationConstraint': region})

            LOGGER.info('Created S3 bucket: {}'.format(default_bucket))
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']

            if error_code == 'BucketAlreadyOwnedByYou':
                pass
            elif error_code == 'OperationAborted' and 'conflicting conditional operation' in message:
                # If this bucket is already being concurrently created, we don't need to create it again.
                pass
            elif error_code == 'TooManyBuckets':
                # Succeed if the default bucket exists
                try:
                    s3.meta.client.head_bucket(Bucket=default_bucket)
                    pass
                except ClientError:
                    raise
            else:
                raise

        self._default_bucket = default_bucket

        return self._default_bucket

    def train(self, image, input_mode, input_config, role, job_name, output_config,
              resource_config, hyperparameters, stop_condition):
        """Create an Amazon SageMaker training job.

        Args:
            image (str): Docker image containing training code.
            input_mode (str): The input mode that the algorithm supports. Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                    a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

            input_config (list): A list of Channel objects. Each channel is a named input source. Please refer to
                 the format details described:
                 https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            job_name (str): Name of the training job being created.
            output_config (dict): The S3 URI where you want to store the training results and optional KMS key ID.
            resource_config (dict): Contains values for ResourceConfig:
            instance_count (int): Number of EC2 instances to use for training.
            instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            hyperparameters (dict): Hyperparameters for model training. The hyperparameters are made accessible as
                a dict[str, str] to the training code on SageMaker. For convenience, this accepts other types for
                keys and values, but ``str()`` will be called to convert them before training.
            stop_condition (dict): Defines when training shall finish. Contains entries that can be understood by the
                service like ``MaxRuntimeInSeconds``.

        Returns:
            str: ARN of the training job, if it is created.
        """

        train_request = {
            'AlgorithmSpecification': {
                'TrainingImage': image,
                'TrainingInputMode': input_mode
            },
            # 'HyperParameters': hyperparameters,
            'InputDataConfig': input_config,
            'OutputDataConfig': output_config,
            'TrainingJobName': job_name,
            "StoppingCondition": stop_condition,
            "ResourceConfig": resource_config,
            "RoleArn": role,
        }

        if hyperparameters and len(hyperparameters) > 0:
            train_request['HyperParameters'] = hyperparameters
        LOGGER.info('Creating training-job with name: {}'.format(job_name))
        LOGGER.debug('train request: {}'.format(json.dumps(train_request, indent=4)))
        self.sagemaker_client.create_training_job(**train_request)

    def create_model(self, name, role, primary_container):
        """Create an Amazon SageMaker ``Model``.

        Specify the S3 location of the model artifacts and Docker image containing
        the inference code. Amazon SageMaker uses this information to deploy the
        model in Amazon SageMaker.

        Args:
            name (str): Name of the Amazon SageMaker ``Model`` to create.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            primary_container (str or dict[str, str]): Docker image which defines the inference code.
                You can also specify the return value of ``sagemaker.container_def()``, which is used to create
                more advanced container configurations, including model containers which need artifacts from S3.

        Returns:
            str: Name of the Amazon SageMaker ``Model`` created.
        """
        role = self.expand_role(role)
        primary_container = _expand_container_def(primary_container)
        LOGGER.info('Creating model with name: {}'.format(name))
        LOGGER.debug("create_model request: {}".format({
            'name': name,
            'role': role,
            'primary_container': primary_container
        }))

        self.sagemaker_client.create_model(ModelName=name,
                                           PrimaryContainer=primary_container,
                                           ExecutionRoleArn=role)

        return name

    def create_model_from_job(self, training_job_name, name=None, role=None, primary_container_image=None,
                              model_data_url=None, env={}):
        """Create an Amazon SageMaker ``Model`` from a SageMaker Training Job.

        Args:
            training_job_name (str): The Amazon SageMaker Training Job name.
            name (str): The name of the SageMaker ``Model`` to create (default: None).
                If not specified, the training job name is used.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, specified either by an IAM role name or
                role ARN. If None, the ``RoleArn`` from the SageMaker Training Job will be used.
            primary_container_image (str): The Docker image reference (default: None). If None, it defaults to
                the Training Image in ``training_job_name``.
            model_data_url (str): S3 location of the model data (default: None). If None, defaults to
                the ``ModelS3Artifacts`` of ``training_job_name``.
            env (dict[string,string]): Model environment variables (default: {}).

        Returns:
            str: The name of the created ``Model``.
        """
        training_job = self.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        name = name or training_job_name
        role = role or training_job['RoleArn']
        primary_container = container_def(
            primary_container_image or training_job['AlgorithmSpecification']['TrainingImage'],
            model_data_url=model_data_url or training_job['ModelArtifacts']['S3ModelArtifacts'],
            env=env)
        return self.create_model(name, role, primary_container)

    def create_endpoint_config(self, name, model_name, initial_instance_count, instance_type):
        """Create an Amazon SageMaker endpoint configuration.

        The endpoint configuration identifies the Amazon SageMaker model (created using the
        ``CreateModel`` API) and the hardware configuration on which to deploy the model. Provide this
        endpoint configuration to the ``CreateEndpoint`` API, which then launches the hardware and deploys the model.

        Args:
            name (str): Name of the Amazon SageMaker endpoint configuration to create.
            model_name (str): Name of the Amazon SageMaker ``Model``.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual number of
                active instances for an endpoint at any given time varies due to autoscaling.
            instance_type (str): Type of EC2 instance to launch, for example, 'ml.c4.xlarge'.

        Returns:
            str: Name of the endpoint point configuration created.
        """
        LOGGER.info('Creating endpoint-config with name {}'.format(name))
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=name,
            ProductionVariants=[{'ModelName': model_name,
                                 'InitialInstanceCount': initial_instance_count,
                                 'InstanceType': instance_type,
                                 'VariantName': 'AllTraffic'}])
        return name

    def create_endpoint(self, endpoint_name, config_name, wait=True):
        """Create an Amazon SageMaker ``Endpoint`` according to the endpoint configuration specified in the request.

        Once the ``Endpoint`` is created, client applications can send requests to obtain inferences.
        The endpoint configuration is created using the ``CreateEndpointConfig`` API.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` being created.
            config_name (str): Name of the Amazon SageMaker endpoint configuration to deploy.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).

        Returns:
            str: Name of the Amazon SageMaker ``Endpoint`` created.
        """
        LOGGER.info('Creating endpoint with name {}'.format(endpoint_name))
        self.sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        if wait:
            self.wait_for_endpoint(endpoint_name)
        return endpoint_name

    def delete_endpoint(self, endpoint_name):
        """Delete an Amazon SageMaker ``Endpoint``.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to delete.
        """
        LOGGER.info('Deleting endpoint with name: {}'.format(endpoint_name))
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    def wait_for_job(self, job, poll=5):
        """Wait for an Amazon SageMaker training job to complete.

        Args:
            job (str): Name of the training job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeTrainingJob`` API.

        Raises:
            ValueError: If the training job fails.
        """
        desc = _wait_until(lambda: _train_done(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc)
        return desc

    def _check_job_status(self, job, desc):
        """Check to see if the job completed successfully and, if not, construct and
        raise a ValueError.

        Args:
            job (str): The name of the job to check.
            desc (dict[str, str]): The result of ``describe_training_job()``.

        Raises:
            ValueError: If the training job fails.
        """
        status = desc['TrainingJobStatus']

        if status != 'Completed':
            reason = desc.get('FailureReason', '(No reason provided)')
            raise ValueError('Error training {}: {} Reason: {}'.format(job, status, reason))

    def wait_for_endpoint(self, endpoint, poll=5):
        """Wait for an Amazon SageMaker endpoint deployment to complete.

        Args:
            endpoint (str): Name of the ``Endpoint`` to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            dict: Return value from the ``DescribeEndpoint`` API.
        """
        desc = _wait_until(lambda: _deploy_done(self.sagemaker_client, endpoint), poll)
        status = desc['EndpointStatus']

        if status != 'InService':
            reason = desc.get('FailureReason', None)
            raise ValueError('Error hosting endpoint {}: {} Reason: {}'.format(endpoint, status, reason))
        return desc

    def endpoint_from_job(self, job_name, initial_instance_count, instance_type,
                          deployment_image=None, name=None, role=None, wait=True,
                          model_environment_vars=None):
        """Create an ``Endpoint`` using the results of a successful training job.

        Specify the job name, Docker image containing the inference code, and hardware configuration to deploy
        the model. Internally the API, creates an Amazon SageMaker model (that describes the model artifacts and
        the Docker image containing inference code), endpoint configuration (describing the hardware to deploy
        for hosting the model), and creates an ``Endpoint`` (launches the EC2 instances and deploys the model on them).
        In response, the API returns the endpoint name to which you can send requests for inferences.

        Args:
            job_name (str): Name of the training job to deploy the results of.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual number of
                active instances for an endpoint at any given time varies due to autoscaling.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
                for example, 'ml.c4.xlarge'.
            deployment_image (str): The Docker image which defines the inference code to be used as the entry point for
                accepting prediction requests. If not specified, uses the image used for the training job.
            name (str): Name of the ``Endpoint`` to create. If not specified, uses the training job name.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).
            model_environment_vars (dict[str, str]): Environment variables to set on the model container
                (default: None).

        Returns:
            str: Name of the ``Endpoint`` that is created.
        """
        job_desc = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        output_url = job_desc['ModelArtifacts']['S3ModelArtifacts']
        deployment_image = deployment_image or job_desc['AlgorithmSpecification']['TrainingImage']
        role = role or job_desc['RoleArn']
        name = name or job_name

        return self.endpoint_from_model_data(model_s3_location=output_url, deployment_image=deployment_image,
                                             initial_instance_count=initial_instance_count, instance_type=instance_type,
                                             name=name, role=role, wait=wait,
                                             model_environment_vars=model_environment_vars)

    def endpoint_from_model_data(self, model_s3_location, deployment_image, initial_instance_count, instance_type,
                                 name=None, role=None, wait=True, model_environment_vars=None):
        """Create and deploy to an ``Endpoint`` using existing model data stored in S3.

        Args:
            model_s3_location (str): S3 URI of the model artifacts to use for the endpoint.
            deployment_image (str): The Docker image which defines the runtime code to be used as
                the entry point for accepting prediction requests.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual number of
                active instances for an endpoint at any given time varies due to autoscaling.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction, e.g. 'ml.c4.xlarge'.
            name (str): Name of the ``Endpoint`` to create. If not specified, uses a name generated by
                combining the image name with a timestamp.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                You must grant sufficient permissions to this role.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).
            model_environment_vars (dict[str, str]): Environment variables to set on the model container
                (default: None).

        Returns:
            str: Name of the ``Endpoint`` that is created.
        """

        model_environment_vars = model_environment_vars or {}
        name = name or name_from_image(deployment_image)

        if _deployment_entity_exists(lambda: self.sagemaker_client.describe_endpoint(EndpointName=name)):
            raise ValueError('Endpoint with name "{}" already exists; please pick a different name.'.format(name))

        if not _deployment_entity_exists(lambda: self.sagemaker_client.describe_model(ModelName=name)):
            self.create_model(name=name,
                              role=role,
                              primary_container=container_def(image=deployment_image,
                                                              model_data_url=model_s3_location,
                                                              env=model_environment_vars))

        if not _deployment_entity_exists(
                lambda: self.sagemaker_client.describe_endpoint_config(EndpointConfigName=name)):
            self.create_endpoint_config(name=name,
                                        model_name=name,
                                        initial_instance_count=initial_instance_count,
                                        instance_type=instance_type)

        self.create_endpoint(endpoint_name=name, config_name=name, wait=wait)
        return name

    def endpoint_from_production_variants(self, name, production_variants, wait=True):
        """Create an SageMaker ``Endpoint`` from a list of production variants.

        Args:
            name (str): The name of the ``Endpoint`` to create.
            production_variants (list[dict[str, str]]): The list of production variants to deploy.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning (default: True).

        Returns:
            str: The name of the created ``Endpoint``.
        """

        if not _deployment_entity_exists(
                lambda: self.sagemaker_client.describe_endpoint_config(EndpointConfigName=name)):
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=name, ProductionVariants=production_variants)
        return self.create_endpoint(endpoint_name=name, config_name=name, wait=wait)

    def expand_role(self, role):
        """Expand an IAM role name into an ARN.

        If the role is already in the form of an ARN, then the role is simply returned. Otherwise we retrieve the full
        ARN and return it.

        Args:
            role (str): An AWS IAM role (either name or full ARN).

        Returns:
            str: The corresponding AWS IAM role ARN.
        """
        if '/' in role:
            return role
        else:
            return boto3.resource("iam").Role(role).arn

    def get_caller_identity_arn(self):
        """Returns the ARN user or role whose credentials are used to call the API.
        Returns:
            (str): The ARN uer or role
        """
        assumed_role = self.boto_session.client('sts').get_caller_identity()['Arn']

        if 'AmazonSageMaker-ExecutionRole' in assumed_role:
            role = re.sub(r'^(.+)sts::(\d+):assumed-role/(.+?)/.*$', r'\1iam::\2:role/service-role/\3', assumed_role)
            return role

        role = re.sub(r'^(.+)sts::(\d+):assumed-role/(.+?)/.*$', r'\1iam::\2:role/\3', assumed_role)
        return role

    def logs_for_job(self, job_name, wait=False, poll=5):  # noqa: C901 - suppress complexity warning for this method
        """Display the logs for a given training job, optionally tailing them until the
        job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
        based on which instance the log entry is from.

        Args:
            job_name (str): Name of the training job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes (default: True).
            poll (int): The interval in seconds between polling for new log entries and job completion (default: 5).

        Raises:
            ValueError: If waiting and the training job fails.
        """

        description = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        instance_count = description['ResourceConfig']['InstanceCount']
        status = description['TrainingJobStatus']

        stream_names = []  # The list of log streams
        positions = {}     # The current position in each stream, map of stream name -> position
        client = self.boto_session.client('logs')
        log_group = '/aws/sagemaker/TrainingJobs'

        job_already_completed = True if status == 'Completed' or status == 'Failed' else False

        state = LogState.TAILING if wait and not job_already_completed else LogState.COMPLETE
        dot = False

        color_wrap = sagemaker.logs.ColorWrap()

        # The loop below implements a state machine that alternates between checking the job status and
        # reading whatever is available in the logs at this point. Note, that if we were called with
        # wait == False, we never check the job status.
        #
        # If wait == TRUE and job is not completed, the initial state is TAILING
        # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is complete).
        #
        # The state table:
        #
        # STATE               ACTIONS                        CONDITION             NEW STATE
        # ----------------    ----------------               -----------------     ----------------
        # TAILING             Read logs, Pause, Get status   Job complete          JOB_COMPLETE
        #                                                    Else                  TAILING
        # JOB_COMPLETE        Read logs, Pause               Any                   COMPLETE
        # COMPLETE            Read logs, Exit                                      N/A
        #
        # Notes:
        # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to Cloudwatch after
        #   the job was marked complete.
        last_describe_job_call = time.time()
        while True:
            if len(stream_names) < instance_count:
                # Log streams are created whenever a container starts writing to stdout/err, so this list
                # may be dynamic until we have a stream for every instance.
                try:
                    streams = client.describe_log_streams(logGroupName=log_group, logStreamNamePrefix=job_name + '/',
                                                          orderBy='LogStreamName', limit=instance_count)
                    stream_names = [s['logStreamName'] for s in streams['logStreams']]
                    positions.update([(s, sagemaker.logs.Position(timestamp=0, skip=0))
                                      for s in stream_names if s not in positions])
                except ClientError as e:
                    # On the very first training job run on an account, there's no log group until
                    # the container starts logging, so ignore any errors thrown about that
                    err = e.response.get('Error', {})
                    if err.get('Code', None) != 'ResourceNotFoundException':
                        raise

            if len(stream_names) > 0:
                if dot:
                    print('')
                    dot = False
                for idx, event in sagemaker.logs.multi_stream_iter(client, log_group, stream_names, positions):
                    color_wrap(idx, event['message'])
                    ts, count = positions[stream_names[idx]]
                    if event['timestamp'] == ts:
                        positions[stream_names[idx]] = sagemaker.logs.Position(timestamp=ts, skip=count + 1)
                    else:
                        positions[stream_names[idx]] = sagemaker.logs.Position(timestamp=event['timestamp'], skip=1)
            else:
                dot = True
                print('.', end='')
                sys.stdout.flush()
            if state == LogState.COMPLETE:
                break

            time.sleep(poll)

            if state == LogState.JOB_COMPLETE:
                state = LogState.COMPLETE
            elif time.time() - last_describe_job_call >= 30:
                description = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
                last_describe_job_call = time.time()

                status = description['TrainingJobStatus']

                if status == 'Completed' or status == 'Failed':
                    state = LogState.JOB_COMPLETE

        if wait:
            self._check_job_status(job_name, description)
            if dot:
                print()
            print('===== Job Complete =====')
            # Customers are not billed for hardware provisioning, so billable time is less than total time
            billable_time = (description['TrainingEndTime'] - description['TrainingStartTime']) * instance_count
            print('Billable seconds:', int(billable_time.total_seconds()) + 1)


def container_def(image, model_data_url=None, env=None):
    """Create a definition for executing a container as part of a SageMaker model.

    Args:
        image (str): Docker image to run for this container.
        model_data_url (str): S3 URI of data required by this container,
            e.g. SageMaker training job model artifacts (default: None).
        env (dict[str, str]): Environment variables to set inside the container (default: None).

    Returns:
        dict[str, str]: A complete container definition object usable with the CreateModel API.
    """
    if env is None:
        env = {}
    c_def = {'Image': image, 'Environment': env}
    if model_data_url:
        c_def['ModelDataUrl'] = model_data_url
    return c_def


def production_variant(model_name, instance_type, initial_instance_count=1, variant_name='AllTraffic',
                       initial_weight=1):
    """Create a production variant description suitable for use in a ``ProductionVariant`` list as part of a
    ``CreateEndpointConfig`` request.

    Args:
        model_name (str): The name of the SageMaker model this production variant references.
        instance_type (str): The EC2 instance type for this production variant. For example, 'ml.c4.8xlarge'.
        initial_instance_count (int): The initial instance count for this production variant (default: 1).
        variant_name (string): The ``VariantName`` of this production variant (default: 'AllTraffic').
        initial_weight (int): The relative ``InitialVariantWeight`` of this production variant (default: 1).

    Returns:
        dict[str, str]: An SageMaker ``ProductionVariant`` description
    """
    return {
        'InstanceType': instance_type,
        'InitialInstanceCount': initial_instance_count,
        'ModelName': model_name,
        'VariantName': variant_name,
        'InitialVariantWeight': initial_weight
    }


def get_execution_role(sagemaker_session=None):
    """Return the role ARN whose credentials are used to call the API.
    Throws an exception if
    Args:
        sagemaker_session(Session): Current sagemaker session
    Returns:
        (str): The role ARN
    """
    if not sagemaker_session:
        sagemaker_session = Session()
    arn = sagemaker_session.get_caller_identity_arn()

    if 'role' in arn:
        return arn
    message = 'The current AWS identity is not a role: {}, therefore it cannot be used as a SageMaker execution role'
    raise ValueError(message.format(arn))


class s3_input(object):
    """Amazon SageMaker channel configurations for S3 data sources.

    Attributes:
        config (dict[str, dict]): A SageMaker ``DataSource`` referencing a SageMaker ``S3DataSource``.
    """

    def __init__(self, s3_data, distribution='FullyReplicated', compression=None,
                 content_type=None, record_wrapping=None, s3_data_type='S3Prefix'):
        """Create a definition for input data used by an SageMaker training job.

        See AWS documentation on the ``CreateTrainingJob`` API for more details on the parameters.

        Args:
            s3_data (str): Defines the location of s3 data to train on.
            distribution (str): Valid values: 'FullyReplicated', 'ShardedByS3Key'
                (default: 'FullyReplicated').
            compression (str): Valid values: 'Gzip', 'Bzip2', 'Lzop' (default: None).
            content_type (str): MIME type of the input data (default: None).
            record_wrapping (str): Valid values: 'RecordIO' (default: None).
            s3_data_type (str): Value values: 'S3Prefix', 'ManifestFile'. If 'S3Prefix', ``s3_data`` defines
                a prefix of s3 objects to train on. All objects with s3 keys beginning with ``s3_data`` will
                be used to train. If 'ManifestFile', then ``s3_data`` defines a single s3 manifest file, listing
                each s3 object to train on. The Manifest file format is described in the SageMaker API documentation:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_S3DataSource.html
        """
        self.config = {
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': distribution,
                    'S3DataType': s3_data_type,
                    'S3Uri': s3_data
                }
            }
        }

        if compression is not None:
            self.config['CompressionType'] = compression
        if content_type is not None:
            self.config['ContentType'] = content_type
        if record_wrapping is not None:
            self.config['RecordWrapperType'] = record_wrapping


def _deployment_entity_exists(describe_fn):
    try:
        describe_fn()
        return True
    except ClientError as ce:
        if not (ce.response['Error']['Code'] == 'ValidationException' and
                'Could not find' in ce.response['Error']['Message']):
            raise ce
        return False


def _train_done(sagemaker_client, job_name):
    training_status_codes = {
        'Created': '-',
        'InProgress': '.',
        'Completed': '!',
        'Failed': '*',
        'Stopping': '>',
        'Stopped': 's',
        'Deleting': 'o',
        'Deleted': 'x'
    }
    in_progress_statuses = ['InProgress', 'Created']

    desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    status = desc['TrainingJobStatus']

    print(training_status_codes.get(status, '?'), end='')
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print('')
    return desc


def _deploy_done(sagemaker_client, endpoint_name):
    hosting_status_codes = {
        "OutOfService": "x",
        "Creating": "-",
        "Updating": "-",
        "InService": "!",
        "RollingBack": "<",
        "Deleting": "o",
        "Failed": "*"
    }
    in_progress_statuses = ['Creating', 'Updating']

    desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = desc['EndpointStatus']

    print(hosting_status_codes.get(status, '?'), end='')
    sys.stdout.flush()

    return None if status in in_progress_statuses else desc


def _wait_until(callable, poll=5):
    result = callable()
    while result is None:
        time.sleep(poll)
        result = callable()
    return result


def _expand_container_def(c_def):
    if isinstance(c_def, six.string_types):
        return container_def(c_def)
    return c_def
