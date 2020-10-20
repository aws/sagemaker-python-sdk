# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import, print_function

import json
import logging
import os
import re
import sys
import time
import warnings

import boto3
import botocore.config
from botocore.exceptions import ClientError
import six

import sagemaker.logs
from sagemaker import vpc_utils

from sagemaker.deprecations import deprecated_class
from sagemaker.inputs import ShuffleConfig, TrainingInput
from sagemaker.user_agent import prepend_user_agent
from sagemaker.utils import (
    name_from_image,
    secondary_training_status_changed,
    secondary_training_status_message,
    sts_regional_endpoint,
)
from sagemaker import exceptions

LOGGER = logging.getLogger("sagemaker")

NOTEBOOK_METADATA_FILE = "/opt/ml/metadata/resource-metadata.json"

_STATUS_CODE_TABLE = {
    "COMPLETED": "Completed",
    "INPROGRESS": "InProgress",
    "FAILED": "Failed",
    "STOPPED": "Stopped",
    "STOPPING": "Stopping",
    "STARTING": "Starting",
}


class LogState(object):
    """Placeholder docstring"""

    STARTING = 1
    WAIT_IN_PROGRESS = 2
    TAILING = 3
    JOB_COMPLETE = 4
    COMPLETE = 5


class Session(object):  # pylint: disable=too-many-public-methods
    """Manage interactions with the Amazon SageMaker APIs and any other AWS services needed.

    This class provides convenient methods for manipulating entities and resources that Amazon
    SageMaker uses, such as training jobs, endpoints, and input datasets in S3.

    AWS service calls are delegated to an underlying Boto3 session, which by default
    is initialized using the AWS configuration chain. When you make an Amazon SageMaker API call
    that accesses an S3 bucket location and one is not specified, the ``Session`` creates a default
    bucket based on a naming convention which includes the current AWS account ID.
    """

    def __init__(
        self,
        boto_session=None,
        sagemaker_client=None,
        sagemaker_runtime_client=None,
        default_bucket=None,
    ):
        """Initialize a SageMaker ``Session``.

        Args:
            boto_session (boto3.session.Session): The underlying Boto3 session which AWS service
                calls are delegated to (default: None). If not provided, one is created with
                default AWS configuration chain.
            sagemaker_client (boto3.SageMaker.Client): Client which makes Amazon SageMaker service
                calls other than ``InvokeEndpoint`` (default: None). Estimators created using this
                ``Session`` use this client. If not provided, one will be created using this
                instance's ``boto_session``.
            sagemaker_runtime_client (boto3.SageMakerRuntime.Client): Client which makes
                ``InvokeEndpoint`` calls to Amazon SageMaker (default: None). Predictors created
                using this ``Session`` use this client. If not provided, one will be created using
                this instance's ``boto_session``.
            default_bucket (str): The default Amazon S3 bucket to be used by this session.
                This will be created the next time an Amazon S3 bucket is needed (by calling
                :func:`default_bucket`).
                If not provided, a default bucket will be created based on the following format:
                "sagemaker-{region}-{aws-account-id}".
                Example: "sagemaker-my-custom-bucket".

        """
        self._default_bucket = None
        self._default_bucket_name_override = default_bucket
        self.s3_resource = None
        self.s3_client = None
        self.config = None

        self._initialize(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            sagemaker_runtime_client=sagemaker_runtime_client,
        )

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client):
        """Initialize this SageMaker Session.

        Creates or uses a boto_session, sagemaker_client and sagemaker_runtime_client.
        Sets the region_name.
        """
        self.boto_session = boto_session or boto3.DEFAULT_SESSION or boto3.Session()

        self._region_name = self.boto_session.region_name
        if self._region_name is None:
            raise ValueError(
                "Must setup local AWS configuration with a region supported by SageMaker."
            )

        self.sagemaker_client = sagemaker_client or self.boto_session.client("sagemaker")
        prepend_user_agent(self.sagemaker_client)

        if sagemaker_runtime_client is not None:
            self.sagemaker_runtime_client = sagemaker_runtime_client
        else:
            config = botocore.config.Config(read_timeout=80)
            self.sagemaker_runtime_client = self.boto_session.client(
                "runtime.sagemaker", config=config
            )

        prepend_user_agent(self.sagemaker_runtime_client)

        self.local_mode = False

    @property
    def boto_region_name(self):
        """Placeholder docstring"""
        return self._region_name

    def upload_data(self, path, bucket=None, key_prefix="data", extra_args=None):
        """Upload local file or directory to S3.

        If a single file is specified for upload, the resulting S3 object key is
        ``{key_prefix}/{filename}`` (filename does not include the local path, if any specified).

        If a directory is specified for upload, the API uploads all content, recursively,
        preserving relative structure of subdirectories. The resulting object key names are:
        ``{key_prefix}/{relative_subdirectory_path}/filename``.

        Args:
            path (str): Path (absolute or relative) of local file or directory to upload.
            bucket (str): Name of the S3 Bucket to upload to (default: None). If not specified, the
                default bucket of the ``Session`` is used (if default bucket does not exist, the
                ``Session`` creates it).
            key_prefix (str): Optional S3 object key name prefix (default: 'data'). S3 uses the
                prefix to create a directory structure for the bucket content that it display in
                the S3 console.
            extra_args (dict): Optional extra arguments that may be passed to the upload operation.
                Similar to ExtraArgs parameter in S3 upload_file function. Please refer to the
                ExtraArgs parameter documentation here:
                https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html#the-extraargs-parameter

        Returns:
            str: The S3 URI of the uploaded file(s). If a file is specified in the path argument,
                the URI format is: ``s3://{bucket name}/{key_prefix}/{original_file_name}``.
                If a directory is specified in the path argument, the URI format is
                ``s3://{bucket name}/{key_prefix}``.
        """
        # Generate a tuple for each file that we want to upload of the form (local_path, s3_key).
        files = []
        key_suffix = None
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for name in filenames:
                    local_path = os.path.join(dirpath, name)
                    s3_relative_prefix = (
                        "" if path == dirpath else os.path.relpath(dirpath, start=path) + "/"
                    )
                    s3_key = "{}/{}{}".format(key_prefix, s3_relative_prefix, name)
                    files.append((local_path, s3_key))
        else:
            _, name = os.path.split(path)
            s3_key = "{}/{}".format(key_prefix, name)
            files.append((path, s3_key))
            key_suffix = name

        bucket = bucket or self.default_bucket()
        if self.s3_resource is None:
            s3 = self.boto_session.resource("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_resource

        for local_path, s3_key in files:
            s3.Object(bucket, s3_key).upload_file(local_path, ExtraArgs=extra_args)

        s3_uri = "s3://{}/{}".format(bucket, key_prefix)
        # If a specific file was used as input (instead of a directory), we return the full S3 key
        # of the uploaded object. This prevents unintentionally using other files under the same
        # prefix during training.
        if key_suffix:
            s3_uri = "{}/{}".format(s3_uri, key_suffix)
        return s3_uri

    def upload_string_as_file_body(self, body, bucket, key, kms_key=None):
        """Upload a string as a file body.

        Args:
            body (str): String representing the body of the file.
            bucket (str): Name of the S3 Bucket to upload to (default: None). If not specified, the
                default bucket of the ``Session`` is used (if default bucket does not exist, the
                ``Session`` creates it).
            key (str): S3 object key. This is the s3 path to the file.
            kms_key (str): The KMS key to use for encrypting the file.

        Returns:
            str: The S3 URI of the uploaded file.
                The URI format is: ``s3://{bucket name}/{key}``.
        """
        if self.s3_resource is None:
            s3 = self.boto_session.resource("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_resource

        s3_object = s3.Object(bucket_name=bucket, key=key)

        if kms_key is not None:
            s3_object.put(Body=body, SSEKMSKeyId=kms_key, ServerSideEncryption="aws:kms")
        else:
            s3_object.put(Body=body)

        s3_uri = "s3://{}/{}".format(bucket, key)
        return s3_uri

    def download_data(self, path, bucket, key_prefix="", extra_args=None):
        """Download file or directory from S3.

        Args:
            path (str): Local path where the file or directory should be downloaded to.
            bucket (str): Name of the S3 Bucket to download from.
            key_prefix (str): Optional S3 object key name prefix.
            extra_args (dict): Optional extra arguments that may be passed to the
                download operation. Please refer to the ExtraArgs parameter in the boto3
                documentation here:
                https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-download-file.html

        Returns:

        """
        # Initialize the S3 client.
        if self.s3_client is None:
            s3 = self.boto_session.client("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_client

        # Initialize the variables used to loop through the contents of the S3 bucket.
        keys = []
        next_token = ""
        base_parameters = {"Bucket": bucket, "Prefix": key_prefix}

        # Loop through the contents of the bucket, 1,000 objects at a time. Gathering all keys into
        # a "keys" list.
        while next_token is not None:
            request_parameters = base_parameters.copy()
            if next_token != "":
                request_parameters.update({"ContinuationToken": next_token})
            response = s3.list_objects_v2(**request_parameters)
            contents = response.get("Contents")
            # For each object, save its key or directory.
            for s3_object in contents:
                key = s3_object.get("Key")
                keys.append(key)
            next_token = response.get("NextContinuationToken")

        # For each object key, create the directory on the local machine if needed, and then
        # download the file.
        for key in keys:
            tail_s3_uri_path = os.path.basename(key_prefix)
            if not os.path.splitext(key_prefix)[1]:
                tail_s3_uri_path = os.path.relpath(key, key_prefix)
            destination_path = os.path.join(path, tail_s3_uri_path)
            if not os.path.exists(os.path.dirname(destination_path)):
                os.makedirs(os.path.dirname(destination_path))
            s3.download_file(
                Bucket=bucket, Key=key, Filename=destination_path, ExtraArgs=extra_args
            )

    def read_s3_file(self, bucket, key_prefix):
        """Read a single file from S3.

        Args:
            bucket (str): Name of the S3 Bucket to download from.
            key_prefix (str): S3 object key name prefix.

        Returns:
            str: The body of the s3 file as a string.

        """
        if self.s3_client is None:
            s3 = self.boto_session.client("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_client

        # Explicitly passing a None kms_key to boto3 throws a validation error.
        s3_object = s3.get_object(Bucket=bucket, Key=key_prefix)

        return s3_object["Body"].read().decode("utf-8")

    def list_s3_files(self, bucket, key_prefix):
        """Lists the S3 files given an S3 bucket and key.

        Args:
            bucket (str): Name of the S3 Bucket to download from.
            key_prefix (str): S3 object key name prefix.

        Returns:
            [str]: The list of files at the S3 path.

        """
        if self.s3_resource is None:
            s3 = self.boto_session.resource("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_resource

        s3_bucket = s3.Bucket(name=bucket)
        s3_objects = s3_bucket.objects.filter(Prefix=key_prefix).all()
        return [s3_object.key for s3_object in s3_objects]

    def default_bucket(self):
        """Return the name of the default bucket to use in relevant Amazon SageMaker interactions.

        Returns:
            str: The name of the default bucket, which is of the form:
                ``sagemaker-{region}-{AWS account ID}``.
        """

        if self._default_bucket:
            return self._default_bucket

        region = self.boto_session.region_name

        default_bucket = self._default_bucket_name_override
        if not default_bucket:
            account = self.boto_session.client(
                "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
            ).get_caller_identity()["Account"]
            default_bucket = "sagemaker-{}-{}".format(region, account)

        self._create_s3_bucket_if_it_does_not_exist(bucket_name=default_bucket, region=region)

        self._default_bucket = default_bucket

        return self._default_bucket

    def _create_s3_bucket_if_it_does_not_exist(self, bucket_name, region):
        """Creates an S3 Bucket if it does not exist.
        Also swallows a few common exceptions that indicate that the bucket already exists or
        that it is being created.

        Args:
            bucket_name (str): Name of the S3 bucket to be created.
            region (str): The region in which to create the bucket.

        Raises:
            botocore.exceptions.ClientError: If S3 throws an unexpected exception during bucket
                creation.
                If the exception is due to the bucket already existing or
                already being created, no exception is raised.

        """
        if self.s3_resource is None:
            s3 = self.boto_session.resource("s3", region_name=region)
        else:
            s3 = self.s3_resource

        bucket = s3.Bucket(name=bucket_name)
        if bucket.creation_date is None:
            try:
                if region == "us-east-1":
                    # 'us-east-1' cannot be specified because it is the default region:
                    # https://github.com/boto/boto3/issues/125
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
                    )

                LOGGER.info("Created S3 bucket: %s", bucket_name)
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                message = e.response["Error"]["Message"]

                if error_code == "BucketAlreadyOwnedByYou":
                    pass
                elif (
                    error_code == "OperationAborted"
                    and "conflicting conditional operation" in message
                ):
                    # If this bucket is already being concurrently created, we don't need to create
                    # it again.
                    pass
                else:
                    raise

    def train(  # noqa: C901
        self,
        input_mode,
        input_config,
        role,
        job_name,
        output_config,
        resource_config,
        vpc_config,
        hyperparameters,
        stop_condition,
        tags,
        metric_definitions,
        enable_network_isolation=False,
        image_uri=None,
        algorithm_arn=None,
        encrypt_inter_container_traffic=False,
        use_spot_instances=False,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        experiment_config=None,
        debugger_rule_configs=None,
        debugger_hook_config=None,
        tensorboard_output_config=None,
        enable_sagemaker_metrics=None,
    ):
        """Create an Amazon SageMaker training job.

        Args:
            input_mode (str): The input mode that the algorithm supports. Valid modes:
                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a
                Unix-named pipe.

            input_config (list): A list of Channel objects. Each channel is a named input source.
                Please refer to the format details described:
                https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. You must grant sufficient permissions to this
                role.
            job_name (str): Name of the training job being created.
            output_config (dict): The S3 URI where you want to store the training results and
                optional KMS key ID.
            resource_config (dict): Contains values for ResourceConfig:
                * instance_count (int): Number of EC2 instances to use for training.
                The key in resource_config is 'InstanceCount'.
                * instance_type (str): Type of EC2 instance to use for training, for example,
                'ml.c4.xlarge'. The key in resource_config is 'InstanceType'.

            vpc_config (dict): Contains values for VpcConfig:
                * subnets (list[str]): List of subnet ids.
                The key in vpc_config is 'Subnets'.
                * security_group_ids (list[str]): List of security group ids.
                The key in vpc_config is 'SecurityGroupIds'.

            hyperparameters (dict): Hyperparameters for model training. The hyperparameters are
                made accessible as a dict[str, str] to the training code on SageMaker. For
                convenience, this accepts other types for keys and values, but ``str()`` will be
                called to convert them before training.
            stop_condition (dict): Defines when training shall finish. Contains entries that can
                be understood by the service like ``MaxRuntimeInSeconds``.
            tags (list[dict]): List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
                used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
                the name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs.
            enable_network_isolation (bool): Whether to request for the training job to run with
                network isolation or not.
            image_uri (str): Docker image containing training code.
            algorithm_arn (str): Algorithm Arn from Marketplace.
            encrypt_inter_container_traffic (bool): Specifies whether traffic between training
                containers is encrypted for the training job (default: ``False``).
            use_spot_instances (bool): whether to use spot instances for training.
            checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
                that the algorithm persists (if any) during training. (default:
                ``None``).
            checkpoint_local_path (str): The local path that the algorithm
                writes its checkpoints to. SageMaker will persist all files
                under this path to `checkpoint_s3_uri` continually during
                training. On job startup the reverse happens - data from the
                s3 location is downloaded to this path before the algorithm is
                started. If the path is unset then SageMaker assumes the
                checkpoints will be provided under `/opt/ml/checkpoints/`.
                (default: ``None``).
            experiment_config (dict): Experiment management configuration. Dictionary contains
                three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                (default: ``None``)
            enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
                Series. For more information see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
                (default: ``None``).

        Returns:
            str: ARN of the training job, if it is created.
        """
        train_request = self._get_train_request(
            input_mode=input_mode,
            input_config=input_config,
            role=role,
            job_name=job_name,
            output_config=output_config,
            resource_config=resource_config,
            vpc_config=vpc_config,
            hyperparameters=hyperparameters,
            stop_condition=stop_condition,
            tags=tags,
            metric_definitions=metric_definitions,
            enable_network_isolation=enable_network_isolation,
            image_uri=image_uri,
            algorithm_arn=algorithm_arn,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            use_spot_instances=use_spot_instances,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            experiment_config=experiment_config,
            debugger_rule_configs=debugger_rule_configs,
            debugger_hook_config=debugger_hook_config,
            tensorboard_output_config=tensorboard_output_config,
            enable_sagemaker_metrics=enable_sagemaker_metrics,
        )
        LOGGER.info("Creating training-job with name: %s", job_name)
        LOGGER.debug("train request: %s", json.dumps(train_request, indent=4))
        self.sagemaker_client.create_training_job(**train_request)

    def _get_train_request(  # noqa: C901
        self,
        input_mode,
        input_config,
        role,
        job_name,
        output_config,
        resource_config,
        vpc_config,
        hyperparameters,
        stop_condition,
        tags,
        metric_definitions,
        enable_network_isolation=False,
        image_uri=None,
        algorithm_arn=None,
        encrypt_inter_container_traffic=False,
        use_spot_instances=False,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        experiment_config=None,
        debugger_rule_configs=None,
        debugger_hook_config=None,
        tensorboard_output_config=None,
        enable_sagemaker_metrics=None,
    ):
        """Constructs a request compatible for creating an Amazon SageMaker training job.

        Args:
            input_mode (str): The input mode that the algorithm supports. Valid modes:
                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a
                Unix-named pipe.

            input_config (list): A list of Channel objects. Each channel is a named input source.
                Please refer to the format details described:
                https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. You must grant sufficient permissions to this
                role.
            job_name (str): Name of the training job being created.
            output_config (dict): The S3 URI where you want to store the training results and
                optional KMS key ID.
            resource_config (dict): Contains values for ResourceConfig:
                * instance_count (int): Number of EC2 instances to use for training.
                The key in resource_config is 'InstanceCount'.
                * instance_type (str): Type of EC2 instance to use for training, for example,
                'ml.c4.xlarge'. The key in resource_config is 'InstanceType'.

            vpc_config (dict): Contains values for VpcConfig:
                * subnets (list[str]): List of subnet ids.
                The key in vpc_config is 'Subnets'.
                * security_group_ids (list[str]): List of security group ids.
                The key in vpc_config is 'SecurityGroupIds'.

            hyperparameters (dict): Hyperparameters for model training. The hyperparameters are
                made accessible as a dict[str, str] to the training code on SageMaker. For
                convenience, this accepts other types for keys and values, but ``str()`` will be
                called to convert them before training.
            stop_condition (dict): Defines when training shall finish. Contains entries that can
                be understood by the service like ``MaxRuntimeInSeconds``.
            tags (list[dict]): List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
                used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
                the name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs.
            enable_network_isolation (bool): Whether to request for the training job to run with
                network isolation or not.
            image_uri (str): Docker image containing training code.
            algorithm_arn (str): Algorithm Arn from Marketplace.
            encrypt_inter_container_traffic (bool): Specifies whether traffic between training
                containers is encrypted for the training job (default: ``False``).
            use_spot_instances (bool): whether to use spot instances for training.
            checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
                that the algorithm persists (if any) during training. (default:
                ``None``).
            checkpoint_local_path (str): The local path that the algorithm
                writes its checkpoints to. SageMaker will persist all files
                under this path to `checkpoint_s3_uri` continually during
                training. On job startup the reverse happens - data from the
                s3 location is downloaded to this path before the algorithm is
                started. If the path is unset then SageMaker assumes the
                checkpoints will be provided under `/opt/ml/checkpoints/`.
                (default: ``None``).
            experiment_config (dict): Experiment management configuration. Dictionary contains
                three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                (default: ``None``)
            enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
                Series. For more information see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
                (default: ``None``).

        Returns:
            Dict: a training request dict
        """
        train_request = {
            "AlgorithmSpecification": {"TrainingInputMode": input_mode},
            "OutputDataConfig": output_config,
            "TrainingJobName": job_name,
            "StoppingCondition": stop_condition,
            "ResourceConfig": resource_config,
            "RoleArn": role,
        }

        if image_uri and algorithm_arn:
            raise ValueError(
                "image_uri and algorithm_arn are mutually exclusive."
                "Both were provided: image_uri: %s algorithm_arn: %s" % (image_uri, algorithm_arn)
            )

        if image_uri is None and algorithm_arn is None:
            raise ValueError("either image_uri or algorithm_arn is required. None was provided.")

        if image_uri is not None:
            train_request["AlgorithmSpecification"]["TrainingImage"] = image_uri

        if algorithm_arn is not None:
            train_request["AlgorithmSpecification"]["AlgorithmName"] = algorithm_arn

        if input_config is not None:
            train_request["InputDataConfig"] = input_config

        if metric_definitions is not None:
            train_request["AlgorithmSpecification"]["MetricDefinitions"] = metric_definitions

        if enable_sagemaker_metrics is not None:
            train_request["AlgorithmSpecification"][
                "EnableSageMakerMetricsTimeSeries"
            ] = enable_sagemaker_metrics

        if hyperparameters and len(hyperparameters) > 0:
            train_request["HyperParameters"] = hyperparameters

        if tags is not None:
            train_request["Tags"] = tags

        if vpc_config is not None:
            train_request["VpcConfig"] = vpc_config

        if experiment_config and len(experiment_config) > 0:
            train_request["ExperimentConfig"] = experiment_config

        if enable_network_isolation:
            train_request["EnableNetworkIsolation"] = enable_network_isolation

        if encrypt_inter_container_traffic:
            train_request["EnableInterContainerTrafficEncryption"] = encrypt_inter_container_traffic

        if use_spot_instances:
            train_request["EnableManagedSpotTraining"] = use_spot_instances

        if checkpoint_s3_uri:
            checkpoint_config = {"S3Uri": checkpoint_s3_uri}
            if checkpoint_local_path:
                checkpoint_config["LocalPath"] = checkpoint_local_path
            train_request["CheckpointConfig"] = checkpoint_config

        if debugger_rule_configs is not None:
            train_request["DebugRuleConfigurations"] = debugger_rule_configs

        if debugger_hook_config is not None:
            train_request["DebugHookConfig"] = debugger_hook_config

        if tensorboard_output_config is not None:
            train_request["TensorBoardOutputConfig"] = tensorboard_output_config

        return train_request

    def process(
        self,
        inputs,
        output_config,
        job_name,
        resources,
        stopping_condition,
        app_specification,
        environment,
        network_config,
        role_arn,
        tags,
        experiment_config=None,
    ):
        """Create an Amazon SageMaker processing job.

        Args:
            inputs ([dict]): List of up to 10 ProcessingInput dictionaries.
            output_config (dict): A config dictionary, which contains a list of up
                to 10 ProcessingOutput dictionaries, as well as an optional KMS key ID.
            job_name (str): The name of the processing job. The name must be unique
                within an AWS Region in an AWS account. Names should have minimum
                length of 1 and maximum length of 63 characters.
            resources (dict): Encapsulates the resources, including ML instances
                and storage, to use for the processing job.
            stopping_condition (dict[str,int]): Specifies a limit to how long
                the processing job can run, in seconds.
            app_specification (dict[str,str]): Configures the processing job to
                run the given image. Details are in the processing container
                specification.
            environment (dict): Environment variables to start the processing
                container with.
            network_config (dict): Specifies networking options, such as network
                traffic encryption between processing containers, whether to allow
                inbound and outbound network calls to and from processing containers,
                and VPC subnets and security groups to use for VPC-enabled processing
                jobs.
            role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
                Amazon SageMaker can assume to perform tasks on your behalf.
            tags ([dict[str,str]]): A list of dictionaries containing key-value
                pairs.
            experiment_config (dict): Experiment management configuration. Dictionary contains
                three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                (default: ``None``)
        """
        process_request = self._get_process_request(
            inputs=inputs,
            output_config=output_config,
            job_name=job_name,
            resources=resources,
            stopping_condition=stopping_condition,
            app_specification=app_specification,
            environment=environment,
            network_config=network_config,
            role_arn=role_arn,
            tags=tags,
            experiment_config=experiment_config,
        )
        LOGGER.info("Creating processing-job with name %s", job_name)
        LOGGER.debug("process request: %s", json.dumps(process_request, indent=4))
        self.sagemaker_client.create_processing_job(**process_request)

    def _get_process_request(
        self,
        inputs,
        output_config,
        job_name,
        resources,
        stopping_condition,
        app_specification,
        environment,
        network_config,
        role_arn,
        tags,
        experiment_config=None,
    ):
        """Constructs a request compatible for an Amazon SageMaker processing job.

        Args:
            inputs ([dict]): List of up to 10 ProcessingInput dictionaries.
            output_config (dict): A config dictionary, which contains a list of up
                to 10 ProcessingOutput dictionaries, as well as an optional KMS key ID.
            job_name (str): The name of the processing job. The name must be unique
                within an AWS Region in an AWS account. Names should have minimum
                length of 1 and maximum length of 63 characters.
            resources (dict): Encapsulates the resources, including ML instances
                and storage, to use for the processing job.
            stopping_condition (dict[str,int]): Specifies a limit to how long
                the processing job can run, in seconds.
            app_specification (dict[str,str]): Configures the processing job to
                run the given image. Details are in the processing container
                specification.
            environment (dict): Environment variables to start the processing
                container with.
            network_config (dict): Specifies networking options, such as network
                traffic encryption between processing containers, whether to allow
                inbound and outbound network calls to and from processing containers,
                and VPC subnets and security groups to use for VPC-enabled processing
                jobs.
            role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
                Amazon SageMaker can assume to perform tasks on your behalf.
            tags ([dict[str,str]]): A list of dictionaries containing key-value
                pairs.
            experiment_config (dict): Experiment management configuration. Dictionary contains
                three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                (default: ``None``)

        Returns:
            Dict: a processing job request dict
        """
        process_request = {
            "ProcessingJobName": job_name,
            "ProcessingResources": resources,
            "AppSpecification": app_specification,
            "RoleArn": role_arn,
        }

        if inputs:
            process_request["ProcessingInputs"] = inputs

        if output_config["Outputs"]:
            process_request["ProcessingOutputConfig"] = output_config

        if environment is not None:
            process_request["Environment"] = environment

        if network_config is not None:
            process_request["NetworkConfig"] = network_config

        if stopping_condition is not None:
            process_request["StoppingCondition"] = stopping_condition

        if tags is not None:
            process_request["Tags"] = tags

        if experiment_config:
            process_request["ExperimentConfig"] = experiment_config

        return process_request

    def create_monitoring_schedule(
        self,
        monitoring_schedule_name,
        schedule_expression,
        statistics_s3_uri,
        constraints_s3_uri,
        monitoring_inputs,
        monitoring_output_config,
        instance_count,
        instance_type,
        volume_size_in_gb,
        volume_kms_key,
        image_uri,
        entrypoint,
        arguments,
        record_preprocessor_source_uri,
        post_analytics_processor_source_uri,
        max_runtime_in_seconds,
        environment,
        network_config,
        role_arn,
        tags,
    ):
        """Create an Amazon SageMaker monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the monitoring schedule. The name must be
                unique within an AWS Region in an AWS account. Names should have a minimum length
                of 1 and a maximum length of 63 characters.
            schedule_expression (str): The cron expression that dictates the monitoring execution
                schedule.
            statistics_s3_uri (str): The S3 uri of the statistics file to use.
            constraints_s3_uri (str): The S3 uri of the constraints file to use.
            monitoring_inputs ([dict]): List of MonitoringInput dictionaries.
            monitoring_output_config (dict): A config dictionary, which contains a list of
                MonitoringOutput dictionaries, as well as an optional KMS key ID.
            instance_count (int): The number of instances to run.
            instance_type (str): The type of instance to run.
            volume_size_in_gb (int): Size of the volume in GB.
            volume_kms_key (str): KMS key to use when encrypting the volume.
            image_uri (str): The image uri to use for monitoring executions.
            entrypoint (str): The entrypoint to the monitoring execution image.
            arguments (str): The arguments to pass to the monitoring execution image.
            record_preprocessor_source_uri (str or None): The S3 uri that points to the script that
                pre-processes the dataset (only applicable to first-party images).
            post_analytics_processor_source_uri (str or None): The S3 uri that points to the script
                that post-processes the dataset (only applicable to first-party images).
            max_runtime_in_seconds (int): Specifies a limit to how long
                the processing job can run, in seconds.
            environment (dict): Environment variables to start the monitoring execution
                container with.
            network_config (dict): Specifies networking options, such as network
                traffic encryption between processing containers, whether to allow
                inbound and outbound network calls to and from processing containers,
                and VPC subnets and security groups to use for VPC-enabled processing
                jobs.
            role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
                Amazon SageMaker can assume to perform tasks on your behalf.
            tags ([dict[str,str]]): A list of dictionaries containing key-value
                pairs.

        """
        monitoring_schedule_request = {
            "MonitoringScheduleName": monitoring_schedule_name,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinition": {
                    "MonitoringInputs": monitoring_inputs,
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": instance_count,
                            "InstanceType": instance_type,
                            "VolumeSizeInGB": volume_size_in_gb,
                        }
                    },
                    "MonitoringAppSpecification": {"ImageUri": image_uri},
                    "RoleArn": role_arn,
                }
            },
        }

        if schedule_expression is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"] = {
                "ScheduleExpression": schedule_expression
            }

        if monitoring_output_config is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringOutputConfig"
            ] = monitoring_output_config

        if statistics_s3_uri is not None or constraints_s3_uri is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ] = {}

        if statistics_s3_uri is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ]["StatisticsResource"] = {"S3Uri": statistics_s3_uri}

        if constraints_s3_uri is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ]["ConstraintsResource"] = {"S3Uri": constraints_s3_uri}

        if record_preprocessor_source_uri is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["RecordPreprocessorSourceUri"] = record_preprocessor_source_uri

        if post_analytics_processor_source_uri is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["PostAnalyticsProcessorSourceUri"] = post_analytics_processor_source_uri

        if entrypoint is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["ContainerEntrypoint"] = entrypoint

        if arguments is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["ContainerArguments"] = arguments

        if volume_kms_key is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringResources"
            ]["ClusterConfig"]["VolumeKmsKeyId"] = volume_kms_key

        if max_runtime_in_seconds is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "StoppingCondition"
            ] = {"MaxRuntimeInSeconds": max_runtime_in_seconds}

        if environment is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "Environment"
            ] = environment

        if network_config is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "NetworkConfig"
            ] = network_config

        if tags is not None:
            monitoring_schedule_request["Tags"] = tags

        LOGGER.info("Creating monitoring schedule name %s.", monitoring_schedule_name)
        LOGGER.debug(
            "monitoring_schedule_request= %s", json.dumps(monitoring_schedule_request, indent=4)
        )
        self.sagemaker_client.create_monitoring_schedule(**monitoring_schedule_request)

    def update_monitoring_schedule(
        self,
        monitoring_schedule_name,
        schedule_expression=None,
        statistics_s3_uri=None,
        constraints_s3_uri=None,
        monitoring_inputs=None,
        monitoring_output_config=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        image_uri=None,
        entrypoint=None,
        arguments=None,
        record_preprocessor_source_uri=None,
        post_analytics_processor_source_uri=None,
        max_runtime_in_seconds=None,
        environment=None,
        network_config=None,
        role_arn=None,
    ):
        """Update an Amazon SageMaker monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the monitoring schedule. The name must be
                unique within an AWS Region in an AWS account. Names should have a minimum length
                of 1 and a maximum length of 63 characters.
            schedule_expression (str): The cron expression that dictates the monitoring execution
                schedule.
            statistics_s3_uri (str): The S3 uri of the statistics file to use.
            constraints_s3_uri (str): The S3 uri of the constraints file to use.
            monitoring_inputs ([dict]): List of MonitoringInput dictionaries.
            monitoring_output_config (dict): A config dictionary, which contains a list of
                MonitoringOutput dictionaries, as well as an optional KMS key ID.
            instance_count (int): The number of instances to run.
            instance_type (str): The type of instance to run.
            volume_size_in_gb (int): Size of the volume in GB.
            volume_kms_key (str): KMS key to use when encrypting the volume.
            image_uri (str): The image uri to use for monitoring executions.
            entrypoint (str): The entrypoint to the monitoring execution image.
            arguments (str): The arguments to pass to the monitoring execution image.
            record_preprocessor_source_uri (str or None): The S3 uri that points to the script that
                pre-processes the dataset (only applicable to first-party images).
            post_analytics_processor_source_uri (str or None): The S3 uri that points to the script
                that post-processes the dataset (only applicable to first-party images).
            max_runtime_in_seconds (int): Specifies a limit to how long
                the processing job can run, in seconds.
            environment (dict): Environment variables to start the monitoring execution
                container with.
            network_config (dict): Specifies networking options, such as network
                traffic encryption between processing containers, whether to allow
                inbound and outbound network calls to and from processing containers,
                and VPC subnets and security groups to use for VPC-enabled processing
                jobs.
            role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
                Amazon SageMaker can assume to perform tasks on your behalf.
            tags ([dict[str,str]]): A list of dictionaries containing key-value
                pairs.

        """
        existing_desc = self.sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

        existing_schedule_config = None
        if (
            existing_desc.get("MonitoringScheduleConfig") is not None
            and existing_desc["MonitoringScheduleConfig"].get("ScheduleConfig") is not None
            and existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
            is not None
        ):
            existing_schedule_config = existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"][
                "ScheduleExpression"
            ]

        request_schedule_expression = schedule_expression or existing_schedule_config
        request_monitoring_inputs = (
            monitoring_inputs
            or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringInputs"
            ]
        )
        request_instance_count = (
            instance_count
            or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringResources"
            ]["ClusterConfig"]["InstanceCount"]
        )
        request_instance_type = (
            instance_type
            or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringResources"
            ]["ClusterConfig"]["InstanceType"]
        )
        request_volume_size_in_gb = (
            volume_size_in_gb
            or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringResources"
            ]["ClusterConfig"]["VolumeSizeInGB"]
        )
        request_image_uri = (
            image_uri
            or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["ImageUri"]
        )
        request_role_arn = (
            role_arn
            or existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["RoleArn"]
        )

        monitoring_schedule_request = {
            "MonitoringScheduleName": monitoring_schedule_name,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinition": {
                    "MonitoringInputs": request_monitoring_inputs,
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": request_instance_count,
                            "InstanceType": request_instance_type,
                            "VolumeSizeInGB": request_volume_size_in_gb,
                        }
                    },
                    "MonitoringAppSpecification": {"ImageUri": request_image_uri},
                    "RoleArn": request_role_arn,
                }
            },
        }

        if existing_schedule_config is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"] = {
                "ScheduleExpression": request_schedule_expression
            }

        existing_monitoring_output_config = existing_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ].get("MonitoringOutputConfig")
        if monitoring_output_config is not None or existing_monitoring_output_config is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringOutputConfig"
            ] = (monitoring_output_config or existing_monitoring_output_config)

        existing_statistics_s3_uri = None
        existing_constraints_s3_uri = None
        if (
            existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"].get(
                "BaselineConfig"
            )
            is not None
        ):
            if (
                existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                    "BaselineConfig"
                ].get("StatisticsResource")
                is not None
            ):
                existing_statistics_s3_uri = existing_desc["MonitoringScheduleConfig"][
                    "MonitoringJobDefinition"
                ]["BaselineConfig"]["StatisticsResource"]["S3Uri"]

            if (
                existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                    "BaselineConfig"
                ].get("ConstraintsResource")
                is not None
            ):
                existing_statistics_s3_uri = existing_desc["MonitoringScheduleConfig"][
                    "MonitoringJobDefinition"
                ]["BaselineConfig"]["ConstraintsResource"]["S3Uri"]

        if (
            statistics_s3_uri is not None
            or constraints_s3_uri is not None
            or existing_statistics_s3_uri is not None
            or existing_constraints_s3_uri is not None
        ):
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ] = {}

        if statistics_s3_uri is not None or existing_statistics_s3_uri is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ]["StatisticsResource"] = {"S3Uri": statistics_s3_uri or existing_statistics_s3_uri}

        if constraints_s3_uri is not None or existing_constraints_s3_uri is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "BaselineConfig"
            ]["ConstraintsResource"] = {"S3Uri": constraints_s3_uri or existing_constraints_s3_uri}

        existing_record_preprocessor_source_uri = existing_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ]["MonitoringAppSpecification"].get("RecordPreprocessorSourceUri")
        if (
            record_preprocessor_source_uri is not None
            or existing_record_preprocessor_source_uri is not None
        ):
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["RecordPreprocessorSourceUri"] = (
                record_preprocessor_source_uri or existing_record_preprocessor_source_uri
            )

        existing_post_analytics_processor_source_uri = existing_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ]["MonitoringAppSpecification"].get("PostAnalyticsProcessorSourceUri")
        if (
            post_analytics_processor_source_uri is not None
            or existing_post_analytics_processor_source_uri is not None
        ):
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["PostAnalyticsProcessorSourceUri"] = (
                post_analytics_processor_source_uri or existing_post_analytics_processor_source_uri
            )

        existing_entrypoint = existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ].get("ContainerEntrypoint")
        if entrypoint is not None or existing_entrypoint is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["ContainerEntrypoint"] = (entrypoint or existing_entrypoint)

        existing_arguments = existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ].get("ContainerArguments")
        if arguments is not None or existing_arguments is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringAppSpecification"
            ]["ContainerArguments"] = (arguments or existing_arguments)

        existing_volume_kms_key = existing_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ]["MonitoringResources"]["ClusterConfig"].get("VolumeKmsKeyId")

        if volume_kms_key is not None or existing_volume_kms_key is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "MonitoringResources"
            ]["ClusterConfig"]["VolumeKmsKeyId"] = (volume_kms_key or existing_volume_kms_key)

        existing_max_runtime_in_seconds = None
        if existing_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"].get(
            "StoppingCondition"
        ):
            existing_max_runtime_in_seconds = existing_desc["MonitoringScheduleConfig"][
                "MonitoringJobDefinition"
            ]["StoppingCondition"].get("MaxRuntimeInSeconds")

        if max_runtime_in_seconds is not None or existing_max_runtime_in_seconds is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "StoppingCondition"
            ] = {"MaxRuntimeInSeconds": max_runtime_in_seconds or existing_max_runtime_in_seconds}

        existing_environment = existing_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ].get("Environment")
        if environment is not None or existing_environment is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "Environment"
            ] = (environment or existing_environment)

        existing_network_config = existing_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ].get("NetworkConfig")
        if network_config is not None or existing_network_config is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "NetworkConfig"
            ] = (network_config or existing_network_config)

        LOGGER.info("Updating monitoring schedule with name: %s .", monitoring_schedule_name)
        LOGGER.debug(
            "monitoring_schedule_request= %s", json.dumps(monitoring_schedule_request, indent=4)
        )
        self.sagemaker_client.update_monitoring_schedule(**monitoring_schedule_request)

    def start_monitoring_schedule(self, monitoring_schedule_name):
        """Starts a monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
                Schedule to start.

        """
        print()
        print("Starting Monitoring Schedule with name: {}".format(monitoring_schedule_name))
        self.sagemaker_client.start_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

    def stop_monitoring_schedule(self, monitoring_schedule_name):
        """Stops a monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
                Schedule to stop.

        """
        print()
        print("Stopping Monitoring Schedule with name: {}".format(monitoring_schedule_name))
        self.sagemaker_client.stop_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

    def delete_monitoring_schedule(self, monitoring_schedule_name):
        """Deletes a monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
                Schedule to delete.

        """
        print()
        print("Deleting Monitoring Schedule with name: {}".format(monitoring_schedule_name))
        self.sagemaker_client.delete_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

    def describe_monitoring_schedule(self, monitoring_schedule_name):
        """Calls the DescribeMonitoringSchedule API for the given monitoring schedule name
        and returns the response.

        Args:
            monitoring_schedule_name (str): The name of the processing job to describe.

        Returns:
            dict: A dictionary response with the processing job description.

        """
        return self.sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

    def list_monitoring_executions(
        self,
        monitoring_schedule_name,
        sort_by="ScheduledTime",
        sort_order="Descending",
        max_results=100,
    ):
        """Lists the monitoring executions associated with the given monitoring_schedule_name.

        Args:
            monitoring_schedule_name (str): The monitoring_schedule_name for which to retrieve the
                monitoring executions.
            sort_by (str): The field to sort by. Can be one of: "CreationTime", "ScheduledTime",
                "Status". Default: "ScheduledTime".
            sort_order (str): The sort order. Can be one of: "Ascending", "Descending".
                Default: "Descending".
            max_results (int): The maximum number of results to return. Must be between 1 and 100.

        Returns:
            dict: Dictionary of monitoring schedule executions.
        """
        response = self.sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=monitoring_schedule_name,
            SortBy=sort_by,
            SortOrder=sort_order,
            MaxResults=max_results,
        )
        return response

    def list_monitoring_schedules(
        self, endpoint_name=None, sort_by="CreationTime", sort_order="Descending", max_results=100
    ):
        """Lists the monitoring executions associated with the given monitoring_schedule_name.

        Args:
            endpoint_name (str): The name of the endpoint to filter on. If not provided, does not
                filter on it. Default: None.
            sort_by (str): The field to sort by. Can be one of: "Name", "CreationTime", "Status".
                Default: "CreationTime".
            sort_order (str): The sort order. Can be one of: "Ascending", "Descending".
                Default: "Descending".
            max_results (int): The maximum number of results to return. Must be between 1 and 100.

        Returns:
            dict: Dictionary of monitoring schedule executions.
        """
        if endpoint_name is not None:
            response = self.sagemaker_client.list_monitoring_schedules(
                EndpointName=endpoint_name,
                SortBy=sort_by,
                SortOrder=sort_order,
                MaxResults=max_results,
            )
        else:
            response = self.sagemaker_client.list_monitoring_schedules(
                SortBy=sort_by, SortOrder=sort_order, MaxResults=max_results
            )

        return response

    def was_processing_job_successful(self, job_name):
        """Calls the DescribeProcessingJob API for the given job name
        and returns the True if the job was successful. False otherwise.

        Args:
            job_name (str): The name of the processing job to describe.

        Returns:
            bool: Whether the processing job was successful.
        """
        job_desc = self.sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
        return job_desc["ProcessingJobStatus"] == "Completed"

    def describe_processing_job(self, job_name):
        """Calls the DescribeProcessingJob API for the given job name
        and returns the response.

        Args:
            job_name (str): The name of the processing job to describe.

        Returns:
            dict: A dictionary response with the processing job description.
        """
        return self.sagemaker_client.describe_processing_job(ProcessingJobName=job_name)

    def stop_processing_job(self, job_name):
        """Calls the StopProcessingJob API for the given job name.

        Args:
            job_name (str): The name of the processing job to stop.

        """
        self.sagemaker_client.stop_processing_job(ProcessingJobName=job_name)

    def stop_training_job(self, job_name):
        """Calls the StopTrainingJob API for the given job name.

        Args:
            job_name (str): The name of the training job to stop.
        """
        self.sagemaker_client.stop_training_job(TrainingJobName=job_name)

    def describe_training_job(self, job_name):
        """Calls the DescribeTrainingJob API for the given job name
        and returns the response.

        Args:
            job_name (str): The name of the training job to describe.

        Returns:
            dict: A dictionary response with the training job description.
        """
        return self.sagemaker_client.describe_training_job(TrainingJobName=job_name)

    def auto_ml(
        self,
        input_config,
        output_config,
        auto_ml_job_config,
        role,
        job_name,
        problem_type=None,
        job_objective=None,
        generate_candidate_definitions_only=False,
        tags=None,
    ):
        """Create an Amazon SageMaker AutoML job.

        Args:
            input_config (list[dict]): A list of Channel objects. Each channel contains "DataSource"
                and "TargetAttributeName", "CompressionType" is an optional field.
            output_config (dict): The S3 URI where you want to store the training results and
                optional KMS key ID.
            auto_ml_job_config (dict): A dict of AutoMLJob config, containing "StoppingCondition",
                "SecurityConfig", optionally contains "VolumeKmsKeyId".
            role (str): The Amazon Resource Name (ARN) of an IAM role that
                Amazon SageMaker can assume to perform tasks on your behalf.
            job_name (str): A string that can be used to identify an AutoMLJob. Each AutoMLJob
                should have a unique job name.
            problem_type (str): The type of problem of this AutoMLJob. Valid values are
                "Regression", "BinaryClassification", "MultiClassClassification". If None,
                SageMaker AutoMLJob will infer the problem type automatically.
            job_objective (dict): AutoMLJob objective, contains "AutoMLJobObjectiveType" (optional),
                "MetricName" and "Value".
            generate_candidate_definitions_only (bool): Indicates whether to only generate candidate
                definitions. If True, AutoML.list_candidates() cannot be called. Default: False.
            tags ([dict[str,str]]): A list of dictionaries containing key-value
                pairs.

        Returns:

        """
        auto_ml_job_request = {
            "AutoMLJobName": job_name,
            "InputDataConfig": input_config,
            "OutputDataConfig": output_config,
            "AutoMLJobConfig": auto_ml_job_config,
            "RoleArn": role,
            "GenerateCandidateDefinitionsOnly": generate_candidate_definitions_only,
        }

        if job_objective is not None:
            auto_ml_job_request["AutoMLJobObjective"] = job_objective
        if problem_type is not None:
            auto_ml_job_request["ProblemType"] = problem_type
        if tags is not None:
            auto_ml_job_request["Tags"] = tags

        LOGGER.info("Creating auto-ml-job with name: %s", job_name)
        LOGGER.debug("auto ml request: %s", json.dumps(auto_ml_job_request, indent=4))
        self.sagemaker_client.create_auto_ml_job(**auto_ml_job_request)

    def describe_auto_ml_job(self, job_name):
        """Calls the DescribeAutoMLJob API for the given job name
        and returns the response.
        Args:
            job_name (str): The name of the AutoML job to describe.
        Returns:
            dict: A dictionary response with the AutoML Job description.
        """
        return self.sagemaker_client.describe_auto_ml_job(AutoMLJobName=job_name)

    def list_candidates(
        self,
        job_name,
        status_equals=None,
        candidate_name=None,
        candidate_arn=None,
        sort_order=None,
        sort_by=None,
        max_results=None,
    ):
        """Returns the list of candidates of an AutoML job for a given name.

        Args:
            job_name (str): The name of the AutoML job. If None, will use object's
                latest_auto_ml_job name.
            status_equals (str): Filter the result with candidate status, values could be
                "Completed", "InProgress", "Failed", "Stopped", "Stopping"
            candidate_name (str): The name of a specified candidate to list.
                Default to None.
            candidate_arn (str): The Arn of a specified candidate to list.
                Default to None.
            sort_order (str): The order that the candidates will be listed in result.
                Default to None.
            sort_by (str): The value that the candidates will be sorted by.
                Default to None.
            max_results (int): The number of candidates will be listed in results,
                between 1 to 100. Default to None. If None, will return all the candidates.
        Returns:
            list: A list of dictionaries with candidates information
        """
        list_candidates_args = {"AutoMLJobName": job_name}

        if status_equals:
            list_candidates_args["StatusEquals"] = status_equals
        if candidate_name:
            list_candidates_args["CandidateNameEquals"] = candidate_name
        if candidate_arn:
            list_candidates_args["CandidateArnEquals"] = candidate_arn
        if sort_order:
            list_candidates_args["SortOrder"] = sort_order
        if sort_by:
            list_candidates_args["SortBy"] = sort_by
        if max_results:
            list_candidates_args["MaxResults"] = max_results

        return self.sagemaker_client.list_candidates_for_auto_ml_job(**list_candidates_args)

    def wait_for_auto_ml_job(self, job, poll=5):
        """Wait for an Amazon SageMaker AutoML job to complete.

        Args:
            job (str): Name of the auto ml job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeAutoMLJob`` API.

        Raises:
            exceptions.UnexpectedStatusException: If the auto ml job fails.
        """
        desc = _wait_until(lambda: _auto_ml_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, "AutoMLJobStatus")
        return desc

    def logs_for_auto_ml_job(  # noqa: C901 - suppress complexity warning for this method
        self, job_name, wait=False, poll=10
    ):
        """Display the logs for a given AutoML job, optionally tailing them until the
        job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
        based on which instance the log entry is from.

        Args:
            job_name (str): Name of the Auto ML job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes
                (default: False).
            poll (int): The interval in seconds between polling for new log entries and job
                completion (default: 5).

        Raises:
            exceptions.UnexpectedStatusException: If waiting and the training job fails.
        """

        description = self.sagemaker_client.describe_auto_ml_job(AutoMLJobName=job_name)

        instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
            self, description, job="AutoML"
        )

        state = _get_initial_job_state(description, "AutoMLJobStatus", wait)

        # The loop below implements a state machine that alternates between checking the job status
        # and reading whatever is available in the logs at this point. Note, that if we were
        # called with wait == False, we never check the job status.
        #
        # If wait == TRUE and job is not completed, the initial state is TAILING
        # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is
        # complete).
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
        # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to
        #   Cloudwatch after the job was marked complete.
        last_describe_job_call = time.time()
        while True:
            _flush_log_streams(
                stream_names,
                instance_count,
                client,
                log_group,
                job_name,
                positions,
                dot,
                color_wrap,
            )
            if state == LogState.COMPLETE:
                break

            time.sleep(poll)

            if state == LogState.JOB_COMPLETE:
                state = LogState.COMPLETE
            elif time.time() - last_describe_job_call >= 30:
                description = self.sagemaker_client.describe_auto_ml_job(AutoMLJobName=job_name)
                last_describe_job_call = time.time()

                status = description["AutoMLJobStatus"]

                if status in ("Completed", "Failed", "Stopped"):
                    print()
                    state = LogState.JOB_COMPLETE

        if wait:
            self._check_job_status(job_name, description, "AutoMLJobStatus")
            if dot:
                print()

    def compile_model(
        self, input_model_config, output_model_config, role, job_name, stop_condition, tags
    ):
        """Create an Amazon SageMaker Neo compilation job.

        Args:
            input_model_config (dict): the trained model and the Amazon S3 location where it is
                stored.
            output_model_config (dict): Identifies the Amazon S3 location where you want Amazon
                SageMaker Neo to save the results of compilation job
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker Neo
                compilation jobs use this role to access model artifacts. You must grant
                sufficient permissions to this role.
            job_name (str): Name of the compilation job being created.
            stop_condition (dict): Defines when compilation job shall finish. Contains entries
                that can be understood by the service like ``MaxRuntimeInSeconds``.
            tags (list[dict]): List of tags for labeling a compile model job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

        Returns:
            str: ARN of the compile model job, if it is created.

        """
        compilation_job_request = {
            "InputConfig": input_model_config,
            "OutputConfig": output_model_config,
            "RoleArn": role,
            "StoppingCondition": stop_condition,
            "CompilationJobName": job_name,
        }

        if tags is not None:
            compilation_job_request["Tags"] = tags

        LOGGER.info("Creating compilation-job with name: %s", job_name)
        self.sagemaker_client.create_compilation_job(**compilation_job_request)

    def tune(  # noqa: C901
        self,
        job_name,
        strategy,
        objective_type,
        objective_metric_name,
        max_jobs,
        max_parallel_jobs,
        parameter_ranges,
        static_hyperparameters,
        input_mode,
        metric_definitions,
        role,
        input_config,
        output_config,
        resource_config,
        stop_condition,
        tags,
        warm_start_config,
        enable_network_isolation=False,
        image_uri=None,
        algorithm_arn=None,
        early_stopping_type="Off",
        encrypt_inter_container_traffic=False,
        vpc_config=None,
        use_spot_instances=False,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
    ):
        """Create an Amazon SageMaker hyperparameter tuning job

        Args:
            job_name (str): Name of the tuning job being created.
            strategy (str): Strategy to be used for hyperparameter estimations.
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize'.
            objective_metric_name (str): Name of the metric for evaluating training jobs.
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
                tuning job.
            max_parallel_jobs (int): Maximum number of parallel training jobs to start.
            parameter_ranges (dict): Dictionary of parameter ranges. These parameter ranges can be
                one of three types: Continuous, Integer, or Categorical.
            static_hyperparameters (dict): Hyperparameters for model training. These
                hyperparameters remain unchanged across all of the training jobs for the
                hyperparameter tuning job. The hyperparameters are made accessible as a dictionary
                for the training code on SageMaker.
            image_uri (str): Docker image URI containing training code.
            algorithm_arn (str): Resource ARN for training algorithm created on or subscribed from
                AWS Marketplace (default: None).
            input_mode (str): The input mode that the algorithm supports. Valid modes:
                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a
                Unix-named pipe.

            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
                used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
                the name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for jobs that don't use an
                Amazon algorithm.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker
                training jobs and APIs that create Amazon SageMaker endpoints use this role to
                access training data and model artifacts. You must grant sufficient permissions
                to this role.
            input_config (list): A list of Channel objects. Each channel is a named input source.
                Please refer to the format details described:
                https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
            output_config (dict): The S3 URI where you want to store the training results and
                optional KMS key ID.
            resource_config (dict): Contains values for ResourceConfig:
                * instance_count (int): Number of EC2 instances to use for training.
                The key in resource_config is 'InstanceCount'.
                * instance_type (str): Type of EC2 instance to use for training, for example,
                'ml.c4.xlarge'. The key in resource_config is 'InstanceType'.

            stop_condition (dict): When training should finish, e.g. ``MaxRuntimeInSeconds``.
            tags (list[dict]): List of tags for labeling the tuning job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            warm_start_config (dict): Configuration defining the type of warm start and
                other required configurations.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off'. If set to 'Off', early stopping will not be
                attempted. If set to 'Auto', early stopping of some training jobs may happen, but
                is not guaranteed to.
            enable_network_isolation (bool): Specifies whether to isolate the training container
                (default: ``False``).
            encrypt_inter_container_traffic (bool): Specifies whether traffic between training
                containers is encrypted for the training jobs started for this hyperparameter
                tuning job (default: ``False``).
            vpc_config (dict): Contains values for VpcConfig (default: None):
                * subnets (list[str]): List of subnet ids.
                The key in vpc_config is 'Subnets'.
                * security_group_ids (list[str]): List of security group ids.
                The key in vpc_config is 'SecurityGroupIds'.
            use_spot_instances (bool): whether to use spot instances for training.
            checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
                that the algorithm persists (if any) during training. (default:
                ``None``).
            checkpoint_local_path (str): The local path that the algorithm
                writes its checkpoints to. SageMaker will persist all files
                under this path to `checkpoint_s3_uri` continually during
                training. On job startup the reverse happens - data from the
                s3 location is downloaded to this path before the algorithm is
                started. If the path is unset then SageMaker assumes the
                checkpoints will be provided under `/opt/ml/checkpoints/`.
                (default: ``None``).

        """

        tune_request = {
            "HyperParameterTuningJobName": job_name,
            "HyperParameterTuningJobConfig": self._map_tuning_config(
                strategy=strategy,
                max_jobs=max_jobs,
                max_parallel_jobs=max_parallel_jobs,
                objective_type=objective_type,
                objective_metric_name=objective_metric_name,
                parameter_ranges=parameter_ranges,
                early_stopping_type=early_stopping_type,
            ),
            "TrainingJobDefinition": self._map_training_config(
                static_hyperparameters=static_hyperparameters,
                role=role,
                input_mode=input_mode,
                image_uri=image_uri,
                algorithm_arn=algorithm_arn,
                metric_definitions=metric_definitions,
                input_config=input_config,
                output_config=output_config,
                resource_config=resource_config,
                vpc_config=vpc_config,
                stop_condition=stop_condition,
                enable_network_isolation=enable_network_isolation,
                encrypt_inter_container_traffic=encrypt_inter_container_traffic,
                use_spot_instances=use_spot_instances,
                checkpoint_s3_uri=checkpoint_s3_uri,
                checkpoint_local_path=checkpoint_local_path,
            ),
        }

        if warm_start_config is not None:
            tune_request["WarmStartConfig"] = warm_start_config

        if tags is not None:
            tune_request["Tags"] = tags

        LOGGER.info("Creating hyperparameter tuning job with name: %s", job_name)
        LOGGER.debug("tune request: %s", json.dumps(tune_request, indent=4))
        self.sagemaker_client.create_hyper_parameter_tuning_job(**tune_request)

    def create_tuning_job(
        self,
        job_name,
        tuning_config,
        training_config=None,
        training_config_list=None,
        warm_start_config=None,
        tags=None,
    ):
        """Create an Amazon SageMaker hyperparameter tuning job. This method supports creating
        tuning jobs with single or multiple training algorithms (estimators), while the ``tune()``
        method above only supports creating tuning jobs with single training algorithm.

        Args:
            job_name (str): Name of the tuning job being created.
            tuning_config (dict): Configuration to launch the tuning job.
            training_config (dict): Configuration to launch training jobs under the tuning job
                using a single algorithm.
            training_config_list (list[dict]): A list of configurations to launch training jobs
                under the tuning job using one or multiple algorithms. Either training_config
                or training_config_list should be provided, but not both.
            warm_start_config (dict): Configuration defining the type of warm start and
                other required configurations.
            tags (list[dict]): List of tags for labeling the tuning job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        """

        if training_config is None and training_config_list is None:
            raise ValueError("Either training_config or training_config_list should be provided.")
        if training_config is not None and training_config_list is not None:
            raise ValueError(
                "Only one of training_config and training_config_list should be provided."
            )

        tune_request = {
            "HyperParameterTuningJobName": job_name,
            "HyperParameterTuningJobConfig": self._map_tuning_config(**tuning_config),
        }

        if training_config is not None:
            tune_request["TrainingJobDefinition"] = self._map_training_config(**training_config)

        if training_config_list is not None:
            tune_request["TrainingJobDefinitions"] = [
                self._map_training_config(**training_cfg) for training_cfg in training_config_list
            ]

        if warm_start_config is not None:
            tune_request["WarmStartConfig"] = warm_start_config

        if tags is not None:
            tune_request["Tags"] = tags

        LOGGER.info("Creating hyperparameter tuning job with name: %s", job_name)
        LOGGER.debug("tune request: %s", json.dumps(tune_request, indent=4))
        self.sagemaker_client.create_hyper_parameter_tuning_job(**tune_request)

    def describe_tuning_job(self, job_name):
        """Calls the DescribeHyperParameterTuningJob API for the given job name
        and returns the response.

            Args:
                job_name (str): The name of the hyperparameter tuning job to describe.

            Returns:
                dict: A dictionary response with the hyperparameter tuning job description.
        """
        return self.sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )

    @classmethod
    def _map_tuning_config(
        cls,
        strategy,
        max_jobs,
        max_parallel_jobs,
        early_stopping_type="Off",
        objective_type=None,
        objective_metric_name=None,
        parameter_ranges=None,
    ):
        """
        Construct tuning job configuration dictionary.

        Args:
            strategy (str): Strategy to be used for hyperparameter estimations.
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
                tuning job.
            max_parallel_jobs (int): Maximum number of parallel training jobs to start.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off'. If set to 'Off', early stopping will not be
                attempted. If set to 'Auto', early stopping of some training jobs may happen,
                but is not guaranteed to.
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize'.
            objective_metric_name (str): Name of the metric for evaluating training jobs.
            parameter_ranges (dict): Dictionary of parameter ranges. These parameter ranges can
                be one of three types: Continuous, Integer, or Categorical.

        Returns:
            A dictionary of tuning job configuration. For format details, please refer to
            HyperParameterTuningJobConfig as described in
            https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job
        """

        tuning_config = {
            "Strategy": strategy,
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": max_jobs,
                "MaxParallelTrainingJobs": max_parallel_jobs,
            },
            "TrainingJobEarlyStoppingType": early_stopping_type,
        }

        tuning_objective = cls._map_tuning_objective(objective_type, objective_metric_name)
        if tuning_objective is not None:
            tuning_config["HyperParameterTuningJobObjective"] = tuning_objective

        if parameter_ranges is not None:
            tuning_config["ParameterRanges"] = parameter_ranges

        return tuning_config

    @classmethod
    def _map_tuning_objective(cls, objective_type, objective_metric_name):
        """
        Construct a dictionary of tuning objective from the arguments

        Args:
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize'.
            objective_metric_name (str): Name of the metric for evaluating training jobs.

        Returns:
            A dictionary of tuning objective. For format details, please refer to
            HyperParameterTuningJobObjective as described in
            https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job
        """

        tuning_objective = None

        if objective_type is not None or objective_metric_name is not None:
            tuning_objective = {}

        if objective_type is not None:
            tuning_objective["Type"] = objective_type

        if objective_metric_name is not None:
            tuning_objective["MetricName"] = objective_metric_name

        return tuning_objective

    @classmethod
    def _map_training_config(
        cls,
        static_hyperparameters,
        input_mode,
        role,
        output_config,
        resource_config,
        stop_condition,
        input_config=None,
        metric_definitions=None,
        image_uri=None,
        algorithm_arn=None,
        vpc_config=None,
        enable_network_isolation=False,
        encrypt_inter_container_traffic=False,
        estimator_name=None,
        objective_type=None,
        objective_metric_name=None,
        parameter_ranges=None,
        use_spot_instances=False,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
    ):
        """
        Construct a dictionary of training job configuration from the arguments

        Args:
            static_hyperparameters (dict): Hyperparameters for model training. These
                hyperparameters remain unchanged across all of the training jobs for the
                hyperparameter tuning job. The hyperparameters are made accessible as a dictionary
                for the training code on SageMaker.
            input_mode (str): The input mode that the algorithm supports. Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                    a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a
                    Unix-named pipe.

            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. You must grant sufficient permissions to
                this role.
            output_config (dict): The S3 URI where you want to store the training results and
                optional KMS key ID.
            resource_config (dict): Contains values for ResourceConfig:

                * instance_count (int): Number of EC2 instances to use for training.
                    The key in resource_config is 'InstanceCount'.
                * instance_type (str): Type of EC2 instance to use for training, for example,
                    'ml.c4.xlarge'. The key in resource_config is 'InstanceType'.

            stop_condition (dict): When training should finish, e.g. ``MaxRuntimeInSeconds``.
            input_config (list): A list of Channel objects. Each channel is a named input source.
                Please refer to the format details described:
                https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
                used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
                the name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for jobs that don't use an
                Amazon algorithm.
            image_uri (str): Docker image URI containing training code.
            algorithm_arn (str): Resource ARN for training algorithm created or subscribed on
                AWS Marketplace
            vpc_config (dict): Contains values for VpcConfig (default: None):

                * subnets (list[str]): List of subnet ids.
                    The key in vpc_config is 'Subnets'.
                * security_group_ids (list[str]): List of security group ids.
                    The key in vpc_config is 'SecurityGroupIds'.

            enable_network_isolation (bool): Specifies whether to isolate the training container
            encrypt_inter_container_traffic (bool): Specifies whether traffic between training
                containers is encrypted for the training jobs started for this hyperparameter
                tuning job (default: ``False``).
            estimator_name (str): Unique name for the estimator.
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize'.
            objective_metric_name (str): Name of the metric for evaluating training jobs.
            parameter_ranges (dict): Dictionary of parameter ranges. These parameter ranges can
                be one of three types: Continuous, Integer, or Categorical.

        Returns:
            A dictionary of training job configuration. For format details, please refer to
            TrainingJobDefinition as described in
            https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job

        """

        training_job_definition = {
            "StaticHyperParameters": static_hyperparameters,
            "RoleArn": role,
            "OutputDataConfig": output_config,
            "ResourceConfig": resource_config,
            "StoppingCondition": stop_condition,
        }

        algorithm_spec = {"TrainingInputMode": input_mode}
        if metric_definitions is not None:
            algorithm_spec["MetricDefinitions"] = metric_definitions

        if algorithm_arn:
            algorithm_spec["AlgorithmName"] = algorithm_arn
        else:
            algorithm_spec["TrainingImage"] = image_uri

        training_job_definition["AlgorithmSpecification"] = algorithm_spec

        if input_config is not None:
            training_job_definition["InputDataConfig"] = input_config

        if vpc_config is not None:
            training_job_definition["VpcConfig"] = vpc_config

        if enable_network_isolation:
            training_job_definition["EnableNetworkIsolation"] = True

        if encrypt_inter_container_traffic:
            training_job_definition["EnableInterContainerTrafficEncryption"] = True

        if use_spot_instances:
            training_job_definition["EnableManagedSpotTraining"] = True

        if checkpoint_s3_uri:
            checkpoint_config = {"S3Uri": checkpoint_s3_uri}
            if checkpoint_local_path:
                checkpoint_config["LocalPath"] = checkpoint_local_path
            training_job_definition["CheckpointConfig"] = checkpoint_config
        if estimator_name is not None:
            training_job_definition["DefinitionName"] = estimator_name

        tuning_objective = cls._map_tuning_objective(objective_type, objective_metric_name)
        if tuning_objective is not None:
            training_job_definition["TuningObjective"] = tuning_objective

        if parameter_ranges is not None:
            training_job_definition["HyperParameterRanges"] = parameter_ranges

        return training_job_definition

    def stop_tuning_job(self, name):
        """Stop the Amazon SageMaker hyperparameter tuning job with the specified name.

        Args:
            name (str): Name of the Amazon SageMaker hyperparameter tuning job.

        Raises:
            ClientError: If an error occurs while trying to stop the hyperparameter tuning job.
        """
        try:
            LOGGER.info("Stopping tuning job: %s", name)
            self.sagemaker_client.stop_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            # allow to pass if the job already stopped
            if error_code == "ValidationException":
                LOGGER.info("Tuning job: %s is already stopped or not running.", name)
            else:
                LOGGER.error(
                    "Error occurred while attempting to stop tuning job: %s. Please try again.",
                    name,
                )
                raise

    def _get_transform_request(
        self,
        job_name,
        model_name,
        strategy,
        max_concurrent_transforms,
        max_payload,
        env,
        input_config,
        output_config,
        resource_config,
        experiment_config,
        tags,
        data_processing,
        model_client_config=None,
    ):
        """Construct an dict can be used to create an Amazon SageMaker transform job.

        Args:
            job_name (str): Name of the transform job being created.
            model_name (str): Name of the SageMaker model being used for the transform job.
            strategy (str): The strategy used to decide how to batch records in a single request.
                Possible values are 'MultiRecord' and 'SingleRecord'.
            max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
                each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP request to the
                container in MB.
            env (dict): Environment variables to be set for use during the transform job.
            input_config (dict): A dictionary describing the input data (and its location) for the
                job.
            output_config (dict): A dictionary describing the output location for the job.
            resource_config (dict): A dictionary describing the resources to complete the job.
            experiment_config (dict): A dictionary describing the experiment configuration for the
                job. Dictionary contains three optional keys,
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
            tags (list[dict]): List of tags for labeling a transform job.
            data_processing(dict): A dictionary describing config for combining the input data and
                transformed data. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            model_client_config (dict): A dictionary describing the model configuration for the
                job. Dictionary contains two optional keys,
                'InvocationsTimeoutInSeconds', and 'InvocationsMaxRetries'.

        Returns:
            Dict: a create transform job request dict
        """
        transform_request = {
            "TransformJobName": job_name,
            "ModelName": model_name,
            "TransformInput": input_config,
            "TransformOutput": output_config,
            "TransformResources": resource_config,
        }

        if strategy is not None:
            transform_request["BatchStrategy"] = strategy

        if max_concurrent_transforms is not None:
            transform_request["MaxConcurrentTransforms"] = max_concurrent_transforms

        if max_payload is not None:
            transform_request["MaxPayloadInMB"] = max_payload

        if env is not None:
            transform_request["Environment"] = env

        if tags is not None:
            transform_request["Tags"] = tags

        if data_processing is not None:
            transform_request["DataProcessing"] = data_processing

        if experiment_config and len(experiment_config) > 0:
            transform_request["ExperimentConfig"] = experiment_config

        if model_client_config and len(model_client_config) > 0:
            transform_request["ModelClientConfig"] = model_client_config

        return transform_request

    def transform(
        self,
        job_name,
        model_name,
        strategy,
        max_concurrent_transforms,
        max_payload,
        env,
        input_config,
        output_config,
        resource_config,
        experiment_config,
        tags,
        data_processing,
        model_client_config=None,
    ):
        """Create an Amazon SageMaker transform job.

        Args:
            job_name (str): Name of the transform job being created.
            model_name (str): Name of the SageMaker model being used for the transform job.
            strategy (str): The strategy used to decide how to batch records in a single request.
                Possible values are 'MultiRecord' and 'SingleRecord'.
            max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
                each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP request to the
                container in MB.
            env (dict): Environment variables to be set for use during the transform job.
            input_config (dict): A dictionary describing the input data (and its location) for the
                job.
            output_config (dict): A dictionary describing the output location for the job.
            resource_config (dict): A dictionary describing the resources to complete the job.
            experiment_config (dict): A dictionary describing the experiment configuration for the
                job. Dictionary contains three optional keys,
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
            tags (list[dict]): List of tags for labeling a transform job.
            data_processing(dict): A dictionary describing config for combining the input data and
                transformed data. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            model_client_config (dict): A dictionary describing the model configuration for the
                job. Dictionary contains two optional keys,
                'InvocationsTimeoutInSeconds', and 'InvocationsMaxRetries'.
        """
        transform_request = self._get_transform_request(
            job_name=job_name,
            model_name=model_name,
            strategy=strategy,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            env=env,
            input_config=input_config,
            output_config=output_config,
            resource_config=resource_config,
            experiment_config=experiment_config,
            tags=tags,
            data_processing=data_processing,
            model_client_config=model_client_config,
        )

        LOGGER.info("Creating transform job with name: %s", job_name)
        LOGGER.debug("Transform request: %s", json.dumps(transform_request, indent=4))
        self.sagemaker_client.create_transform_job(**transform_request)

    def _create_model_request(
        self,
        name,
        role,
        container_defs,
        vpc_config=None,
        enable_network_isolation=False,
        primary_container=None,
        tags=None,
    ):  # pylint: disable=redefined-outer-name
        """Placeholder docstring"""
        if container_defs and primary_container:
            raise ValueError("Both container_defs and primary_container can not be passed as input")

        if primary_container:
            msg = (
                "primary_container is going to be deprecated in a future release. Please use "
                "container_defs instead."
            )
            warnings.warn(msg, DeprecationWarning)
            container_defs = primary_container

        role = self.expand_role(role)

        if isinstance(container_defs, list):
            container_definition = container_defs
        else:
            container_definition = _expand_container_def(container_defs)

        request = {"ModelName": name, "ExecutionRoleArn": role}
        if isinstance(container_definition, list):
            request["Containers"] = container_definition
        else:
            request["PrimaryContainer"] = container_definition
        if tags:
            request["Tags"] = tags

        if vpc_config:
            request["VpcConfig"] = vpc_config

        if enable_network_isolation:
            request["EnableNetworkIsolation"] = True

        return request

    def create_model(
        self,
        name,
        role,
        container_defs,
        vpc_config=None,
        enable_network_isolation=False,
        primary_container=None,
        tags=None,
    ):
        """Create an Amazon SageMaker ``Model``.
        Specify the S3 location of the model artifacts and Docker image containing
        the inference code. Amazon SageMaker uses this information to deploy the
        model in Amazon SageMaker. This method can also be used to create a Model for an Inference
        Pipeline if you pass the list of container definitions through the containers parameter.

        Args:
            name (str): Name of the Amazon SageMaker ``Model`` to create.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. You must grant sufficient permissions to this
                role.
            container_defs (list[dict[str, str]] or [dict[str, str]]): A single container
                definition or a list of container definitions which will be invoked sequentially
                while performing the prediction. If the list contains only one container, then
                it'll be passed to SageMaker Hosting as the ``PrimaryContainer`` and otherwise,
                it'll be passed as ``Containers``.You can also specify the  return value of
                ``sagemaker.get_container_def()`` or ``sagemaker.pipeline_container_def()``,
                which will used to create more advanced container configurations, including model
                containers which need artifacts from S3.
            vpc_config (dict[str, list[str]]): The VpcConfig set on the model (default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            enable_network_isolation (bool): Wether the model requires network isolation or not.
            primary_container (str or dict[str, str]): Docker image which defines the inference
                code. You can also specify the return value of ``sagemaker.container_def()``,
                which is used to create more advanced container configurations, including model
                containers which need artifacts from S3. This field is deprecated, please use
                container_defs instead.
            tags(List[dict[str, str]]): Optional. The list of tags to add to the model.

        Example:
            >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
            For more information about tags, see https://boto3.amazonaws.com/v1/documentation\
            /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags


        Returns:
            str: Name of the Amazon SageMaker ``Model`` created.
        """
        create_model_request = self._create_model_request(
            name=name,
            role=role,
            container_defs=container_defs,
            vpc_config=vpc_config,
            enable_network_isolation=enable_network_isolation,
            primary_container=primary_container,
            tags=tags,
        )
        LOGGER.info("Creating model with name: %s", name)
        LOGGER.debug("CreateModel request: %s", json.dumps(create_model_request, indent=4))

        try:
            self.sagemaker_client.create_model(**create_model_request)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]

            if (
                error_code == "ValidationException"
                and "Cannot create already existing model" in message
            ):
                LOGGER.warning("Using already existing model: %s", name)
            else:
                raise

        return name

    def create_model_from_job(
        self,
        training_job_name,
        name=None,
        role=None,
        image_uri=None,
        model_data_url=None,
        env=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        tags=None,
    ):
        """Create an Amazon SageMaker ``Model`` from a SageMaker Training Job.

        Args:
            training_job_name (str): The Amazon SageMaker Training Job name.
            name (str): The name of the SageMaker ``Model`` to create (default: None).
                If not specified, the training job name is used.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, specified either
                by an IAM role name or role ARN. If None, the ``RoleArn`` from the SageMaker
                Training Job will be used.
            image_uri (str): The Docker image URI (default: None). If None, it
                defaults to the training image URI from ``training_job_name``.
            model_data_url (str): S3 location of the model data (default: None). If None, defaults
                to the ``ModelS3Artifacts`` of ``training_job_name``.
            env (dict[string,string]): Model environment variables (default: {}).
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
                model.
                Default: use VpcConfig from training job.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            tags(List[dict[str, str]]): Optional. The list of tags to add to the model.
                For more, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

        Returns:
            str: The name of the created ``Model``.
        """
        training_job = self.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        name = name or training_job_name
        role = role or training_job["RoleArn"]
        env = env or {}
        primary_container = container_def(
            image_uri or training_job["AlgorithmSpecification"]["TrainingImage"],
            model_data_url=model_data_url or training_job["ModelArtifacts"]["S3ModelArtifacts"],
            env=env,
        )
        vpc_config = _vpc_config_from_training_job(training_job, vpc_config_override)
        return self.create_model(name, role, primary_container, vpc_config=vpc_config, tags=tags)

    def create_model_package_from_algorithm(self, name, description, algorithm_arn, model_data):
        """Create a SageMaker Model Package from the results of training with an Algorithm Package

        Args:
            name (str): ModelPackage name
            description (str): Model Package description
            algorithm_arn (str): arn or name of the algorithm used for training.
            model_data (str): s3 URI to the model artifacts produced by training
        """
        request = {
            "ModelPackageName": name,
            "ModelPackageDescription": description,
            "SourceAlgorithmSpecification": {
                "SourceAlgorithms": [{"AlgorithmName": algorithm_arn, "ModelDataUrl": model_data}]
            },
        }
        try:
            LOGGER.info("Creating model package with name: %s", name)
            self.sagemaker_client.create_model_package(**request)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]

            if error_code == "ValidationException" and "ModelPackage already exists" in message:
                LOGGER.warning("Using already existing model package: %s", name)
            else:
                raise

    def wait_for_model_package(self, model_package_name, poll=5):
        """Wait for an Amazon SageMaker endpoint deployment to complete.

        Args:
            endpoint (str): Name of the ``Endpoint`` to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            dict: Return value from the ``DescribeEndpoint`` API.
        """
        desc = _wait_until(
            lambda: _create_model_package_status(self.sagemaker_client, model_package_name), poll
        )
        status = desc["ModelPackageStatus"]

        if status != "Completed":
            reason = desc.get("FailureReason", None)
            raise exceptions.UnexpectedStatusException(
                message="Error creating model package {package}: {status} Reason: {reason}".format(
                    package=model_package_name, status=status, reason=reason
                ),
                allowed_statuses=["Completed"],
                actual_status=status,
            )
        return desc

    def create_endpoint_config(
        self,
        name,
        model_name,
        initial_instance_count,
        instance_type,
        accelerator_type=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
    ):
        """Create an Amazon SageMaker endpoint configuration.

        The endpoint configuration identifies the Amazon SageMaker model (created using the
        ``CreateModel`` API) and the hardware configuration on which to deploy the model. Provide
        this endpoint configuration to the ``CreateEndpoint`` API, which then launches the
        hardware and deploys the model.

        Args:
            name (str): Name of the Amazon SageMaker endpoint configuration to create.
            model_name (str): Name of the Amazon SageMaker ``Model``.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual
                number of active instances for an endpoint at any given time varies due to
                autoscaling.
            instance_type (str): Type of EC2 instance to launch, for example, 'ml.c4.xlarge'.
            accelerator_type (str): Type of Elastic Inference accelerator to attach to the
                instance. For example, 'ml.eia1.medium'.
                For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            tags(List[dict[str, str]]): Optional. The list of tags to add to the endpoint config.
            kms_key (str): The KMS key that is used to encrypt the data on the storage volume
                attached to the instance hosting the endpoint.
            data_capture_config_dict (dict): Specifies configuration related to Endpoint data
                capture for use with Amazon SageMaker Model Monitoring. Default: None.

        Example:
            >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
            For more information about tags, see
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags

        Returns:
            str: Name of the endpoint point configuration created.

        """
        LOGGER.info("Creating endpoint-config with name %s", name)

        tags = tags or []

        request = {
            "EndpointConfigName": name,
            "ProductionVariants": [
                production_variant(
                    model_name,
                    instance_type,
                    initial_instance_count,
                    accelerator_type=accelerator_type,
                )
            ],
        }

        if tags is not None:
            request["Tags"] = tags

        if kms_key is not None:
            request["KmsKeyId"] = kms_key

        if data_capture_config_dict is not None:
            request["DataCaptureConfig"] = data_capture_config_dict

        self.sagemaker_client.create_endpoint_config(**request)
        return name

    def create_endpoint_config_from_existing(
        self,
        existing_config_name,
        new_config_name,
        new_tags=None,
        new_kms_key=None,
        new_data_capture_config_dict=None,
        new_production_variants=None,
    ):
        """Create an Amazon SageMaker endpoint configuration from an existing one. Updating any
        values that were passed in.

        The endpoint configuration identifies the Amazon SageMaker model (created using the
        ``CreateModel`` API) and the hardware configuration on which to deploy the model. Provide
        this endpoint configuration to the ``CreateEndpoint`` API, which then launches the
        hardware and deploys the model.

        Args:
            new_config_name (str): Name of the Amazon SageMaker endpoint configuration to create.
            existing_config_name (str): Name of the existing Amazon SageMaker endpoint
                configuration.
            new_tags (list[dict[str, str]]): Optional. The list of tags to add to the endpoint
                config. If not specified, the tags of the existing endpoint configuration are used.
                If any of the existing tags are reserved AWS ones (i.e. begin with "aws"),
                they are not carried over to the new endpoint configuration.
            new_kms_key (str): The KMS key that is used to encrypt the data on the storage volume
                attached to the instance hosting the endpoint (default: None). If not specified,
                the KMS key of the existing endpoint configuration is used.
            new_data_capture_config_dict (dict): Specifies configuration related to Endpoint data
                capture for use with Amazon SageMaker Model Monitoring (default: None).
                If not specified, the data capture configuration of the existing
                endpoint configuration is used.
            new_production_variants (list[dict]): The configuration for which model(s) to host and
                the resources to deploy for hosting the model(s). If not specified,
                the ``ProductionVariants`` of the existing endpoint configuration is used.

        Returns:
            str: Name of the endpoint point configuration created.

        """
        LOGGER.info("Creating endpoint-config with name %s", new_config_name)

        existing_endpoint_config_desc = self.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=existing_config_name
        )

        request = {
            "EndpointConfigName": new_config_name,
        }

        request["ProductionVariants"] = (
            new_production_variants or existing_endpoint_config_desc["ProductionVariants"]
        )

        request_tags = new_tags or self.list_tags(
            existing_endpoint_config_desc["EndpointConfigArn"]
        )
        if request_tags:
            request["Tags"] = request_tags

        if new_kms_key is not None or existing_endpoint_config_desc.get("KmsKeyId") is not None:
            request["KmsKeyId"] = new_kms_key or existing_endpoint_config_desc.get("KmsKeyId")

        request_data_capture_config_dict = (
            new_data_capture_config_dict or existing_endpoint_config_desc.get("DataCaptureConfig")
        )

        if request_data_capture_config_dict is not None:
            request["DataCaptureConfig"] = request_data_capture_config_dict

        self.sagemaker_client.create_endpoint_config(**request)

    def create_endpoint(self, endpoint_name, config_name, tags=None, wait=True):
        """Create an Amazon SageMaker ``Endpoint`` according to the endpoint configuration
        specified in the request.

        Once the ``Endpoint`` is created, client applications can send requests to obtain
        inferences. The endpoint configuration is created using the ``CreateEndpointConfig`` API.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` being created.
            config_name (str): Name of the Amazon SageMaker endpoint configuration to deploy.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).

        Returns:
            str: Name of the Amazon SageMaker ``Endpoint`` created.
        """
        LOGGER.info("Creating endpoint with name %s", endpoint_name)

        tags = tags or []

        self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=tags
        )
        if wait:
            self.wait_for_endpoint(endpoint_name)
        return endpoint_name

    def update_endpoint(self, endpoint_name, endpoint_config_name, wait=True):
        """Update an Amazon SageMaker ``Endpoint`` according to the endpoint configuration
        specified in the request

        Raise an error if endpoint with endpoint_name does not exist.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to update.
            endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to
                deploy.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).

        Returns:
            str: Name of the Amazon SageMaker ``Endpoint`` being updated.

        Raises:
            ValueError: if the endpoint does not already exist
        """
        if not _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        ):
            raise ValueError(
                "Endpoint with name '{}' does not exist; please use an "
                "existing endpoint name".format(endpoint_name)
            )

        self.sagemaker_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )

        if wait:
            self.wait_for_endpoint(endpoint_name)
        return endpoint_name

    def delete_endpoint(self, endpoint_name):
        """Delete an Amazon SageMaker ``Endpoint``.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to delete.
        """
        LOGGER.info("Deleting endpoint with name: %s", endpoint_name)
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    def delete_endpoint_config(self, endpoint_config_name):
        """Delete an Amazon SageMaker endpoint configuration.

        Args:
            endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to
                delete.
        """
        LOGGER.info("Deleting endpoint configuration with name: %s", endpoint_config_name)
        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

    def delete_model(self, model_name):
        """Delete an Amazon SageMaker Model.

        Args:
            model_name (str): Name of the Amazon SageMaker model to delete.

        """
        LOGGER.info("Deleting model with name: %s", model_name)
        self.sagemaker_client.delete_model(ModelName=model_name)

    def list_tags(self, resource_arn, max_results=50):
        """List the tags given an Amazon Resource Name

        Args:
            resource_arn (str): The Amazon Resource Name (ARN) for which to get the tags list.
            max_results (int): The maximum number of results to include in a single page.
                This method takes care of that abstraction and returns a full list.

        """
        tags_list = []

        try:
            list_tags_response = self.sagemaker_client.list_tags(
                ResourceArn=resource_arn, MaxResults=max_results
            )
            tags_list = tags_list + list_tags_response["Tags"]

            next_token = list_tags_response.get("nextToken")
            while next_token is not None:
                list_tags_response = self.sagemaker_client.list_tags(
                    ResourceArn=resource_arn, MaxResults=max_results, NextToken=next_token
                )
                tags_list = tags_list + list_tags_response["Tags"]
                next_token = list_tags_response.get("nextToken")

            non_aws_tags = []
            for tag in tags_list:
                if "aws:" not in tag["Key"]:
                    non_aws_tags.append(tag)
            return non_aws_tags
        except ClientError as error:
            print("Error retrieving tags. resource_arn: {}".format(resource_arn))
            raise error

    def wait_for_job(self, job, poll=5):
        """Wait for an Amazon SageMaker training job to complete.

        Args:
            job (str): Name of the training job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeTrainingJob`` API.

        Raises:
            exceptions.UnexpectedStatusException: If the training job fails.

        """
        desc = _wait_until_training_done(
            lambda last_desc: _train_done(self.sagemaker_client, job, last_desc), None, poll
        )
        self._check_job_status(job, desc, "TrainingJobStatus")
        return desc

    def wait_for_processing_job(self, job, poll=5):
        """Wait for an Amazon SageMaker Processing job to complete.

        Args:
            job (str): Name of the processing job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeProcessingJob`` API.

        Raises:
            exceptions.UnexpectedStatusException: If the compilation job fails.
        """
        desc = _wait_until(lambda: _processing_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, "ProcessingJobStatus")
        return desc

    def wait_for_compilation_job(self, job, poll=5):
        """Wait for an Amazon SageMaker Neo compilation job to complete.

        Args:
            job (str): Name of the compilation job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeCompilationJob`` API.

        Raises:
            exceptions.UnexpectedStatusException: If the compilation job fails.
        """
        desc = _wait_until(lambda: _compilation_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, "CompilationJobStatus")
        return desc

    def wait_for_tuning_job(self, job, poll=5):
        """Wait for an Amazon SageMaker hyperparameter tuning job to complete.

        Args:
            job (str): Name of the tuning job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeHyperParameterTuningJob`` API.

        Raises:
            exceptions.UnexpectedStatusException: If the hyperparameter tuning job fails.
        """
        desc = _wait_until(lambda: _tuning_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, "HyperParameterTuningJobStatus")
        return desc

    def describe_transform_job(self, job_name):
        """Calls the DescribeTransformJob API for the given job name
        and returns the response.

        Args:
            job_name (str): The name of the transform job to describe.

        Returns:
            dict: A dictionary response with the transform job description.
        """
        return self.sagemaker_client.describe_transform_job(TransformJobName=job_name)

    def wait_for_transform_job(self, job, poll=5):
        """Wait for an Amazon SageMaker transform job to complete.

        Args:
            job (str): Name of the transform job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeTransformJob`` API.

        Raises:
            exceptions.UnexpectedStatusException: If the transform job fails.
        """
        desc = _wait_until(lambda: _transform_job_status(self.sagemaker_client, job), poll)
        self._check_job_status(job, desc, "TransformJobStatus")
        return desc

    def stop_transform_job(self, name):
        """Stop the Amazon SageMaker hyperparameter tuning job with the specified name.

        Args:
            name (str): Name of the Amazon SageMaker batch transform job.

        Raises:
            ClientError: If an error occurs while trying to stop the batch transform job.
        """
        try:
            LOGGER.info("Stopping transform job: %s", name)
            self.sagemaker_client.stop_transform_job(TransformJobName=name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            # allow to pass if the job already stopped
            if error_code == "ValidationException":
                LOGGER.info("Transform job: %s is already stopped or not running.", name)
            else:
                LOGGER.error("Error occurred while attempting to stop transform job: %s.", name)
                raise

    def _check_job_status(self, job, desc, status_key_name):
        """Check to see if the job completed successfully and, if not, construct and
        raise a exceptions.UnexpectedStatusException.

        Args:
            job (str): The name of the job to check.
            desc (dict[str, str]): The result of ``describe_training_job()``.
            status_key_name (str): Status key name to check for.

        Raises:
            exceptions.UnexpectedStatusException: If the training job fails.
        """
        status = desc[status_key_name]
        # If the status is capital case, then convert it to Camel case
        status = _STATUS_CODE_TABLE.get(status, status)

        if status not in ("Completed", "Stopped"):
            reason = desc.get("FailureReason", "(No reason provided)")
            job_type = status_key_name.replace("JobStatus", " job")
            raise exceptions.UnexpectedStatusException(
                message="Error for {job_type} {job_name}: {status}. Reason: {reason}".format(
                    job_type=job_type, job_name=job, status=status, reason=reason
                ),
                allowed_statuses=["Completed", "Stopped"],
                actual_status=status,
            )

    def wait_for_endpoint(self, endpoint, poll=30):
        """Wait for an Amazon SageMaker endpoint deployment to complete.

        Args:
            endpoint (str): Name of the ``Endpoint`` to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            dict: Return value from the ``DescribeEndpoint`` API.
        """
        desc = _wait_until(lambda: _deploy_done(self.sagemaker_client, endpoint), poll)
        status = desc["EndpointStatus"]

        if status != "InService":
            reason = desc.get("FailureReason", None)
            raise exceptions.UnexpectedStatusException(
                message="Error hosting endpoint {endpoint}: {status}. Reason: {reason}.".format(
                    endpoint=endpoint, status=status, reason=reason
                ),
                allowed_statuses=["InService"],
                actual_status=status,
            )
        return desc

    def endpoint_from_job(
        self,
        job_name,
        initial_instance_count,
        instance_type,
        image_uri=None,
        name=None,
        role=None,
        wait=True,
        model_environment_vars=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        accelerator_type=None,
        data_capture_config=None,
    ):
        """Create an ``Endpoint`` using the results of a successful training job.

        Specify the job name, Docker image containing the inference code, and hardware
        configuration to deploy the model. Internally the API, creates an Amazon SageMaker model
        (that describes the model artifacts and the Docker image containing inference code),
        endpoint configuration (describing the hardware to deploy for hosting the model), and
        creates an ``Endpoint`` (launches the EC2 instances and deploys the model on them). In
        response, the API returns the endpoint name to which you can send requests for inferences.

        Args:
            job_name (str): Name of the training job to deploy the results of.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual
                number of active instances for an endpoint at any given time varies due to
                autoscaling.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
                for example, 'ml.c4.xlarge'.
            image_uri (str): The Docker image which defines the inference code to be used
                as the entry point for accepting prediction requests. If not specified, uses the
                image used for the training job.
            name (str): Name of the ``Endpoint`` to create. If not specified, uses the training job
                name.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts. You must grant sufficient permissions to this
                role.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).
            model_environment_vars (dict[str, str]): Environment variables to set on the model
                container (default: None).
            vpc_config_override (dict[str, list[str]]): Overrides VpcConfig set on the model.
                Default: use VpcConfig from training job.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            accelerator_type (str): Type of Elastic Inference accelerator to attach to the
                instance. For example, 'ml.eia1.medium'.
                For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.

        Returns:
            str: Name of the ``Endpoint`` that is created.

        """
        job_desc = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        output_url = job_desc["ModelArtifacts"]["S3ModelArtifacts"]
        image_uri = image_uri or job_desc["AlgorithmSpecification"]["TrainingImage"]
        role = role or job_desc["RoleArn"]
        name = name or job_name
        vpc_config_override = _vpc_config_from_training_job(job_desc, vpc_config_override)

        return self.endpoint_from_model_data(
            model_s3_location=output_url,
            image_uri=image_uri,
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            name=name,
            role=role,
            wait=wait,
            model_environment_vars=model_environment_vars,
            model_vpc_config=vpc_config_override,
            accelerator_type=accelerator_type,
            data_capture_config=data_capture_config,
        )

    def endpoint_from_model_data(
        self,
        model_s3_location,
        image_uri,
        initial_instance_count,
        instance_type,
        name=None,
        role=None,
        wait=True,
        model_environment_vars=None,
        model_vpc_config=None,
        accelerator_type=None,
        data_capture_config=None,
    ):
        """Create and deploy to an ``Endpoint`` using existing model data stored in S3.

        Args:
            model_s3_location (str): S3 URI of the model artifacts to use for the endpoint.
            image_uri (str): The Docker image URI which defines the runtime code to be
                used as the entry point for accepting prediction requests.
            initial_instance_count (int): Minimum number of EC2 instances to launch. The actual
                number of active instances for an endpoint at any given time varies due to
                autoscaling.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
                e.g. 'ml.c4.xlarge'.
            name (str): Name of the ``Endpoint`` to create. If not specified, uses a name
                generated by combining the image name with a timestamp.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts.
                You must grant sufficient permissions to this role.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).
            model_environment_vars (dict[str, str]): Environment variables to set on the model
                container (default: None).
            model_vpc_config (dict[str, list[str]]): The VpcConfig set on the model (default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            accelerator_type (str): Type of Elastic Inference accelerator to attach to the instance.
                For example, 'ml.eia1.medium'.
                For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.

        Returns:
            str: Name of the ``Endpoint`` that is created.

        """
        model_environment_vars = model_environment_vars or {}
        name = name or name_from_image(image_uri)
        model_vpc_config = vpc_utils.sanitize(model_vpc_config)

        if _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_endpoint(EndpointName=name)
        ):
            raise ValueError(
                'Endpoint with name "{}" already exists; please pick a different name.'.format(name)
            )

        if not _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_model(ModelName=name)
        ):
            primary_container = container_def(
                image_uri=image_uri,
                model_data_url=model_s3_location,
                env=model_environment_vars,
            )
            self.create_model(
                name=name, role=role, container_defs=primary_container, vpc_config=model_vpc_config
            )

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        if not _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_endpoint_config(EndpointConfigName=name)
        ):
            self.create_endpoint_config(
                name=name,
                model_name=name,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                data_capture_config_dict=data_capture_config_dict,
            )

        self.create_endpoint(endpoint_name=name, config_name=name, wait=wait)
        return name

    def endpoint_from_production_variants(
        self,
        name,
        production_variants,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
    ):
        """Create an SageMaker ``Endpoint`` from a list of production variants.

        Args:
            name (str): The name of the ``Endpoint`` to create.
            production_variants (list[dict[str, str]]): The list of production variants to deploy.
            tags (list[dict[str, str]]): A list of key-value pairs for tagging the endpoint
                (default: None).
            kms_key (str): The KMS key that is used to encrypt the data on the storage volume
                attached to the instance hosting the endpoint.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).
            data_capture_config_dict (dict): Specifies configuration related to Endpoint data
                capture for use with Amazon SageMaker Model Monitoring. Default: None.

        Returns:
            str: The name of the created ``Endpoint``.

        """
        if not _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_endpoint_config(EndpointConfigName=name)
        ):
            config_options = {"EndpointConfigName": name, "ProductionVariants": production_variants}
            if tags:
                config_options["Tags"] = tags
            if kms_key:
                config_options["KmsKeyId"] = kms_key
            if data_capture_config_dict is not None:
                config_options["DataCaptureConfig"] = data_capture_config_dict

            self.sagemaker_client.create_endpoint_config(**config_options)
        return self.create_endpoint(endpoint_name=name, config_name=name, tags=tags, wait=wait)

    def expand_role(self, role):
        """Expand an IAM role name into an ARN.

        If the role is already in the form of an ARN, then the role is simply returned. Otherwise
        we retrieve the full ARN and return it.

        Args:
            role (str): An AWS IAM role (either name or full ARN).

        Returns:
            str: The corresponding AWS IAM role ARN.
        """
        if "/" in role:
            return role
        return self.boto_session.resource("iam").Role(role).arn

    def get_caller_identity_arn(self):
        """Returns the ARN user or role whose credentials are used to call the API.

        Returns:
            str: The ARN user or role
        """
        if os.path.exists(NOTEBOOK_METADATA_FILE):
            with open(NOTEBOOK_METADATA_FILE, "rb") as f:
                instance_name = json.loads(f.read())["ResourceName"]
            try:
                instance_desc = self.sagemaker_client.describe_notebook_instance(
                    NotebookInstanceName=instance_name
                )
                return instance_desc["RoleArn"]
            except ClientError:
                LOGGER.debug(
                    "Couldn't call 'describe_notebook_instance' to get the Role "
                    "ARN of the instance %s.",
                    instance_name,
                )

        assumed_role = self.boto_session.client(
            "sts",
            region_name=self.boto_region_name,
            endpoint_url=sts_regional_endpoint(self.boto_region_name),
        ).get_caller_identity()["Arn"]

        if "AmazonSageMaker-ExecutionRole" in assumed_role:
            role = re.sub(
                r"^(.+)sts::(\d+):assumed-role/(.+?)/.*$",
                r"\1iam::\2:role/service-role/\3",
                assumed_role,
            )
            return role

        role = re.sub(r"^(.+)sts::(\d+):assumed-role/(.+?)/.*$", r"\1iam::\2:role/\3", assumed_role)

        # Call IAM to get the role's path
        role_name = role[role.rfind("/") + 1 :]
        try:
            role = self.boto_session.client("iam").get_role(RoleName=role_name)["Role"]["Arn"]
        except ClientError:
            LOGGER.warning(
                "Couldn't call 'get_role' to get Role ARN from role name %s to get Role path.",
                role_name,
            )

        return role

    def logs_for_job(  # noqa: C901 - suppress complexity warning for this method
        self, job_name, wait=False, poll=10, log_type="All"
    ):
        """Display the logs for a given training job, optionally tailing them until the
        job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
        based on which instance the log entry is from.

        Args:
            job_name (str): Name of the training job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes
                (default: False).
            poll (int): The interval in seconds between polling for new log entries and job
                completion (default: 5).

        Raises:
            exceptions.UnexpectedStatusException: If waiting and the training job fails.
        """

        description = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        print(secondary_training_status_message(description, None), end="")

        instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
            self, description, job="Training"
        )

        state = _get_initial_job_state(description, "TrainingJobStatus", wait)

        # The loop below implements a state machine that alternates between checking the job status
        # and reading whatever is available in the logs at this point. Note, that if we were
        # called with wait == False, we never check the job status.
        #
        # If wait == TRUE and job is not completed, the initial state is TAILING
        # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is
        # complete).
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
        # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to
        #   Cloudwatch after the job was marked complete.
        last_describe_job_call = time.time()
        last_description = description
        last_debug_rule_statuses = None

        while True:
            _flush_log_streams(
                stream_names,
                instance_count,
                client,
                log_group,
                job_name,
                positions,
                dot,
                color_wrap,
            )
            if state == LogState.COMPLETE:
                break

            time.sleep(poll)

            if state == LogState.JOB_COMPLETE:
                state = LogState.COMPLETE
            elif time.time() - last_describe_job_call >= 30:
                description = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
                last_describe_job_call = time.time()

                if secondary_training_status_changed(description, last_description):
                    print()
                    print(secondary_training_status_message(description, last_description), end="")
                    last_description = description

                status = description["TrainingJobStatus"]

                if status in ("Completed", "Failed", "Stopped"):
                    print()
                    state = LogState.JOB_COMPLETE

                # Print prettified logs related to the status of SageMaker Debugger rules.
                debug_rule_statuses = description.get("DebugRuleEvaluationStatuses", {})
                if (
                    debug_rule_statuses
                    and _debug_rule_statuses_changed(debug_rule_statuses, last_debug_rule_statuses)
                    and (log_type in {"All", "Rules"})
                ):
                    print()
                    print("********* Debugger Rule Status *********")
                    print("*")
                    for status in debug_rule_statuses:
                        rule_log = "* {:>18}: {:<18}".format(
                            status["RuleConfigurationName"], status["RuleEvaluationStatus"]
                        )
                        print(rule_log)
                    print("*")
                    print("*" * 40)

                    last_debug_rule_statuses = debug_rule_statuses

        if wait:
            self._check_job_status(job_name, description, "TrainingJobStatus")
            if dot:
                print()
            # Customers are not billed for hardware provisioning, so billable time is less than
            # total time
            training_time = description.get("TrainingTimeInSeconds")
            billable_time = description.get("BillableTimeInSeconds")
            if training_time is not None:
                print("Training seconds:", training_time * instance_count)
            if billable_time is not None:
                print("Billable seconds:", billable_time * instance_count)
                if description.get("EnableManagedSpotTraining"):
                    saving = (1 - float(billable_time) / training_time) * 100
                    print("Managed Spot Training savings: {:.1f}%".format(saving))

    def logs_for_processing_job(self, job_name, wait=False, poll=10):
        """Display the logs for a given processing job, optionally tailing them until the
        job is complete.

        Args:
            job_name (str): Name of the processing job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes
                (default: False).
            poll (int): The interval in seconds between polling for new log entries and job
                completion (default: 5).

        Raises:
            ValueError: If the processing job fails.
        """

        description = self.sagemaker_client.describe_processing_job(ProcessingJobName=job_name)

        instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
            self, description, job="Processing"
        )

        state = _get_initial_job_state(description, "ProcessingJobStatus", wait)

        # The loop below implements a state machine that alternates between checking the job status
        # and reading whatever is available in the logs at this point. Note, that if we were
        # called with wait == False, we never check the job status.
        #
        # If wait == TRUE and job is not completed, the initial state is TAILING
        # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is
        # complete).
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
        # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to
        #   Cloudwatch after the job was marked complete.
        last_describe_job_call = time.time()
        while True:
            _flush_log_streams(
                stream_names,
                instance_count,
                client,
                log_group,
                job_name,
                positions,
                dot,
                color_wrap,
            )
            if state == LogState.COMPLETE:
                break

            time.sleep(poll)

            if state == LogState.JOB_COMPLETE:
                state = LogState.COMPLETE
            elif time.time() - last_describe_job_call >= 30:
                description = self.sagemaker_client.describe_processing_job(
                    ProcessingJobName=job_name
                )
                last_describe_job_call = time.time()

                status = description["ProcessingJobStatus"]

                if status in ("Completed", "Failed", "Stopped"):
                    print()
                    state = LogState.JOB_COMPLETE

        if wait:
            self._check_job_status(job_name, description, "ProcessingJobStatus")
            if dot:
                print()

    def logs_for_transform_job(self, job_name, wait=False, poll=10):
        """Display the logs for a given transform job, optionally tailing them until the
        job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
        based on which instance the log entry is from.

        Args:
            job_name (str): Name of the transform job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes
                (default: False).
            poll (int): The interval in seconds between polling for new log entries and job
                completion (default: 5).

        Raises:
            ValueError: If the transform job fails.
        """

        description = self.sagemaker_client.describe_transform_job(TransformJobName=job_name)

        instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
            self, description, job="Transform"
        )

        state = _get_initial_job_state(description, "TransformJobStatus", wait)

        # The loop below implements a state machine that alternates between checking the job status
        # and reading whatever is available in the logs at this point. Note, that if we were
        # called with wait == False, we never check the job status.
        #
        # If wait == TRUE and job is not completed, the initial state is TAILING
        # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is
        # complete).
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
        # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to
        #   Cloudwatch after the job was marked complete.
        last_describe_job_call = time.time()
        while True:
            _flush_log_streams(
                stream_names,
                instance_count,
                client,
                log_group,
                job_name,
                positions,
                dot,
                color_wrap,
            )
            if state == LogState.COMPLETE:
                break

            time.sleep(poll)

            if state == LogState.JOB_COMPLETE:
                state = LogState.COMPLETE
            elif time.time() - last_describe_job_call >= 30:
                description = self.sagemaker_client.describe_transform_job(
                    TransformJobName=job_name
                )
                last_describe_job_call = time.time()

                status = description["TransformJobStatus"]

                if status in ("Completed", "Failed", "Stopped"):
                    print()
                    state = LogState.JOB_COMPLETE

        if wait:
            self._check_job_status(job_name, description, "TransformJobStatus")
            if dot:
                print()


def container_def(image_uri, model_data_url=None, env=None, container_mode=None):
    """Create a definition for executing a container as part of a SageMaker model.

    Args:
        image_uri (str): Docker image URI to run for this container.
        model_data_url (str): S3 URI of data required by this container,
            e.g. SageMaker training job model artifacts (default: None).
        env (dict[str, str]): Environment variables to set inside the container (default: None).
        container_mode (str): The model container mode. Valid modes:
                * MultiModel: Indicates that model container can support hosting multiple models
                * SingleModel: Indicates that model container can support hosting a single model
                This is the default model container mode when container_mode = None
    Returns:
        dict[str, str]: A complete container definition object usable with the CreateModel API if
        passed via `PrimaryContainers` field.
    """
    if env is None:
        env = {}
    c_def = {"Image": image_uri, "Environment": env}
    if model_data_url:
        c_def["ModelDataUrl"] = model_data_url
    if container_mode:
        c_def["Mode"] = container_mode
    return c_def


def pipeline_container_def(models, instance_type=None):
    """Create a definition for executing a pipeline of containers as part of a SageMaker model.

    Args:
        models (list[sagemaker.Model]): this will be a list of ``sagemaker.Model`` objects in the
            order the inference should be invoked.
        instance_type (str): The EC2 instance type to deploy this Model to. For example,
            'ml.p2.xlarge' (default: None).

    Returns:
        list[dict[str, str]]: list of container definition objects usable with with the
            CreateModel API for inference pipelines if passed via `Containers` field.
    """
    c_defs = []  # should contain list of container definitions in the same order customer passed
    for model in models:
        c_defs.append(model.prepare_container_def(instance_type))
    return c_defs


def production_variant(
    model_name,
    instance_type,
    initial_instance_count=1,
    variant_name="AllTraffic",
    initial_weight=1,
    accelerator_type=None,
):
    """Create a production variant description suitable for use in a ``ProductionVariant`` list as
    part of a ``CreateEndpointConfig`` request.

    Args:
        model_name (str): The name of the SageMaker model this production variant references.
        instance_type (str): The EC2 instance type for this production variant. For example,
            'ml.c4.8xlarge'.
        initial_instance_count (int): The initial instance count for this production variant
            (default: 1).
        variant_name (string): The ``VariantName`` of this production variant
            (default: 'AllTraffic').
        initial_weight (int): The relative ``InitialVariantWeight`` of this production variant
            (default: 1).
        accelerator_type (str): Type of Elastic Inference accelerator for this production variant.
            For example, 'ml.eia1.medium'.
            For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

    Returns:
        dict[str, str]: An SageMaker ``ProductionVariant`` description
    """
    production_variant_configuration = {
        "ModelName": model_name,
        "InstanceType": instance_type,
        "InitialInstanceCount": initial_instance_count,
        "VariantName": variant_name,
        "InitialVariantWeight": initial_weight,
    }

    if accelerator_type:
        production_variant_configuration["AcceleratorType"] = accelerator_type

    return production_variant_configuration


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

    if ":role/" in arn:
        return arn
    message = (
        "The current AWS identity is not a role: {}, therefore it cannot be used as a "
        "SageMaker execution role"
    )
    raise ValueError(message.format(arn))


def _deployment_entity_exists(describe_fn):
    """Placeholder docstring"""
    try:
        describe_fn()
        return True
    except ClientError as ce:
        error_code = ce.response["Error"]["Code"]
        if not (
            error_code == "ValidationException"
            and "Could not find" in ce.response["Error"]["Message"]
        ):
            raise ce
        return False


def _train_done(sagemaker_client, job_name, last_desc):
    """Placeholder docstring"""
    in_progress_statuses = ["InProgress", "Created"]

    desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    status = desc["TrainingJobStatus"]

    if secondary_training_status_changed(desc, last_desc):
        print()
        print(secondary_training_status_message(desc, last_desc), end="")
    else:
        print(".", end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return desc, False

    print()
    return desc, True


def _processing_job_status(sagemaker_client, job_name):
    """Prints the job status for the given processing job name.
    Returns the job description.

    Args:
        sagemaker_client: The boto3 SageMaker client.
        job_name (str): The name of the job for which the status
            is requested.

    Returns:
        dict: The processing job description.

    """
    compile_status_codes = {
        "Completed": "!",
        "InProgress": ".",
        "Failed": "*",
        "Stopped": "s",
        "Stopping": "_",
    }
    in_progress_statuses = ["InProgress", "Stopping", "Starting"]

    desc = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
    status = desc["ProcessingJobStatus"]

    status = _STATUS_CODE_TABLE.get(status, status)
    print(compile_status_codes.get(status, "?"), end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    return desc


def _compilation_job_status(sagemaker_client, job_name):
    """Placeholder docstring"""
    compile_status_codes = {
        "Completed": "!",
        "InProgress": ".",
        "Failed": "*",
        "Stopped": "s",
        "Stopping": "_",
    }
    in_progress_statuses = ["InProgress", "Stopping", "Starting"]

    desc = sagemaker_client.describe_compilation_job(CompilationJobName=job_name)
    status = desc["CompilationJobStatus"]

    status = _STATUS_CODE_TABLE.get(status, status)
    print(compile_status_codes.get(status, "?"), end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    return desc


def _tuning_job_status(sagemaker_client, job_name):
    """Placeholder docstring"""
    tuning_status_codes = {
        "Completed": "!",
        "InProgress": ".",
        "Failed": "*",
        "Stopped": "s",
        "Stopping": "_",
    }
    in_progress_statuses = ["InProgress", "Stopping"]

    desc = sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=job_name
    )
    status = desc["HyperParameterTuningJobStatus"]

    print(tuning_status_codes.get(status, "?"), end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print("")
    return desc


def _transform_job_status(sagemaker_client, job_name):
    """Placeholder docstring"""
    transform_job_status_codes = {
        "Completed": "!",
        "InProgress": ".",
        "Failed": "*",
        "Stopped": "s",
        "Stopping": "_",
    }
    in_progress_statuses = ["InProgress", "Stopping"]

    desc = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    status = desc["TransformJobStatus"]

    print(transform_job_status_codes.get(status, "?"), end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print("")
    return desc


def _auto_ml_job_status(sagemaker_client, job_name):
    """Placeholder docstring"""
    auto_ml_job_status_codes = {
        "Completed": "!",
        "InProgress": ".",
        "Failed": "*",
        "Stopped": "s",
        "Stopping": "_",
    }
    in_progress_statuses = ["InProgress", "Stopping"]

    desc = sagemaker_client.describe_auto_ml_job(AutoMLJobName=job_name)
    status = desc["AutoMLJobStatus"]

    print(auto_ml_job_status_codes.get(status, "?"), end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print("")
    return desc


def _create_model_package_status(sagemaker_client, model_package_name):
    """Placeholder docstring"""
    in_progress_statuses = ["InProgress", "Pending"]

    desc = sagemaker_client.describe_model_package(ModelPackageName=model_package_name)
    status = desc["ModelPackageStatus"]
    print(".", end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print("")
    return desc


def _deploy_done(sagemaker_client, endpoint_name):
    """Placeholder docstring"""
    hosting_status_codes = {
        "OutOfService": "x",
        "Creating": "-",
        "Updating": "-",
        "InService": "!",
        "RollingBack": "<",
        "Deleting": "o",
        "Failed": "*",
    }
    in_progress_statuses = ["Creating", "Updating"]

    desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = desc["EndpointStatus"]

    print(hosting_status_codes.get(status, "?"), end="")
    sys.stdout.flush()

    return None if status in in_progress_statuses else desc


def _wait_until_training_done(callable_fn, desc, poll=5):
    """Placeholder docstring"""
    job_desc, finished = callable_fn(desc)
    while not finished:
        time.sleep(poll)
        job_desc, finished = callable_fn(job_desc)
    return job_desc


def _wait_until(callable_fn, poll=5):
    """Placeholder docstring"""
    result = callable_fn()
    while result is None:
        time.sleep(poll)
        result = callable_fn()
    return result


def _expand_container_def(c_def):
    """Placeholder docstring"""
    if isinstance(c_def, six.string_types):
        return container_def(c_def)
    return c_def


def _vpc_config_from_training_job(
    training_job_desc, vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT
):
    """Placeholder docstring"""
    if vpc_config_override is vpc_utils.VPC_CONFIG_DEFAULT:
        return training_job_desc.get(vpc_utils.VPC_CONFIG_KEY)
    return vpc_utils.sanitize(vpc_config_override)


def _get_initial_job_state(description, status_key, wait):
    """Placeholder docstring"""
    status = description[status_key]
    job_already_completed = status in ("Completed", "Failed", "Stopped")
    return LogState.TAILING if wait and not job_already_completed else LogState.COMPLETE


def _debug_rule_statuses_changed(current_statuses, last_statuses):
    """Checks the rule evaluation statuses for SageMaker Debugger rules."""
    if not last_statuses:
        return True

    for current, last in zip(current_statuses, last_statuses):
        if (current["RuleConfigurationName"] == last["RuleConfigurationName"]) and (
            current["RuleEvaluationStatus"] != last["RuleEvaluationStatus"]
        ):
            return True

    return False


def _logs_init(sagemaker_session, description, job):
    """Placeholder docstring"""
    if job == "Training":
        instance_count = description["ResourceConfig"]["InstanceCount"]
    elif job == "Transform":
        instance_count = description["TransformResources"]["InstanceCount"]
    elif job == "Processing":
        instance_count = description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
    elif job == "AutoML":
        instance_count = 0

    stream_names = []  # The list of log streams
    positions = {}  # The current position in each stream, map of stream name -> position

    # Increase retries allowed (from default of 4), as we don't want waiting for a training job
    # to be interrupted by a transient exception.
    config = botocore.config.Config(retries={"max_attempts": 15})
    client = sagemaker_session.boto_session.client("logs", config=config)
    log_group = "/aws/sagemaker/" + job + "Jobs"

    dot = False

    color_wrap = sagemaker.logs.ColorWrap()

    return instance_count, stream_names, positions, client, log_group, dot, color_wrap


def _flush_log_streams(
    stream_names, instance_count, client, log_group, job_name, positions, dot, color_wrap
):
    """Placeholder docstring"""
    if len(stream_names) < instance_count:
        # Log streams are created whenever a container starts writing to stdout/err, so this list
        # may be dynamic until we have a stream for every instance.
        try:
            streams = client.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name + "/",
                orderBy="LogStreamName",
                limit=min(instance_count, 50),
            )
            stream_names = [s["logStreamName"] for s in streams["logStreams"]]

            while "nextToken" in streams:
                streams = client.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=job_name + "/",
                    orderBy="LogStreamName",
                    limit=50,
                )

                stream_names.extend([s["logStreamName"] for s in streams["logStreams"]])

            positions.update(
                [
                    (s, sagemaker.logs.Position(timestamp=0, skip=0))
                    for s in stream_names
                    if s not in positions
                ]
            )
        except ClientError as e:
            # On the very first training job run on an account, there's no log group until
            # the container starts logging, so ignore any errors thrown about that
            err = e.response.get("Error", {})
            if err.get("Code", None) != "ResourceNotFoundException":
                raise

    if len(stream_names) > 0:
        if dot:
            print("")
            dot = False
        for idx, event in sagemaker.logs.multi_stream_iter(
            client, log_group, stream_names, positions
        ):
            color_wrap(idx, event["message"])
            ts, count = positions[stream_names[idx]]
            if event["timestamp"] == ts:
                positions[stream_names[idx]] = sagemaker.logs.Position(timestamp=ts, skip=count + 1)
            else:
                positions[stream_names[idx]] = sagemaker.logs.Position(
                    timestamp=event["timestamp"], skip=1
                )
    else:
        dot = True
        print(".", end="")
        sys.stdout.flush()


s3_input = deprecated_class(TrainingInput, "sagemaker.session.s3_input")
ShuffleConfig = deprecated_class(ShuffleConfig, "sagemaker.session.ShuffleConfig")
