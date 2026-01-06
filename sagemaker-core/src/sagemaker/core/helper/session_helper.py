# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import, annotations, print_function

import json
import logging
import os
import re
import sys
import time
import uuid
import six
import warnings
from functools import reduce
from typing import Dict, Any, Optional, List

import boto3
import botocore
import botocore.config
from botocore.exceptions import ClientError
from sagemaker.core import exceptions
from sagemaker.core.common_utils import (
    secondary_training_status_changed,
    secondary_training_status_message,
)
import sagemaker.core.logs
from sagemaker.core.session_settings import SessionSettings
from sagemaker.core.common_utils import (
    secondary_training_status_changed,
    secondary_training_status_message,
    sts_regional_endpoint,
    retries,
    resolve_value_from_config,
    get_sagemaker_config_value,
    resolve_class_attribute_from_config,
    resolve_nested_dict_value_from_config,
    update_nested_dictionary_with_values_from_config,
    update_list_of_dicts_with_values_from_config,
    format_tags,
    Tags,
    TagsDict,
    instance_supports_kms,
    create_paginator_config,
)

from sagemaker.core.config.config_utils import _log_sagemaker_config_merge
from sagemaker.core._studio import _append_project_tags
from sagemaker.core.config.config import load_sagemaker_config, validate_sagemaker_config
from sagemaker.core.config.config_schema import (
    KEY,
    TRANSFORM_JOB,
    TRANSFORM_JOB_ENVIRONMENT_PATH,
    TRANSFORM_JOB_KMS_KEY_ID_PATH,
    TRANSFORM_OUTPUT_KMS_KEY_ID_PATH,
    VOLUME_KMS_KEY_ID,
    TRANSFORM_JOB_VOLUME_KMS_KEY_ID_PATH,
    MODEL,
    MODEL_CONTAINERS_PATH,
    MODEL_EXECUTION_ROLE_ARN_PATH,
    MODEL_ENABLE_NETWORK_ISOLATION_PATH,
    MODEL_PRIMARY_CONTAINER_PATH,
    MODEL_VPC_CONFIG_PATH,
    ENDPOINT_CONFIG_PRODUCTION_VARIANTS_PATH,
    KMS_KEY_ID,
    ENDPOINT_CONFIG_KMS_KEY_ID_PATH,
    ENDPOINT_CONFIG,
    ENDPOINT_CONFIG_DATA_CAPTURE_PATH,
    ENDPOINT_CONFIG_ASYNC_INFERENCE_PATH,
    ENDPOINT_CONFIG_VPC_CONFIG_PATH,
    ENDPOINT_CONFIG_ENABLE_NETWORK_ISOLATION_PATH,
    ENDPOINT_CONFIG_EXECUTION_ROLE_ARN_PATH,
    ENDPOINT,
    INFERENCE_COMPONENT,
    SAGEMAKER,
    TAGS,
    SESSION_DEFAULT_S3_BUCKET_PATH,
    SESSION_DEFAULT_S3_OBJECT_KEY_PREFIX_PATH,
)

# Setting LOGGER for backward compatibility, in case users import it...
logger = LOGGER = logging.getLogger("sagemaker")

NOTEBOOK_METADATA_FILE = "/opt/ml/metadata/resource-metadata.json"
MODEL_MONITOR_ONE_TIME_SCHEDULE = "NOW"
_STATUS_CODE_TABLE = {
    "COMPLETED": "Completed",
    "INPROGRESS": "InProgress",
    "IN_PROGRESS": "InProgress",
    "FAILED": "Failed",
    "STOPPED": "Stopped",
    "STOPPING": "Stopping",
    "STARTING": "Starting",
    "PENDING": "Pending",
}
EP_LOGGER_POLL = 30
DEFAULT_EP_POLL = 30


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
        sagemaker_featurestore_runtime_client=None,
        default_bucket=None,
        sagemaker_config: dict = None,
        settings=None,
        sagemaker_metrics_client=None,
        default_bucket_prefix: str = None,
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
            sagemaker_featurestore_runtime_client (boto3.SageMakerFeatureStoreRuntime.Client):
                Client which makes SageMaker FeatureStore record related calls to Amazon SageMaker
                (default: None). If not provided, one will be created using
                this instance's ``boto_session``.
            default_bucket (str): The default Amazon S3 bucket to be used by this session.
                This will be created the next time an Amazon S3 bucket is needed (by calling
                :func:`default_bucket`).
                If not provided, it will be fetched from the sagemaker_config. If not configured
                there either, a default bucket will be created based on the following format:
                "sagemaker-{region}-{aws-account-id}".
                Example: "sagemaker-my-custom-bucket".
            sagemaker_metrics_client (boto3.SageMakerMetrics.Client):
                Client which makes SageMaker Metrics related calls to Amazon SageMaker
                (default: None). If not provided, one will be created using
                this instance's ``boto_session``.
            default_bucket_prefix (str): The default prefix to use for S3 Object Keys. (default:
                None). If provided and where applicable, it will be used by the SDK to construct
                default S3 URIs, in the format:
                `s3://{default_bucket}/{default_bucket_prefix}/<rest of object key>`
                This parameter can also be specified via `{sagemaker_config}` instead of here. If
                not provided here or within `{sagemaker_config}`, default S3 URIs will have the
                format: `s3://{default_bucket}/<rest of object key>`
        """

        # sagemaker_config is validated and initialized inside :func:`_initialize`,
        # so if default_bucket is None and the sagemaker_config has a default S3 bucket configured,
        # _default_bucket_name_override will be set again inside :func:`_initialize`.
        self.endpoint_arn = None
        self._default_bucket = None
        self._default_bucket_name_override = default_bucket
        # this may also be set again inside :func:`_initialize` if it is None
        self.default_bucket_prefix = default_bucket_prefix
        self._default_bucket_set_by_sdk = False

        self.s3_resource = None
        self.s3_client = None
        self.resource_groups_client = None
        self.resource_group_tagging_client = None
        self._config = None
        self.lambda_client = None
        self.settings = settings if settings else SessionSettings()

        self._initialize(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            sagemaker_runtime_client=sagemaker_runtime_client,
            sagemaker_featurestore_runtime_client=sagemaker_featurestore_runtime_client,
            sagemaker_metrics_client=sagemaker_metrics_client,
            sagemaker_config=sagemaker_config,
        )

    def _initialize(
        self,
        boto_session,
        sagemaker_client,
        sagemaker_runtime_client,
        sagemaker_featurestore_runtime_client,
        sagemaker_metrics_client,
        sagemaker_config: dict = None,
    ):
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

        # Make use of user_agent_extra field of the botocore_config object
        # to append SageMaker Python SDK specific user_agent suffix
        # to the current User-Agent header value from boto3
        # This config will also make sure that user_agent never fails to log the User-Agent string
        # even if boto User-Agent header format is updated in the future
        # Ref: https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html

        # Create sagemaker_client with the botocore_config object
        # This config is customized to append SageMaker Python SDK specific user_agent suffix
        if sagemaker_client is not None:
            self.sagemaker_client = sagemaker_client
        else:
            from sagemaker.core.user_agent import get_user_agent_extra_suffix
            config = botocore.config.Config(user_agent_extra=get_user_agent_extra_suffix())
            self.sagemaker_client = self.boto_session.client("sagemaker", config=config)

        if sagemaker_runtime_client is not None:
            self.sagemaker_runtime_client = sagemaker_runtime_client
        else:
            config = botocore.config.Config(read_timeout=80)
            self.sagemaker_runtime_client = self.boto_session.client(
                "runtime.sagemaker", config=config
            )

        if sagemaker_featurestore_runtime_client:
            self.sagemaker_featurestore_runtime_client = sagemaker_featurestore_runtime_client
        else:
            self.sagemaker_featurestore_runtime_client = self.boto_session.client(
                "sagemaker-featurestore-runtime"
            )

        if sagemaker_metrics_client:
            self.sagemaker_metrics_client = sagemaker_metrics_client
        else:
            self.sagemaker_metrics_client = self.boto_session.client("sagemaker-metrics")

        self.s3_client = self.boto_session.client("s3", region_name=self.boto_region_name)
        self.s3_resource = self.boto_session.resource("s3", region_name=self.boto_region_name)

        self.local_mode = False

        if sagemaker_config:
            validate_sagemaker_config(sagemaker_config)
            self.sagemaker_config = sagemaker_config
        else:
            # self.s3_resource might be None. If it is None, load_sagemaker_config will
            # create a default S3 resource, but only if it needs to fetch from S3
            self.sagemaker_config = load_sagemaker_config(s3_resource=self.s3_resource)

        # after sagemaker_config initialization, update self._default_bucket_name_override if needed
        self._default_bucket_name_override = resolve_value_from_config(
            direct_input=self._default_bucket_name_override,
            config_path=SESSION_DEFAULT_S3_BUCKET_PATH,
            sagemaker_session=self,
        )
        # after sagemaker_config initialization, update self.default_bucket_prefix if needed
        self.default_bucket_prefix = resolve_value_from_config(
            direct_input=self.default_bucket_prefix,
            config_path=SESSION_DEFAULT_S3_OBJECT_KEY_PREFIX_PATH,
            sagemaker_session=self,
        )

    def account_id(self) -> str:
        """Get the AWS account id of the caller.

        Returns:
            AWS account ID.
        """
        region = self.boto_session.region_name
        sts_client = self.boto_session.client(
            "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
        )
        return sts_client.get_caller_identity()["Account"]

    @property
    def config(self) -> Dict | None:
        """The config for the local mode, unused in a normal session"""
        return self._config

    @config.setter
    def config(self, value: Dict | None):
        """The config for the local mode, unused in a normal session"""
        self._config = value

    @property
    def boto_region_name(self):
        """Placeholder docstring"""
        return self._region_name

    def get_caller_identity_arn(self):
        """Returns the ARN user or role whose credentials are used to call the API.

        Returns:
            str: The ARN user or role
        """
        if os.path.exists(NOTEBOOK_METADATA_FILE):
            with open(NOTEBOOK_METADATA_FILE, "rb") as f:
                metadata = json.loads(f.read())
                instance_name = metadata.get("ResourceName")
                domain_id = metadata.get("DomainId")
                user_profile_name = metadata.get("UserProfileName")
                execution_role_arn = metadata.get("ExecutionRoleArn")
            try:
                # find execution role from the metadata file if present
                if execution_role_arn is not None:
                    return execution_role_arn

                if domain_id is None:
                    instance_desc = self.sagemaker_client.describe_notebook_instance(
                        NotebookInstanceName=instance_name
                    )
                    return instance_desc["RoleArn"]

                user_profile_desc = self.sagemaker_client.describe_user_profile(
                    DomainId=domain_id, UserProfileName=user_profile_name
                )

                # First, try to find role in userSettings
                if user_profile_desc.get("UserSettings", {}).get("ExecutionRole"):
                    return user_profile_desc["UserSettings"]["ExecutionRole"]

                # If not found, fallback to the domain
                domain_desc = self.sagemaker_client.describe_domain(DomainId=domain_id)
                return domain_desc["DefaultUserSettings"]["ExecutionRole"]
            except ClientError:
                logger.debug(
                    "Couldn't call 'describe_notebook_instance' to get the Role "
                    "ARN of the instance %s.",
                    instance_name,
                )

        assumed_role = self.boto_session.client(
            "sts",
            region_name=self.boto_region_name,
            endpoint_url=sts_regional_endpoint(self.boto_region_name),
        ).get_caller_identity()["Arn"]

        role = re.sub(r"^(.+)sts::(\d+):assumed-role/(.+?)/.*$", r"\1iam::\2:role/\3", assumed_role)

        # Call IAM to get the role's path
        role_name = role[role.rfind("/") + 1 :]
        try:
            role = self.boto_session.client("iam").get_role(RoleName=role_name)["Role"]["Arn"]
        except ClientError:
            logger.warning(
                "Couldn't call 'get_role' to get Role ARN from role name %s to get Role path.",
                role_name,
            )

            # This conditional has been present since the inception of SageMaker
            # Guessing this conditional's purpose was to handle lack of IAM permissions
            # https://github.com/aws/sagemaker-python-sdk/issues/2089#issuecomment-791802713
            if "AmazonSageMaker-ExecutionRole" in assumed_role:
                logger.warning(
                    "Assuming role was created in SageMaker AWS console, "
                    "as the name contains `AmazonSageMaker-ExecutionRole`. "
                    "Defaulting to Role ARN with service-role in path. "
                    "If this Role ARN is incorrect, please add "
                    "IAM read permissions to your role or supply the "
                    "Role Arn directly."
                )
                role = re.sub(
                    r"^(.+)sts::(\d+):assumed-role/(.+?)/.*$",
                    r"\1iam::\2:role/service-role/\3",
                    assumed_role,
                )

        return role

    def upload_data(self, path, bucket=None, key_prefix="data", callback=None, extra_args=None):
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
        bucket, key_prefix = self.determine_bucket_and_prefix(
            bucket=bucket, key_prefix=key_prefix, sagemaker_session=self
        )

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

        if self.s3_resource is None:
            s3 = self.boto_session.resource("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_resource

        for local_path, s3_key in files:
            s3.Object(bucket, s3_key).upload_file(
                local_path, Callback=callback, ExtraArgs=extra_args
            )

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
            list[str]: List of local paths of downloaded files
        """
        # Initialize the S3 client.
        if self.s3_client is None:
            s3 = self.boto_session.client("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_client

        # Initialize the variables used to loop through the contents of the S3 bucket.
        keys = []
        directories = []
        next_token = ""
        base_parameters = {"Bucket": bucket, "Prefix": key_prefix}

        # Loop through the contents of the bucket, 1,000 objects at a time. Gathering all keys into
        # a "keys" list.
        while next_token is not None:
            request_parameters = base_parameters.copy()
            if next_token != "":
                request_parameters.update({"ContinuationToken": next_token})
            response = s3.list_objects_v2(**request_parameters)
            contents = response.get("Contents", None)
            if not contents:
                logger.info(
                    "Nothing to download from bucket: %s, key_prefix: %s.", bucket, key_prefix
                )
                return []
            # For each object, save its key or directory.
            for s3_object in contents:
                key: str = s3_object.get("Key")
                obj_size = s3_object.get("Size")
                if key.endswith("/") and int(obj_size) == 0:
                    directories.append(os.path.join(path, key))
                else:
                    keys.append(key)
            next_token = response.get("NextContinuationToken")

        # For each object key, create the directory on the local machine if needed, and then
        # download the file.
        downloaded_paths = []
        for dir_path in directories:
            os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        for key in keys:
            tail_s3_uri_path = os.path.basename(key)
            if not os.path.splitext(key_prefix)[1]:
                tail_s3_uri_path = os.path.relpath(key, key_prefix)
            destination_path = os.path.join(path, tail_s3_uri_path)
            if not os.path.exists(os.path.dirname(destination_path)):
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            s3.download_file(
                Bucket=bucket, Key=key, Filename=destination_path, ExtraArgs=extra_args
            )
            downloaded_paths.append(destination_path)
        return downloaded_paths

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

        This function will create the s3 bucket if it does not exist.

        Returns:
            str: The name of the default bucket. If the name was not explicitly specified through
                the Session or sagemaker_config, the bucket will take the form:
                ``sagemaker-{region}-{AWS account ID}``.
        """

        if self._default_bucket:
            return self._default_bucket

        region = self.boto_session.region_name

        default_bucket = self._default_bucket_name_override
        if not default_bucket:
            default_bucket = self.generate_default_sagemaker_bucket_name(self.boto_session)
            self._default_bucket_set_by_sdk = True

        self._create_s3_bucket_if_it_does_not_exist(
            bucket_name=default_bucket,
            region=region,
        )

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
            self.general_bucket_check_if_user_has_permission(bucket_name, s3, bucket, region, True)

        elif self._default_bucket_set_by_sdk:
            self.general_bucket_check_if_user_has_permission(bucket_name, s3, bucket, region, False)

            expected_bucket_owner_id = self.account_id()
            self.expected_bucket_owner_id_bucket_check(bucket_name, s3, expected_bucket_owner_id)

    def expected_bucket_owner_id_bucket_check(self, bucket_name, s3, expected_bucket_owner_id):
        """Checks if the bucket belongs to a particular owner and throws a Client Error if it is not

        Args:
            bucket_name (str): Name of the S3 bucket
            s3 (str): S3 object from boto session
            expected_bucket_owner_id (str): Owner ID string

        """
        try:
            if self.default_bucket_prefix:
                s3.meta.client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=self.default_bucket_prefix,
                    ExpectedBucketOwner=expected_bucket_owner_id,
                )
            else:
                s3.meta.client.head_bucket(
                    Bucket=bucket_name, ExpectedBucketOwner=expected_bucket_owner_id
                )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]
            if error_code == "403" and message == "Forbidden":
                LOGGER.error(
                    "Since default_bucket param was not set, SageMaker Python SDK tried to use "
                    "%s bucket. "
                    "This bucket cannot be configured to use as it is not owned by Account %s. "
                    "To unblock it's recommended to use custom default_bucket "
                    "parameter in sagemaker.Session",
                    bucket_name,
                    expected_bucket_owner_id,
                )
                raise

    def general_bucket_check_if_user_has_permission(
        self, bucket_name, s3, bucket, region, bucket_creation_date_none
    ):
        """Checks if the person running has the permissions to the bucket

        If there is any other error that comes up with calling head bucket, it is raised up here
        If there is no bucket , it will create one

        Args:
            bucket_name (str): Name of the S3 bucket
            s3 (str): S3 object from boto session
            region (str): The region in which to create the bucket.
            bucket_creation_date_none (bool):Indicating whether S3 bucket already exists or not
        """
        try:
            if self.default_bucket_prefix:
                s3.meta.client.list_objects_v2(
                    Bucket=bucket_name, Prefix=self.default_bucket_prefix
                )
            else:
                s3.meta.client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]
            # bucket does not exist or forbidden to access
            if bucket_creation_date_none:
                if error_code == "404" and message == "Not Found":
                    self.create_bucket_for_not_exist_error(bucket_name, region, s3)
                elif error_code == "403" and message == "Forbidden":
                    LOGGER.error(
                        "Bucket %s exists, but access is forbidden. Please try again after "
                        "adding appropriate access.",
                        bucket.name,
                    )
                    raise
                else:
                    raise

    def create_bucket_for_not_exist_error(self, bucket_name, region, s3):
        """Creates the S3 bucket in the given region

        Args:
            bucket_name (str): Name of the S3 bucket
            s3 (str): S3 object from boto session
            region (str): The region in which to create the bucket.
        """
        # bucket does not exist, create one
        try:
            if region == "us-east-1":
                # 'us-east-1' cannot be specified because it is the default region:
                # https://github.com/boto/boto3/issues/125
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )

            logger.info("Created S3 bucket: %s", bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]

            if error_code == "OperationAborted" and "conflicting conditional operation" in message:
                # If this bucket is already being concurrently created,
                # we don't need to create it again.
                pass
            else:
                raise

    def generate_default_sagemaker_bucket_name(self, boto_session):
        """Generates a name for the default sagemaker S3 bucket.

        Args:
            boto_session (boto3.session.Session): The underlying Boto3 session which AWS service
        """
        region = boto_session.region_name
        account = boto_session.client(
            "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
        ).get_caller_identity()["Account"]
        return "sagemaker-{}-{}".format(region, account)

    def determine_bucket_and_prefix(
        self, bucket: Optional[str] = None, key_prefix: Optional[str] = None, sagemaker_session=None
    ):
        """Helper function that returns the correct S3 bucket and prefix to use depending on the inputs.

        Args:
            bucket (Optional[str]): S3 Bucket to use (if it exists)
            key_prefix (Optional[str]): S3 Object Key Prefix to use or append to (if it exists)
            sagemaker_session (sagemaker.core.session.Session): Session to fetch a default bucket and
                prefix from, if bucket doesn't exist. Expected to exist

        Returns: The correct S3 Bucket and S3 Object Key Prefix that should be used
        """
        if bucket:
            final_bucket = bucket
            final_key_prefix = key_prefix
        else:
            final_bucket = sagemaker_session.default_bucket()

            # default_bucket_prefix (if it exists) should be appended if (and only if) 'bucket' does not
            # exist and we are using the Session's default_bucket.
            final_key_prefix = s3_path_join(sagemaker_session.default_bucket_prefix, key_prefix)

        # We should not append default_bucket_prefix even if the bucket exists but is equal to the
        # default_bucket, because either:
        # (1) the bucket was explicitly passed in by the user and just happens to be the same as the
        # default_bucket (in which case we don't want to change the user's input), or
        # (2) the default_bucket was fetched from Session earlier already (and the default prefix
        # should have been fetched then as well), and then this function was
        # called with it. If we appended the default prefix here, we would be appending it more than
        # once in total.

        return final_bucket, final_key_prefix

    def _append_sagemaker_config_tags(self, tags: List[TagsDict], config_path_to_tags: str):
        """Appends tags specified in the sagemaker_config to the given list of tags.

        To minimize the chance of duplicate tags being applied, this is intended to be used
        immediately before calls to sagemaker_client, rather than during initialization of
        classes like EstimatorBase.

        Args:
            tags: The list of tags to append to.
            config_path_to_tags: The path to look up tags in the config.

        Returns:
            A list of tags.
        """
        config_tags = get_sagemaker_config_value(self, config_path_to_tags)

        if config_tags is None or len(config_tags) == 0:
            return tags

        all_tags = tags or []
        for config_tag in config_tags:
            config_tag_key = config_tag[KEY]
            if not any(tag.get("Key", None) == config_tag_key for tag in all_tags):
                # This check prevents new tags with duplicate keys from being added
                # (to prevent API failure and/or overwriting of tags). If there is a conflict,
                # the user-provided tag should take precedence over the config-provided tag.
                # Note: this does not check user-provided tags for conflicts with other
                # user-provided tags.
                all_tags.append(config_tag)

        _log_sagemaker_config_merge(
            source_value=tags,
            config_value=config_tags,
            merged_source_and_config_value=all_tags,
            config_key_path=config_path_to_tags,
        )

        return all_tags

    def endpoint_from_production_variants(
        self,
        name,
        production_variants,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        explainer_config_dict=None,
        live_logging=False,
        vpc_config=None,
        enable_network_isolation=None,
        role=None,
    ):
        """Create an SageMaker ``Endpoint`` from a list of production variants.

        Args:
            name (str): The name of the ``Endpoint`` to create.
            production_variants (list[dict[str, str]]): The list of production variants to deploy.
            tags (Optional[Tags]): A list of key-value pairs for tagging the endpoint
                (default: None).
            kms_key (str): The KMS key that is used to encrypt the data on the storage volume
                attached to the instance hosting the endpoint.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).
            data_capture_config_dict (dict): Specifies configuration related to Endpoint data
                capture for use with Amazon SageMaker Model Monitoring. Default: None.
            async_inference_config_dict (dict) : specifies configuration related to async endpoint.
                Use this configuration when trying to create async endpoint and make async inference
                (default: None)
            explainer_config_dict (dict) : Specifies configuration related to explainer.
                Use this configuration when trying to use online explainability.
                (default: None).
            vpc_config (dict[str, list[str]]:
                The VpcConfig set on the model (default: None).
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            enable_network_isolation (Boolean): Default False.
                If True, enables network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                (default: None).
        Returns:
            str: The name of the created ``Endpoint``.
        """

        supports_kms = any(
            [
                instance_supports_kms(production_variant["InstanceType"])
                for production_variant in production_variants
                if "InstanceType" in production_variant
            ]
        )

        update_list_of_dicts_with_values_from_config(
            production_variants,
            ENDPOINT_CONFIG_PRODUCTION_VARIANTS_PATH,
            required_key_paths=["CoreDumpConfig.DestinationS3Uri"],
            sagemaker_session=self,
        )

        config_options = {"EndpointConfigName": name, "ProductionVariants": production_variants}

        kms_key = (
            resolve_value_from_config(
                kms_key, ENDPOINT_CONFIG_KMS_KEY_ID_PATH, sagemaker_session=self
            )
            if supports_kms
            else kms_key
        )

        vpc_config = resolve_value_from_config(
            vpc_config,
            ENDPOINT_CONFIG_VPC_CONFIG_PATH,
            sagemaker_session=self,
        )

        enable_network_isolation = resolve_value_from_config(
            enable_network_isolation,
            ENDPOINT_CONFIG_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self,
        )

        role = resolve_value_from_config(
            role,
            ENDPOINT_CONFIG_EXECUTION_ROLE_ARN_PATH,
            sagemaker_session=self,
            sagemaker_config=load_sagemaker_config() if (self is None) else None,
        )

        # For Amazon SageMaker inference component based endpoint, it will not pass
        # Model names during endpoint creation. Instead, ExecutionRoleArn will be
        # needed in the endpoint config to create Endpoint
        model_names = [pv["ModelName"] for pv in production_variants if "ModelName" in pv]
        if len(model_names) == 0:
            # Currently, SageMaker Python SDK allow using RoleName to deploy models.
            # Use expand_role method to handle this situation.
            role = self.expand_role(role)
            config_options["ExecutionRoleArn"] = role
        endpoint_config_tags = _append_project_tags(format_tags(tags))
        endpoint_tags = _append_project_tags(format_tags(tags))

        endpoint_config_tags = self._append_sagemaker_config_tags(
            endpoint_config_tags, "{}.{}.{}".format(SAGEMAKER, ENDPOINT_CONFIG, TAGS)
        )
        if endpoint_config_tags:
            config_options["Tags"] = endpoint_config_tags
        if kms_key:
            config_options["KmsKeyId"] = kms_key
        if data_capture_config_dict is not None:
            inferred_data_capture_config_dict = update_nested_dictionary_with_values_from_config(
                data_capture_config_dict, ENDPOINT_CONFIG_DATA_CAPTURE_PATH, sagemaker_session=self
            )
            config_options["DataCaptureConfig"] = inferred_data_capture_config_dict
        if async_inference_config_dict is not None:
            inferred_async_inference_config_dict = update_nested_dictionary_with_values_from_config(
                async_inference_config_dict,
                ENDPOINT_CONFIG_ASYNC_INFERENCE_PATH,
                sagemaker_session=self,
            )
            config_options["AsyncInferenceConfig"] = inferred_async_inference_config_dict
        if explainer_config_dict is not None:
            config_options["ExplainerConfig"] = explainer_config_dict
        if vpc_config is not None:
            config_options["VpcConfig"] = vpc_config
        if enable_network_isolation is not None:
            config_options["EnableNetworkIsolation"] = enable_network_isolation
        if role is not None:
            config_options["ExecutionRoleArn"] = role

        logger.info("Creating endpoint-config with name %s", name)
        self.sagemaker_client.create_endpoint_config(**config_options)

        return self.create_endpoint(
            endpoint_name=name,
            config_name=name,
            tags=endpoint_tags,
            wait=wait,
            live_logging=live_logging,
        )

    def create_endpoint(self, endpoint_name, config_name, tags=None, wait=True, live_logging=False):
        """Create an Amazon SageMaker ``Endpoint`` according to the configuration in the request.

        Once the ``Endpoint`` is created, client applications can send requests to obtain
        inferences. The endpoint configuration is created using the ``CreateEndpointConfig`` API.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` being created.
            config_name (str): Name of the Amazon SageMaker endpoint configuration to deploy.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).
            tags (Optional[Tags]): A list of key-value pairs for tagging the endpoint
                (default: None).

        Returns:
            str: Name of the Amazon SageMaker ``Endpoint`` created.

        Raises:
            botocore.exceptions.ClientError: If Sagemaker throws an exception while creating
            endpoint.
        """
        logger.info("Creating endpoint with name %s", endpoint_name)

        tags = format_tags(tags) or []
        tags = _append_project_tags(tags)
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, ENDPOINT, TAGS)
        )
        try:
            res = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=tags
            )
            if res:
                self.endpoint_arn = res["EndpointArn"]

            if wait:
                self.wait_for_endpoint(endpoint_name, live_logging=live_logging)
            return endpoint_name
        except Exception as e:
            troubleshooting = (
                "https://docs.aws.amazon.com/sagemaker/latest/dg/"
                "sagemaker-python-sdk-troubleshooting.html"
                "#sagemaker-python-sdk-troubleshooting-create-endpoint"
            )
            logger.error(
                "Please check the troubleshooting guide for common errors: %s", troubleshooting
            )
            raise e

    def wait_for_endpoint(self, endpoint, poll=DEFAULT_EP_POLL, live_logging=False):
        """Wait for an Amazon SageMaker endpoint deployment to complete.

        Args:
            endpoint (str): Name of the ``Endpoint`` to wait for.
            poll (int): Polling interval in seconds (default: 30).

        Raises:
            exceptions.CapacityError: If the endpoint creation job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the endpoint creation job fails.

        Returns:
            dict: Return value from the ``DescribeEndpoint`` API.
        """

        if not live_logging or not _has_permission_for_live_logging(self.boto_session, endpoint):
            desc = _wait_until(lambda: _deploy_done(self.sagemaker_client, endpoint), poll)
        else:
            cloudwatch_client = self.boto_session.client("logs")
            paginator = cloudwatch_client.get_paginator("filter_log_events")
            paginator_config = create_paginator_config()
            desc = _wait_until(
                lambda: _live_logging_deploy_done(
                    self.sagemaker_client, endpoint, paginator, paginator_config, EP_LOGGER_POLL
                ),
                poll=EP_LOGGER_POLL,
            )
        status = desc["EndpointStatus"]

        if status != "InService":
            reason = desc.get("FailureReason", None)
            trouble_shooting = (
                "Try changing the instance type or reference the troubleshooting page "
                "https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-troubleshooting"
                ".html"
            )
            message = "Error hosting endpoint {}: {}. Reason: {}. {}".format(
                endpoint, status, reason, trouble_shooting
            )
            if "CapacityError" in str(reason):
                raise exceptions.CapacityError(
                    message=message,
                    allowed_statuses=["InService"],
                    actual_status=status,
                )
            raise exceptions.UnexpectedStatusException(
                message=message,
                allowed_statuses=["InService"],
                actual_status=status,
            )
        return desc

    def create_inference_component(
        self,
        inference_component_name: str,
        endpoint_name: str,
        variant_name: str,
        specification: Dict[str, Any],
        runtime_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Tags] = None,
        wait: bool = True,
    ):
        """Create an Amazon SageMaker Inference Component.

        Args:
            inference_component_name (str): Name of the Amazon SageMaker inference component
                to create.
            endpoint_name (str): Name of the Amazon SageMaker endpoint that the inference component
                will deploy to.
            variant_name (str): Name of the Amazon SageMaker variant that the inference component
                will deploy to.
            specification (Dict[str, Any]): The inference component specification.
            runtime_config (Optional[Dict[str, Any]]): Optional. The inference component
                runtime configuration. (Default: None).
            tags (Optional[Tags]): Optional. Either a dictionary or a list
                of dictionaries containing key-value pairs. (Default: None).
            wait (bool) : Optional. Wait for the inference component to finish being created before
                returning a value. (Default: True).

        Returns:
            str: Name of the Amazon SageMaker ``InferenceComponent`` if created.
        """
        LOGGER.info(
            "Creating inference component with name %s for endpoint %s",
            inference_component_name,
            endpoint_name,
        )

        if runtime_config is None:
            runtime_config = {"CopyCount": 1}

        request = {
            "InferenceComponentName": inference_component_name,
            "EndpointName": endpoint_name,
            "VariantName": variant_name,
            "Specification": specification,
            "RuntimeConfig": runtime_config,
        }

        tags = format_tags(tags)
        tags = _append_project_tags(tags)
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, INFERENCE_COMPONENT, TAGS)
        )
        if tags and len(tags) != 0:
            request["Tags"] = tags

        self.sagemaker_client.create_inference_component(**request)
        if wait:
            self.wait_for_inference_component(inference_component_name)
        return inference_component_name

    def wait_for_inference_component(self, inference_component_name, poll=20):
        """Wait for an Amazon SageMaker ``Inference Component`` deployment to complete.

        Args:
            inference_component_name (str): Name of the ``Inference Component`` to wait for.
            poll (int): Polling interval in seconds (default: 20).

        Raises:
            exceptions.CapacityError: If the inference component creation fails with CapacityError.
            exceptions.UnexpectedStatusException: If the inference component creation fails.

        Returns:
            dict: Return value from the ``DescribeInferenceComponent`` API.
        """
        desc = _wait_until(
            lambda: self._inference_component_done(self.sagemaker_client, inference_component_name),
            poll,
        )
        status = desc["InferenceComponentStatus"]

        if status != "InService":
            message = f"Error creating inference component '{inference_component_name}'"
            reason = desc.get("FailureReason")
            if reason:
                message = f"{message}: {reason}"
            if "CapacityError" in str(reason):
                raise exceptions.CapacityError(
                    message=message,
                    allowed_statuses=["InService"],
                    actual_status=status,
                )
            raise exceptions.UnexpectedStatusException(
                message=message,
                allowed_statuses=["InService"],
                actual_status=status,
            )
        return desc

    def describe_inference_component(self, inference_component_name):
        """Describe an Amazon SageMaker ``InferenceComponent``

        Args:
            inference_component_name (str): Name of the Amazon SageMaker ``InferenceComponent``.

        Returns:
            dict[str,str]: Inference component details.
        """

        return self.sagemaker_client.describe_inference_component(
            InferenceComponentName=inference_component_name
        )

    def _inference_component_done(self, sagemaker_client, inference_component_name):
        """Check if creation of inference component is done.

        Args:
            sagemaker_client (boto3.SageMaker.Client): Client which makes Amazon SageMaker
                service calls
            inference_component_name (str): Name of the Amazon SageMaker ``InferenceComponent``.
        Returns:
            dict[str,str]: Inference component details.
        """

        create_inference_component_codes = {
            "InService": "!",
            "Creating": "-",
            "Updating": "-",
            "Failed": "*",
            "Deleting": "o",
        }
        in_progress_statuses = ["Creating", "Updating", "Deleting"]

        desc = sagemaker_client.describe_inference_component(
            InferenceComponentName=inference_component_name
        )
        status = desc["InferenceComponentStatus"]

        print(create_inference_component_codes.get(status, "?"), end="", flush=True)

        return None if status in in_progress_statuses else desc

    def update_endpoint(self, endpoint_name, endpoint_config_name, wait=True):
        """Update an Amazon SageMaker ``Endpoint`` , Raise an error endpoint_name does not exist.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to update.
            endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to
                deploy.
            wait (bool): Whether to wait for the endpoint deployment to complete before returning
                (default: True).

        Returns:
            str: Name of the Amazon SageMaker ``Endpoint`` being updated.

        Raises:
            - ValueError: if the endpoint does not already exist
            - botocore.exceptions.ClientError: If SageMaker throws an error while
            creating endpoint config, describing endpoint or updating endpoint
        """
        if not _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        ):
            raise ValueError(
                "Endpoint with name '{}' does not exist; please use an "
                "existing endpoint name".format(endpoint_name)
            )

        try:

            res = self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
            )
            if res:
                self.endpoint_arn = res["EndpointArn"]

            if wait:
                self.wait_for_endpoint(endpoint_name)
            return endpoint_name
        except Exception as e:
            troubleshooting = (
                "https://docs.aws.amazon.com/sagemaker/latest/dg/"
                "sagemaker-python-sdk-troubleshooting.html"
                "#sagemaker-python-sdk-troubleshooting-update-endpoint"
            )
            logger.error(
                "Please check the troubleshooting guide for common errors: %s", troubleshooting
            )
            raise e

    def endpoint_in_service_or_not(self, endpoint_name: str):
        """Check whether an Amazon SageMaker ``Endpoint``` is in IN_SERVICE status.

        Raise any exception that is not recognized as "not found".

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to
            check status.

        Returns:
            bool: True if ``Endpoint`` is IN_SERVICE, False if ``Endpoint`` not exists
            or it's in other status.

        Raises:

        """
        try:
            desc = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = desc["EndpointStatus"]
            if status == "InService":
                return True
            return False

        except botocore.exceptions.ClientError as e:
            str_err = str(e).lower()
            if "could not find" in str_err or "not found" in str_err:
                return False
            raise

    def _intercept_create_request(
        self,
        request: Dict,
        create,
        func_name: str = None,
        # pylint: disable=unused-argument
    ):
        """This function intercepts the create job request.

        PipelineSession inherits this Session class and will override
        this function to intercept the create request.

        Args:
            request (dict): the create job request
            create (functor): a functor calls the sagemaker client create method
            func_name (str): the name of the function needed intercepting
        """
        return create(request)

    def _create_inference_recommendations_job_request(
        self,
        role: str,
        job_name: str,
        job_description: str,
        framework: str,
        sample_payload_url: str,
        supported_content_types: List[str],
        tags: Optional[Tags],
        model_name: str = None,
        model_package_version_arn: str = None,
        job_duration_in_seconds: int = None,
        job_type: str = "Default",
        framework_version: str = None,
        nearest_model_name: str = None,
        supported_instance_types: List[str] = None,
        endpoint_configurations: List[Dict[str, Any]] = None,
        traffic_pattern: Dict[str, Any] = None,
        stopping_conditions: Dict[str, Any] = None,
        resource_limit: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Get request dictionary for CreateInferenceRecommendationsJob API.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifacts.
                You must grant sufficient permissions to this role.
            job_name (str): The name of the Inference Recommendations Job.
            job_description (str): A description of the Inference Recommendations Job.
            framework (str): The machine learning framework of the Image URI.
            sample_payload_url (str): The S3 path where the sample payload is stored.
            supported_content_types (List[str]): The supported MIME types for the input data.
            model_name (str): Name of the Amazon SageMaker ``Model`` to be used.
            model_package_version_arn (str): The Amazon Resource Name (ARN) of a
                versioned model package.
            job_duration_in_seconds (int): The maximum job duration that a job
                can run for. Will be used for `Advanced` jobs.
            job_type (str): The type of job being run. Must either be `Default` or `Advanced`.
            framework_version (str): The framework version of the Image URI.
            nearest_model_name (str): The name of a pre-trained machine learning model
                benchmarked by Amazon SageMaker Inference Recommender that matches your model.
            supported_instance_types (List[str]): A list of the instance types that are used
                to generate inferences in real-time.
            tags (Optional[Tags]): Tags used to identify where
                the Inference Recommendatons Call was made from.
            endpoint_configurations (List[Dict[str, any]]): Specifies the endpoint configurations
                to use for a job. Will be used for `Advanced` jobs.
            traffic_pattern (Dict[str, any]): Specifies the traffic pattern for the job.
                Will be used for `Advanced` jobs.
            stopping_conditions (Dict[str, any]): A set of conditions for stopping a
                recommendation job.
                If any of the conditions are met, the job is automatically stopped.
                Will be used for `Advanced` jobs.
            resource_limit (Dict[str, any]): Defines the resource limit for the job.
                Will be used for `Advanced` jobs.
        Returns:
            Dict[str, Any]: request dictionary for the CreateInferenceRecommendationsJob API
        """

        containerConfig = {
            "Domain": "MACHINE_LEARNING",
            "Task": "OTHER",
            "Framework": framework,
            "PayloadConfig": {
                "SamplePayloadUrl": sample_payload_url,
                "SupportedContentTypes": supported_content_types,
            },
        }

        if framework_version:
            containerConfig["FrameworkVersion"] = framework_version
        if nearest_model_name:
            containerConfig["NearestModelName"] = nearest_model_name
        if supported_instance_types:
            containerConfig["SupportedInstanceTypes"] = supported_instance_types

        request = {
            "JobName": job_name,
            "JobType": job_type,
            "RoleArn": role,
            "InputConfig": {
                "ContainerConfig": containerConfig,
            },
            "Tags": format_tags(tags),
        }

        request.get("InputConfig").update(
            {"ModelPackageVersionArn": model_package_version_arn}
            if model_package_version_arn
            else {"ModelName": model_name}
        )

        if job_description:
            request["JobDescription"] = job_description
        if job_duration_in_seconds:
            request["InputConfig"]["JobDurationInSeconds"] = job_duration_in_seconds

        if job_type == "Advanced":
            if stopping_conditions:
                request["StoppingConditions"] = stopping_conditions
            if resource_limit:
                request["InputConfig"]["ResourceLimit"] = resource_limit
            if traffic_pattern:
                request["InputConfig"]["TrafficPattern"] = traffic_pattern
            if endpoint_configurations:
                request["InputConfig"]["EndpointConfigurations"] = endpoint_configurations

        return request

    def create_inference_recommendations_job(
        self,
        role: str,
        sample_payload_url: str,
        supported_content_types: List[str],
        job_name: str = None,
        job_type: str = "Default",
        model_name: str = None,
        model_package_version_arn: str = None,
        job_duration_in_seconds: int = None,
        nearest_model_name: str = None,
        supported_instance_types: List[str] = None,
        framework: str = None,
        framework_version: str = None,
        endpoint_configurations: List[Dict[str, any]] = None,
        traffic_pattern: Dict[str, any] = None,
        stopping_conditions: Dict[str, any] = None,
        resource_limit: Dict[str, any] = None,
    ):
        """Creates an Inference Recommendations Job

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
                jobs and APIs that create Amazon SageMaker endpoints use this role to access
                training data and model artifactså.
                You must grant sufficient permissions to this role.
            sample_payload_url (str): The S3 path where the sample payload is stored.
            supported_content_types (List[str]): The supported MIME types for the input data.
            model_name (str): Name of the Amazon SageMaker ``Model`` to be used.
            model_package_version_arn (str): The Amazon Resource Name (ARN) of a
                versioned model package.
            job_name (str): The name of the job being run.
            job_type (str): The type of job being run. Must either be `Default` or `Advanced`.
            job_duration_in_seconds (int): The maximum job duration that a job
                can run for. Will be used for `Advanced` jobs.
            nearest_model_name (str): The name of a pre-trained machine learning model
                benchmarked by Amazon SageMaker Inference Recommender that matches your model.
            supported_instance_types (List[str]): A list of the instance types that are used
                to generate inferences in real-time.
            framework (str): The machine learning framework of the Image URI.
            framework_version (str): The framework version of the Image URI.
            endpoint_configurations (List[Dict[str, any]]): Specifies the endpoint configurations
                to use for a job. Will be used for `Advanced` jobs.
            traffic_pattern (Dict[str, any]): Specifies the traffic pattern for the job.
                Will be used for `Advanced` jobs.
            stopping_conditions (Dict[str, any]): A set of conditions for stopping a
                recommendation job.
                If any of the conditions are met, the job is automatically stopped.
                Will be used for `Advanced` jobs.
            resource_limit (Dict[str, any]): Defines the resource limit for the job.
                Will be used for `Advanced` jobs.
        Returns:
            str: The name of the job created. In the form of `SMPYTHONSDK-<timestamp>`
        """

        if model_name is None and model_package_version_arn is None:
            raise ValueError("Please provide either model_name or model_package_version_arn.")

        if model_name is not None and model_package_version_arn is not None:
            raise ValueError("Please provide either model_name or model_package_version_arn.")

        if not job_name:
            unique_tail = uuid.uuid4()
            job_name = "SMPYTHONSDK-" + str(unique_tail)
        job_description = "#python-sdk-create"

        tags = [{"Key": "ClientType", "Value": "PythonSDK-RightSize"}]

        create_inference_recommendations_job_request = (
            self._create_inference_recommendations_job_request(
                role=role,
                model_name=model_name,
                model_package_version_arn=model_package_version_arn,
                job_name=job_name,
                job_type=job_type,
                job_duration_in_seconds=job_duration_in_seconds,
                job_description=job_description,
                framework=framework,
                framework_version=framework_version,
                nearest_model_name=nearest_model_name,
                sample_payload_url=sample_payload_url,
                supported_content_types=supported_content_types,
                supported_instance_types=supported_instance_types,
                endpoint_configurations=endpoint_configurations,
                traffic_pattern=traffic_pattern,
                stopping_conditions=stopping_conditions,
                resource_limit=resource_limit,
                tags=tags,
            )
        )

        def submit(request):
            logger.info("Creating Inference Recommendations job with name: %s", job_name)
            logger.debug("process request: %s", json.dumps(request, indent=4))
            self.sagemaker_client.create_inference_recommendations_job(**request)

        self._intercept_create_request(
            create_inference_recommendations_job_request,
            submit,
            self.create_inference_recommendations_job.__name__,
        )
        return job_name

    def wait_for_inference_recommendations_job(
        self, job_name: str, poll: int = 120, log_level: str = "Verbose"
    ) -> Dict[str, Any]:
        """Wait for an Amazon SageMaker Inference Recommender job to complete.

        Args:
            job_name (str): Name of the Inference Recommender job to wait for.
            poll (int): Polling interval in seconds (default: 120).
            log_level (str): The level of verbosity for the logs.
            Can be "Quiet" or "Verbose" (default: "Quiet").

        Returns:
            (dict): Return value from the ``DescribeInferenceRecommendationsJob`` API.

        Raises:
            exceptions.CapacityError: If the Inference Recommender job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the Inference Recommender job fails.
        """
        if log_level == "Quiet":
            _wait_until(
                lambda: _describe_inference_recommendations_job_status(
                    self.sagemaker_client, job_name
                ),
                poll,
            )
        elif log_level == "Verbose":
            _display_inference_recommendations_job_steps_status(
                self, self.sagemaker_client, job_name
            )
        else:
            raise ValueError("log_level must be either Quiet or Verbose")
        desc = _describe_inference_recommendations_job_status(self.sagemaker_client, job_name)
        _check_job_status(job_name, desc, "Status")
        return desc

    def delete_model(self, model_name):
        """Delete an Amazon SageMaker Model.

        Args:
            model_name (str): Name of the Amazon SageMaker model to delete.
        """
        logger.info("Deleting model with name: %s", model_name)
        self.sagemaker_client.delete_model(ModelName=model_name)

    def delete_endpoint(self, endpoint_name):
        """Delete an Amazon SageMaker ``Endpoint``.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to delete.
        """
        logger.info("Deleting endpoint with name: %s", endpoint_name)
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    def delete_endpoint_config(self, endpoint_config_name):
        """Delete an Amazon SageMaker endpoint configuration.

        Args:
            endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to
                delete.
        """
        logger.info("Deleting endpoint configuration with name: %s", endpoint_config_name)
        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

    def wait_for_optimization_job(self, job, poll=5):
        """Wait for an Amazon SageMaker Optimization job to complete.

        Args:
            job (str): Name of optimization job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeOptimizationJob`` API.

        Raises:
            exceptions.ResourceNotFound: If optimization job fails with CapacityError.
            exceptions.UnexpectedStatusException: If optimization job fails.
        """
        desc = _wait_until(lambda: _optimization_job_status(self.sagemaker_client, job), poll)
        _check_job_status(job, desc, "OptimizationJobStatus")
        return desc

    def update_inference_component(
        self, inference_component_name, specification=None, runtime_config=None, wait=True
    ):
        """Update an Amazon SageMaker ``InferenceComponent``

        Args:
            inference_component_name (str): Name of the Amazon SageMaker ``InferenceComponent``.
            specification ([dict[str,int]]): Resource configuration. Optional.
                Example: {
                "MinMemoryRequiredInMb": 1024,
                "NumberOfCpuCoresRequired": 1,
                "NumberOfAcceleratorDevicesRequired": 1,
                "MaxMemoryRequiredInMb": 4096,
                },
            runtime_config ([dict[str,int]]): Number of copies. Optional.
                Default: {
                "copyCount": 1
                }
            wait: Wait for inference component to be created before return. Optional. Default is
                True.

        Return:
            str: inference component name

        Raises:
            ValueError: If the inference_component_name does not exist.
        """
        if not _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_inference_component(
                InferenceComponentName=inference_component_name
            )
        ):
            raise ValueError(
                "InferenceComponent with name '{}' does not exist; please use an "
                "existing model name".format(inference_component_name)
            )

        request = {
            "InferenceComponentName": inference_component_name,
            "Specification": specification,
            "RuntimeConfig": runtime_config,
        }

        self.sagemaker_client.update_inference_component(**request)

        if wait:
            self.wait_for_inference_component(inference_component_name)
        return inference_component_name

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
            update_list_of_dicts_with_values_from_config(
                container_defs, MODEL_CONTAINERS_PATH, sagemaker_session=self
            )
            container_definition = container_defs
        else:
            container_definition = _expand_container_def(container_defs)
            container_definition = update_nested_dictionary_with_values_from_config(
                container_definition, MODEL_PRIMARY_CONTAINER_PATH, sagemaker_session=self
            )

        request = {"ModelName": name, "ExecutionRoleArn": role}
        if isinstance(container_definition, list):
            request["Containers"] = container_definition
        elif "ModelPackageName" in container_definition:
            request["Containers"] = [container_definition]
        else:
            request["PrimaryContainer"] = container_definition

        if tags:
            request["Tags"] = format_tags(tags)

        if vpc_config:
            request["VpcConfig"] = vpc_config

        if enable_network_isolation:
            # enable_network_isolation may be a pipeline variable which is
            # parsed in execution time
            request["EnableNetworkIsolation"] = enable_network_isolation

        return request

    def create_model(
        self,
        name,
        role=None,
        container_defs=None,
        vpc_config=None,
        enable_network_isolation=None,
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
            enable_network_isolation (bool): Whether the model requires network isolation or not.
            primary_container (str or dict[str, str]): Docker image which defines the inference
                code. You can also specify the return value of ``sagemaker.container_def()``,
                which is used to create more advanced container configurations, including model
                containers which need artifacts from S3. This field is deprecated, please use
                container_defs instead.
            tags(Optional[Tags]): Optional. The list of tags to add to the model.

        Example:
            >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
            For more information about tags, see https://boto3.amazonaws.com/v1/documentation\
            /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags

        Returns:
            str: Name of the Amazon SageMaker ``Model`` created.
        """
        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(tags, "{}.{}.{}".format(SAGEMAKER, MODEL, TAGS))
        role = resolve_value_from_config(
            role, MODEL_EXECUTION_ROLE_ARN_PATH, sagemaker_session=self
        )
        vpc_config = resolve_value_from_config(
            vpc_config, MODEL_VPC_CONFIG_PATH, sagemaker_session=self
        )
        enable_network_isolation = resolve_value_from_config(
            direct_input=enable_network_isolation,
            config_path=MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            default_value=False,
            sagemaker_session=self,
        )

        # Due to ambuiguity in container_defs which accepts both a single
        # container definition(dtype: dict) and a list of container definitions (dtype: list),
        # we need to inject environment variables into the container_defs in the helper function
        # _create_model_request.
        create_model_request = self._create_model_request(
            name=name,
            role=role,
            container_defs=container_defs,
            vpc_config=vpc_config,
            enable_network_isolation=enable_network_isolation,
            primary_container=primary_container,
            tags=tags,
        )

        def submit(request):
            logger.info("Creating model with name: %s", name)
            logger.debug("CreateModel request: %s", json.dumps(request, indent=4))
            try:
                self.sagemaker_client.create_model(**request)
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                message = e.response["Error"]["Message"]
                if (
                    error_code == "ValidationException"
                    and "Cannot create already existing model" in message
                ):
                    logger.warning("Using already existing model: %s", name)
                else:
                    raise

        self._intercept_create_request(create_model_request, submit, self.create_model.__name__)
        return name

    def create_model_package_from_algorithm(self, name, description, algorithm_arn, model_data):
        """Create a SageMaker Model Package from the results of training with an Algorithm Package.

        Args:
            name (str): ModelPackage name
            description (str): Model Package description
            algorithm_arn (str): arn or name of the algorithm used for training.
            model_data (str or dict[str, Any]): s3 URI or a dictionary representing a
            ``ModelDataSource`` to the model artifacts produced by training
        """
        sourceAlgorithm = {"AlgorithmName": algorithm_arn}
        if isinstance(model_data, dict):
            sourceAlgorithm["ModelDataSource"] = model_data
        else:
            sourceAlgorithm["ModelDataUrl"] = model_data

        request = {
            "ModelPackageName": name,
            "ModelPackageDescription": description,
            "SourceAlgorithmSpecification": {"SourceAlgorithms": [sourceAlgorithm]},
        }
        try:
            logger.info("Creating model package with name: %s", name)
            self.sagemaker_client.create_model_package(**request)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]

            if error_code == "ValidationException" and "ModelPackage already exists" in message:
                logger.warning("Using already existing model package: %s", name)
            else:
                raise

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
    

def _expand_container_def(c_def):
    """Placeholder docstring"""
    if isinstance(c_def, six.string_types):
        return container_def(c_def)
    return c_def


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


def s3_path_join(*args, with_end_slash: bool = False):
    """Returns the arguments joined by a slash ("/"), similar to ``os.path.join()`` (on Unix).

    Behavior of this function:
    - If the first argument is "s3://", then that is preserved.
    - The output by default will have no slashes at the beginning or end. There is one exception
        (see `with_end_slash`). For example, `s3_path_join("/foo", "bar/")` will yield
        `"foo/bar"` and `s3_path_join("foo", "bar", with_end_slash=True)` will yield `"foo/bar/"`
    - Any repeat slashes will be removed in the output (except for "s3://" if provided at the
        beginning). For example, `s3_path_join("s3://", "//foo/", "/bar///baz")` will yield
        `"s3://foo/bar/baz"`.
    - Empty or None arguments will be skipped. For example
        `s3_path_join("foo", "", None, "bar")` will yield `"foo/bar"`

    Alternatives to this function that are NOT recommended for S3 paths:
    - `os.path.join(...)` will have different behavior on Unix machines vs non-Unix machines
    - `pathlib.PurePosixPath(...)` will apply potentially unintended simplification of single
        dots (".") and root directories. (for example
        `pathlib.PurePosixPath("foo", "/bar/./", "baz")` would yield `"/bar/baz"`)
    - `"{}/{}/{}".format(...)` and similar may result in unintended repeat slashes

    Args:
        *args: The strings to join with a slash.
        with_end_slash (bool): (default: False) If true and if the path is not empty, appends a "/"
            to the end of the path

    Returns:
        str: The joined string, without a slash at the end unless with_end_slash is True.
    """
    delimiter = "/"

    non_empty_args = list(filter(lambda item: item is not None and item != "", args))

    merged_path = ""
    for index, path in enumerate(non_empty_args):
        if (
            index == 0
            or (merged_path and merged_path[-1] == delimiter)
            or (path and path[0] == delimiter)
        ):
            # dont need to add an extra slash because either this is the beginning of the string,
            # or one (or more) slash already exists
            merged_path += path
        else:
            merged_path += delimiter + path

    if with_end_slash and merged_path and merged_path[-1] != delimiter:
        merged_path += delimiter

    # At this point, merged_path may include slashes at the beginning and/or end. And some of the
    # provided args may have had duplicate slashes inside or at the ends.
    # For backwards compatibility reasons, these need to be filtered out (done below). In the
    # future, if there is a desire to support multiple slashes for S3 paths throughout the SDK,
    # one option is to create a new optional argument (or a new function) that only executes the
    # logic above.
    filtered_path = merged_path

    # remove duplicate slashes
    if filtered_path:

        def duplicate_delimiter_remover(sequence, next_char):
            if sequence[-1] == delimiter and next_char == delimiter:
                return sequence
            return sequence + next_char

        if filtered_path.startswith("s3://"):
            filtered_path = reduce(
                duplicate_delimiter_remover, filtered_path[5:], filtered_path[:5]
            )
        else:
            filtered_path = reduce(duplicate_delimiter_remover, filtered_path)

    # remove beginning slashes
    filtered_path = filtered_path.lstrip(delimiter)

    # remove end slashes
    if not with_end_slash and filtered_path != "s3://":
        filtered_path = filtered_path.rstrip(delimiter)

    return filtered_path


def botocore_resolver():
    """Get the DNS suffix for the given region.

    Args:
        region (str): AWS region name

    Returns:
        str: the DNS suffix
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data("endpoints"))


def sts_regional_endpoint(region):
    """Get the AWS STS endpoint specific for the given region.

    We need this function because the AWS SDK does not yet honor
    the ``region_name`` parameter when creating an AWS STS client.

    For the list of regional endpoints, see
    https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_enable-regions.html#id_credentials_region-endpoints.

    Args:
        region (str): AWS region name

    Returns:
        str: AWS STS regional endpoint
    """
    endpoint_data = botocore_resolver().construct_endpoint("sts", region)
    if region == "il-central-1" and not endpoint_data:
        endpoint_data = {"hostname": "sts.{}.amazonaws.com".format(region)}
    return "https://{}".format(endpoint_data["hostname"])


def get_execution_role(sagemaker_session=None, use_default=False):
    """Return the role ARN whose credentials are used to call the API.

    Throws an exception if role doesn't exist.

    Args:
        sagemaker_session (Session): Current sagemaker session.
        use_default (bool): Use a default role if ``get_caller_identity_arn`` does not
            return a correct role. This default role will be created if needed.
            Defaults to ``False``.

    Returns:
        (str): The role ARN
    """
    if not sagemaker_session:
        sagemaker_session = Session()
    arn = sagemaker_session.get_caller_identity_arn()

    if ":role/" in arn:
        return arn

    if use_default:
        default_role_name = "AmazonSageMaker-DefaultRole"

        LOGGER.warning("Using default role: %s", default_role_name)

        boto3_session = sagemaker_session.boto_session
        permissions_policy = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": ["sagemaker.amazonaws.com"]},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        )
        iam_client = boto3_session.client("iam")
        try:
            iam_client.get_role(RoleName=default_role_name)
        except iam_client.exceptions.NoSuchEntityException:
            iam_client.create_role(
                RoleName=default_role_name, AssumeRolePolicyDocument=str(permissions_policy)
            )

            LOGGER.warning("Created new sagemaker execution role: %s", default_role_name)

        iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            RoleName=default_role_name,
        )
        return iam_client.get_role(RoleName=default_role_name)["Role"]["Arn"]

    message = (
        "The current AWS identity is not a role: {}, therefore it cannot be used as a "
        "SageMaker execution role"
    )
    raise ValueError(message.format(arn))


def get_add_model_package_inference_args(
    model_package_arn,
    name,
    containers=None,
    content_types=None,
    response_types=None,
    inference_instances=None,
    transform_instances=None,
    description=None,
):
    """Get request dictionary for UpdateModelPackage API for additional inference.

    Args:
        model_package_arn (str): Arn for the model package.
        name (str): Name to identify the additional inference specification
        containers (dict): The Amazon ECR registry path of the Docker image
            that contains the inference code.
        image_uris (List[str]): The ECR path where inference code is stored.
        description (str): Description for the additional inference specification
        content_types (list[str]): The supported MIME types
            for the input data.
        response_types (list[str]): The supported MIME types
            for the output data.
        inference_instances (list[str]): A list of the instance
            types that are used to generate inferences in real-time (default: None).
        transform_instances (list[str]): A list of the instance
            types on which a transformation job can be run or on which an endpoint can be
            deployed (default: None).
    """

    request_dict = {}
    if containers is not None:
        inference_specification = {
            "Containers": containers,
        }

        if name is not None:
            inference_specification.update({"Name": name})

        if description is not None:
            inference_specification.update({"Description": description})
        if content_types is not None:
            inference_specification.update(
                {
                    "SupportedContentTypes": content_types,
                }
            )
        if response_types is not None:
            inference_specification.update(
                {
                    "SupportedResponseMIMETypes": response_types,
                }
            )
        if inference_instances is not None:
            inference_specification.update(
                {
                    "SupportedRealtimeInferenceInstanceTypes": inference_instances,
                }
            )
        if transform_instances is not None:
            inference_specification.update(
                {
                    "SupportedTransformInstanceTypes": transform_instances,
                }
            )
        request_dict["AdditionalInferenceSpecificationsToAdd"] = [inference_specification]
        request_dict.update({"ModelPackageArn": model_package_arn})
    return request_dict


def get_update_model_package_inference_args(
    model_package_arn,
    containers=None,
    content_types=None,
    response_types=None,
    inference_instances=None,
    transform_instances=None,
):
    """Get request dictionary for UpdateModelPackage API for inference specification.

    Args:
        model_package_arn (str): Arn for the model package.
        containers (dict): The Amazon ECR registry path of the Docker image
            that contains the inference code.
        content_types (list[str]): The supported MIME types
            for the input data.
        response_types (list[str]): The supported MIME types
            for the output data.
        inference_instances (list[str]): A list of the instance
            types that are used to generate inferences in real-time (default: None).
        transform_instances (list[str]): A list of the instance
            types on which a transformation job can be run or on which an endpoint can be
            deployed (default: None).
    """

    request_dict = {}
    if containers is not None:
        inference_specification = {
            "Containers": containers,
        }
        if content_types is not None:
            inference_specification.update(
                {
                    "SupportedContentTypes": content_types,
                }
            )
        if response_types is not None:
            inference_specification.update(
                {
                    "SupportedResponseMIMETypes": response_types,
                }
            )
        if inference_instances is not None:
            inference_specification.update(
                {
                    "SupportedRealtimeInferenceInstanceTypes": inference_instances,
                }
            )
        if transform_instances is not None:
            inference_specification.update(
                {
                    "SupportedTransformInstanceTypes": transform_instances,
                }
            )
        request_dict["InferenceSpecification"] = inference_specification
        request_dict.update({"ModelPackageArn": model_package_arn})
    return request_dict


def _logs_for_job(  # noqa: C901 - suppress complexity warning for this method
    sagemaker_session, job_name, wait=False, poll=10, log_type="All", timeout=None
):
    """Display logs for a given training job, optionally tailing them until job is complete.

    If the output is a tty or a Jupyter cell, it will be color-coded
    based on which instance the log entry is from.

    Args:
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions.
        job_name (str): Name of the training job to display the logs for.
        wait (bool): Whether to keep looking for new log entries until the job completes
            (default: False).
        poll (int): The interval in seconds between polling for new log entries and job
            completion (default: 5).
        log_type ([str]): A list of strings specifying which logs to print. Acceptable
            strings are "All", "None", "Training", or "Rules". To maintain backwards
            compatibility, boolean values are also accepted and converted to strings.
        timeout (int): Timeout in seconds to wait until the job is completed. ``None`` by
            default.
    Returns:
        Last call to sagemaker DescribeTrainingJob
    Raises:
        exceptions.CapacityError: If the training job fails with CapacityError.
        exceptions.UnexpectedStatusException: If waiting and the training job fails.
    """
    sagemaker_client = sagemaker_session.sagemaker_client
    request_end_time = time.time() + timeout if timeout else None
    description = _wait_until(
        lambda: sagemaker_client.describe_training_job(TrainingJobName=job_name)
    )
    print(secondary_training_status_message(description, None), end="")

    instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
        sagemaker_session.boto_session, description, job="Training"
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
    last_profiler_rule_statuses = None

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
        if timeout and time.time() > request_end_time:
            print("Timeout Exceeded. {} seconds elapsed.".format(timeout))
            break

        if state == LogState.COMPLETE:
            break

        time.sleep(poll)

        if state == LogState.JOB_COMPLETE:
            state = LogState.COMPLETE
        elif time.time() - last_describe_job_call >= 30:
            description = sagemaker_client.describe_training_job(TrainingJobName=job_name)
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
                and _rule_statuses_changed(debug_rule_statuses, last_debug_rule_statuses)
                and (log_type in {"All", "Rules"})
            ):
                for status in debug_rule_statuses:
                    rule_log = (
                        f"{status['RuleConfigurationName']}: {status['RuleEvaluationStatus']}"
                    )
                    print(rule_log)

                last_debug_rule_statuses = debug_rule_statuses

            # Print prettified logs related to the status of SageMaker Profiler rules.
            profiler_rule_statuses = description.get("ProfilerRuleEvaluationStatuses", {})
            if (
                profiler_rule_statuses
                and _rule_statuses_changed(profiler_rule_statuses, last_profiler_rule_statuses)
                and (log_type in {"All", "Rules"})
            ):
                for status in profiler_rule_statuses:
                    rule_log = (
                        f"{status['RuleConfigurationName']}: {status['RuleEvaluationStatus']}"
                    )
                    print(rule_log)

                last_profiler_rule_statuses = profiler_rule_statuses

    if wait:
        _check_job_status(job_name, description, "TrainingJobStatus")
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
    return last_description


def _check_job_status(job, desc, status_key_name):
    """Check to see if the job completed successfully.

    If not, construct and raise a exceptions. (UnexpectedStatusException).

    Args:
        job (str): The name of the job to check.
        desc (dict[str, str]): The result of ``describe_training_job()``.
        status_key_name (str): Status key name to check for.

    Raises:
        exceptions.CapacityError: If the training job fails with CapacityError.
        exceptions.UnexpectedStatusException: If the training job fails.
    """
    status = desc[status_key_name]
    # If the status is capital case, then convert it to Camel case
    status = _STATUS_CODE_TABLE.get(status, status)

    if status == "Stopped":
        logger.warning(
            "Job ended with status 'Stopped' rather than 'Completed'. "
            "This could mean the job timed out or stopped early for some other reason: "
            "Consider checking whether it completed as you expect."
        )
    elif status != "Completed":
        reason = desc.get("FailureReason", "(No reason provided)")
        job_type = status_key_name.replace("JobStatus", " job")
        troubleshooting = (
            "https://docs.aws.amazon.com/sagemaker/latest/dg/"
            "sagemaker-python-sdk-troubleshooting.html"
        )
        message = (
            "Error for {job_type} {job_name}: {status}. Reason: {reason}. "
            "Check troubleshooting guide for common errors: {troubleshooting}"
        ).format(
            job_type=job_type,
            job_name=job,
            status=status,
            reason=reason,
            troubleshooting=troubleshooting,
        )
        if "CapacityError" in str(reason):
            raise exceptions.CapacityError(
                message=message,
                allowed_statuses=["Completed", "Stopped"],
                actual_status=status,
            )
        raise exceptions.UnexpectedStatusException(
            message=message,
            allowed_statuses=["Completed", "Stopped"],
            actual_status=status,
        )


def _logs_init(boto_session, description, job):
    """Placeholder docstring"""
    if job == "Training":
        if "InstanceGroups" in description["ResourceConfig"]:
            instance_count = 0
            for instanceGroup in description["ResourceConfig"]["InstanceGroups"]:
                instance_count += instanceGroup["InstanceCount"]
        else:
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
    client = boto_session.client("logs", config=config)
    log_group = "/aws/sagemaker/" + job + "Jobs"

    dot = False

    color_wrap = sagemaker.core.logs.ColorWrap()

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
                    (s, sagemaker.core.logs.Position(timestamp=0, skip=0))
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
        for idx, event in sagemaker.core.logs.multi_stream_iter(
            client, log_group, stream_names, positions
        ):
            color_wrap(idx, event["message"])
            ts, count = positions[stream_names[idx]]
            if event["timestamp"] == ts:
                positions[stream_names[idx]] = sagemaker.core.logs.Position(
                    timestamp=ts, skip=count + 1
                )
            else:
                positions[stream_names[idx]] = sagemaker.core.logs.Position(
                    timestamp=event["timestamp"], skip=1
                )
    else:
        dot = True
        print(".", end="")
        sys.stdout.flush()


def _wait_until(callable_fn, poll=5):
    """Placeholder docstring"""
    elapsed_time = 0
    result = None
    while result is None:
        try:
            elapsed_time += poll
            time.sleep(poll)
            result = callable_fn()
        except botocore.exceptions.ClientError as err:
            # For initial 5 mins we accept/pass AccessDeniedException.
            # The reason is to await tag propagation to avoid false AccessDenied claims for an
            # access policy based on resource tags, The caveat here is for true AccessDenied
            # cases the routine will fail after 5 mins
            if err.response["Error"]["Code"] == "AccessDeniedException" and elapsed_time <= 300:
                logger.warning(
                    "Received AccessDeniedException. This could mean the IAM role does not "
                    "have the resource permissions, in which case please add resource access "
                    "and retry. For cases where the role has tag based resource policy, "
                    "continuing to wait for tag propagation.."
                )
                continue
            raise err
    return result


def _get_initial_job_state(description, status_key, wait):
    """Placeholder docstring"""
    status = description[status_key]
    job_already_completed = status in ("Completed", "Failed", "Stopped")
    return LogState.TAILING if wait and not job_already_completed else LogState.COMPLETE


def _rule_statuses_changed(current_statuses, last_statuses):
    """Checks the rule evaluation statuses for SageMaker Debugger and Profiler rules."""
    if not last_statuses:
        return True

    for current, last in zip(current_statuses, last_statuses):
        if (current["RuleConfigurationName"] == last["RuleConfigurationName"]) and (
            current["RuleEvaluationStatus"] != last["RuleEvaluationStatus"]
        ):
            return True

    return False


def update_args(args: Dict[str, Any], **kwargs):
    """Updates the request arguments dict with the value if populated.

    This is to handle the case that the service API doesn't like NoneTypes for argument values.

    Args:
        request_args (Dict[str, Any]): the request arguments dict
        kwargs: key, value pairs to update the args dict
    """
    for key, value in kwargs.items():
        if value is not None:
            args.update({key: value})


def production_variant(
    model_name=None,
    instance_type=None,
    initial_instance_count=None,
    variant_name="AllTraffic",
    initial_weight=1,
    accelerator_type=None,
    serverless_inference_config=None,
    volume_size=None,
    model_data_download_timeout=None,
    container_startup_health_check_timeout=None,
    managed_instance_scaling=None,
    routing_config=None,
    inference_ami_version=None,
):
    """Create a production variant description suitable for use in a ``ProductionVariant`` list.

    This is also part of a ``CreateEndpointConfig`` request.

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
        serverless_inference_config (dict): Specifies configuration dict related to serverless
            endpoint. The dict is converted from sagemaker.model_monitor.ServerlessInferenceConfig
            object (default: None)
        volume_size (int): The size, in GB, of the ML storage volume attached to individual
            inference instance associated with the production variant. Currenly only Amazon EBS
            gp2 storage volumes are supported.
        model_data_download_timeout (int): The timeout value, in seconds, to download and extract
            model data from Amazon S3 to the individual inference instance associated with this
            production variant.
        container_startup_health_check_timeout (int): The timeout value, in seconds, for your
            inference container to pass health check by SageMaker Hosting. For more information
            about health check see:
            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
            #your-algorithms
            -inference-algo-ping-requests

    Returns:
        dict[str, str]: An SageMaker ``ProductionVariant`` description
    """
    production_variant_configuration = {
        "VariantName": variant_name,
    }
    if model_name:
        production_variant_configuration["ModelName"] = model_name
        production_variant_configuration["InitialVariantWeight"] = initial_weight

    if accelerator_type:
        production_variant_configuration["AcceleratorType"] = accelerator_type

    if managed_instance_scaling:
        production_variant_configuration["ManagedInstanceScaling"] = managed_instance_scaling

    if serverless_inference_config:
        production_variant_configuration["ServerlessConfig"] = serverless_inference_config
    else:
        initial_instance_count = initial_instance_count or 1
        production_variant_configuration["InitialInstanceCount"] = initial_instance_count
        production_variant_configuration["InstanceType"] = instance_type
        update_args(
            production_variant_configuration,
            VolumeSizeInGB=volume_size,
            ModelDataDownloadTimeoutInSeconds=model_data_download_timeout,
            ContainerStartupHealthCheckTimeoutInSeconds=container_startup_health_check_timeout,
            RoutingConfig=routing_config,
        )

    if inference_ami_version:
        production_variant_configuration["InferenceAmiVersion"] = inference_ami_version

    return production_variant_configuration


def _has_permission_for_live_logging(boto_session, endpoint_name) -> bool:
    """Validate if customer's role has the right permission to access logs from CloudWatch"""
    try:
        cloudwatch_client = boto_session.client("logs")
        cloudwatch_client.filter_log_events(
            logGroupName=f"/aws/sagemaker/Endpoints/{endpoint_name}",
            logStreamNamePrefix="AllTraffic/",
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            LOGGER.warning(
                ("Failed to enable live logging: %s. Fallback to default logging..."),
                e,
            )

            return False
        return True


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


def _live_logging_deploy_done(sagemaker_client, endpoint_name, paginator, paginator_config, poll):
    """Placeholder docstring"""
    stop = False
    endpoint_status = None
    try:
        desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = desc["EndpointStatus"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            LOGGER.debug("Waiting for endpoint to become visible")
            return None
        raise e

    try:
        # if endpoint is in an invalid state -> set stop to true, sleep, and flush the logs
        if endpoint_status != "Creating":
            stop = True
            if endpoint_status == "InService":
                LOGGER.info("Created endpoint with name %s. Waiting for it to be InService", endpoint_name)
            else:
                time.sleep(poll)

        pages = paginator.paginate(
            logGroupName=f"/aws/sagemaker/Endpoints/{endpoint_name}",
            logStreamNamePrefix="AllTraffic/",
            PaginationConfig=paginator_config,
        )

        for page in pages:
            if "nextToken" in page:
                paginator_config["StartingToken"] = page["nextToken"]
                for event in page["events"]:
                    LOGGER.info(event["message"])
            else:
                LOGGER.debug("No log events available")

        # if stop is true -> return the describe response and stop polling
        if stop:
            return desc
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            LOGGER.debug("Waiting for endpoint log group to appear")
            return None
        raise e

    return None


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


def get_log_events_for_inference_recommender(cw_client, log_group_name, log_stream_name):
    """Retrieves log events from the specified CloudWatch log group and log stream.

    Args:
        cw_client (boto3.client): A boto3 CloudWatch client.
        log_group_name (str): The name of the CloudWatch log group.
        log_stream_name (str): The name of the CloudWatch log stream.

    Returns:
        (dict): A dictionary containing log events from CloudWatch log group and log stream.
    """
    print("Fetching logs from CloudWatch...", flush=True)
    for _ in retries(
        max_retry_count=30,  # 30*10 = 5min
        exception_message_prefix="Waiting for cloudwatch stream to appear. ",
        seconds_to_sleep=10,
    ):
        try:
            return cw_client.get_log_events(
                logGroupName=log_group_name, logStreamName=log_stream_name
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                pass


def _describe_inference_recommendations_job_status(sagemaker_client, job_name: str):
    """Describes the status of a job and returns the job description.

    Args:
        sagemaker_client (boto3.client.sagemaker): A SageMaker client.
        job_name (str): The name of the job.

    Returns:
        dict: The job description, or None if the job is still in progress.
    """
    inference_recommendations_job_status_codes = {
        "PENDING": ".",
        "IN_PROGRESS": ".",
        "COMPLETED": "!",
        "FAILED": "*",
        "STOPPING": "_",
        "STOPPED": "s",
    }
    in_progress_statuses = {"PENDING", "IN_PROGRESS", "STOPPING"}

    desc = sagemaker_client.describe_inference_recommendations_job(JobName=job_name)
    status = desc["Status"]

    print(inference_recommendations_job_status_codes.get(status, "?"), end="", flush=True)

    if status in in_progress_statuses:
        return None

    print("")
    return desc


def _display_inference_recommendations_job_steps_status(
    sagemaker_session, sagemaker_client, job_name: str, poll: int = 60
):
    """Placeholder docstring"""
    cloudwatch_client = sagemaker_session.boto_session.client("logs")
    in_progress_statuses = {"PENDING", "IN_PROGRESS", "STOPPING"}
    log_group_name = "/aws/sagemaker/InferenceRecommendationsJobs"
    log_stream_name = job_name + "/execution"

    initial_logs_batch = get_log_events_for_inference_recommender(
        cloudwatch_client, log_group_name, log_stream_name
    )
    print(f"Retrieved logStream: {log_stream_name} from logGroup: {log_group_name}", flush=True)
    events = initial_logs_batch["events"]
    print(*[event["message"] for event in events], sep="\n", flush=True)

    next_forward_token = initial_logs_batch["nextForwardToken"] if events else None
    flush_remaining = True
    while True:
        logs_batch = (
            cloudwatch_client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                nextToken=next_forward_token,
            )
            if next_forward_token
            else cloudwatch_client.get_log_events(
                logGroupName=log_group_name, logStreamName=log_stream_name
            )
        )

        events = logs_batch["events"]

        desc = sagemaker_client.describe_inference_recommendations_job(JobName=job_name)
        status = desc["Status"]

        if not events:
            if status in in_progress_statuses:
                time.sleep(poll)
                continue
            if flush_remaining:
                flush_remaining = False
                time.sleep(poll)
                continue

        next_forward_token = logs_batch["nextForwardToken"]
        print(*[event["message"] for event in events], sep="\n", flush=True)

        if status not in in_progress_statuses:
            break

        time.sleep(poll)


def _optimization_job_status(sagemaker_client, job_name):
    """Placeholder docstring"""
    optimization_job_status_codes = {
        "INPROGRESS": ".",
        "COMPLETED": "!",
        "FAILED": "*",
        "STARTING": ".",
        "STOPPING": "_",
        "STOPPED": "s",
    }
    in_progress_statuses = ["INPROGRESS", "STARTING", "STOPPING"]

    desc = sagemaker_client.describe_optimization_job(OptimizationJobName=job_name)
    status = desc["OptimizationJobStatus"]

    print(optimization_job_status_codes.get(status, "?"), end="")
    sys.stdout.flush()

    if status in in_progress_statuses:
        return None

    print("")
    return desc


def container_def(
    image_uri,
    model_data_url=None,
    env=None,
    container_mode=None,
    image_config=None,
    accept_eula=None,
    additional_model_data_sources=None,
    model_reference_arn=None,
):
    """Create a definition for executing a container as part of a SageMaker model.

    Args:
        image_uri (str): Docker image URI to run for this container.
        model_data_url (str or dict[str, Any]): S3 location of model data required by this
            container, e.g. SageMaker training job model artifacts. It can either be a string
            representing S3 URI of model data, or a dictionary representing a
            ``ModelDataSource`` object. (default: None).
        env (dict[str, str]): Environment variables to set inside the container (default: None).
        container_mode (str): The model container mode. Valid modes:
                * MultiModel: Indicates that model container can support hosting multiple models
                * SingleModel: Indicates that model container can support hosting a single model
                This is the default model container mode when container_mode = None
        image_config (dict[str, str]): Specifies whether the image of model container is pulled
            from ECR, or private registry in your VPC. By default it is set to pull model
            container image from ECR. (default: None).
        accept_eula (bool): For models that require a Model Access Config, specify True or
            False to indicate whether model terms of use have been accepted.
            The `accept_eula` value must be explicitly defined as `True` in order to
            accept the end-user license agreement (EULA) that some
            models require. (Default: None).
        additional_model_data_sources (PipelineVariable or dict): Additional location
                of SageMaker model data (default: None).

    Returns:
        dict[str, str]: A complete container definition object usable with the CreateModel API if
        passed via `PrimaryContainers` field.
    """
    if env is None:
        env = {}
    c_def = {"Image": image_uri, "Environment": env}

    if additional_model_data_sources:
        c_def["AdditionalModelDataSources"] = additional_model_data_sources

    if isinstance(model_data_url, str) and (
        not (model_data_url.startswith("s3://") and model_data_url.endswith("tar.gz"))
        or accept_eula is None
    ):
        c_def["ModelDataUrl"] = model_data_url

    elif isinstance(model_data_url, (dict, str)):
        if isinstance(model_data_url, dict):
            c_def["ModelDataSource"] = model_data_url
        else:
            c_def["ModelDataSource"] = {
                "S3DataSource": {
                    "S3Uri": model_data_url,
                    "S3DataType": "S3Object",
                    "CompressionType": "Gzip",
                }
            }
        if accept_eula is not None:
            c_def["ModelDataSource"]["S3DataSource"]["ModelAccessConfig"] = {
                "AcceptEula": accept_eula
            }
        if model_reference_arn:
            c_def["ModelDataSource"]["S3DataSource"]["HubAccessConfig"] = {
                "HubContentArn": model_reference_arn
            }

    elif model_data_url is not None:
        c_def["ModelDataUrl"] = model_data_url

    if container_mode:
        c_def["Mode"] = container_mode
    if image_config:
        c_def["ImageConfig"] = image_config
    return c_def