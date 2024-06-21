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
"""Placeholder docstring"""
from __future__ import absolute_import, annotations, print_function

import json
import logging
import os
import re
import sys
import time
import typing
import warnings
import uuid
import datetime

from copy import deepcopy
from typing import List, Dict, Any, Sequence, Optional

import boto3
import botocore
import botocore.config
from botocore.exceptions import ClientError
import six
from sagemaker.utils import instance_supports_kms, create_paginator_config

import sagemaker.logs
from sagemaker import vpc_utils, s3_utils
from sagemaker._studio import _append_project_tags
from sagemaker.config import load_sagemaker_config, validate_sagemaker_config
from sagemaker.config import (
    KEY,
    TRAINING_JOB,
    TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    TRAINING_JOB_ROLE_ARN_PATH,
    TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    TRAINING_JOB_ENVIRONMENT_PATH,
    TRAINING_JOB_VPC_CONFIG_PATH,
    TRAINING_JOB_OUTPUT_DATA_CONFIG_PATH,
    TRAINING_JOB_RESOURCE_CONFIG_PATH,
    TRAINING_JOB_PROFILE_CONFIG_PATH,
    PROCESSING_JOB_INPUTS_PATH,
    PROCESSING_JOB,
    PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    PROCESSING_JOB_ENVIRONMENT_PATH,
    PROCESSING_JOB_ROLE_ARN_PATH,
    PROCESSING_JOB_NETWORK_CONFIG_PATH,
    PROCESSING_OUTPUT_CONFIG_PATH,
    PROCESSING_JOB_PROCESSING_RESOURCES_PATH,
    MONITORING_JOB_ENVIRONMENT_PATH,
    MONITORING_JOB_ROLE_ARN_PATH,
    MONITORING_JOB_VOLUME_KMS_KEY_ID_PATH,
    MONITORING_JOB_NETWORK_CONFIG_PATH,
    MONITORING_JOB_OUTPUT_KMS_KEY_ID_PATH,
    MONITORING_SCHEDULE,
    MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH,
    AUTO_ML_ROLE_ARN_PATH,
    AUTO_ML_V2_ROLE_ARN_PATH,
    AUTO_ML_OUTPUT_CONFIG_PATH,
    AUTO_ML_V2_OUTPUT_CONFIG_PATH,
    AUTO_ML_JOB_CONFIG_PATH,
    AUTO_ML_JOB,
    AUTO_ML_JOB_V2,
    COMPILATION_JOB_ROLE_ARN_PATH,
    COMPILATION_JOB_OUTPUT_CONFIG_PATH,
    COMPILATION_JOB_VPC_CONFIG_PATH,
    COMPILATION_JOB,
    EDGE_PACKAGING_ROLE_ARN_PATH,
    EDGE_PACKAGING_OUTPUT_CONFIG_PATH,
    EDGE_PACKAGING_RESOURCE_KEY_PATH,
    EDGE_PACKAGING_JOB,
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
    MODEL_PRIMARY_CONTAINER_ENVIRONMENT_PATH,
    MODEL_VPC_CONFIG_PATH,
    MODEL_PACKAGE_VALIDATION_ROLE_PATH,
    VALIDATION_ROLE,
    VALIDATION_PROFILES,
    MODEL_PACKAGE_INFERENCE_SPECIFICATION_CONTAINERS_PATH,
    MODEL_PACKAGE_VALIDATION_PROFILES_PATH,
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
    FEATURE_GROUP,
    TAGS,
    FEATURE_GROUP_ROLE_ARN_PATH,
    FEATURE_GROUP_ONLINE_STORE_CONFIG_PATH,
    FEATURE_GROUP_OFFLINE_STORE_CONFIG_PATH,
    SESSION_DEFAULT_S3_BUCKET_PATH,
    SESSION_DEFAULT_S3_OBJECT_KEY_PREFIX_PATH,
)
from sagemaker.config.config_utils import _log_sagemaker_config_merge
from sagemaker.deprecations import deprecated_class
from sagemaker.enums import EndpointType
from sagemaker.inputs import ShuffleConfig, TrainingInput, BatchDataCaptureConfig
from sagemaker.user_agent import get_user_agent_extra_suffix
from sagemaker.utils import (
    name_from_image,
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
)
from sagemaker import exceptions
from sagemaker.session_settings import SessionSettings
from sagemaker.utils import can_model_package_source_uri_autopopulate

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
EP_LOGGER_POLL = 10
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
        settings=None,
        sagemaker_metrics_client=None,
        sagemaker_config: dict = None,
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
            settings (sagemaker.session_settings.SessionSettings): Optional. Set of optional
                parameters to apply to the session.
            sagemaker_metrics_client (boto3.SageMakerMetrics.Client):
                Client which makes SageMaker Metrics related calls to Amazon SageMaker
                (default: None). If not provided, one will be created using
                this instance's ``boto_session``.
            sagemaker_config (dict): A dictionary containing default values for the
                SageMaker Python SDK. (default: None). The dictionary must adhere to the schema
                defined at `~sagemaker.config.config_schema.SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA`.
                If sagemaker_config is not provided and configuration files exist (at the default
                paths for admins and users, or paths set through the environment variables
                SAGEMAKER_ADMIN_CONFIG_OVERRIDE and SAGEMAKER_USER_CONFIG_OVERRIDE),
                a new dictionary will be generated from those configuration files. Alternatively,
                this dictionary can be generated by calling
                :func:`~sagemaker.config.load_sagemaker_config` and then be provided to the
                Session.
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
        botocore_config = botocore.config.Config(user_agent_extra=get_user_agent_extra_suffix())

        # Create sagemaker_client with the botocore_config object
        # This config is customized to append SageMaker Python SDK specific user_agent suffix
        self.sagemaker_client = sagemaker_client or self.boto_session.client(
            "sagemaker", config=botocore_config
        )

        if sagemaker_runtime_client is not None:
            self.sagemaker_runtime_client = sagemaker_runtime_client
        else:
            config = botocore.config.Config(
                read_timeout=80, user_agent_extra=get_user_agent_extra_suffix()
            )
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
            self.sagemaker_metrics_client = self.boto_session.client(
                "sagemaker-metrics", config=botocore_config
            )

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
        bucket, key_prefix = s3_utils.determine_bucket_and_prefix(
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
            default_bucket = generate_default_sagemaker_bucket_name(self.boto_session)
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

    def train(  # noqa: C901
        self,
        input_mode,
        input_config,
        role=None,
        job_name=None,
        output_config=None,
        resource_config=None,
        vpc_config=None,
        hyperparameters=None,
        stop_condition=None,
        tags=None,
        metric_definitions=None,
        enable_network_isolation=None,
        image_uri=None,
        training_image_config=None,
        infra_check_config=None,
        container_entry_point=None,
        container_arguments=None,
        algorithm_arn=None,
        encrypt_inter_container_traffic=None,
        use_spot_instances=False,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        experiment_config=None,
        debugger_rule_configs=None,
        debugger_hook_config=None,
        tensorboard_output_config=None,
        enable_sagemaker_metrics=None,
        profiler_rule_configs=None,
        profiler_config=None,
        environment: Optional[Dict[str, str]] = None,
        retry_strategy=None,
        remote_debug_config=None,
        session_chaining_config=None,
    ):
        """Create an Amazon SageMaker training job.

        Args:
            input_mode (str): The input mode that the algorithm supports. Valid modes:
                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a
                Unix-named pipe.
                * 'FastFile' - Amazon SageMaker streams data from S3 on demand instead of
                downloading the entire dataset before training begins.
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
            tags (Optional[Tags]): Tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
                used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
                the name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs.
            enable_network_isolation (bool): Whether to request for the training job to run with
                network isolation or not.
            image_uri (str): Docker image containing training code.
            training_image_config(dict): Training image configuration.
                Optionally, the dict can contain 'TrainingRepositoryAccessMode' and
                'TrainingRepositoryCredentialsProviderArn' (under 'TrainingRepositoryAuthConfig').
                For example,

                .. code:: python

                    training_image_config = {
                        "TrainingRepositoryAccessMode": "Vpc",
                        "TrainingRepositoryAuthConfig": {
                            "TrainingRepositoryCredentialsProviderArn":
                              "arn:aws:lambda:us-west-2:1234567890:function:test"
                        },
                    }

                If TrainingRepositoryAccessMode is set to Vpc, the training image is accessed
                through a private Docker registry in customer Vpc. If it's set to Platform or None,
                the training image is accessed through ECR.
                If TrainingRepositoryCredentialsProviderArn is provided, the credentials to
                authenticate to the private Docker registry will be retrieved from this AWS Lambda
                function. (default: ``None``). When it's set to None, SageMaker will not do
                authentication before pulling the image in the private Docker registry.
            container_entry_point (List[str]): Optional. The entrypoint script for a Docker
                container used to run a training job. This script takes precedence over
                the default train processing instructions.
            container_arguments (List[str]): Optional. The arguments for a container used to run
                a training job.
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
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain four keys:
                'ExperimentName', 'TrialName',  'TrialComponentDisplayName' and 'RunName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
                * `RunName` is used to record an experiment run.
            enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
                Series. For more information see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html
                #SageMaker-Type
                -AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
                (default: ``None``).
            profiler_rule_configs (list[dict]): A list of profiler rule
                configurations.src/sagemaker/lineage/artifact.py:285
            profiler_config (dict): Configuration for how profiling information is emitted
                with SageMaker Profiler. (default: ``None``).
            remote_debug_config(dict): Configuration for RemoteDebug. (default: ``None``)
                The dict can contain 'EnableRemoteDebug'(bool).
                For example,

                .. code:: python

                    remote_debug_config = {
                        "EnableRemoteDebug": True,
                    }
            session_chaining_config(dict): Configuration for SessionChaining. (default: ``None``)
                The dict can contain 'EnableSessionTagChaining'(bool).
                For example,

                .. code:: python

                    session_chaining_config = {
                        "EnableSessionTagChaining": True,
                    }
            environment (dict[str, str]) : Environment variables to be set for
                use during training job (default: ``None``)
            retry_strategy(dict): Defines RetryStrategy for InternalServerFailures.
                * max_retry_attsmpts (int): Number of times a job should be retried.
                The key in RetryStrategy is 'MaxRetryAttempts'.
            infra_check_config(dict): Infra check configuration.
                Optionally, the dict can contain 'EnableInfraCheck'(bool).
                For example,

                .. code:: python

                    infra_check_config = {
                        "EnableInfraCheck": True,
                    }
        Returns:
            str: ARN of the training job, if it is created.
        """
        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, TRAINING_JOB, TAGS)
        )

        _encrypt_inter_container_traffic = resolve_value_from_config(
            direct_input=encrypt_inter_container_traffic,
            config_path=TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
            default_value=False,
            sagemaker_session=self,
        )
        role = resolve_value_from_config(role, TRAINING_JOB_ROLE_ARN_PATH, sagemaker_session=self)
        enable_network_isolation = resolve_value_from_config(
            direct_input=enable_network_isolation,
            config_path=TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
            default_value=False,
            sagemaker_session=self,
        )
        inferred_vpc_config = update_nested_dictionary_with_values_from_config(
            vpc_config, TRAINING_JOB_VPC_CONFIG_PATH, sagemaker_session=self
        )
        inferred_output_config = update_nested_dictionary_with_values_from_config(
            output_config, TRAINING_JOB_OUTPUT_DATA_CONFIG_PATH, sagemaker_session=self
        )
        customer_supplied_kms_key = "VolumeKmsKeyId" in resource_config
        inferred_resource_config = update_nested_dictionary_with_values_from_config(
            resource_config, TRAINING_JOB_RESOURCE_CONFIG_PATH, sagemaker_session=self
        )
        inferred_profiler_config = update_nested_dictionary_with_values_from_config(
            profiler_config, TRAINING_JOB_PROFILE_CONFIG_PATH, sagemaker_session=self
        )
        if (
            not customer_supplied_kms_key
            and "InstanceType" in inferred_resource_config
            and not instance_supports_kms(inferred_resource_config["InstanceType"])
            and "VolumeKmsKeyId" in inferred_resource_config
        ):
            del inferred_resource_config["VolumeKmsKeyId"]

        environment = resolve_value_from_config(
            direct_input=environment,
            config_path=TRAINING_JOB_ENVIRONMENT_PATH,
            default_value=None,
            sagemaker_session=self,
        )
        train_request = self._get_train_request(
            input_mode=input_mode,
            input_config=input_config,
            role=role,
            job_name=job_name,
            output_config=inferred_output_config,
            resource_config=inferred_resource_config,
            vpc_config=inferred_vpc_config,
            hyperparameters=hyperparameters,
            stop_condition=stop_condition,
            tags=tags,
            metric_definitions=metric_definitions,
            enable_network_isolation=enable_network_isolation,
            image_uri=image_uri,
            training_image_config=training_image_config,
            infra_check_config=infra_check_config,
            container_entry_point=container_entry_point,
            container_arguments=container_arguments,
            algorithm_arn=algorithm_arn,
            encrypt_inter_container_traffic=_encrypt_inter_container_traffic,
            use_spot_instances=use_spot_instances,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            experiment_config=experiment_config,
            debugger_rule_configs=debugger_rule_configs,
            debugger_hook_config=debugger_hook_config,
            tensorboard_output_config=tensorboard_output_config,
            enable_sagemaker_metrics=enable_sagemaker_metrics,
            profiler_rule_configs=profiler_rule_configs,
            profiler_config=inferred_profiler_config,
            remote_debug_config=remote_debug_config,
            session_chaining_config=session_chaining_config,
            environment=environment,
            retry_strategy=retry_strategy,
        )

        def submit(request):
            logger.info("Creating training-job with name: %s", job_name)
            logger.debug("train request: %s", json.dumps(request, indent=4))
            self.sagemaker_client.create_training_job(**request)

        self._intercept_create_request(train_request, submit, self.train.__name__)

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
        training_image_config=None,
        infra_check_config=None,
        container_entry_point=None,
        container_arguments=None,
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
        profiler_rule_configs=None,
        profiler_config=None,
        remote_debug_config=None,
        session_chaining_config=None,
        environment=None,
        retry_strategy=None,
    ):
        """Constructs a request compatible for creating an Amazon SageMaker training job.

        Args:
            input_mode (str): The input mode that the algorithm supports. Valid modes:
                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                a directory in the Docker container.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a
                Unix-named pipe.
                * 'FastFile' - Amazon SageMaker streams data from S3 on demand instead of
                downloading the entire dataset before training begins.
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
            training_image_config(dict): Training image configuration.
                Optionally, the dict can contain 'TrainingRepositoryAccessMode' and
                'TrainingRepositoryCredentialsProviderArn' (under 'TrainingRepositoryAuthConfig').
                For example,

                .. code:: python

                    training_image_config = {
                        "TrainingRepositoryAccessMode": "Vpc",
                        "TrainingRepositoryAuthConfig": {
                            "TrainingRepositoryCredentialsProviderArn":
                              "arn:aws:lambda:us-west-2:1234567890:function:test"
                        },
                    }

                If TrainingRepositoryAccessMode is set to Vpc, the training image is accessed
                through a private Docker registry in customer Vpc. If it's set to Platform or None,
                the training image is accessed through ECR.
                If TrainingRepositoryCredentialsProviderArn is provided, the credentials to
                authenticate to the private Docker registry will be retrieved from this AWS Lambda
                function. (default: ``None``). When it's set to None, SageMaker will not do
                authentication before pulling the image in the private Docker registry.
            container_entry_point (List[str]): Optional. The entrypoint script for a Docker
                container used to run a training job. This script takes precedence over
                the default train processing instructions.
            container_arguments (List[str]): Optional. The arguments for a container used to run
                a training job.
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
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain four keys:
                'ExperimentName', 'TrialName', 'TrialComponentDisplayName' and 'RunName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
                * `RunName` is used to record an experiment run.
            enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
                Series. For more information see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html
                #SageMaker-Type
                -AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
                (default: ``None``).
            profiler_rule_configs (list[dict]): A list of profiler rule configurations.
            profiler_config(dict): Configuration for how profiling information is emitted with
                SageMaker Profiler. (default: ``None``).
            remote_debug_config(dict): Configuration for RemoteDebug. (default: ``None``)
                The dict can contain 'EnableRemoteDebug'(bool).
                For example,

                .. code:: python

                    remote_debug_config = {
                        "EnableRemoteDebug": True,
                    }
            session_chaining_config(dict): Configuration for SessionChaining. (default: ``None``)
                The dict can contain 'EnableSessionTagChaining'(bool).
                For example,

                .. code:: python

                    session_chaining_config = {
                        "EnableSessionTagChaining": True,
                    }
            environment (dict[str, str]) : Environment variables to be set for
                use during training job (default: ``None``)
            retry_strategy(dict): Defines RetryStrategy for InternalServerFailures.
                * max_retry_attsmpts (int): Number of times a job should be retried.
                The key in RetryStrategy is 'MaxRetryAttempts'.

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

        if training_image_config is not None:
            train_request["AlgorithmSpecification"]["TrainingImageConfig"] = training_image_config

        if infra_check_config is not None:
            train_request["InfraCheckConfig"] = infra_check_config

        if container_entry_point is not None:
            train_request["AlgorithmSpecification"]["ContainerEntrypoint"] = container_entry_point

        if container_arguments is not None:
            train_request["AlgorithmSpecification"]["ContainerArguments"] = container_arguments

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

        if environment is not None:
            train_request["Environment"] = environment

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
            # estimator.use_spot_instances may be a Pipeline ParameterBoolean object
            # which is parsed during the Pipeline execution runtime
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

        if profiler_rule_configs is not None:
            train_request["ProfilerRuleConfigurations"] = profiler_rule_configs

        if profiler_config is not None:
            train_request["ProfilerConfig"] = profiler_config

        if remote_debug_config is not None:
            train_request["RemoteDebugConfig"] = remote_debug_config

        if session_chaining_config is not None:
            train_request["SessionChainingConfig"] = session_chaining_config

        if retry_strategy is not None:
            train_request["RetryStrategy"] = retry_strategy

        return train_request

    def update_training_job(
        self,
        job_name,
        profiler_rule_configs=None,
        profiler_config=None,
        resource_config=None,
        remote_debug_config=None,
    ):
        """Calls the UpdateTrainingJob API for the given job name and returns the response.

        Args:
            job_name (str): Name of the training job being updated.
            profiler_rule_configs (list): List of profiler rule configurations. (default: ``None``).
            profiler_config(dict): Configuration for how profiling information is emitted with
                SageMaker Profiler. (default: ``None``).
            resource_config (dict): Configuration of the resources for the training job. You can
                update the keep-alive period if the warm pool status is `Available`. No other fields
                can be updated. (default: ``None``).
            remote_debug_config(dict): Configuration for RemoteDebug. (default: ``None``)
                The dict can contain 'EnableRemoteDebug'(bool).
                For example,

                .. code:: python

                    remote_debug_config = {
                        "EnableRemoteDebug": True,
                    }
        """
        # No injections from sagemaker_config because the UpdateTrainingJob API's resource_config
        # object accepts fewer parameters than the CreateTrainingJob API, and none that the
        # sagemaker_config currently supports
        inferred_profiler_config = update_nested_dictionary_with_values_from_config(
            profiler_config, TRAINING_JOB_PROFILE_CONFIG_PATH, sagemaker_session=self
        )
        update_training_job_request = self._get_update_training_job_request(
            job_name=job_name,
            profiler_rule_configs=profiler_rule_configs,
            profiler_config=inferred_profiler_config,
            resource_config=resource_config,
            remote_debug_config=remote_debug_config,
        )
        logger.info("Updating training job with name %s", job_name)
        logger.debug("Update request: %s", json.dumps(update_training_job_request, indent=4))
        self.sagemaker_client.update_training_job(**update_training_job_request)

    def _get_update_training_job_request(
        self,
        job_name,
        profiler_rule_configs=None,
        profiler_config=None,
        resource_config=None,
        remote_debug_config=None,
    ):
        """Constructs a request compatible for updating an Amazon SageMaker training job.

        Args:
            job_name (str): Name of the training job being updated.
            profiler_rule_configs (list): List of profiler rule configurations. (default: ``None``).
            profiler_config(dict): Configuration for how profiling information is emitted with
                SageMaker Profiler. (default: ``None``).
            resource_config (dict): Configuration of the resources for the training job. You can
                update the keep-alive period if the warm pool status is `Available`. No other fields
                can be updated. (default: ``None``).
            remote_debug_config(dict): Configuration for RemoteDebug. (default: ``None``)
                The dict can contain 'EnableRemoteDebug'(bool).
                For example,

                .. code:: python

                    remote_debug_config = {
                        "EnableRemoteDebug": True,
                    }

        Returns:
            Dict: an update training request dict
        """
        update_training_job_request = {
            "TrainingJobName": job_name,
        }

        if profiler_rule_configs is not None:
            update_training_job_request["ProfilerRuleConfigurations"] = profiler_rule_configs

        if profiler_config is not None:
            update_training_job_request["ProfilerConfig"] = profiler_config

        if resource_config is not None:
            update_training_job_request["ResourceConfig"] = resource_config

        if remote_debug_config is not None:
            update_training_job_request["RemoteDebugConfig"] = remote_debug_config

        return update_training_job_request

    def process(
        self,
        inputs,
        output_config,
        job_name,
        resources,
        stopping_condition,
        app_specification,
        environment: Optional[Dict[str, str]] = None,
        network_config=None,
        role_arn=None,
        tags=None,
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
            tags (Optional[Tags]): A list of dictionaries containing key-value
                pairs.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
        """
        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, PROCESSING_JOB, TAGS)
        )

        network_config = resolve_nested_dict_value_from_config(
            network_config,
            ["EnableInterContainerTrafficEncryption"],
            PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
            sagemaker_session=self,
        )
        # Processing Input can either have AthenaDatasetDefinition or RedshiftDatasetDefinition
        # or neither, but not both
        union_key_paths_for_dataset_definition = [
            [
                "DatasetDefinition",
                "S3Input",
            ],
            [
                "DatasetDefinition.AthenaDatasetDefinition",
                "DatasetDefinition.RedshiftDatasetDefinition",
            ],
        ]
        update_list_of_dicts_with_values_from_config(
            inputs,
            PROCESSING_JOB_INPUTS_PATH,
            union_key_paths=union_key_paths_for_dataset_definition,
            sagemaker_session=self,
        )
        role_arn = resolve_value_from_config(
            role_arn, PROCESSING_JOB_ROLE_ARN_PATH, sagemaker_session=self
        )
        inferred_network_config_from_config = update_nested_dictionary_with_values_from_config(
            network_config, PROCESSING_JOB_NETWORK_CONFIG_PATH, sagemaker_session=self
        )
        inferred_output_config = update_nested_dictionary_with_values_from_config(
            output_config, PROCESSING_OUTPUT_CONFIG_PATH, sagemaker_session=self
        )
        inferred_resources_config = update_nested_dictionary_with_values_from_config(
            resources, PROCESSING_JOB_PROCESSING_RESOURCES_PATH, sagemaker_session=self
        )
        environment = resolve_value_from_config(
            direct_input=environment,
            config_path=PROCESSING_JOB_ENVIRONMENT_PATH,
            default_value=None,
            sagemaker_session=self,
        )
        process_request = self._get_process_request(
            inputs=inputs,
            output_config=inferred_output_config,
            job_name=job_name,
            resources=inferred_resources_config,
            stopping_condition=stopping_condition,
            app_specification=app_specification,
            environment=environment,
            network_config=inferred_network_config_from_config,
            role_arn=role_arn,
            tags=tags,
            experiment_config=experiment_config,
        )

        def submit(request):
            logger.info("Creating processing-job with name %s", job_name)
            logger.debug("process request: %s", json.dumps(request, indent=4))
            self.sagemaker_client.create_processing_job(**request)

        self._intercept_create_request(process_request, submit, self.process.__name__)

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
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.

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
        tags=None,
        data_analysis_start_time=None,
        data_analysis_end_time=None,
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
            tags (Optional[Tags]): A list of dictionaries containing key-value
                pairs.
            data_analysis_start_time (str): Start time for the data analysis window
                for the one time monitoring schedule (NOW), e.g. "-PT1H"
            data_analysis_end_time (str): End time for the data analysis window
                for the one time monitoring schedule (NOW), e.g. "-PT1H"
        """
        role_arn = resolve_value_from_config(
            role_arn, MONITORING_JOB_ROLE_ARN_PATH, sagemaker_session=self
        )
        volume_kms_key = resolve_value_from_config(
            volume_kms_key, MONITORING_JOB_VOLUME_KMS_KEY_ID_PATH, sagemaker_session=self
        )
        inferred_network_config_from_config = update_nested_dictionary_with_values_from_config(
            network_config, MONITORING_JOB_NETWORK_CONFIG_PATH, sagemaker_session=self
        )
        environment = resolve_value_from_config(
            direct_input=environment,
            config_path=MONITORING_JOB_ENVIRONMENT_PATH,
            default_value=None,
            sagemaker_session=self,
        )
        monitoring_schedule_request = {
            "MonitoringScheduleName": monitoring_schedule_name,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinition": {
                    "Environment": environment,
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
                "ScheduleExpression": schedule_expression,
            }
            if data_analysis_start_time is not None:
                monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                    "DataAnalysisStartTime"
                ] = data_analysis_start_time

            if data_analysis_end_time is not None:
                monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                    "DataAnalysisEndTime"
                ] = data_analysis_end_time

        if monitoring_output_config is not None:
            kms_key_from_config = resolve_value_from_config(
                config_path=MONITORING_JOB_OUTPUT_KMS_KEY_ID_PATH, sagemaker_session=self
            )
            if KMS_KEY_ID not in monitoring_output_config and kms_key_from_config:
                monitoring_output_config[KMS_KEY_ID] = kms_key_from_config
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

        if inferred_network_config_from_config is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "NetworkConfig"
            ] = inferred_network_config_from_config

        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, MONITORING_SCHEDULE, TAGS)
        )
        if tags is not None:
            monitoring_schedule_request["Tags"] = tags

        logger.info("Creating monitoring schedule name %s.", monitoring_schedule_name)
        logger.debug(
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
        data_analysis_start_time=None,
        data_analysis_end_time=None,
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
            data_analysis_start_time (str): Start time for the data analysis window
                for the one time monitoring schedule (NOW), e.g. "-PT1H"
            data_analysis_end_time (str): End time for the data analysis window
                for the one time monitoring schedule (NOW), e.g. "-PT1H"
        """
        existing_desc = self.sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

        existing_schedule_config = None
        existing_data_analysis_start_time = None
        existing_data_analysis_end_time = None

        if (
            existing_desc.get("MonitoringScheduleConfig") is not None
            and existing_desc["MonitoringScheduleConfig"].get("ScheduleConfig") is not None
            and existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
            is not None
        ):
            existing_schedule_config = existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"][
                "ScheduleExpression"
            ]
            if (
                existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"].get(
                    "DataAnalysisStartTime"
                )
                is not None
            ):
                existing_data_analysis_start_time = existing_desc["MonitoringScheduleConfig"][
                    "ScheduleConfig"
                ]["DataAnalysisStartTime"]
            if (
                existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"].get(
                    "DataAnalysisEndTime"
                )
                is not None
            ):
                existing_data_analysis_end_time = existing_desc["MonitoringScheduleConfig"][
                    "ScheduleConfig"
                ]["DataAnalysisEndTime"]

        request_schedule_expression = schedule_expression or existing_schedule_config
        request_data_analysis_start_time = (
            data_analysis_start_time or existing_data_analysis_start_time
        )
        request_data_analysis_end_time = data_analysis_end_time or existing_data_analysis_end_time

        if request_schedule_expression == MODEL_MONITOR_ONE_TIME_SCHEDULE and (
            request_data_analysis_start_time is None or request_data_analysis_end_time is None
        ):
            message = (
                "Both data_analysis_start_time and data_analysis_end_time are required "
                "for one time monitoring schedule "
            )
            LOGGER.error(message)
            raise ValueError(message)

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
                "ScheduleExpression": request_schedule_expression,
            }

            if request_data_analysis_start_time is not None:
                monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                    "DataAnalysisStartTime"
                ] = request_data_analysis_start_time

            if request_data_analysis_end_time is not None:
                monitoring_schedule_request["MonitoringScheduleConfig"]["ScheduleConfig"][
                    "DataAnalysisEndTime"
                ] = request_data_analysis_end_time

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

        _network_config = network_config or existing_network_config
        _network_config = resolve_nested_dict_value_from_config(
            _network_config,
            ["EnableInterContainerTrafficEncryption"],
            MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH,
            sagemaker_session=self,
        )
        if _network_config is not None:
            monitoring_schedule_request["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
                "NetworkConfig"
            ] = _network_config

        logger.info("Updating monitoring schedule with name: %s .", monitoring_schedule_name)
        logger.debug(
            "monitoring_schedule_request= %s", json.dumps(monitoring_schedule_request, indent=4)
        )
        self.sagemaker_client.update_monitoring_schedule(**monitoring_schedule_request)

    def start_monitoring_schedule(self, monitoring_schedule_name):
        """Starts a monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
                Schedule to start.
        """
        logger.info("Starting Monitoring Schedule with name: %s", monitoring_schedule_name)
        self.sagemaker_client.start_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

    def stop_monitoring_schedule(self, monitoring_schedule_name):
        """Stops a monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
                Schedule to stop.
        """
        logger.info("Stopping Monitoring Schedule with name: %s", monitoring_schedule_name)
        self.sagemaker_client.stop_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

    def delete_monitoring_schedule(self, monitoring_schedule_name):
        """Deletes a monitoring schedule.

        Args:
            monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
                Schedule to delete.
        """
        logger.info("Deleting Monitoring Schedule with name: %s", monitoring_schedule_name)
        self.sagemaker_client.delete_monitoring_schedule(
            MonitoringScheduleName=monitoring_schedule_name
        )

    def describe_monitoring_schedule(self, monitoring_schedule_name):
        """Calls the DescribeMonitoringSchedule API for given name and returns the response.

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

    def update_monitoring_alert(
        self,
        monitoring_schedule_name: str,
        monitoring_alert_name: str,
        data_points_to_alert: int,
        evaluation_period: int,
    ):
        """Update the monitoring alerts associated with the given schedule_name and alert_name

        Args:
            monitoring_schedule_name (str): The name of the monitoring schedule to update.
            monitoring_alert_name (str): The name of the monitoring alert to update.
            data_points_to_alert (int):  The data point to alert.
            evaluation_period (int): The period to evaluate the alert status.

        Returns:
            dict: A dict represents the update alert response.
        """
        return self.sagemaker_client.update_monitoring_alert(
            MonitoringScheduleName=monitoring_schedule_name,
            MonitoringAlertName=monitoring_alert_name,
            DatapointsToAlert=data_points_to_alert,
            EvaluationPeriod=evaluation_period,
        )

    def list_monitoring_alerts(
        self,
        monitoring_schedule_name: str,
        next_token: Optional[str] = None,
        max_results: Optional[int] = 10,
    ) -> Dict:
        """Lists the monitoring alerts associated with the given monitoring_schedule_name.

        Args:
            monitoring_schedule_name (str): The name of the monitoring schedule to filter on.
                If not provided, does not filter on it.
            next_token (Optional[str]):  The pagination token. Default: None
            max_results (Optional[int]): The maximum number of results to return.
                Must be between 1 and 100. Default: 10

        Returns:
            dict: list of monitoring alerts.
        """
        params = {
            "MonitoringScheduleName": monitoring_schedule_name,
            "MaxResults": max_results,
        }
        if next_token:
            params.update({"NextToken": next_token})

        return self.sagemaker_client.list_monitoring_alerts(**params)

    def list_monitoring_alert_history(
        self,
        monitoring_schedule_name: Optional[str] = None,
        monitoring_alert_name: Optional[str] = None,
        sort_by: Optional[str] = "CreationTime",
        sort_order: Optional[str] = "Descending",
        next_token: Optional[str] = None,
        max_results: Optional[int] = 10,
        creation_time_before: Optional[str] = None,
        creation_time_after: Optional[str] = None,
        status_equals: Optional[str] = None,
    ) -> Dict:
        """Lists the alert history associated with the given schedule_name and alert_name.

        Args:
            monitoring_schedule_name (Optional[str]): The name of the monitoring_schedule_name
                to filter on. If not provided, does not filter on it. Default: None.
            monitoring_alert_name (Optional[str]): The name of the monitoring_alert_name
                to filter on. If not provided, does not filter on it. Default: None.
            sort_by (Optional[str]): sort_by (str): The field to sort by.
                Can be one of: "Name", "CreationTime" Default: "CreationTime".
            sort_order (Optional[str]): The sort order. Can be one of: "Ascending", "Descending".
                Default: "Descending".
            next_token (Optional[str]):  The pagination token. Default: None
            max_results (Optional[int]): The maximum number of results to return.
                Must be between 1 and 100. Default: 10.
            creation_time_before (Optional[str]): A filter to filter alert history before a time
            creation_time_after (Optional[str]): A filter to filter alert history after a time
                Default: None.
            status_equals (Optional[str]): A filter to filter alert history by status
                Default: None.

        Returns:
            dict: list of monitoring alert history.
        """
        params = {
            "MonitoringScheduleName": monitoring_schedule_name,
            "SortBy": sort_by,
            "SortOrder": sort_order,
            "MaxResults": max_results,
        }
        if monitoring_alert_name:
            params.update({"MonitoringAlertName": monitoring_alert_name})
        if creation_time_before:
            params.update({"CreationTimeBefore": creation_time_before})
        if creation_time_after:
            params.update({"CreationTimeAfter": creation_time_after})
        if status_equals:
            params.update({"StatusEquals": status_equals})
        if next_token:
            params.update({"NextToken": next_token})

        return self.sagemaker_client.list_monitoring_alert_history(**params)

    def was_processing_job_successful(self, job_name):
        """Calls the DescribeProcessingJob API for the given job name.

        It returns True if job was successful.

        Args:
            job_name (str): The name of the processing job to describe.

        Returns:
            bool: Whether the processing job was successful.
        """
        job_desc = self.sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
        return job_desc["ProcessingJobStatus"] == "Completed"

    def describe_processing_job(self, job_name):
        """Calls the DescribeProcessingJob API for the given job name and returns the response.

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
        """Calls the DescribeTrainingJob API for the given job name and returns the response.

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
        role=None,
        job_name=None,
        problem_type=None,
        job_objective=None,
        generate_candidate_definitions_only=False,
        tags=None,
        model_deploy_config=None,
    ):
        """Create an Amazon SageMaker AutoML job.

        Args:
            input_config (list[dict]): A list of Channel objects. Each channel contains "DataSource"
                and "TargetAttributeName", "CompressionType" and "SampleWeightAttributeName" are
                optional fields.
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
            tags (Optional[Tags]): A list of dictionaries containing key-value
                pairs.
            model_deploy_config (dict): Specifies how to generate the endpoint name
                for an automatic one-click Autopilot model deployment.
                Contains "AutoGenerateEndpointName" and "EndpointName"
        """

        role = resolve_value_from_config(role, AUTO_ML_ROLE_ARN_PATH, sagemaker_session=self)
        inferred_output_config = update_nested_dictionary_with_values_from_config(
            output_config, AUTO_ML_OUTPUT_CONFIG_PATH, sagemaker_session=self
        )
        inferred_automl_job_config = update_nested_dictionary_with_values_from_config(
            auto_ml_job_config, AUTO_ML_JOB_CONFIG_PATH, sagemaker_session=self
        )
        auto_ml_job_request = self._get_auto_ml_request(
            input_config=input_config,
            output_config=inferred_output_config,
            auto_ml_job_config=inferred_automl_job_config,
            role=role,
            job_name=job_name,
            problem_type=problem_type,
            job_objective=job_objective,
            generate_candidate_definitions_only=generate_candidate_definitions_only,
            tags=format_tags(tags),
            model_deploy_config=model_deploy_config,
        )

        def submit(request):
            logger.info("Creating auto-ml-job with name: %s", job_name)
            logger.debug("auto ml request: %s", json.dumps(request), indent=4)
            self.sagemaker_client.create_auto_ml_job(**request)

        self._intercept_create_request(auto_ml_job_request, submit, self.auto_ml.__name__)

    def _get_auto_ml_request(
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
        model_deploy_config=None,
    ):
        """Constructs a request compatible for creating an Amazon SageMaker AutoML job.

        Args:
            input_config (list[dict]): A list of Channel objects. Each channel contains "DataSource"
                and "TargetAttributeName", "CompressionType" and "SampleWeightAttributeName" are
                optional fields.
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
            tags (Optional[Tags]): A list of dictionaries containing key-value
                pairs.
            model_deploy_config (dict): Specifies how to generate the endpoint name
                for an automatic one-click Autopilot model deployment.
                Contains "AutoGenerateEndpointName" and "EndpointName"

        Returns:
            Dict: a automl request dict
        """
        auto_ml_job_request = {
            "AutoMLJobName": job_name,
            "InputDataConfig": input_config,
            "OutputDataConfig": output_config,
            "AutoMLJobConfig": auto_ml_job_config,
            "RoleArn": role,
            "GenerateCandidateDefinitionsOnly": generate_candidate_definitions_only,
        }
        if model_deploy_config is not None:
            auto_ml_job_request["ModelDeployConfig"] = model_deploy_config

        if job_objective is not None:
            auto_ml_job_request["AutoMLJobObjective"] = job_objective
        if problem_type is not None:
            auto_ml_job_request["ProblemType"] = problem_type

        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, AUTO_ML_JOB, TAGS)
        )
        if tags is not None:
            auto_ml_job_request["Tags"] = tags

        return auto_ml_job_request

    def describe_auto_ml_job(self, job_name):
        """Calls the DescribeAutoMLJob API for the given job name and returns the response.

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
            exceptions.CapacityError: If the auto ml job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the auto ml job fails.
        """
        desc = _wait_until(lambda: _auto_ml_job_status(self.sagemaker_client, job), poll)
        _check_job_status(job, desc, "AutoMLJobStatus")
        return desc

    def logs_for_auto_ml_job(  # noqa: C901 - suppress complexity warning for this method
        self, job_name, wait=False, poll=10
    ):
        """Display logs for a given AutoML job, optionally tailing them until job is complete.

        If the output is a tty or a Jupyter cell, it will be color-coded
        based on which instance the log entry is from.

        Args:
            job_name (str): Name of the Auto ML job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes
                (default: False).
            poll (int): The interval in seconds between polling for new log entries and job
                completion (default: 5).

        Raises:
            exceptions.CapacityError: If waiting and auto ml job fails with CapacityError.
            exceptions.UnexpectedStatusException: If waiting and auto ml job fails.
        """

        description = _wait_until(lambda: self.describe_auto_ml_job_v2(job_name), poll)

        instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
            self.boto_session, description, job="AutoML"
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
                description = self.sagemaker_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                last_describe_job_call = time.time()

                status = description["AutoMLJobStatus"]

                if status in ("Completed", "Failed", "Stopped"):
                    print()
                    state = LogState.JOB_COMPLETE

        if wait:
            _check_job_status(job_name, description, "AutoMLJobStatus")
            if dot:
                print()

    def create_auto_ml_v2(
        self,
        input_config,
        job_name,
        problem_config,
        output_config,
        job_objective=None,
        model_deploy_config=None,
        data_split_config=None,
        role=None,
        security_config=None,
        tags=None,
    ):
        """Create an Amazon SageMaker AutoMLV2 job.

        Args:
            input_config (list[dict]): A list of AutoMLDataChannel objects.
                Each channel contains "DataSource" and other optional fields.
            job_name (str): A string that can be used to identify an AutoMLJob. Each AutoMLJob
                should have a unique job name.
            problem_config (object): A collection of settings specific
                to the problem type used to configure an AutoML job V2.
                There must be one and only one config of the following type.
                Supported problem types are:

                - Image Classification (sagemaker.automl.automlv2.ImageClassificationJobConfig),
                - Tabular (sagemaker.automl.automlv2.TabularJobConfig),
                - Text Classification (sagemaker.automl.automlv2.TextClassificationJobConfig),
                - Text Generation (TextGenerationJobConfig),
                - Time Series Forecasting (
                    sagemaker.automl.automlv2.TimeSeriesForecastingJobConfig).

            output_config (dict): The S3 URI where you want to store the training results and
                optional KMS key ID.
            job_objective (dict): AutoMLJob objective, contains "AutoMLJobObjectiveType" (optional),
                "MetricName" and "Value".
            model_deploy_config (dict): Specifies how to generate the endpoint name
                for an automatic one-click Autopilot model deployment.
                Contains "AutoGenerateEndpointName" and "EndpointName"
            data_split_config (dict): This structure specifies how to split the data
                into train and validation datasets.
            role (str): The Amazon Resource Name (ARN) of an IAM role that
                Amazon SageMaker can assume to perform tasks on your behalf.
            security_config (dict): The security configuration for traffic encryption
                or Amazon VPC settings.
            tags (Optional[Tags]): A list of dictionaries containing key-value
                pairs.
        """

        role = resolve_value_from_config(role, AUTO_ML_V2_ROLE_ARN_PATH, sagemaker_session=self)
        inferred_output_config = update_nested_dictionary_with_values_from_config(
            output_config, AUTO_ML_V2_OUTPUT_CONFIG_PATH, sagemaker_session=self
        )

        auto_ml_job_v2_request = self._get_auto_ml_request_v2(
            input_config=input_config,
            job_name=job_name,
            problem_config=problem_config,
            output_config=inferred_output_config,
            role=role,
            job_objective=job_objective,
            model_deploy_config=model_deploy_config,
            data_split_config=data_split_config,
            security_config=security_config,
            tags=format_tags(tags),
        )

        def submit(request):
            logger.info("Creating auto-ml-v2-job with name: %s", job_name)
            logger.debug("auto ml v2 request: %s", json.dumps(request), indent=4)
            print(json.dumps(request))
            self.sagemaker_client.create_auto_ml_job_v2(**request)

        self._intercept_create_request(
            auto_ml_job_v2_request, submit, self.create_auto_ml_v2.__name__
        )

    def _get_auto_ml_request_v2(
        self,
        input_config,
        output_config,
        job_name,
        problem_config,
        role,
        job_objective=None,
        model_deploy_config=None,
        data_split_config=None,
        security_config=None,
        tags=None,
    ):
        """Constructs a request compatible for creating an Amazon SageMaker AutoML job.

        Args:
            input_config (list[dict]): A list of Channel objects. Each channel contains "DataSource"
                and "TargetAttributeName", "CompressionType" and "SampleWeightAttributeName" are
                optional fields.
            output_config (dict): The S3 URI where you want to store the training results and
                optional KMS key ID.
            job_name (str): A string that can be used to identify an AutoMLJob. Each AutoMLJob
                should have a unique job name.
            problem_config (object): A collection of settings specific
                to the problem type used to configure an AutoML job V2.
                There must be one and only one config of the following type.
                Supported problem types are:

                - Image Classification (sagemaker.automl.automlv2.ImageClassificationJobConfig),
                - Tabular (sagemaker.automl.automlv2.TabularJobConfig),
                - Text Classification (sagemaker.automl.automlv2.TextClassificationJobConfig),
                - Text Generation (TextGenerationJobConfig),
                - Time Series Forecasting (
                    sagemaker.automl.automlv2.TimeSeriesForecastingJobConfig).

            role (str): The Amazon Resource Name (ARN) of an IAM role that
                Amazon SageMaker can assume to perform tasks on your behalf.
            job_objective (dict): AutoMLJob objective, contains "AutoMLJobObjectiveType" (optional),
                "MetricName" and "Value".
            model_deploy_config (dict): Specifies how to generate the endpoint name
                for an automatic one-click Autopilot model deployment.
                Contains "AutoGenerateEndpointName" and "EndpointName"
            data_split_config (dict): This structure specifies how to split the data
                into train and validation datasets.
            security_config (dict): The security configuration for traffic encryption
                or Amazon VPC settings.
            tags (Optional[Tags]): A list of dictionaries containing key-value
                pairs.

        Returns:
            Dict: a automl v2 request dict
        """
        auto_ml_job_v2_request = {
            "AutoMLJobName": job_name,
            "AutoMLJobInputDataConfig": input_config,
            "OutputDataConfig": output_config,
            "AutoMLProblemTypeConfig": problem_config,
            "RoleArn": role,
        }
        if job_objective is not None:
            auto_ml_job_v2_request["AutoMLJobObjective"] = job_objective
        if model_deploy_config is not None:
            auto_ml_job_v2_request["ModelDeployConfig"] = model_deploy_config
        if data_split_config is not None:
            auto_ml_job_v2_request["DataSplitConfig"] = data_split_config
        if security_config is not None:
            auto_ml_job_v2_request["SecurityConfig"] = security_config

        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, AUTO_ML_JOB_V2, TAGS)
        )
        if tags is not None:
            auto_ml_job_v2_request["Tags"] = tags

        return auto_ml_job_v2_request

    # Done
    def describe_auto_ml_job_v2(self, job_name):
        """Calls the DescribeAutoMLJobV2 API for the given job name and returns the response.

        Args:
            job_name (str): The name of the AutoML job to describe.

        Returns:
            dict: A dictionary response with the AutoMLV2 Job description.
        """
        return self.sagemaker_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)

    def compile_model(
        self,
        input_model_config,
        output_model_config,
        role=None,
        job_name=None,
        stop_condition=None,
        tags=None,
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
            tags (Optional[Tags]): List of tags for labeling a compile model job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

        Returns:
            str: ARN of the compile model job, if it is created.
        """
        role = resolve_value_from_config(
            role, COMPILATION_JOB_ROLE_ARN_PATH, sagemaker_session=self
        )
        inferred_output_model_config = update_nested_dictionary_with_values_from_config(
            output_model_config, COMPILATION_JOB_OUTPUT_CONFIG_PATH, sagemaker_session=self
        )
        vpc_config = resolve_value_from_config(
            config_path=COMPILATION_JOB_VPC_CONFIG_PATH, sagemaker_session=self
        )
        compilation_job_request = {
            "InputConfig": input_model_config,
            "OutputConfig": inferred_output_model_config,
            "RoleArn": role,
            "StoppingCondition": stop_condition,
            "CompilationJobName": job_name,
        }
        if vpc_config:
            compilation_job_request["VpcConfig"] = vpc_config

        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, COMPILATION_JOB, TAGS)
        )
        if tags is not None:
            compilation_job_request["Tags"] = tags

        logger.info("Creating compilation-job with name: %s", job_name)
        self.sagemaker_client.create_compilation_job(**compilation_job_request)

    def package_model_for_edge(
        self,
        output_model_config,
        role=None,
        job_name=None,
        compilation_job_name=None,
        model_name=None,
        model_version=None,
        resource_key=None,
        tags=None,
    ):
        """Create an Amazon SageMaker Edge packaging job.

        Args:
            output_model_config (dict): Identifies the Amazon S3 location where you want Amazon
                SageMaker Edge to save the results of edge packaging job
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker Edge
                edge packaging jobs use this role to access model artifacts. You must grant
                sufficient permissions to this role.
            job_name (str): Name of the edge packaging job being created.
            compilation_job_name (str): Name of the compilation job being created.
            resource_key (str): KMS key to encrypt the disk used to package the job
            tags (Optional[Tags]): List of tags for labeling a compile model job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        """
        role = resolve_value_from_config(role, EDGE_PACKAGING_ROLE_ARN_PATH, sagemaker_session=self)
        inferred_output_model_config = update_nested_dictionary_with_values_from_config(
            output_model_config, EDGE_PACKAGING_OUTPUT_CONFIG_PATH, sagemaker_session=self
        )
        edge_packaging_job_request = {
            "OutputConfig": inferred_output_model_config,
            "RoleArn": role,
            "ModelName": model_name,
            "ModelVersion": model_version,
            "EdgePackagingJobName": job_name,
            "CompilationJobName": compilation_job_name,
        }
        resource_key = resolve_value_from_config(
            resource_key, EDGE_PACKAGING_RESOURCE_KEY_PATH, sagemaker_session=self
        )
        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, EDGE_PACKAGING_JOB, TAGS)
        )
        if tags is not None:
            edge_packaging_job_request["Tags"] = tags
        if resource_key is not None:
            edge_packaging_job_request["ResourceKey"] = resource_key

        logger.info("Creating edge-packaging-job with name: %s", job_name)
        self.sagemaker_client.create_edge_packaging_job(**edge_packaging_job_request)

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
        max_runtime_in_seconds=None,
        strategy_config=None,
        completion_criteria_config=None,
        enable_network_isolation=False,
        image_uri=None,
        algorithm_arn=None,
        early_stopping_type="Off",
        encrypt_inter_container_traffic=False,
        vpc_config=None,
        use_spot_instances=False,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        random_seed=None,
        environment=None,
        hpo_resource_config=None,
        autotune=False,
        auto_parameters=None,
    ):
        """Create an Amazon SageMaker hyperparameter tuning job.

        Args:
            job_name (str): Name of the tuning job being created.
            strategy (str): Strategy to be used for hyperparameter estimations.
            strategy_config (dict): A configuration for the hyperparameter tuning
                job optimisation strategy.
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
                * 'FastFile' - Amazon SageMaker streams data from S3 on demand instead of
                downloading the entire dataset before training begins.
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
            max_runtime_in_seconds (int or PipelineVariable): The maximum time in seconds
                that a training job launched by a hyperparameter tuning job can run.
            completion_criteria_config (sagemaker.tuner.TuningJobCompletionCriteriaConfig): A
                configuration for the completion criteria.
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
            random_seed (int): An initial value used to initialize a pseudo-random number generator.
                Setting a random seed will make the hyperparameter tuning search strategies to
                produce more consistent configurations for the same tuning job. (default: ``None``).
            environment (dict[str, str]) : Environment variables to be set for
                use during training jobs (default: ``None``)
            hpo_resource_config (dict): The configuration for the hyperparameter tuning resources,
                including the compute instances and storage volumes, used for training jobs launched
                by the tuning job, where you must specify either
                instance_configs or instance_count + instance_type + volume_size:
                * instance_count (int): Number of EC2 instances to use for training.
                The key in resource_config is 'InstanceCount'.
                * instance_type (str): Type of EC2 instance to use for training, for example,
                'ml.c4.xlarge'. The key in resource_config is 'InstanceType'.
                * volume_size (int or PipelineVariable): The volume size in GB of the data to be
                processed for hyperparameter optimisation
                * instance_configs (List[InstanceConfig]): A list containing the configuration(s)
                for one or more resources for processing hyperparameter jobs. These resources
                include compute instances and storage volumes to use in model training jobs.
                * volume_kms_key_id: The AWS Key Management Service (AWS KMS) key
                that Amazon SageMaker uses to encrypt data on the storage
                volume attached to the ML compute instance(s) that run the training job.
            autotune (bool): Whether the parameter ranges or other unset settings of a tuning job
                should be chosen automatically (default: False).
            auto_parameters (dict[str, str]): Dictionary of auto parameters. The keys are names
                of auto parameters and values are example values of auto parameters
                (default: ``None``).
        """

        tune_request = {
            "HyperParameterTuningJobName": job_name,
            "HyperParameterTuningJobConfig": self._map_tuning_config(
                strategy=strategy,
                max_jobs=max_jobs,
                max_parallel_jobs=max_parallel_jobs,
                max_runtime_in_seconds=max_runtime_in_seconds,
                objective_type=objective_type,
                objective_metric_name=objective_metric_name,
                parameter_ranges=parameter_ranges,
                early_stopping_type=early_stopping_type,
                random_seed=random_seed,
                strategy_config=strategy_config,
                completion_criteria_config=completion_criteria_config,
                auto_parameters=auto_parameters,
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
                hpo_resource_config=hpo_resource_config,
                vpc_config=vpc_config,
                stop_condition=stop_condition,
                enable_network_isolation=enable_network_isolation,
                encrypt_inter_container_traffic=encrypt_inter_container_traffic,
                use_spot_instances=use_spot_instances,
                checkpoint_s3_uri=checkpoint_s3_uri,
                checkpoint_local_path=checkpoint_local_path,
                environment=environment,
            ),
        }

        if warm_start_config is not None:
            tune_request["WarmStartConfig"] = warm_start_config

        if autotune:
            tune_request["Autotune"] = {"Mode": "Enabled"}

        tags = _append_project_tags(tags)
        if tags is not None:
            tune_request["Tags"] = tags

        logger.info("Creating hyperparameter tuning job with name: %s", job_name)
        logger.debug("tune request: %s", json.dumps(tune_request, indent=4))
        self.sagemaker_client.create_hyper_parameter_tuning_job(**tune_request)

    def create_tuning_job(
        self,
        job_name,
        tuning_config,
        training_config=None,
        training_config_list=None,
        warm_start_config=None,
        tags=None,
        autotune=False,
    ):
        """Create an Amazon SageMaker hyperparameter tuning job.

        This method supports creating tuning jobs with single or multiple training algorithms
        (estimators), while the ``tune()`` method above only supports creating tuning jobs
        with single training algorithm.

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
            tags (Optional[Tags]): List of tags for labeling the tuning job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            autotune (bool): Whether the parameter ranges or other unset settings of a tuning job
                should be chosen automatically.
        """

        if training_config is None and training_config_list is None:
            raise ValueError("Either training_config or training_config_list should be provided.")
        if training_config is not None and training_config_list is not None:
            raise ValueError(
                "Only one of training_config and training_config_list should be provided."
            )

        tune_request = self._get_tuning_request(
            job_name=job_name,
            tuning_config=tuning_config,
            training_config=training_config,
            training_config_list=training_config_list,
            warm_start_config=warm_start_config,
            tags=format_tags(tags),
            autotune=autotune,
        )

        def submit(request):
            logger.info("Creating hyperparameter tuning job with name: %s", job_name)
            logger.debug("tune request: %s", json.dumps(request, indent=4))
            self.sagemaker_client.create_hyper_parameter_tuning_job(**request)

        self._intercept_create_request(tune_request, submit, self.create_tuning_job.__name__)

    def _get_tuning_request(
        self,
        job_name,
        tuning_config,
        training_config=None,
        training_config_list=None,
        warm_start_config=None,
        tags=None,
        autotune=False,
    ):
        """Construct CreateHyperParameterTuningJob request

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
            tags (Optional[Tags]): List of tags for labeling the tuning job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            autotune (bool): Whether the parameter ranges or other unset settings of a tuning job
                should be chosen automatically.
        Returns:
            dict: A dictionary for CreateHyperParameterTuningJob request
        """
        tune_request = {
            "HyperParameterTuningJobName": job_name,
            "HyperParameterTuningJobConfig": self._map_tuning_config(**tuning_config),
        }
        if autotune:
            tune_request["Autotune"] = {"Mode": "Enabled"}

        if training_config is not None:
            tune_request["TrainingJobDefinition"] = self._map_training_config(**training_config)

        if training_config_list is not None:
            tune_request["TrainingJobDefinitions"] = [
                self._map_training_config(**training_cfg) for training_cfg in training_config_list
            ]

        if warm_start_config is not None:
            tune_request["WarmStartConfig"] = warm_start_config

        tags = _append_project_tags(format_tags(tags))
        if tags is not None:
            tune_request["Tags"] = tags

        return tune_request

    def describe_tuning_job(self, job_name):
        """Calls DescribeHyperParameterTuningJob API for the given job name, returns the response.

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
        max_runtime_in_seconds=None,
        early_stopping_type="Off",
        objective_type=None,
        objective_metric_name=None,
        parameter_ranges=None,
        random_seed=None,
        strategy_config=None,
        completion_criteria_config=None,
        auto_parameters=None,
    ):
        """Construct tuning job configuration dictionary.

        Args:
            strategy (str): Strategy to be used for hyperparameter estimations.
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
                tuning job.
            max_parallel_jobs (int): Maximum number of parallel training jobs to start.
            max_runtime_in_seconds (int or PipelineVariable): The maximum time in seconds
                that a training job launched by a hyperparameter tuning job can run.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off'. If set to 'Off', early stopping will not be
                attempted. If set to 'Auto', early stopping of some training jobs may happen,
                but is not guaranteed to.
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize'.
            objective_metric_name (str): Name of the metric for evaluating training jobs.
            parameter_ranges (dict): Dictionary of parameter ranges. These parameter ranges can
                be one of three types: Continuous, Integer, or Categorical.
            random_seed (int): An initial value used to initialize a pseudo-random number generator.
                Setting a random seed will make the hyperparameter tuning search strategies to
                produce more consistent configurations for the same tuning job.
            strategy_config (dict): A configuration for the hyperparameter tuning job optimisation
                strategy.
            completion_criteria_config (dict): A configuration
                for the completion criteria.
            auto_parameters (dict): Dictionary of auto parameters. The keys are names of auto
                parameters and valeus are example values of auto parameters.

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

        if max_runtime_in_seconds is not None:
            tuning_config["ResourceLimits"]["MaxRuntimeInSeconds"] = max_runtime_in_seconds

        if random_seed is not None:
            tuning_config["RandomSeed"] = random_seed

        tuning_objective = cls._map_tuning_objective(objective_type, objective_metric_name)
        if tuning_objective is not None:
            tuning_config["HyperParameterTuningJobObjective"] = tuning_objective

        if parameter_ranges is not None:
            tuning_config["ParameterRanges"] = parameter_ranges

        if auto_parameters is not None:
            if parameter_ranges is None:
                tuning_config["ParameterRanges"] = {}
            tuning_config["ParameterRanges"]["AutoParameters"] = [
                {"Name": name, "ValueHint": value} for name, value in auto_parameters.items()
            ]

        if strategy_config is not None:
            tuning_config["StrategyConfig"] = strategy_config

        if completion_criteria_config is not None:
            tuning_config["TuningJobCompletionCriteria"] = completion_criteria_config
        return tuning_config

    @classmethod
    def _map_tuning_objective(cls, objective_type, objective_metric_name):
        """Construct a dictionary of tuning objective from the arguments.

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
        stop_condition,
        input_config=None,
        resource_config=None,
        hpo_resource_config=None,
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
        max_retry_attempts=None,
        environment=None,
        auto_parameters=None,
    ):
        """Construct a dictionary of training job configuration from the arguments.

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
                * 'FastFile' - Amazon SageMaker streams data from S3 on demand instead of
                    downloading the entire dataset before training begins.
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
            max_retry_attempts (int): The number of times to retry the job.
            environment (dict[str, str]) : Environment variables to be set for
                use during training jobs (default: ``None``)

        Returns:
            A dictionary of training job configuration. For format details, please refer to
            TrainingJobDefinition as described in
            https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job
        """
        if hpo_resource_config is not None:
            resource_config_map = {"HyperParameterTuningResourceConfig": hpo_resource_config}
        else:
            resource_config_map = {"ResourceConfig": resource_config}

        training_job_definition = {
            "StaticHyperParameters": static_hyperparameters,
            "RoleArn": role,
            "OutputDataConfig": output_config,
            "StoppingCondition": stop_condition,
            **resource_config_map,
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
            training_job_definition["EnableNetworkIsolation"] = enable_network_isolation

        if encrypt_inter_container_traffic:
            training_job_definition["EnableInterContainerTrafficEncryption"] = (
                encrypt_inter_container_traffic
            )

        if use_spot_instances:
            # use_spot_instances may be a Pipeline ParameterBoolean object
            # which is parsed during the Pipeline execution runtime
            training_job_definition["EnableManagedSpotTraining"] = use_spot_instances

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

        if auto_parameters is not None:
            if parameter_ranges is None:
                training_job_definition["HyperParameterRanges"] = {}
            training_job_definition["HyperParameterRanges"]["AutoParameters"] = [
                {"Name": name, "ValueHint": value} for name, value in auto_parameters.items()
            ]

        if max_retry_attempts is not None:
            training_job_definition["RetryStrategy"] = {"MaximumRetryAttempts": max_retry_attempts}

        if environment is not None:
            training_job_definition["Environment"] = environment
        return training_job_definition

    def stop_tuning_job(self, name):
        """Stop the Amazon SageMaker hyperparameter tuning job with the specified name.

        Args:
            name (str): Name of the Amazon SageMaker hyperparameter tuning job.

        Raises:
            ClientError: If an error occurs while trying to stop the hyperparameter tuning job.
        """
        try:
            logger.info("Stopping tuning job: %s", name)
            self.sagemaker_client.stop_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            # allow to pass if the job already stopped
            if error_code == "ValidationException":
                logger.info("Tuning job: %s is already stopped or not running.", name)
            else:
                logger.error(
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
        batch_data_capture_config: BatchDataCaptureConfig = None,
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
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
            tags (list[dict]): List of tags for labeling a transform job.
            data_processing(dict): A dictionary describing config for combining the input data and
                transformed data. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            model_client_config (dict): A dictionary describing the model configuration for the
                job. Dictionary contains two optional keys,
                'InvocationsTimeoutInSeconds', and 'InvocationsMaxRetries'.
            batch_data_capture_config (BatchDataCaptureConfig): Configuration object which
                specifies the configurations related to the batch data capture for the transform job
                (default: None)

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

        if batch_data_capture_config is not None:
            transform_request["DataCaptureConfig"] = batch_data_capture_config._to_request_dict()

        return transform_request

    def transform(
        self,
        job_name,
        model_name,
        strategy,
        max_concurrent_transforms,
        max_payload,
        input_config,
        output_config,
        resource_config,
        experiment_config,
        env: Optional[Dict[str, str]] = None,
        tags=None,
        data_processing=None,
        model_client_config=None,
        batch_data_capture_config: BatchDataCaptureConfig = None,
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
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
            tags (Optional[Tags]): List of tags for labeling a transform job.
            data_processing(dict): A dictionary describing config for combining the input data and
                transformed data. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            model_client_config (dict): A dictionary describing the model configuration for the
                job. Dictionary contains two optional keys,
                'InvocationsTimeoutInSeconds', and 'InvocationsMaxRetries'.
            batch_data_capture_config (BatchDataCaptureConfig): Configuration object which
                specifies the configurations related to the batch data capture for the transform job
        """
        tags = _append_project_tags(format_tags(tags))
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, TRANSFORM_JOB, TAGS)
        )
        batch_data_capture_config = resolve_class_attribute_from_config(
            None,
            batch_data_capture_config,
            "kms_key_id",
            TRANSFORM_JOB_KMS_KEY_ID_PATH,
            sagemaker_session=self,
        )
        output_config = resolve_nested_dict_value_from_config(
            output_config, [KMS_KEY_ID], TRANSFORM_OUTPUT_KMS_KEY_ID_PATH, sagemaker_session=self
        )
        resource_config = resolve_nested_dict_value_from_config(
            resource_config,
            [VOLUME_KMS_KEY_ID],
            TRANSFORM_JOB_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self,
        )
        env = resolve_value_from_config(
            direct_input=env,
            config_path=TRANSFORM_JOB_ENVIRONMENT_PATH,
            default_value=None,
            sagemaker_session=self,
        )

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
            batch_data_capture_config=batch_data_capture_config,
        )

        def submit(request):
            logger.info("Creating transform job with name: %s", job_name)
            logger.debug("Transform request: %s", json.dumps(request, indent=4))
            self.sagemaker_client.create_transform_job(**request)

        self._intercept_create_request(transform_request, submit, self.transform.__name__)

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

    def create_model_from_job(
        self,
        training_job_name,
        name=None,
        role=None,
        image_uri=None,
        model_data_url=None,
        env=None,
        enable_network_isolation=None,
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
            enable_network_isolation (bool): Whether the model requires network isolation or not.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
                model.
                Default: use VpcConfig from training job.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            tags(Optional[Tags]): Optional. The list of tags to add to the model.
                For more, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

        Returns:
            str: The name of the created ``Model``.
        """
        training_job = self.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        name = name or training_job_name
        role = role or training_job["RoleArn"]
        role = resolve_value_from_config(
            role, MODEL_EXECUTION_ROLE_ARN_PATH, training_job["RoleArn"], self
        )
        enable_network_isolation = resolve_value_from_config(
            direct_input=enable_network_isolation,
            config_path=MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            default_value=False,
            sagemaker_session=self,
        )
        env = resolve_value_from_config(
            env,
            MODEL_PRIMARY_CONTAINER_ENVIRONMENT_PATH,
            default_value={},
            sagemaker_session=self,
        )
        primary_container = container_def(
            image_uri or training_job["AlgorithmSpecification"]["TrainingImage"],
            model_data_url=model_data_url or self._gen_s3_model_data_source(training_job),
            env=env,
        )
        vpc_config = _vpc_config_from_training_job(training_job, vpc_config_override)
        vpc_config = resolve_value_from_config(
            vpc_config, MODEL_VPC_CONFIG_PATH, sagemaker_session=self
        )
        return self.create_model(
            name,
            role,
            primary_container,
            enable_network_isolation=enable_network_isolation,
            vpc_config=vpc_config,
            tags=format_tags(tags),
        )

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

    def create_model_package_from_containers(
        self,
        containers=None,
        content_types=None,
        response_types=None,
        inference_instances=None,
        transform_instances=None,
        model_package_name=None,
        model_package_group_name=None,
        model_metrics=None,
        metadata_properties=None,
        marketplace_cert=False,
        approval_status="PendingManualApproval",
        description=None,
        drift_check_baselines=None,
        customer_metadata_properties=None,
        validation_specification=None,
        domain=None,
        sample_payload_url=None,
        task=None,
        skip_model_validation="None",
        source_uri=None,
        model_card=None,
    ):
        """Get request dictionary for CreateModelPackage API.

        Args:
            containers (list): A list of inference containers that can be used for inference
                specifications of Model Package (default: None).
            content_types (list): The supported MIME types for the input data (default: None).
            response_types (list): The supported MIME types for the output data (default: None).
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time (default: None).
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed (default: None).
            model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
                using `model_package_name` makes the Model Package un-versioned (default: None).
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object (default: None)
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace (default: False).
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval" (default: "PendingManualApproval").
            description (str): Model Package description (default: None).
            drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
            customer_metadata_properties (dict[str, str]): A dictionary of key-value paired
                metadata properties (default: None).
            domain (str): Domain values can be "COMPUTER_VISION", "NATURAL_LANGUAGE_PROCESSING",
                "MACHINE_LEARNING" (default: None).
            sample_payload_url (str): The S3 path where the sample payload is stored
                (default: None).
            task (str): Task values which are supported by Inference Recommender are "FILL_MASK",
                "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION", "IMAGE_SEGMENTATION",
                "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
            skip_model_validation (str): Indicates if you want to skip model validation.
                Values can be "All" or "None" (default: None).
            source_uri (str): The URI of the source for the model package (default: None).
            model_card (ModeCard or ModelPackageModelCard): document contains qualitative and
                quantitative information about a model (default: None).
        """
        if containers:
            # Containers are provided. Now we can merge missing entries from config.
            # If Containers are not provided, it is safe to ignore. This is because,
            # if this object is provided to the API, then Image is required for Containers.
            # That is not supported by the config now. So if we merge values from config,
            # then API will throw an exception. In the future, when SageMaker Config starts
            # supporting other parameters we can add that.
            update_list_of_dicts_with_values_from_config(
                containers,
                MODEL_PACKAGE_INFERENCE_SPECIFICATION_CONTAINERS_PATH,
                required_key_paths=["Image"],
                sagemaker_session=self,
            )

        if validation_specification:
            # ValidationSpecification is provided. Now we can merge missing entries from config.
            # If ValidationSpecification is not provided, it is safe to ignore. This is because,
            # if this object is provided to the API, then both ValidationProfiles and ValidationRole
            # are required and for ValidationProfile, ProfileName is a required parameter. That is
            # not supported by the config now. So if we merge values from config, then API will
            # throw an exception. In the future, when SageMaker Config starts supporting other
            # parameters we can add that.
            validation_role = resolve_value_from_config(
                validation_specification.get(VALIDATION_ROLE, None),
                MODEL_PACKAGE_VALIDATION_ROLE_PATH,
                sagemaker_session=self,
            )
            validation_specification[VALIDATION_ROLE] = validation_role
            validation_profiles = validation_specification.get(VALIDATION_PROFILES, [])
            update_list_of_dicts_with_values_from_config(
                validation_profiles,
                MODEL_PACKAGE_VALIDATION_PROFILES_PATH,
                required_key_paths=["ProfileName", "TransformJobDefinition"],
                sagemaker_session=self,
            )
        model_pkg_request = get_create_model_package_request(
            model_package_name,
            model_package_group_name,
            containers,
            content_types,
            response_types,
            inference_instances,
            transform_instances,
            model_metrics,
            metadata_properties,
            marketplace_cert,
            approval_status,
            description,
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties=customer_metadata_properties,
            validation_specification=validation_specification,
            domain=domain,
            sample_payload_url=sample_payload_url,
            task=task,
            skip_model_validation=skip_model_validation,
            source_uri=source_uri,
            model_card=model_card,
        )

        def submit(request):
            if model_package_group_name is not None and not model_package_group_name.startswith(
                "arn:"
            ):
                _create_resource(
                    lambda: self.sagemaker_client.create_model_package_group(
                        ModelPackageGroupName=request["ModelPackageGroupName"]
                    )
                )
            if "SourceUri" in request and request["SourceUri"] is not None:
                # Remove inference spec from request if the
                # given source uri can lead to auto-population of it
                if can_model_package_source_uri_autopopulate(request["SourceUri"]):
                    if "InferenceSpecification" in request:
                        del request["InferenceSpecification"]
                    return self.sagemaker_client.create_model_package(**request)
                # If source uri can't autopopulate,
                # first create model package with just the inference spec
                # and then update model package with the source uri.
                # Done this way because passing source uri and inference spec together
                # in create/update model package is not allowed in the base sdk.
                request_source_uri = request["SourceUri"]
                del request["SourceUri"]
                model_package = self.sagemaker_client.create_model_package(**request)
                update_source_uri_args = {
                    "ModelPackageArn": model_package.get("ModelPackageArn"),
                    "SourceUri": request_source_uri,
                }
                return self.sagemaker_client.update_model_package(**update_source_uri_args)
            return self.sagemaker_client.create_model_package(**request)

        return self._intercept_create_request(
            model_pkg_request, submit, self.create_model_package_from_containers.__name__
        )

    def wait_for_model_package(self, model_package_name, poll=5):
        """Wait for an Amazon SageMaker endpoint deployment to complete.

        Args:
            endpoint (str): Name of the ``Endpoint`` to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            dict: Return value from the ``DescribeEndpoint`` API.

        Raises:
            exceptions.CapacityError: If the Model Package job fails with CapacityError.
            exceptions.UnexpectedStatusException: If waiting and the Model Package job fails.
        """
        desc = _wait_until(
            lambda: _create_model_package_status(self.sagemaker_client, model_package_name), poll
        )
        status = desc["ModelPackageStatus"]

        if status != "Completed":
            reason = desc.get("FailureReason", None)
            message = "Error creating model package {package}: {status} Reason: {reason}".format(
                package=model_package_name, status=status, reason=reason
            )
            if "CapacityError" in str(reason):
                raise exceptions.CapacityError(
                    message=message,
                    allowed_statuses=["InService"],
                    actual_status=status,
                )
            raise exceptions.UnexpectedStatusException(
                message=message,
                allowed_statuses=["Completed"],
                actual_status=status,
            )
        return desc

    def describe_model(self, name):
        """Calls the DescribeModel API for the given model name.

        Args:
            name (str): The name of the SageMaker model.

        Returns:
            dict: A dictionary response with the model description.
        """
        return self.sagemaker_client.describe_model(ModelName=name)

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
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        explainer_config_dict=None,
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
            tags(Optional[Tags]): Optional. The list of tags to add to the endpoint config.
            kms_key (str): The KMS key that is used to encrypt the data on the storage volume
                attached to the instance hosting the endpoint.
            data_capture_config_dict (dict): Specifies configuration related to Endpoint data
                capture for use with Amazon SageMaker Model Monitoring. Default: None.
            volume_size (int): The size, in GB, of the ML storage volume attached to individual
                inference instance associated with the production variant. Currenly only Amazon EBS
                gp2 storage volumes are supported.
            model_data_download_timeout (int): The timeout value, in seconds, to download and
                extract model data from Amazon S3 to the individual inference instance associated
                with this production variant.
            container_startup_health_check_timeout (int): The timeout value, in seconds, for your
                inference container to pass health check by SageMaker Hosting. For more information
                about health check see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code
                .html#your-algorithms
                -inference-algo-ping-requests
            explainer_config_dict (dict): Specifies configuration to enable explainers.
                Default: None.

        Example:
            >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
            For more information about tags, see
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker
            .html#SageMaker
            .Client.add_tags

        Returns:
            str: Name of the endpoint point configuration created.
        """
        logger.info("Creating endpoint-config with name %s", name)

        tags = format_tags(tags) or []
        provided_production_variant = production_variant(
            model_name,
            instance_type,
            initial_instance_count,
            accelerator_type=accelerator_type,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
        )
        production_variants = [provided_production_variant]
        # Currently we just inject CoreDumpConfig.KmsKeyId from the config for production variant.
        # But if that parameter is injected, then CoreDumpConfig.DestinationS3Uri needs to be
        # present.
        # But SageMaker Python SDK doesn't support CoreDumpConfig.DestinationS3Uri.
        update_list_of_dicts_with_values_from_config(
            production_variants,
            ENDPOINT_CONFIG_PRODUCTION_VARIANTS_PATH,
            required_key_paths=["CoreDumpConfig.DestinationS3Uri"],
            sagemaker_session=self,
        )
        request = {
            "EndpointConfigName": name,
            "ProductionVariants": production_variants,
        }

        tags = _append_project_tags(tags)
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, ENDPOINT_CONFIG, TAGS)
        )
        if tags is not None:
            request["Tags"] = tags
        kms_key = (
            resolve_value_from_config(
                kms_key, ENDPOINT_CONFIG_KMS_KEY_ID_PATH, sagemaker_session=self
            )
            if instance_supports_kms(instance_type)
            else kms_key
        )
        if kms_key is not None:
            request["KmsKeyId"] = kms_key

        if data_capture_config_dict is not None:
            inferred_data_capture_config_dict = update_nested_dictionary_with_values_from_config(
                data_capture_config_dict, ENDPOINT_CONFIG_DATA_CAPTURE_PATH, sagemaker_session=self
            )
            request["DataCaptureConfig"] = inferred_data_capture_config_dict

        if explainer_config_dict is not None:
            request["ExplainerConfig"] = explainer_config_dict

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
        new_explainer_config_dict=None,
        endpoint_type=EndpointType.MODEL_BASED,
    ):
        """Create an Amazon SageMaker endpoint configuration from an existing one.

        It also updates any values that were passed in.
        The endpoint configuration identifies the Amazon SageMaker model (created using the
        ``CreateModel`` API) and the hardware configuration on which to deploy the model. Provide
        this endpoint configuration to the ``CreateEndpoint`` API, which then launches the
        hardware and deploys the model.

        Args:
            new_config_name (str): Name of the Amazon SageMaker endpoint configuration to create.
            existing_config_name (str): Name of the existing Amazon SageMaker endpoint
                configuration.
            new_tags (Optional[Tags]): Optional. The list of tags to add to the endpoint
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
            new_explainer_config_dict (dict): Specifies configuration to enable explainers.
                (default: None). If not specified, the explainer configuration of the existing
                endpoint configuration is used.
            endpoint_type (EndpointType): The Endpoint type to determine whether model will be
                deployed as Amazon SageMaker Inference Component to the endpoint

        Returns:
            str: Name of the endpoint point configuration created.
        """
        logger.info("Creating endpoint-config with name %s", new_config_name)

        existing_endpoint_config_desc = self.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=existing_config_name
        )

        request = {
            "EndpointConfigName": new_config_name,
        }

        production_variants = (
            new_production_variants or existing_endpoint_config_desc["ProductionVariants"]
        )
        if endpoint_type == EndpointType.INFERENCE_COMPONENT_BASED:
            # Make a copy of Production variants and remove the InitialVariantWeight
            # in the copy
            copy_production_variants = deepcopy(production_variants)
            for prod_var in copy_production_variants:
                if "InitialVariantWeight" in prod_var:
                    del prod_var["InitialVariantWeight"]
            # assign production_variants with copy_production_variants,
            # so it will not have InitialVariantWeight and the new_production_variants
            # will not be mutated
            production_variants = copy_production_variants

        update_list_of_dicts_with_values_from_config(
            production_variants,
            ENDPOINT_CONFIG_PRODUCTION_VARIANTS_PATH,
            required_key_paths=["CoreDumpConfig.DestinationS3Uri"],
            sagemaker_session=self,
        )
        request["ProductionVariants"] = production_variants

        for pv in production_variants:
            if "ModelName" not in pv or not pv["ModelName"]:
                request["ExecutionRoleArn"] = self.get_caller_identity_arn()

        request_tags = format_tags(new_tags) or self.list_tags(
            existing_endpoint_config_desc["EndpointConfigArn"]
        )
        request_tags = _append_project_tags(request_tags)
        request_tags = self._append_sagemaker_config_tags(
            request_tags, "{}.{}.{}".format(SAGEMAKER, ENDPOINT_CONFIG, TAGS)
        )
        if request_tags:
            request["Tags"] = request_tags

        if new_kms_key is not None or existing_endpoint_config_desc.get("KmsKeyId") is not None:
            request["KmsKeyId"] = new_kms_key or existing_endpoint_config_desc.get("KmsKeyId")

        supports_kms = any(
            [
                instance_supports_kms(production_variant["InstanceType"])
                for production_variant in production_variants
                if "InstanceType" in production_variant
            ]
        )

        if KMS_KEY_ID not in request and supports_kms:
            kms_key_from_config = resolve_value_from_config(
                config_path=ENDPOINT_CONFIG_KMS_KEY_ID_PATH, sagemaker_session=self
            )
            if kms_key_from_config:
                request[KMS_KEY_ID] = kms_key_from_config

        request_data_capture_config_dict = (
            new_data_capture_config_dict or existing_endpoint_config_desc.get("DataCaptureConfig")
        )

        if request_data_capture_config_dict is not None:
            inferred_data_capture_config_dict = update_nested_dictionary_with_values_from_config(
                request_data_capture_config_dict,
                ENDPOINT_CONFIG_DATA_CAPTURE_PATH,
                sagemaker_session=self,
            )
            request["DataCaptureConfig"] = inferred_data_capture_config_dict

        async_inference_config_dict = existing_endpoint_config_desc.get(
            "AsyncInferenceConfig", None
        )
        if async_inference_config_dict is not None:
            inferred_async_inference_config_dict = update_nested_dictionary_with_values_from_config(
                async_inference_config_dict,
                ENDPOINT_CONFIG_ASYNC_INFERENCE_PATH,
                sagemaker_session=self,
            )
            request["AsyncInferenceConfig"] = inferred_async_inference_config_dict

        request_explainer_config_dict = (
            new_explainer_config_dict or existing_endpoint_config_desc.get("ExplainerConfig", None)
        )

        if request_explainer_config_dict is not None:
            request["ExplainerConfig"] = request_explainer_config_dict

        self.sagemaker_client.create_endpoint_config(**request)

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
        """
        logger.info("Creating endpoint with name %s", endpoint_name)

        tags = format_tags(tags) or []
        tags = _append_project_tags(tags)
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, ENDPOINT, TAGS)
        )

        res = self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=tags
        )
        if res:
            self.endpoint_arn = res["EndpointArn"]

        if wait:
            self.wait_for_endpoint(endpoint_name, live_logging=live_logging)
        return endpoint_name

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
            ValueError: if the endpoint does not already exist
        """
        if not _deployment_entity_exists(
            lambda: self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        ):
            raise ValueError(
                "Endpoint with name '{}' does not exist; please use an "
                "existing endpoint name".format(endpoint_name)
            )

        res = self.sagemaker_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        if res:
            self.endpoint_arn = res["EndpointArn"]

        if wait:
            self.wait_for_endpoint(endpoint_name)
        return endpoint_name

    def is_inference_component_based_endpoint(self, endpoint_name):
        """Returns 'True' if endpoint is inference-component-based, 'False' otherwise.

        An endpoint is inference component based if and only if the associated endpoint config
        has a role associated with it and no production variants with a ``ModelName`` field.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to determine
                if inference-component-based.
        """
        describe_endpoint_response = self.describe_endpoint(endpoint_name)
        endpoint_config_name = describe_endpoint_response["EndpointConfigName"]
        describe_endpoint_config_response = self.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        production_variants = describe_endpoint_config_response.get("ProductionVariants", [])
        execution_role_arn = describe_endpoint_config_response.get("ExecutionRoleArn")
        if len(production_variants) == 0:
            return False
        return execution_role_arn is not None and all(
            production_variant.get("ModelName") is None
            for production_variant in production_variants
        )

    def describe_endpoint(self, endpoint_name):
        """Describe an Amazon SageMaker ``Endpoint``.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to describe.
        """
        response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response

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

    def wait_for_inference_component_deletion(self, inference_component_name: str, poll: int = 20):
        """Wait for an Amazon SageMaker ``Inference Component`` deployment to complete.

        Args:
            inference_component_name (str): Name of the inference component to wait for.
            poll (int): Polling interval in seconds (default: 20).

        """
        _wait_until(
            lambda: self._inference_component_deletion_done(
                self.sagemaker_client, inference_component_name
            ),
            poll,
        )

    def _inference_component_deletion_done(self, sagemaker_client, inference_component_name: str):
        """Check if deletion of inference component is done.

        Args:
            sagemaker_client (boto3.SageMaker.Client): Client which makes Amazon SageMaker
                service calls
            inference_component_name (str): Name of the Amazon SageMaker inference component.
        Returns:
            bool: True if deletion is done. None otherwise.
        """
        try:
            sagemaker_client.describe_inference_component(
                InferenceComponentName=inference_component_name
            )
        except botocore.exceptions.ClientError as e:
            str_err = str(e).lower()
            if "could not find" in str_err or "not found" in str_err:
                return True
            raise
        return None

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

    def delete_inference_component(self, inference_component_name: str, wait: bool = False):
        """Deletes a InferenceComponent.

        Args:
            inference_component_name (str): Name of the Amazon SageMaker ``InferenceComponent``
                to delete.
            wait (bool): wait delete inference component operation finish. (default: False)

        Return:
            None
        """

        LOGGER.info("Deleting inference component with name: %s", inference_component_name)
        self.sagemaker_client.delete_inference_component(
            InferenceComponentName=inference_component_name
        )
        if wait:
            _wait_until(
                lambda: self._inference_component_deletion_done(
                    self.sagemaker_client, inference_component_name
                ),
                poll=20,
            )

    def list_and_paginate_inference_component_names_associated_with_endpoint(
        self,
        endpoint_name: str,
    ) -> List[str]:
        """List inference component names for an endpoint, concatenating paginated results.

        Args:
            endpoint_name (str): A string that matches the name of the
                endpoint to which the inference components are deployed.
        """
        inference_component_names: List[str] = []
        next_token: Optional[str] = None
        first_iteration: bool = True

        while first_iteration or next_token:
            first_iteration = False
            list_inference_components_response = self.list_inference_components(
                endpoint_name_equals=endpoint_name, next_token=next_token
            )
            inference_component_names.extend(
                [
                    ic["InferenceComponentName"]
                    for ic in list_inference_components_response["InferenceComponents"]
                ]
            )
            next_token = list_inference_components_response.get("NextToken")

        return sorted(inference_component_names)

    def list_inference_components(
        self,
        endpoint_name_equals: Optional[str] = None,
        variant_name_equals: Optional[str] = None,
        name_contains: Optional[str] = None,
        creation_time_after: Optional[datetime.datetime] = None,
        creation_time_before: Optional[datetime.datetime] = None,
        last_modified_time_after: Optional[datetime.datetime] = None,
        last_modified_time_before: Optional[datetime.datetime] = None,
        status_equals: Optional[str] = None,
        sort_order: Optional[str] = None,
        sort_by: Optional[str] = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List inference components under current endpoint.

        Args:
            endpoint_name_equals (str): Optional. A string that matches the name of the
                endpoint that the inference components are deployed to. (Default: None).
            variant_name_equals (str): Optional. A string that matches the name of the variant
                associated with the inference components. (Default: None).
            name_contains (str): Optional. A string that partially matches the names of one or
                more inference components. Filters inference components by name.
                (Default: None).
            creation_time_after (datetime.datetime): Optional. Use this parameter to
                search for inference components that were created after a specific date and time.
                (Default: None).
            creation_time_before (datetime.datetime): Optional. Use this parameter to
                search for inference components that were created before a specific date and time.
                (Default: None).
            last_modified_time_after (datetime.datetime): Optional. Use this parameter to
                search for inference components that were last modified after a specific date
                and time. (Default: None).
            last_modified_time_before (datetime.datetime): Optional. Use this parameter to
                search for inference components that were last modified before a specific date
                and time. (Default: None).
            status_equals (str): Optional. The inference component status to apply as a filter
                to the response. (Default: None).
            sort_order (str): Optional. The order in which inference components are listed in the
                response. (Default: None).
            sort_by (str): Optional. The value on which the inference component list is
                sorted. (Default: None).
            max_results (int): Optional. The maximum number of results returned by
                list_inference_components. (Default: None).
            next_token (str): Optional. A token to resume pagination of list_inference_components
                results. (Default: None).

        Return:
            [dict[str,Any]]: Dict of inference component details.
        """

        list_inference_component_args = {}

        def check_object(key, value):
            if value is not None:
                list_inference_component_args[key] = value

        check_object("SortBy", sort_by)
        check_object("SortOrder", sort_order)
        check_object("NameContains", name_contains)
        check_object("CreationTimeBefore", creation_time_before)
        check_object("CreationTimeAfter", creation_time_after)
        check_object("LastModifiedTimeBefore", last_modified_time_before)
        check_object("LastModifiedTimeAfter", last_modified_time_after)
        check_object("StatusEquals", status_equals)
        check_object("EndpointNameEquals", endpoint_name_equals)
        check_object("VariantNameEquals", variant_name_equals)
        check_object("NextToken", next_token)
        check_object("MaxResults", max_results)

        return self.sagemaker_client.list_inference_components(**list_inference_component_args)

    def delete_model(self, model_name):
        """Delete an Amazon SageMaker Model.

        Args:
            model_name (str): Name of the Amazon SageMaker model to delete.
        """
        logger.info("Deleting model with name: %s", model_name)
        self.sagemaker_client.delete_model(ModelName=model_name)

    def list_group_resources(self, group, filters, next_token: str = ""):
        """To list group resources with given filters

        Args:
            group (str): The name or the ARN of the group.
            filters (list): Filters that needs to be applied to the list operation.
        """
        self.resource_groups_client = self.resource_groups_client or self.boto_session.client(
            "resource-groups"
        )
        return self.resource_groups_client.list_group_resources(
            Group=group, Filters=filters, NextToken=next_token
        )

    def delete_resource_group(self, group):
        """To delete a resource group

        Args:
            group (str): The name or the ARN of the resource group to delete.
        """
        self.resource_groups_client = self.resource_groups_client or self.boto_session.client(
            "resource-groups"
        )
        return self.resource_groups_client.delete_group(Group=group)

    def get_resource_group_query(self, group):
        """To get the group query for an AWS Resource Group

        Args:
            group (str): The name or the ARN of the resource group to query.
        """
        self.resource_groups_client = self.resource_groups_client or self.boto_session.client(
            "resource-groups"
        )
        return self.resource_groups_client.get_group_query(Group=group)

    def get_tagging_resources(self, tag_filters, resource_type_filters):
        """To list the complete resources for a particular resource group tag

        tag_filters: filters for the tag
        resource_type_filters: resource filter for the tag
        """
        self.resource_group_tagging_client = (
            self.resource_group_tagging_client
            or self.boto_session.client("resourcegroupstaggingapi")
        )
        resource_list = []

        try:
            resource_tag_response = self.resource_group_tagging_client.get_resources(
                TagFilters=tag_filters, ResourceTypeFilters=resource_type_filters
            )

            resource_list = resource_list + resource_tag_response["ResourceTagMappingList"]

            next_token = resource_tag_response.get("PaginationToken")
            while next_token is not None and next_token != "":
                resource_tag_response = self.resource_group_tagging_client.get_resources(
                    TagFilters=tag_filters,
                    ResourceTypeFilters=resource_type_filters,
                    NextToken=next_token,
                )
                resource_list = resource_list + resource_tag_response["ResourceTagMappingList"]
                next_token = resource_tag_response.get("PaginationToken")

            return resource_list
        except ClientError as error:
            raise error

    def create_group(self, name, resource_query, tags):
        """To create a AWS Resource Group

        Args:
            name (str): The name of the group, which is also the identifier of the group.
            resource_query (str): The resource query that determines
                which AWS resources are members of this group
            tags (dict): The Tags to be attached to the Resource Group
        """
        self.resource_groups_client = self.resource_groups_client or self.boto_session.client(
            "resource-groups"
        )

        return self.resource_groups_client.create_group(
            Name=name, ResourceQuery=resource_query, Tags=tags
        )

    def list_tags(self, resource_arn, max_results=50):
        """List the tags given an Amazon Resource Name.

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
            logger.error("Error retrieving tags. resource_arn: %s", resource_arn)
            raise error

    def wait_for_job(self, job, poll=5):
        """Wait for an Amazon SageMaker training job to complete.

        Args:
            job (str): Name of the training job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeTrainingJob`` API.

        Raises:
            exceptions.CapacityError: If the training job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the training job fails.
        """
        desc = _wait_until_training_done(
            lambda last_desc: _train_done(self.sagemaker_client, job, last_desc), None, poll
        )
        _check_job_status(job, desc, "TrainingJobStatus")
        return desc

    def wait_for_processing_job(self, job, poll=5):
        """Wait for an Amazon SageMaker Processing job to complete.

        Args:
            job (str): Name of the processing job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeProcessingJob`` API.

        Raises:
            exceptions.CapacityError: If the processing job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the processing job fails.
        """
        desc = _wait_until(lambda: _processing_job_status(self.sagemaker_client, job), poll)
        _check_job_status(job, desc, "ProcessingJobStatus")
        return desc

    def wait_for_compilation_job(self, job, poll=5):
        """Wait for an Amazon SageMaker Neo compilation job to complete.

        Args:
            job (str): Name of the compilation job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeCompilationJob`` API.

        Raises:
            exceptions.CapacityError: If the compilation job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the compilation job fails.
        """
        desc = _wait_until(lambda: _compilation_job_status(self.sagemaker_client, job), poll)
        _check_job_status(job, desc, "CompilationJobStatus")
        return desc

    def wait_for_edge_packaging_job(self, job, poll=5):
        """Wait for an Amazon SageMaker Edge packaging job to complete.

        Args:
            job (str): Name of the edge packaging job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeEdgePackagingJob`` API.

        Raises:
            exceptions.CapacityError: If the edge packaging job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the edge packaging job fails.
        """
        desc = _wait_until(lambda: _edge_packaging_job_status(self.sagemaker_client, job), poll)
        _check_job_status(job, desc, "EdgePackagingJobStatus")
        return desc

    def wait_for_tuning_job(self, job, poll=5):
        """Wait for an Amazon SageMaker hyperparameter tuning job to complete.

        Args:
            job (str): Name of the tuning job to wait for.
            poll (int): Polling interval in seconds (default: 5).

        Returns:
            (dict): Return value from the ``DescribeHyperParameterTuningJob`` API.

        Raises:
            exceptions.CapacityError: If the hyperparameter tuning job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the hyperparameter tuning job fails.
        """
        desc = _wait_until(lambda: _tuning_job_status(self.sagemaker_client, job), poll)
        _check_job_status(job, desc, "HyperParameterTuningJobStatus")
        return desc

    def describe_transform_job(self, job_name):
        """Calls the DescribeTransformJob API for the given job name and returns the response.

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
            exceptions.CapacityError: If the transform job fails with CapacityError.
            exceptions.UnexpectedStatusException: If the transform job fails.
        """
        desc = _wait_until(lambda: _transform_job_status(self.sagemaker_client, job), poll)
        _check_job_status(job, desc, "TransformJobStatus")
        return desc

    def stop_transform_job(self, name):
        """Stop the Amazon SageMaker hyperparameter tuning job with the specified name.

        Args:
            name (str): Name of the Amazon SageMaker batch transform job.

        Raises:
            ClientError: If an error occurs while trying to stop the batch transform job.
        """
        try:
            logger.info("Stopping transform job: %s", name)
            self.sagemaker_client.stop_transform_job(TransformJobName=name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            # allow to pass if the job already stopped
            if error_code == "ValidationException":
                logger.info("Transform job: %s is already stopped or not running.", name)
            else:
                logger.error("Error occurred while attempting to stop transform job: %s.", name)
                raise

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
        model_s3_location = self._gen_s3_model_data_source(job_desc)
        image_uri = image_uri or job_desc["AlgorithmSpecification"]["TrainingImage"]
        role = role or job_desc["RoleArn"]
        name = name or job_name
        vpc_config_override = _vpc_config_from_training_job(job_desc, vpc_config_override)

        return self.endpoint_from_model_data(
            model_s3_location=model_s3_location,
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

    def _gen_s3_model_data_source(self, training_job_spec):
        """Generates ``ModelDataSource`` value from given DescribeTrainingJob API response.

        Args:
            training_job_spec (dict): SageMaker DescribeTrainingJob API response.

        Returns:
            dict: A ``ModelDataSource`` value.
        """
        model_data_s3_uri = training_job_spec["ModelArtifacts"]["S3ModelArtifacts"]
        compression_type = training_job_spec.get("OutputDataConfig", {}).get(
            "CompressionType", "GZIP"
        )
        # See https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_OutputDataConfig.html
        # and https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_S3ModelDataSource.html
        if compression_type in {"NONE", "GZIP"}:
            model_compression_type = compression_type.title()
        else:
            raise ValueError(
                f'Unrecognized training job output data compression type "{compression_type}"'
            )
        s3_model_data_type = "S3Object" if model_compression_type == "Gzip" else "S3Prefix"
        # if model data is in S3Prefix type and has no trailing forward slash in its URI,
        # append one so that it meets SageMaker Hosting's mandate for deploying uncompressed model.
        if s3_model_data_type == "S3Prefix" and not model_data_s3_uri.endswith("/"):
            model_data_s3_uri += "/"
        return {
            "S3DataSource": {
                "S3Uri": model_data_s3_uri,
                "S3DataType": s3_model_data_type,
                "CompressionType": model_compression_type,
            }
        }

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
        tags=None,
    ):
        """Create and deploy to an ``Endpoint`` using existing model data stored in S3.

        Args:
            model_s3_location (str or dict): S3 location of the model artifacts
                to use for the endpoint.
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
            tags (Optional[Tags]): A list of key-value pairs for tagging the endpoint
                (default: None).

        Returns:
            str: Name of the ``Endpoint`` that is created.
        """
        model_environment_vars = model_environment_vars or {}
        name = name or name_from_image(image_uri)
        model_vpc_config = vpc_utils.sanitize(model_vpc_config)
        endpoint_config_tags = _append_project_tags(format_tags(tags))
        endpoint_tags = _append_project_tags(format_tags(tags))
        endpoint_config_tags = self._append_sagemaker_config_tags(
            endpoint_config_tags, "{}.{}.{}".format(SAGEMAKER, ENDPOINT_CONFIG, TAGS)
        )
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

        _create_resource(
            lambda: self.create_endpoint_config(
                name=name,
                model_name=name,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                data_capture_config_dict=data_capture_config_dict,
                tags=endpoint_config_tags,
            )
        )

        # to make change backwards compatible
        response = _create_resource(
            lambda: self.create_endpoint(
                endpoint_name=name, config_name=name, tags=endpoint_tags, wait=wait
            )
        )
        if not response:
            raise ValueError(
                'Endpoint with name "{}" already exists; please pick a different name.'.format(name)
            )

        return name

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
                metadata = json.loads(f.read())
                instance_name = metadata.get("ResourceName")
                domain_id = metadata.get("DomainId")
                user_profile_name = metadata.get("UserProfileName")
                execution_role_arn = metadata.get("ExecutionRoleArn")
            try:
                if domain_id is None:
                    instance_desc = self.sagemaker_client.describe_notebook_instance(
                        NotebookInstanceName=instance_name
                    )
                    return instance_desc["RoleArn"]

                # find execution role from the metadata file if present
                if execution_role_arn is not None:
                    return execution_role_arn

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

    def logs_for_job(self, job_name, wait=False, poll=10, log_type="All", timeout=None):
        """Display logs for a given training job, optionally tailing them until job is complete.

        If the output is a tty or a Jupyter cell, it will be color-coded
        based on which instance the log entry is from.

        Args:
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
        Raises:
            exceptions.CapacityError: If the training job fails with CapacityError.
            exceptions.UnexpectedStatusException: If waiting and the training job fails.
        """
        _logs_for_job(self, job_name, wait, poll, log_type, timeout)

    def logs_for_processing_job(self, job_name, wait=False, poll=10):
        """Display logs for a given processing job, optionally tailing them until the is complete.

        Args:
            job_name (str): Name of the processing job to display the logs for.
            wait (bool): Whether to keep looking for new log entries until the job completes
                (default: False).
            poll (int): The interval in seconds between polling for new log entries and job
                completion (default: 5).

        Raises:
            ValueError: If the processing job fails.
        """

        description = _wait_until(lambda: self.describe_processing_job(job_name), poll)

        instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
            self.boto_session, description, job="Processing"
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
            _check_job_status(job_name, description, "ProcessingJobStatus")
            if dot:
                print()

    def logs_for_transform_job(self, job_name, wait=False, poll=10):
        """Display logs for a given training job, optionally tailing them until job is complete.

        If the output is a tty or a Jupyter cell, it will be color-coded
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

        description = _wait_until(lambda: self.describe_transform_job(job_name), poll)

        instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
            self.boto_session, description, job="Transform"
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
            _check_job_status(job_name, description, "TransformJobStatus")
            if dot:
                print()

    def delete_feature_group(self, feature_group_name: str):
        """Deletes a FeatureGroup in the FeatureStore service.

        Args:
            feature_group_name (str): name of the feature group to be deleted.
        """
        self.sagemaker_client.delete_feature_group(FeatureGroupName=feature_group_name)

    def create_feature_group(
        self,
        feature_group_name: str,
        record_identifier_name: str,
        event_time_feature_name: str,
        feature_definitions: Sequence[Dict[str, str]],
        role_arn: str = None,
        online_store_config: Dict[str, str] = None,
        offline_store_config: Dict[str, str] = None,
        throughput_config: Dict[str, Any] = None,
        description: str = None,
        tags: Optional[Tags] = None,
    ) -> Dict[str, Any]:
        """Creates a FeatureGroup in the FeatureStore service.

        Args:
            feature_group_name (str): name of the FeatureGroup.
            record_identifier_name (str): name of the record identifier feature.
            event_time_feature_name (str): name of the event time feature.
            feature_definitions (Sequence[Dict[str, str]]): list of feature definitions.
            role_arn (str): ARN of the role will be used to execute the api.
            online_store_config (Dict[str, str]): dict contains configuration of the
                feature online store.
            offline_store_config (Dict[str, str]): dict contains configuration of the
                feature offline store.
            throughput_config (Dict[str, str]): dict contains throughput configuration
                for the feature group.
            description (str): description of the FeatureGroup.
            tags (Optional[Tags]): tags for labeling a FeatureGroup.

        Returns:
            Response dict from service.
        """
        tags = format_tags(tags)
        tags = _append_project_tags(tags)
        tags = self._append_sagemaker_config_tags(
            tags, "{}.{}.{}".format(SAGEMAKER, FEATURE_GROUP, TAGS)
        )
        role_arn = resolve_value_from_config(
            role_arn, FEATURE_GROUP_ROLE_ARN_PATH, sagemaker_session=self
        )
        inferred_online_store_from_config = update_nested_dictionary_with_values_from_config(
            online_store_config, FEATURE_GROUP_ONLINE_STORE_CONFIG_PATH, sagemaker_session=self
        )
        if inferred_online_store_from_config is not None:
            # OnlineStore should be handled differently because if you set KmsKeyId, then you
            # need to set EnableOnlineStore key as well
            inferred_online_store_from_config["EnableOnlineStore"] = True
        inferred_offline_store_from_config = update_nested_dictionary_with_values_from_config(
            offline_store_config, FEATURE_GROUP_OFFLINE_STORE_CONFIG_PATH, sagemaker_session=self
        )
        kwargs = dict(
            FeatureGroupName=feature_group_name,
            RecordIdentifierFeatureName=record_identifier_name,
            EventTimeFeatureName=event_time_feature_name,
            FeatureDefinitions=feature_definitions,
            RoleArn=role_arn,
        )
        update_args(
            kwargs,
            OnlineStoreConfig=inferred_online_store_from_config,
            OfflineStoreConfig=inferred_offline_store_from_config,
            ThroughputConfig=throughput_config,
            Description=description,
            Tags=tags,
        )
        return self.sagemaker_client.create_feature_group(**kwargs)

    def describe_feature_group(
        self,
        feature_group_name: str,
        next_token: str = None,
    ) -> Dict[str, Any]:
        """Describe a FeatureGroup by name in FeatureStore service.

        Args:
            feature_group_name (str): name of the FeatureGroup to describe.
            next_token (str): next_token to get next page of features.
        Returns:
            Response dict from service.
        """

        kwargs = dict(FeatureGroupName=feature_group_name)
        update_args(kwargs, NextToken=next_token)
        return self.sagemaker_client.describe_feature_group(**kwargs)

    def update_feature_group(
        self,
        feature_group_name: str,
        feature_additions: Sequence[Dict[str, str]] = None,
        online_store_config: Dict[str, any] = None,
        throughput_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Update a FeatureGroup

            Supports modifications like adding new features from the given feature definitions,
            updating online store and throughput configurations.

        Args:
            feature_group_name (str): name of the FeatureGroup to update.
            feature_additions (Sequence[Dict[str, str]): list of feature definitions to be updated.
            online_store_config (Dict[str, Any]): updates to online store config
            throughput_config (Dict[str, Any]): target throughput configuration of the feature group
        Returns:
            Response dict from service.
        """
        update_req = {"FeatureGroupName": feature_group_name}
        if online_store_config is not None:
            update_req["OnlineStoreConfig"] = online_store_config

        if throughput_config is not None:
            update_req["ThroughputConfig"] = throughput_config

        if feature_additions is not None:
            update_req["FeatureAdditions"] = feature_additions

        return self.sagemaker_client.update_feature_group(**update_req)

    def list_feature_groups(
        self,
        name_contains,
        feature_group_status_equals,
        offline_store_status_equals,
        creation_time_after,
        creation_time_before,
        sort_order,
        sort_by,
        max_results,
        next_token,
    ) -> Dict[str, Any]:
        """List all FeatureGroups satisfying given filters.

        Args:
            name_contains (str): A string that partially matches one or more FeatureGroups' names.
                Filters FeatureGroups by name.
            feature_group_status_equals (str): A FeatureGroup status.
                Filters FeatureGroups by FeatureGroup status.
            offline_store_status_equals (str): An OfflineStore status.
                Filters FeatureGroups by OfflineStore status.
            creation_time_after (datetime.datetime): Use this parameter to search for FeatureGroups
                created after a specific date and time.
            creation_time_before (datetime.datetime): Use this parameter to search for FeatureGroups
                created before a specific date and time.
            sort_order (str): The order in which FeatureGroups are listed.
            sort_by (str): The value on which the FeatureGroup list is sorted.
            max_results (int): The maximum number of results returned by ListFeatureGroups.
            next_token (str): A token to resume pagination of ListFeatureGroups results.
        Returns:
            Response dict from service.
        """
        list_feature_groups_args = {}

        def check_object(key, value):
            if value is not None:
                list_feature_groups_args[key] = value

        check_object("NameContains", name_contains)
        check_object("FeatureGroupStatusEquals", feature_group_status_equals)
        check_object("OfflineStoreStatusEquals", offline_store_status_equals)
        check_object("CreationTimeAfter", creation_time_after)
        check_object("CreationTimeBefore", creation_time_before)
        check_object("SortOrder", sort_order)
        check_object("SortBy", sort_by)
        check_object("MaxResults", max_results)
        check_object("NextToken", next_token)

        return self.sagemaker_client.list_feature_groups(**list_feature_groups_args)

    def update_feature_metadata(
        self,
        feature_group_name: str,
        feature_name: str,
        description: str = None,
        parameter_additions: Sequence[Dict[str, str]] = None,
        parameter_removals: Sequence[str] = None,
    ) -> Dict[str, Any]:
        """Update a feature metadata and add/remove metadata.

        Args:
            feature_group_name (str): name of the FeatureGroup to update.
            feature_name (str): name of the feature to update.
            description (str): description of the feature to update.
            parameter_additions (Sequence[Dict[str, str]): list of feature parameter to be added.
            parameter_removals (Sequence[Dict[str, str]): list of feature parameter to be removed.
        Returns:
            Response dict from service.
        """

        request = {
            "FeatureGroupName": feature_group_name,
            "FeatureName": feature_name,
        }

        if description is not None:
            request["Description"] = description
        if parameter_additions is not None:
            request["ParameterAdditions"] = parameter_additions
        if parameter_removals is not None:
            request["ParameterRemovals"] = parameter_removals

        return self.sagemaker_client.update_feature_metadata(**request)

    def describe_feature_metadata(
        self, feature_group_name: str, feature_name: str
    ) -> Dict[str, Any]:
        """Describe feature metadata by feature name in FeatureStore service.

        Args:
            feature_group_name (str): name of the FeatureGroup.
            feature_name (str): name of the feature.
        Returns:
            Response dict from service.
        """

        return self.sagemaker_client.describe_feature_metadata(
            FeatureGroupName=feature_group_name, FeatureName=feature_name
        )

    def search(
        self,
        resource: str,
        search_expression: Dict[str, any] = None,
        sort_by: str = None,
        sort_order: str = None,
        next_token: str = None,
        max_results: int = None,
    ) -> Dict[str, Any]:
        """Search for SageMaker resources satisfying given filters.

        Args:
            resource (str): The name of the Amazon SageMaker resource to search for.
            search_expression (Dict[str, any]): A Boolean conditional statement. Resources must
                satisfy this condition to be included in search results.
            sort_by (str): The name of the resource property used to sort the ``SearchResults``.
                The default is ``LastModifiedTime``.
            sort_order (str): How ``SearchResults`` are ordered.
                Valid values are ``Ascending`` or ``Descending``. The default is ``Descending``.
                next_token (str): If more than ``MaxResults`` resources match the specified
                ``SearchExpression``, the response includes a ``NextToken``. The ``NextToken`` can
                be passed to the next ``SearchRequest`` to continue retrieving results.
            max_results (int): The maximum number of results to return.

        Returns:
            Response dict from service.
        """
        search_args = {"Resource": resource}

        if search_expression:
            search_args["SearchExpression"] = search_expression
        if sort_by:
            search_args["SortBy"] = sort_by
        if sort_order:
            search_args["SortOrder"] = sort_order
        if next_token:
            search_args["NextToken"] = next_token
        if max_results:
            search_args["MaxResults"] = max_results

        return self.sagemaker_client.search(**search_args)

    def put_record(
        self,
        feature_group_name: str,
        record: Sequence[Dict[str, str]],
        target_stores: Sequence[str] = None,
        ttl_duration: Dict[str, Any] = None,
    ):
        """Puts a single record in the FeatureGroup.

        Args:
            feature_group_name (str): Name of the FeatureGroup.
            record (Sequence[Dict[str, str]]): List of FeatureValue dicts to be ingested
                into FeatureStore.
            target_stores (Sequence[str]): Optional. List of target stores to put the record.
            ttl_duration (Dict[str, str]): Optional. Time-to-Live (TTL) duration for the record.

        Returns:
            Response dict from service.
        """

        params = {
            "FeatureGroupName": feature_group_name,
            "Record": record,
        }

        if ttl_duration:
            params["TtlDuration"] = ttl_duration

        if target_stores:
            params["TargetStores"] = target_stores

        return self.sagemaker_featurestore_runtime_client.put_record(**params)

    def delete_record(
        self,
        feature_group_name: str,
        record_identifier_value_as_string: str,
        event_time: str,
        deletion_mode: str = None,
    ):
        """Deletes a single record from the FeatureGroup.

        Args:
            feature_group_name (str): name of the FeatureGroup.
            record_identifier_value_as_string (str): name of the record identifier.
            event_time (str): a timestamp indicating when the deletion event occurred.
            deletion_mode: (str): deletion mode for deleting record.
        """
        return self.sagemaker_featurestore_runtime_client.delete_record(
            FeatureGroupName=feature_group_name,
            RecordIdentifierValueAsString=record_identifier_value_as_string,
            EventTime=event_time,
            DeletionMode=deletion_mode,
        )

    def get_record(
        self,
        record_identifier_value_as_string: str,
        feature_group_name: str,
        feature_names: Sequence[str],
        expiration_time_response: str = None,
    ) -> Dict[str, Sequence[Dict[str, str]]]:
        """Gets a single record in the FeatureGroup.

        Args:
            record_identifier_value_as_string (str): name of the record identifier.
            feature_group_name (str): name of the FeatureGroup.
            feature_names (Sequence[str]): list of feature names.
            expiration_time_response (str): the field of expiration time response
                to toggle returning of expiresAt.
        """
        get_record_args = {
            "FeatureGroupName": feature_group_name,
            "RecordIdentifierValueAsString": record_identifier_value_as_string,
        }

        if expiration_time_response:
            get_record_args["ExpirationTimeResponse"] = expiration_time_response

        if feature_names:
            get_record_args["FeatureNames"] = feature_names

        return self.sagemaker_featurestore_runtime_client.get_record(**get_record_args)

    def batch_get_record(
        self,
        identifiers: Sequence[Dict[str, Any]],
        expiration_time_response: str = None,
    ) -> Dict[str, Any]:
        """Gets a batch of record from FeatureStore.

        Args:
            identifiers (Sequence[Dict[str, Any]]): list of identifiers to uniquely identify records
                in FeatureStore.
            expiration_time_response (str): the field of expiration time response
                to toggle returning of expiresAt.

        Returns:
            Response dict from service.
        """
        batch_get_record_args = {"Identifiers": identifiers}

        if expiration_time_response:
            batch_get_record_args["ExpirationTimeResponse"] = expiration_time_response

        return self.sagemaker_featurestore_runtime_client.batch_get_record(**batch_get_record_args)

    def start_query_execution(
        self,
        catalog: str,
        database: str,
        query_string: str,
        output_location: str,
        kms_key: str = None,
        workgroup: str = None,
    ) -> Dict[str, str]:
        """Start Athena query execution.

        Args:
            catalog (str): name of the data catalog.
            database (str): name of the data catalog database.
            query_string (str): SQL expression.
            output_location (str): S3 location of the output file.
            kms_key (str): KMS key id will be used to encrypt the result if given.
            workgroup (str): The name of the workgroup in which the query is being started.
            If the workgroup is not specified, the default workgroup is used.

        Returns:
            Response dict from the service.
        """
        kwargs = dict(
            QueryString=query_string, QueryExecutionContext=dict(Catalog=catalog, Database=database)
        )
        result_config = dict(OutputLocation=output_location)
        if kms_key:
            result_config.update(
                EncryptionConfiguration=dict(EncryptionOption="SSE_KMS", KmsKey=kms_key)
            )
        kwargs.update(ResultConfiguration=result_config)

        if workgroup:
            kwargs.update(WorkGroup=workgroup)

        athena_client = self.boto_session.client("athena", region_name=self.boto_region_name)
        return athena_client.start_query_execution(**kwargs)

    def get_query_execution(self, query_execution_id: str) -> Dict[str, Any]:
        """Get execution status of the Athena query.

        Args:
            query_execution_id (str): execution ID of the Athena query.
        """
        athena_client = self.boto_session.client("athena", region_name=self.boto_region_name)
        return athena_client.get_query_execution(QueryExecutionId=query_execution_id)

    def wait_for_athena_query(self, query_execution_id: str, poll: int = 5):
        """Wait for Athena query to finish.

        Args:
             query_execution_id (str): execution ID of the Athena query.
             poll (int): time interval to poll get_query_execution API.
        """
        query_state = (
            self.get_query_execution(query_execution_id=query_execution_id)
            .get("QueryExecution")
            .get("Status")
            .get("State")
        )
        while query_state not in ("SUCCEEDED", "FAILED"):
            logger.info("Query %s is being executed.", query_execution_id)
            time.sleep(poll)
            query_state = (
                self.get_query_execution(query_execution_id=query_execution_id)
                .get("QueryExecution")
                .get("Status")
                .get("State")
            )
        if query_state == "SUCCEEDED":
            logger.info("Query %s successfully executed.", query_execution_id)
        else:
            logger.error("Failed to execute query %s.", query_execution_id)

    def download_athena_query_result(
        self,
        bucket: str,
        prefix: str,
        query_execution_id: str,
        filename: str,
    ):
        """Download query result file from S3.

        Args:
            bucket (str): name of the S3 bucket where the result file is stored.
            prefix (str): S3 prefix of the result file.
            query_execution_id (str): execution ID of the Athena query.
            filename (str): name of the downloaded file.
        """
        if self.s3_client is None:
            s3 = self.boto_session.client("s3", region_name=self.boto_region_name)
        else:
            s3 = self.s3_client
        s3.download_file(Bucket=bucket, Key=f"{prefix}/{query_execution_id}.csv", Filename=filename)

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

    def _intercept_create_request(
        self,
        request: typing.Dict,
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
                training data and model artifacts.
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

    def create_presigned_mlflow_tracking_server_url(
        self,
        tracking_server_name: str,
        expires_in_seconds: int = None,
        session_expiration_duration_in_seconds: int = None,
    ) -> Dict[str, Any]:
        """Creates a Presigned Url to acess the Mlflow UI.

        Args:
            tracking_server_name (str): Name of the Mlflow Tracking Server.
            expires_in_seconds (int): Expiration duration of the URL.
            session_expiration_duration_in_seconds (int): Session duration of the URL.
        Returns:
            (dict): Return value from the ``CreatePresignedMlflowTrackingServerUrl`` API.

        """

        create_presigned_url_args = {"TrackingServerName": tracking_server_name}
        if expires_in_seconds is not None:
            create_presigned_url_args["ExpiresInSeconds"] = expires_in_seconds

        if session_expiration_duration_in_seconds is not None:
            create_presigned_url_args["SessionExpirationDurationInSeconds"] = (
                session_expiration_duration_in_seconds
            )

        return self.sagemaker_client.create_presigned_mlflow_tracking_server_url(
            **create_presigned_url_args
        )

    def create_hub(
        self,
        hub_name: str,
        hub_description: str,
        hub_display_name: str = None,
        hub_search_keywords: List[str] = None,
        s3_storage_config: Dict[str, Any] = None,
        tags: List[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Creates a SageMaker Hub

        Args:
            hub_name (str): The name of the Hub to create.
            hub_description (str): A description of the Hub.
            hub_display_name (str): The display name of the Hub.
            hub_search_keywords (list): The searchable keywords for the Hub.
            s3_storage_config (S3StorageConfig): The Amazon S3 storage configuration for the Hub.
            tags (list): Any tags to associate with the Hub.

        Returns:
            (dict): Return value from the ``CreateHub`` API.
        """
        request = {"HubName": hub_name, "HubDescription": hub_description}

        if hub_display_name:
            request["HubDisplayName"] = hub_display_name
        else:
            request["HubDisplayName"] = hub_name

        if hub_search_keywords:
            request["HubSearchKeywords"] = hub_search_keywords
        if s3_storage_config:
            request["S3StorageConfig"] = s3_storage_config
        if tags:
            request["Tags"] = tags

        return self.sagemaker_client.create_hub(**request)

    def describe_hub(self, hub_name: str) -> Dict[str, Any]:
        """Describes a SageMaker Hub

        Args:
            hub_name (str): The name of the hub to describe.

        Returns:
            (dict): Return value for ``DescribeHub`` API
        """
        request = {"HubName": hub_name}

        return self.sagemaker_client.describe_hub(**request)

    def list_hubs(
        self,
        creation_time_after: str = None,
        creation_time_before: str = None,
        max_results: int = None,
        max_schema_version: str = None,
        name_contains: str = None,
        next_token: str = None,
        sort_by: str = None,
        sort_order: str = None,
    ) -> Dict[str, Any]:
        """Lists all existing SageMaker Hubs

        Args:
            creation_time_after (str): Only list HubContent that was created after
                the time specified.
            creation_time_before (str): Only list HubContent that was created
                before the time specified.
            max_results (int): The maximum amount of HubContent to list.
            max_schema_version (str): The upper bound of the HubContentSchemaVersion.
            name_contains (str): Only list HubContent if the name contains the specified string.
            next_token (str): If the response to a previous ``ListHubContents`` request was
                truncated, the response includes a ``NextToken``. To retrieve the next set of
                hub content, use the token in the next request.
            sort_by (str): Sort HubContent versions by either name or creation time.
            sort_order (str): Sort Hubs by ascending or descending order.
        Returns:
            (dict): Return value for ``ListHubs`` API
        """
        request = {}
        if creation_time_after:
            request["CreationTimeAfter"] = creation_time_after
        if creation_time_before:
            request["CreationTimeBefore"] = creation_time_before
        if max_results:
            request["MaxResults"] = max_results
        if max_schema_version:
            request["MaxSchemaVersion"] = max_schema_version
        if name_contains:
            request["NameContains"] = name_contains
        if next_token:
            request["NextToken"] = next_token
        if sort_by:
            request["SortBy"] = sort_by
        if sort_order:
            request["SortOrder"] = sort_order

        return self.sagemaker_client.list_hubs(**request)

    def list_hub_contents(
        self,
        hub_name: str,
        hub_content_type: str,
        creation_time_after: str = None,
        creation_time_before: str = None,
        max_results: int = None,
        max_schema_version: str = None,
        name_contains: str = None,
        next_token: str = None,
        sort_by: str = None,
        sort_order: str = None,
    ) -> Dict[str, Any]:
        """Lists the HubContents in a SageMaker Hub

        Args:
            hub_name (str): The name of the Hub to list the contents of.
            hub_content_type (str): The type of the HubContent to list.
            creation_time_after (str): Only list HubContent that was created after the
                time specified.
            creation_time_before (str): Only list HubContent that was created before the
                time specified.
            max_results (int): The maximum amount of HubContent to list.
            max_schema_version (str): The upper bound of the HubContentSchemaVersion.
            name_contains (str): Only list HubContent if the name contains the specified string.
            next_token (str): If the response to a previous ``ListHubContents`` request was
                truncated, the response includes a ``NextToken``. To retrieve the next set of
                hub content, use the token in the next request.
            sort_by (str): Sort HubContent versions by either name or creation time.
            sort_order (str): Sort Hubs by ascending or descending order.
        Returns:
            (dict): Return value for ``ListHubContents`` API
        """
        request = {"HubName": hub_name, "HubContentType": hub_content_type}
        if creation_time_after:
            request["CreationTimeAfter"] = creation_time_after
        if creation_time_before:
            request["CreationTimeBefore"] = creation_time_before
        if max_results:
            request["MaxResults"] = max_results
        if max_schema_version:
            request["MaxSchemaVersion"] = max_schema_version
        if name_contains:
            request["NameContains"] = name_contains
        if next_token:
            request["NextToken"] = next_token
        if sort_by:
            request["SortBy"] = sort_by
        if sort_order:
            request["SortOrder"] = sort_order

        return self.sagemaker_client.list_hub_contents(**request)

    def delete_hub(self, hub_name: str) -> None:
        """Deletes a SageMaker Hub

        Args:
            hub_name (str): The name of the hub to delete.
        """
        request = {"HubName": hub_name}

        return self.sagemaker_client.delete_hub(**request)

    def create_hub_content_reference(
        self,
        hub_name: str,
        source_hub_content_arn: str,
        hub_content_name: str = None,
        min_version: str = None,
    ) -> Dict[str, str]:
        """Creates a given HubContent reference in a SageMaker Hub

        Args:
            hub_name (str): The name of the Hub that you want to delete content in.
            source_hub_content_arn (str): Hub content arn in the public/source Hub.
            hub_content_name (str): The name of the reference that you want to add to the Hub.
            min_version (str): A minimum version of the hub content to add to the Hub.

        Returns:
            (dict): Return value for ``CreateHubContentReference`` API
        """

        request = {"HubName": hub_name, "SageMakerPublicHubContentArn": source_hub_content_arn}

        if hub_content_name:
            request["HubContentName"] = hub_content_name
        if min_version:
            request["MinVersion"] = min_version

        return self.sagemaker_client.create_hub_content_reference(**request)

    def delete_hub_content_reference(
        self, hub_name: str, hub_content_type: str, hub_content_name: str
    ) -> None:
        """Deletes a given HubContent reference in a SageMaker Hub

        Args:
            hub_name (str): The name of the Hub that you want to delete content in.
            hub_content_type (str): The type of the content that you want to delete from a Hub.
            hub_content_name (str): The name of the content that you want to delete from a Hub.
        """
        request = {
            "HubName": hub_name,
            "HubContentType": hub_content_type,
            "HubContentName": hub_content_name,
        }

        return self.sagemaker_client.delete_hub_content_reference(**request)

    def describe_hub_content(
        self,
        hub_content_name: str,
        hub_content_type: str,
        hub_name: str,
        hub_content_version: str = None,
    ) -> Dict[str, Any]:
        """Describes a HubContent in a SageMaker Hub

        Args:
            hub_content_name (str): The name of the HubContent to describe.
            hub_content_type (str): The type of HubContent in the Hub.
            hub_name (str): The name of the Hub that contains the HubContent to describe.
            hub_content_version (str): The version of the HubContent to describe

        Returns:
            (dict): Return value for ``DescribeHubContent`` API
        """
        request = {
            "HubContentName": hub_content_name,
            "HubContentType": hub_content_type,
            "HubName": hub_name,
        }
        if hub_content_version:
            request["HubContentVersion"] = hub_content_version

        return self.sagemaker_client.describe_hub_content(**request)

    def list_hub_content_versions(
        self,
        hub_name,
        hub_content_type: str,
        hub_content_name: str,
        min_version: str = None,
        max_schema_version: str = None,
        creation_time_after: str = None,
        creation_time_before: str = None,
        max_results: int = None,
        next_token: str = None,
        sort_by: str = None,
        sort_order: str = None,
    ) -> Dict[str, Any]:
        """List all versions of a HubContent in a SageMaker Hub

        Args:
            hub_content_name (str): The name of the HubContent to describe.
            hub_content_type (str): The type of HubContent in the Hub.
            hub_name (str): The name of the Hub that contains the HubContent to describe.

        Returns:
            (dict): Return value for ``DescribeHubContent`` API
        """

        request = {
            "HubName": hub_name,
            "HubContentName": hub_content_name,
            "HubContentType": hub_content_type,
        }

        if min_version:
            request["MinVersion"] = min_version
        if creation_time_after:
            request["CreationTimeAfter"] = creation_time_after
        if creation_time_before:
            request["CreationTimeBefore"] = creation_time_before
        if max_results:
            request["MaxResults"] = max_results
        if max_schema_version:
            request["MaxSchemaVersion"] = max_schema_version
        if next_token:
            request["NextToken"] = next_token
        if sort_by:
            request["SortBy"] = sort_by
        if sort_order:
            request["SortOrder"] = sort_order

        return self.sagemaker_client.list_hub_content_versions(**request)


def get_model_package_args(
    content_types=None,
    response_types=None,
    inference_instances=None,
    transform_instances=None,
    model_package_name=None,
    model_package_group_name=None,
    model_data=None,
    image_uri=None,
    model_metrics=None,
    metadata_properties=None,
    marketplace_cert=False,
    approval_status=None,
    description=None,
    tags=None,
    container_def_list=None,
    drift_check_baselines=None,
    customer_metadata_properties=None,
    validation_specification=None,
    domain=None,
    sample_payload_url=None,
    task=None,
    skip_model_validation=None,
    source_uri=None,
    model_card=None,
):
    """Get arguments for create_model_package method.

    Args:
        content_types (list): The supported MIME types for the input data.
        response_types (list): The supported MIME types for the output data.
        inference_instances (list): A list of the instance types that are used to
            generate inferences in real-time (default: None).
        transform_instances (list): A list of the instance types on which a transformation
            job can be run or on which an endpoint can be deployed (default: None).
        model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
            using `model_package_name` makes the Model Package un-versioned (default: None).
        model_package_group_name (str): Model Package Group name, exclusive to
            `model_package_name`, using `model_package_group_name` makes the Model Package
        model_data (str): s3 URI to the model artifacts from training (default: None).
        image_uri (str): Inference image uri for the container. Model class' self.image will
            be used if it is None (default: None).
        model_metrics (ModelMetrics): ModelMetrics object (default: None).
        metadata_properties (MetadataProperties): MetadataProperties object (default: None).
        marketplace_cert (bool): A boolean value indicating if the Model Package is certified
            for AWS Marketplace (default: False).
        approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
            or "PendingManualApproval" (default: "PendingManualApproval").
        description (str): Model Package description (default: None).
        tags (Optional[Tags]): A list of dictionaries containing key-value pairs
            (default: None).
        container_def_list (list): A list of container defintiions (default: None).
        drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
        customer_metadata_properties (dict[str, str]): A dictionary of key-value paired
            metadata properties (default: None).
        domain (str): Domain values can be "COMPUTER_VISION", "NATURAL_LANGUAGE_PROCESSING",
            "MACHINE_LEARNING" (default: None).
        sample_payload_url (str): The S3 path where the sample payload is stored (default: None).
        task (str): Task values which are supported by Inference Recommender are "FILL_MASK",
            "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION", "IMAGE_SEGMENTATION",
            "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
        skip_model_validation (str): Indicates if you want to skip model validation.
            Values can be "All" or "None" (default: None).
        source_uri (str): The URI of the source for the model package (default: None).
        model_card (ModeCard or ModelPackageModelCard): document contains qualitative and
                quantitative information about a model (default: None).

    Returns:
        dict: A dictionary of method argument names and values.
    """
    if container_def_list is not None:
        containers = container_def_list
    else:
        container = {
            "Image": image_uri,
        }
        if model_data is not None:
            container["ModelDataUrl"] = model_data

        containers = [container]

    model_package_args = {
        "containers": containers,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "marketplace_cert": marketplace_cert,
    }

    if content_types is not None:
        model_package_args["content_types"] = content_types
    if response_types is not None:
        model_package_args["response_types"] = response_types
    if model_package_name is not None:
        model_package_args["model_package_name"] = model_package_name
    if model_package_group_name is not None:
        model_package_args["model_package_group_name"] = model_package_group_name
    if model_metrics is not None:
        model_package_args["model_metrics"] = model_metrics._to_request_dict()
    if drift_check_baselines is not None:
        model_package_args["drift_check_baselines"] = drift_check_baselines._to_request_dict()
    if metadata_properties is not None:
        model_package_args["metadata_properties"] = metadata_properties._to_request_dict()
    if approval_status is not None:
        model_package_args["approval_status"] = approval_status
    if description is not None:
        model_package_args["description"] = description
    if tags is not None:
        model_package_args["tags"] = format_tags(tags)
    if customer_metadata_properties is not None:
        model_package_args["customer_metadata_properties"] = customer_metadata_properties
    if validation_specification is not None:
        model_package_args["validation_specification"] = validation_specification
    if domain is not None:
        model_package_args["domain"] = domain
    if sample_payload_url is not None:
        model_package_args["sample_payload_url"] = sample_payload_url
    if task is not None:
        model_package_args["task"] = task
    if skip_model_validation is not None:
        model_package_args["skip_model_validation"] = skip_model_validation
    if source_uri is not None:
        model_package_args["source_uri"] = source_uri
    if model_card is not None:
        original_req = model_card._create_request_args()
        if original_req.get("ModelCardName") is not None:
            del original_req["ModelCardName"]
        if original_req.get("Content") is not None:
            original_req["ModelCardContent"] = original_req["Content"]
            del original_req["Content"]
        model_package_args["model_card"] = original_req
    return model_package_args


def get_create_model_package_request(
    model_package_name=None,
    model_package_group_name=None,
    containers=None,
    content_types=None,
    response_types=None,
    inference_instances=None,
    transform_instances=None,
    model_metrics=None,
    metadata_properties=None,
    marketplace_cert=False,
    approval_status="PendingManualApproval",
    description=None,
    tags=None,
    drift_check_baselines=None,
    customer_metadata_properties=None,
    validation_specification=None,
    domain=None,
    sample_payload_url=None,
    task=None,
    skip_model_validation="None",
    source_uri=None,
    model_card=None,
):
    """Get request dictionary for CreateModelPackage API.

    Args:
        model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
            using `model_package_name` makes the Model Package un-versioned (default: None).
        model_package_group_name (str): Model Package Group name, exclusive to
            `model_package_name`, using `model_package_group_name` makes the Model Package
        containers (list): A list of inference containers that can be used for inference
            specifications of Model Package (default: None).
        content_types (list): The supported MIME types for the input data (default: None).
        response_types (list): The supported MIME types for the output data (default: None).
        inference_instances (list): A list of the instance types that are used to
            generate inferences in real-time (default: None).
        transform_instances (list): A list of the instance types on which a transformation
            job can be run or on which an endpoint can be deployed (default: None).
        model_metrics (ModelMetrics): ModelMetrics object (default: None).
        metadata_properties (MetadataProperties): MetadataProperties object (default: None).
        marketplace_cert (bool): A boolean value indicating if the Model Package is certified
            for AWS Marketplace (default: False).
        approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
            or "PendingManualApproval" (default: "PendingManualApproval").
        description (str): Model Package description (default: None).
        tags (Optional[Tags]): A list of dictionaries containing key-value pairs
            (default: None).
        drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
        customer_metadata_properties (dict[str, str]): A dictionary of key-value paired
            metadata properties (default: None).
        domain (str): Domain values can be "COMPUTER_VISION", "NATURAL_LANGUAGE_PROCESSING",
            "MACHINE_LEARNING" (default: None).
        sample_payload_url (str): The S3 path where the sample payload is stored (default: None).
        task (str): Task values which are supported by Inference Recommender are "FILL_MASK",
            "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION", "IMAGE_SEGMENTATION",
            "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
        skip_model_validation (str): Indicates if you want to skip model validation.
            Values can be "All" or "None" (default: None).
        source_uri (str): The URI of the source for the model package (default: None).
        model_card (ModeCard or ModelPackageModelCard): document contains qualitative and
                quantitative information about a model (default: None).
    """

    if all([model_package_name, model_package_group_name]):
        raise ValueError(
            "model_package_name and model_package_group_name cannot be present at the " "same time."
        )
    if all([model_package_name, source_uri]):
        raise ValueError(
            "Un-versioned SageMaker Model Package currently cannot be " "created with source_uri."
        )
    if (containers is not None) and all(
        [
            model_package_name,
            any(
                [
                    (("ModelDataSource" in c) and (c["ModelDataSource"] is not None))
                    for c in containers
                ]
            ),
        ]
    ):
        raise ValueError(
            "Un-versioned SageMaker Model Package currently cannot be "
            "created with ModelDataSource."
        )
    request_dict = {}
    if model_package_name is not None:
        request_dict["ModelPackageName"] = model_package_name
    if model_package_group_name is not None:
        request_dict["ModelPackageGroupName"] = model_package_group_name
    if description is not None:
        request_dict["ModelPackageDescription"] = description
    if tags is not None:
        request_dict["Tags"] = format_tags(tags)
    if model_metrics:
        request_dict["ModelMetrics"] = model_metrics
    if drift_check_baselines:
        request_dict["DriftCheckBaselines"] = drift_check_baselines
    if metadata_properties:
        request_dict["MetadataProperties"] = metadata_properties
    if customer_metadata_properties is not None:
        request_dict["CustomerMetadataProperties"] = customer_metadata_properties
    if validation_specification:
        request_dict["ValidationSpecification"] = validation_specification
    if domain is not None:
        request_dict["Domain"] = domain
    if sample_payload_url is not None:
        request_dict["SamplePayloadUrl"] = sample_payload_url
    if task is not None:
        request_dict["Task"] = task
    if source_uri is not None:
        request_dict["SourceUri"] = source_uri
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
        if model_package_group_name is not None:
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
        else:
            if not all([inference_instances, transform_instances]):
                raise ValueError(
                    "inference_instances and transform_instances "
                    "must be provided if model_package_group_name is not present."
                )
            inference_specification.update(
                {
                    "SupportedRealtimeInferenceInstanceTypes": inference_instances,
                    "SupportedTransformInstanceTypes": transform_instances,
                }
            )
        request_dict["InferenceSpecification"] = inference_specification
    request_dict["CertifyForMarketplace"] = marketplace_cert
    request_dict["ModelApprovalStatus"] = approval_status
    request_dict["SkipModelValidation"] = skip_model_validation
    if model_card is not None:
        request_dict["ModelCard"] = model_card

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


def container_def(
    image_uri,
    model_data_url=None,
    env=None,
    container_mode=None,
    image_config=None,
    accept_eula=None,
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

    Returns:
        dict[str, str]: A complete container definition object usable with the CreateModel API if
        passed via `PrimaryContainers` field.
    """
    if env is None:
        env = {}
    c_def = {"Image": image_uri, "Environment": env}

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

    return production_variant_configuration


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


def generate_default_sagemaker_bucket_name(boto_session):
    """Generates a name for the default sagemaker S3 bucket.

    Args:
        boto_session (boto3.session.Session): The underlying Boto3 session which AWS service
    """
    region = boto_session.region_name
    account = boto_session.client(
        "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
    ).get_caller_identity()["Account"]
    return "sagemaker-{}-{}".format(region, account)


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


def _create_resource(create_fn):
    """Call create function and accepts/pass when resource already exists.

    This is a helper function to use an existing resource if found when creating.

    Args:
        create_fn: Create resource function.

    Returns:
        (bool): True if new resource was created, False if resource already exists.
    """
    try:
        create_fn()
        # create function succeeded, resource does not exist already
        return True
    except ClientError as ce:
        error_code = ce.response["Error"]["Code"]
        error_message = ce.response["Error"]["Message"]
        already_exists_exceptions = ["ValidationException", "ResourceInUse"]
        already_exists_msg_patterns = ["Cannot create already existing", "already exists"]
        if not (
            error_code in already_exists_exceptions
            and any(p in error_message for p in already_exists_msg_patterns)
        ):
            raise ce
        # no new resource created as resource already exists
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


def _edge_packaging_job_status(sagemaker_client, job_name):
    """Process the current status of a packaging job.

    Args:
        sagemaker_client (boto3.client.sagemaker): a sagemaker client
        job_name (str): the name of the job to inspect.

    Returns:
        Dict: the status of the edge packaging job
    """
    package_status_codes = {
        "Completed": "!",
        "InProgress": ".",
        "Failed": "*",
        "Stopped": "s",
        "Stopping": "_",
    }
    in_progress_statuses = ["InProgress", "Stopping", "Starting"]

    desc = sagemaker_client.describe_edge_packaging_job(EdgePackagingJobName=job_name)
    status = desc["EdgePackagingJobStatus"]

    status = _STATUS_CODE_TABLE.get(status, status)
    print(package_status_codes.get(status, "?"), end="")
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


# live log on endpoint in creation
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
                LOGGER.info("Created endpoint with name %s", endpoint_name)
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


def _wait_until_training_done(callable_fn, desc, poll=5):
    """Placeholder docstring"""
    elapsed_time = 0
    finished = None
    job_desc = desc
    while not finished:
        try:
            elapsed_time += poll
            time.sleep(poll)
            job_desc, finished = callable_fn(job_desc)
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
    return job_desc


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
    description = sagemaker_client.describe_training_job(TrainingJobName=job_name)
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
        message = "Error for {job_type} {job_name}: {status}. Reason: {reason}".format(
            job_type=job_type, job_name=job, status=status, reason=reason
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


s3_input = deprecated_class(TrainingInput, "sagemaker.session.s3_input")
ShuffleConfig = deprecated_class(ShuffleConfig, "sagemaker.session.ShuffleConfig")
