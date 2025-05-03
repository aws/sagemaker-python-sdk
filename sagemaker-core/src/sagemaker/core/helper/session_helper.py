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
from functools import reduce
from typing import Dict, Optional

import boto3
import botocore
import botocore.config
from botocore.exceptions import ClientError

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

        self._initialize(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            sagemaker_runtime_client=sagemaker_runtime_client,
            sagemaker_featurestore_runtime_client=sagemaker_featurestore_runtime_client,
            sagemaker_metrics_client=sagemaker_metrics_client,
        )

    def _initialize(
        self,
        boto_session,
        sagemaker_client,
        sagemaker_runtime_client,
        sagemaker_featurestore_runtime_client,
        sagemaker_metrics_client,
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
        self.sagemaker_client = sagemaker_client or self.boto_session.client("sagemaker")

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
