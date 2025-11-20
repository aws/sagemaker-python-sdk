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


import os
import jsonschema
import boto3
import yaml
import pathlib

from functools import lru_cache
from typing import List
from platformdirs import site_config_dir, user_config_dir

from botocore.utils import merge_dicts
from six.moves.urllib.parse import urlparse
from sagemaker.core.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA
from sagemaker.core.utils.exceptions import (
    LocalConfigNotFoundError,
    S3ConfigNotFoundError,
    IntelligentDefaultsError,
    ConfigSchemaValidationError,
)
from sagemaker.core.utils.utils import get_textual_rich_logger

logger = get_textual_rich_logger(__name__)


_APP_NAME = "sagemaker"
# The default name of the config file.
_CONFIG_FILE_NAME = "config.yaml"
# The default config file location of the Administrator provided config file. This path can be
# overridden with `SAGEMAKER_CORE_ADMIN_CONFIG_OVERRIDE` environment variable.
_DEFAULT_ADMIN_CONFIG_FILE_PATH = os.path.join(site_config_dir(_APP_NAME), _CONFIG_FILE_NAME)
# The default config file location of the user provided config file. This path can be
# overridden with `SAGEMAKER_USER_CONFIG_OVERRIDE` environment variable.
_DEFAULT_USER_CONFIG_FILE_PATH = os.path.join(user_config_dir(_APP_NAME), _CONFIG_FILE_NAME)
# The default config file location of the local mode.
_DEFAULT_LOCAL_MODE_CONFIG_FILE_PATH = os.path.join(
    os.path.expanduser("~"), ".sagemaker", _CONFIG_FILE_NAME
)
ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE = "SAGEMAKER_CORE_ADMIN_CONFIG_OVERRIDE"
ENV_VARIABLE_USER_CONFIG_OVERRIDE = "SAGEMAKER_CORE_USER_CONFIG_OVERRIDE"

S3_PREFIX = "s3://"


def load_default_configs(additional_config_paths: List[str] = None, s3_resource=None):
    default_config_path = os.getenv(
        ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE, _DEFAULT_ADMIN_CONFIG_FILE_PATH
    )
    user_config_path = os.getenv(ENV_VARIABLE_USER_CONFIG_OVERRIDE, _DEFAULT_USER_CONFIG_FILE_PATH)

    config_paths = [default_config_path, user_config_path]
    if additional_config_paths:
        config_paths += additional_config_paths
    config_paths = list(filter(lambda item: item is not None, config_paths))
    merged_config = {}
    for file_path in config_paths:
        config_from_file = {}
        if file_path.startswith(S3_PREFIX):
            config_from_file = _load_config_from_s3(file_path, s3_resource)
        else:
            try:
                config_from_file = _load_config_from_file(file_path)
            except ValueError:
                error = LocalConfigNotFoundError(file_path=file_path)
                if file_path not in (
                    _DEFAULT_ADMIN_CONFIG_FILE_PATH,
                    _DEFAULT_USER_CONFIG_FILE_PATH,
                ):
                    # Throw exception only when User provided file path is invalid.
                    # If there are no files in the Default config file locations, don't throw
                    # Exceptions.
                    raise error

                logger.debug(error)
        if config_from_file:
            try:
                validate_sagemaker_config(config_from_file)
            except jsonschema.exceptions.ValidationError as error:
                raise ConfigSchemaValidationError(file_path=file_path, message=str(error))
            merge_dicts(merged_config, config_from_file)
            logger.debug("Fetched defaults config from location: %s", file_path)
        else:
            logger.debug("Not applying SDK defaults from location: %s", file_path)

    return merged_config


def validate_sagemaker_config(sagemaker_config: dict = None):
    """Validates whether a given dictionary adheres to the schema.

    Args:
        sagemaker_config: A dictionary containing default values for the
                SageMaker Python SDK. (default: None).
    """
    jsonschema.validate(sagemaker_config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def _load_config_from_s3(s3_uri, s3_resource_for_config) -> dict:
    """Placeholder docstring"""
    if not s3_resource_for_config:
        # Constructing a default Boto3 S3 Resource from a default Boto3 session.
        boto_session = boto3.DEFAULT_SESSION or boto3.Session()
        boto_region_name = boto_session.region_name
        if boto_region_name is None:
            raise IntelligentDefaultsError(
                message=(
                    "Valid region is not provided in the Boto3 session."
                    + "Setup local AWS configuration with a valid region supported by SageMaker."
                )
            )
        s3_resource_for_config = boto_session.resource("s3", region_name=boto_region_name)

    logger.debug("Fetching defaults config from location: %s", s3_uri)
    inferred_s3_uri = _get_inferred_s3_uri(s3_uri, s3_resource_for_config)
    parsed_url = urlparse(inferred_s3_uri)
    bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
    s3_object = s3_resource_for_config.Object(bucket, key_prefix)
    s3_file_content = s3_object.get()["Body"].read()
    return yaml.safe_load(s3_file_content.decode("utf-8"))


def _get_inferred_s3_uri(s3_uri, s3_resource_for_config):
    """Placeholder docstring"""
    parsed_url = urlparse(s3_uri)
    bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
    s3_bucket = s3_resource_for_config.Bucket(name=bucket)
    s3_objects = s3_bucket.objects.filter(Prefix=key_prefix).all()
    s3_files_with_same_prefix = [
        "{}{}/{}".format(S3_PREFIX, bucket, s3_object.key) for s3_object in s3_objects
    ]
    if len(s3_files_with_same_prefix) == 0:
        # Customer provided us with an incorrect s3 path.
        raise S3ConfigNotFoundError(
            s3_uri=s3_uri,
            message="Provide a valid S3 URI in the format s3://<bucket>/<key-prefix>/{_CONFIG_FILE_NAME}.",
        )
    if len(s3_files_with_same_prefix) > 1:
        # Customer has provided us with a S3 URI which points to a directory
        # search for s3://<bucket>/directory-key-prefix/config.yaml
        inferred_s3_uri = str(pathlib.PurePosixPath(s3_uri, _CONFIG_FILE_NAME)).replace(
            "s3:/", "s3://"
        )
        if inferred_s3_uri not in s3_files_with_same_prefix:
            # We don't know which file we should be operating with.
            raise S3ConfigNotFoundError(
                s3_uri=s3_uri,
                message="Provide an S3 URI pointing to a directory that contains a {_CONFIG_FILE_NAME} file.",
            )
        # Customer has a config.yaml present in the directory that was provided as the S3 URI
        return inferred_s3_uri
    return s3_uri


def _load_config_from_file(file_path: str) -> dict:
    """Placeholder docstring"""
    inferred_file_path = file_path
    if os.path.isdir(file_path):
        inferred_file_path = os.path.join(file_path, _CONFIG_FILE_NAME)
    if not os.path.exists(inferred_file_path):
        raise ValueError
    logger.debug("Fetching defaults config from location: %s", file_path)
    with open(inferred_file_path, "r") as f:
        content = yaml.safe_load(f)
    return content


@lru_cache(maxsize=None)
def load_default_configs_for_resource_name(resource_name: str):
    configs_data = load_default_configs()
    if not configs_data:
        logger.debug("No default configurations found for resource: %s", resource_name)
        return {}
    return configs_data["SageMaker"]["PythonSDK"]["Resources"].get(resource_name)


def get_config_value(attribute, resource_defaults, global_defaults):
    if resource_defaults and attribute in resource_defaults:
        return resource_defaults[attribute]
    if global_defaults and attribute in global_defaults:
        return global_defaults[attribute]
    logger.debug(
        f"Configurable value {attribute} not entered in parameters or present in the Config"
    )
    return None
