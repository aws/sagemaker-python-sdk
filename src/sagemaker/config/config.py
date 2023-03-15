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
"""This module configures the default values for SageMaker Python SDK.

It supports loading Config files from local file system/S3.
The schema of the Config file is dictated in config_schema.py in the same module.

"""
from __future__ import absolute_import

import pathlib
import logging
import os
from typing import List
import boto3
import yaml
from jsonschema import validate
from platformdirs import site_config_dir, user_config_dir
from botocore.utils import merge_dicts
from six.moves.urllib.parse import urlparse
from sagemaker.config.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA

logger = logging.getLogger("sagemaker")

_APP_NAME = "sagemaker"
# The default config file location of the Administrator provided Config file. This path can be
# overridden with `SAGEMAKER_ADMIN_CONFIG_OVERRIDE` environment variable.
_DEFAULT_ADMIN_CONFIG_FILE_PATH = os.path.join(site_config_dir(_APP_NAME), "config.yaml")
# The default config file location of the user provided Config file. This path can be
# overridden with `SAGEMAKER_USER_CONFIG_OVERRIDE` environment variable.
_DEFAULT_USER_CONFIG_FILE_PATH = os.path.join(user_config_dir(_APP_NAME), "config.yaml")

ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE = "SAGEMAKER_ADMIN_CONFIG_OVERRIDE"
ENV_VARIABLE_USER_CONFIG_OVERRIDE = "SAGEMAKER_USER_CONFIG_OVERRIDE"

_BOTO_SESSION = boto3.DEFAULT_SESSION or boto3.Session()
# The default Boto3 S3 Resource. This is constructed from the default Boto3 session. This will be
# used to fetch SageMakerConfig from S3. Users can override this by passing their own S3 Resource
# as the constructor parameter for SageMakerConfig.
_DEFAULT_S3_RESOURCE = _BOTO_SESSION.resource("s3")
S3_PREFIX = "s3://"


class SageMakerConfig(object):
    """SageMakerConfig class encapsulates the Config for SageMaker Python SDK.

    Usages:
    This class will be integrated with sagemaker.session.Session. Users of SageMaker Python SDK
    will have the ability to pass a SageMakerConfig object to sagemaker.session.Session. If
    SageMakerConfig object is not provided by the user, then sagemaker.session.Session will
    create its own SageMakerConfig object.

    Note: Once sagemaker.session.Session is initialized, it will operate with the configuration
    values at that instant. If the users wish to alter configuration files/file paths after
    sagemaker.session.Session is initialized, then that will not be reflected in
    sagemaker.session.Session. They would have to re-initialize sagemaker.session.Session to
    pick the latest changes.

    """

    def __init__(self, additional_config_paths: List[str] = None, s3_resource=_DEFAULT_S3_RESOURCE):
        """Constructor for SageMakerConfig.

        By default, it will first look for Config files in paths that are dictated by
        _DEFAULT_ADMIN_CONFIG_FILE_PATH and _DEFAULT_USER_CONFIG_FILE_PATH.

        Users can override the _DEFAULT_ADMIN_CONFIG_FILE_PATH and _DEFAULT_USER_CONFIG_FILE_PATH
        by using environment variables - SAGEMAKER_ADMIN_CONFIG_OVERRIDE and
        SAGEMAKER_USER_CONFIG_OVERRIDE

        Additional Configuration file paths can also be provided as a constructor parameter.

        This constructor will then
        * Load each config file.
        * It will validate the schema of the config files.
        * It will perform the merge operation in the same order.

        This constructor will throw exceptions for the following cases:
        * Schema validation fails for one/more config files.
        * When the config file is not a proper YAML file.
        * Any S3 related issues that arises while fetching config file from S3. This includes
        permission issues, S3 Object is not found in the specified S3 URI.
        * File doesn't exist in a path that was specified by the user as part of environment
        variable/ additional_config_paths. This doesn't include
        _DEFAULT_ADMIN_CONFIG_FILE_PATH and _DEFAULT_USER_CONFIG_FILE_PATH


        Args:
            additional_config_paths: List of Config file paths.
                These paths can be one of the following:
                * Local file path
                * Local directory path (in this case, we will look for config.yaml in that
                directory)
                * S3 URI of the config file
                * S3 URI of the directory containing the config file (in this case, we will look for
                config.yaml in that directory)
                Note: S3 URI follows the format s3://<bucket>/<Key prefix>
            s3_resource: Corresponds to boto3 S3 resource. This will be used to fetch Config
            files from S3. If it is not provided, we will create a default s3 resource
            See :py:meth:`boto3.session.Session.resource`. This argument is not needed if the
            config files are present in the local file system

        """
        default_config_path = os.getenv(
            ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE, _DEFAULT_ADMIN_CONFIG_FILE_PATH
        )
        user_config_path = os.getenv(
            ENV_VARIABLE_USER_CONFIG_OVERRIDE, _DEFAULT_USER_CONFIG_FILE_PATH
        )
        self._config_paths = [default_config_path, user_config_path]
        if additional_config_paths:
            self._config_paths += additional_config_paths
        self._config_paths = list(filter(lambda item: item is not None, self._config_paths))
        self._config = _load_config_files(self._config_paths, s3_resource)

    @property
    def config_paths(self) -> List[str]:
        """Getter for Config paths.

        Returns:
            List[str]: This corresponds to the list of config file paths.
        """
        return self._config_paths

    @property
    def config(self) -> dict:
        """Getter for the configuration object.

        Returns:
            dict: A dictionary representing the configurations that were loaded from the config
            file(s).
        """
        return self._config


def _load_config_files(file_paths: List[str], s3_resource_for_config) -> dict:
    """This method loads all the config files from the paths that were provided as Inputs.

    Note: Supported Config file locations are Local File System and S3.

    This method will throw exceptions for the following cases:
        * Schema validation fails for one/more config files.
        * When the config file is not a proper YAML file.
        * Any S3 related issues that arises while fetching config file from S3. This includes
        permission issues, S3 Object is not found in the specified S3 URI.
        * File doesn't exist in a path that was specified by the user as part of environment
        variable/ additional_config_paths. This doesn't include
        _DEFAULT_ADMIN_CONFIG_FILE_PATH and _DEFAULT_USER_CONFIG_FILE_PATH

    Args:
        file_paths(List[str]): The list of paths corresponding to the config file. Note: This
        path can either be a Local File System path or it can be a S3 URI.
        s3_resource_for_config: Corresponds to boto3 S3 resource. This will be used to fetch Config
        files from S3. See :py:meth:`boto3.session.Session.resource`.

    Returns:
        dict: A dictionary representing the configurations that were loaded from the config files.

    """
    merged_config = {}
    for file_path in file_paths:
        config_from_file = {}
        if file_path.startswith(S3_PREFIX):
            config_from_file = _load_config_from_s3(file_path, s3_resource_for_config)
        else:
            try:
                config_from_file = _load_config_from_file(file_path)
            except ValueError:
                if file_path not in (
                    _DEFAULT_ADMIN_CONFIG_FILE_PATH,
                    _DEFAULT_USER_CONFIG_FILE_PATH,
                ):
                    # Throw exception only when User provided file path is invalid.
                    # If there are no files in the Default config file locations, don't throw
                    # Exceptions.
                    raise
        if config_from_file:
            validate(config_from_file, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)
            merge_dicts(merged_config, config_from_file)
    return merged_config


def _load_config_from_file(file_path: str) -> dict:
    """This method loads the config file from the path that was specified as parameter.

    If the path that was provided, corresponds to a directory then this method will try to search
    for 'config.yaml' in that directory. Note: We will not be doing any recursive search.

    Args:
        file_path(str): The file path from which the Config file needs to be loaded.

    Returns:
        dict: A dictionary representing the configurations that were loaded from the config file.

    This method will throw Exceptions for the following cases:
    * When the config file is not a proper YAML file.
    * File doesn't exist in a path that was specified by the consumer.
    """
    inferred_file_path = file_path
    if os.path.isdir(file_path):
        inferred_file_path = os.path.join(file_path, "config.yaml")
    if not os.path.exists(inferred_file_path):
        raise ValueError(
            f"Unable to load config file from the location: {file_path} Please"
            f" provide a valid file path"
        )
    logger.debug("Fetching configuration file from the path: %s", file_path)
    return yaml.safe_load(open(inferred_file_path, "r"))


def _load_config_from_s3(s3_uri, s3_resource_for_config) -> dict:
    """This method loads the config file from the S3 URI that was specified as parameter.

    If the S3 URI that was provided, corresponds to a directory then this method will try to
    search for 'config.yaml' in that directory. Note: We will not be doing any recursive search.

    Args:
        s3_uri(str): The S3 URI of the config file.
            Note: S3 URI follows the format s3://<bucket>/<Key prefix>
        s3_resource_for_config: Corresponds to boto3 S3 resource. This will be used to fetch Config
        files from S3. See :py:meth:`boto3.session.Session.resource`.

    Returns:
        dict: A dictionary representing the configurations that were loaded from the config file.

    This method will throw Exceptions for the following cases:
    * If Boto3 S3 resource is not provided.
    * When the config file is not a proper YAML file.
    * If the method is unable to retrieve the list of all the S3 files with the same prefix
    * If there are no S3 files with that prefix.
    * If a folder in S3 bucket is provided as s3_uri, and if it doesn't have config.yaml,
        then we will throw an Exception.
    """
    if not s3_resource_for_config:
        raise RuntimeError("Please provide a S3 client for loading the config")
    logger.debug("Fetching configuration file from the S3 URI: %s", s3_uri)
    inferred_s3_uri = _get_inferred_s3_uri(s3_uri, s3_resource_for_config)
    parsed_url = urlparse(inferred_s3_uri)
    bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
    s3_object = s3_resource_for_config.Object(bucket, key_prefix)
    s3_file_content = s3_object.get()["Body"].read()
    return yaml.safe_load(s3_file_content.decode("utf-8"))


def _get_inferred_s3_uri(s3_uri, s3_resource_for_config):
    """Verifies whether the given S3 URI exists and returns the URI.

    If there are multiple S3 objects with the same key prefix,
    then this method will verify whether S3 URI + /config.yaml exists.
    s3://example-bucket/somekeyprefix/config.yaml

    Args:
        s3_uri (str) : An S3 uri that refers to a location in which config file is present.
            s3_uri must start with 's3://'.
            An example s3_uri: 's3://example-bucket/config.yaml'.
        s3_resource_for_config: Corresponds to boto3 S3 resource. This will be used to fetch Config
            files from S3.
            See :py:meth:`boto3.session.Session.resource`

    Returns:
        str: Valid S3 URI of the Config file. None if it doesn't exist.

    This method will throw Exceptions for the following cases:
    * If the method is unable to retrieve the list of all the S3 files with the same prefix
    * If there are no S3 files with that prefix.
    * If a folder in S3 bucket is provided as s3_uri, and if it doesn't have config.yaml,
    then we will throw an Exception.
    """
    parsed_url = urlparse(s3_uri)
    bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
    try:
        s3_bucket = s3_resource_for_config.Bucket(name=bucket)
        s3_objects = s3_bucket.objects.filter(Prefix=key_prefix).all()
        s3_files_with_same_prefix = [
            "{}{}/{}".format(S3_PREFIX, bucket, s3_object.key) for s3_object in s3_objects
        ]
    except Exception as e:  # pylint: disable=W0703
        # if customers didn't provide us with a valid S3 File/insufficient read permission,
        # We will fail hard.
        raise RuntimeError(f"Unable to read from S3 with URI: {s3_uri} due to {e}")
    if len(s3_files_with_same_prefix) == 0:
        # Customer provided us with an incorrect s3 path.
        raise ValueError("Please provide a valid s3 path instead of {}".format(s3_uri))
    if len(s3_files_with_same_prefix) > 1:
        # Customer has provided us with a S3 URI which points to a directory
        # search for s3://<bucket>/directory-key-prefix/config.yaml
        inferred_s3_uri = str(pathlib.PurePosixPath(s3_uri, "config.yaml")).replace("s3:/", "s3://")
        if inferred_s3_uri not in s3_files_with_same_prefix:
            # We don't know which file we should be operating with.
            raise ValueError("Please provide a S3 URI which has config.yaml in the directory")
        # Customer has a config.yaml present in the directory that was provided as the S3 URI
        return inferred_s3_uri
    return s3_uri
