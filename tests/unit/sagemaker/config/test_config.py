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
from __future__ import absolute_import

import os
import pytest
import yaml
import logging
from mock import Mock, MagicMock

from sagemaker.config.config import (
    load_sagemaker_config,
    logger,
    _DEFAULT_ADMIN_CONFIG_FILE_PATH,
    _DEFAULT_USER_CONFIG_FILE_PATH,
)
from jsonschema import exceptions
from yaml.constructor import ConstructorError


@pytest.fixture()
def config_file_as_yaml(get_data_dir):
    config_file_path = os.path.join(get_data_dir, "config.yaml")
    return open(config_file_path, "r").read()


@pytest.fixture()
def expected_merged_config(get_data_dir):
    expected_merged_config_file_path = os.path.join(
        get_data_dir, "expected_output_config_after_merge.yaml"
    )
    return yaml.safe_load(open(expected_merged_config_file_path, "r").read())


def test_config_when_default_config_file_and_user_config_file_is_not_found():
    assert load_sagemaker_config() == {}


def test_config_when_overriden_default_config_file_is_not_found(get_data_dir):
    fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = fake_config_file_path
    with pytest.raises(ValueError):
        load_sagemaker_config()
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]


def test_invalid_config_file_which_has_python_code(get_data_dir):
    invalid_config_file_path = os.path.join(get_data_dir, "config_file_with_code.yaml")
    # no exceptions will be thrown with yaml.unsafe_load
    yaml.unsafe_load(open(invalid_config_file_path, "r"))
    # PyYAML will throw exceptions for yaml.safe_load. SageMaker Config is using
    # yaml.safe_load internally
    with pytest.raises(ConstructorError) as exception_info:
        load_sagemaker_config(additional_config_paths=[invalid_config_file_path])
    assert "python/object/apply:eval" in str(exception_info.value)


def test_config_when_additional_config_file_path_is_not_found(get_data_dir):
    fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
    with pytest.raises(ValueError):
        load_sagemaker_config(additional_config_paths=[fake_config_file_path])


def test_config_factory_when_override_user_config_file_is_not_found(get_data_dir):
    fake_additional_override_config_file_path = os.path.join(
        get_data_dir, "additional-config-not-found.yaml"
    )
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = fake_additional_override_config_file_path
    with pytest.raises(ValueError):
        load_sagemaker_config()
    del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]


def test_default_config_file_with_invalid_schema(get_data_dir):
    config_file_path = os.path.join(get_data_dir, "invalid_config_file.yaml")
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = config_file_path
    with pytest.raises(exceptions.ValidationError):
        load_sagemaker_config()
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]


def test_default_config_file_when_directory_is_provided_as_the_path(
    get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
):
    # This will try to load config.yaml file from that directory if present.
    expected_config = base_config_with_schema
    expected_config["SageMaker"] = valid_config_with_all_the_scopes
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
    assert expected_config == load_sagemaker_config()
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]


def test_additional_config_paths_when_directory_is_provided(
    get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
):
    # This will try to load config.yaml file from that directory if present.
    expected_config = base_config_with_schema
    expected_config["SageMaker"] = valid_config_with_all_the_scopes
    assert expected_config == load_sagemaker_config(additional_config_paths=[get_data_dir])


def test_default_config_file_when_path_is_provided_as_environment_variable(
    get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
):
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
    # This will try to load config.yaml file from that directory if present.
    expected_config = base_config_with_schema
    expected_config["SageMaker"] = valid_config_with_all_the_scopes
    assert expected_config == load_sagemaker_config()
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]


def test_merge_behavior_when_additional_config_file_path_is_not_found(
    get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
):
    valid_config_file_path = os.path.join(get_data_dir, "config.yaml")
    fake_additional_override_config_file_path = os.path.join(
        get_data_dir, "additional-config-not-found.yaml"
    )
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = valid_config_file_path
    with pytest.raises(ValueError):
        load_sagemaker_config(additional_config_paths=[fake_additional_override_config_file_path])
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]


def test_merge_behavior(get_data_dir, expected_merged_config):
    valid_config_file_path = os.path.join(get_data_dir, "sample_config_for_merge.yaml")
    additional_override_config_file_path = os.path.join(
        get_data_dir, "sample_additional_config_for_merge.yaml"
    )
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = valid_config_file_path
    assert expected_merged_config == load_sagemaker_config(
        additional_config_paths=[additional_override_config_file_path]
    )
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = additional_override_config_file_path
    assert expected_merged_config == load_sagemaker_config()
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
    del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]


def test_s3_config_file(
    config_file_as_yaml, valid_config_with_all_the_scopes, base_config_with_schema, s3_resource_mock
):
    config_file_bucket = "config-file-bucket"
    config_file_s3_prefix = "config/config.yaml"
    list_file_entry_mock = Mock()
    list_file_entry_mock.key = config_file_s3_prefix
    s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
        Prefix=config_file_s3_prefix
    ).all.return_value = [list_file_entry_mock]
    response_body_mock = MagicMock()
    response_body_mock.read.return_value = config_file_as_yaml.encode("utf-8")
    s3_resource_mock.Object(config_file_bucket, config_file_s3_prefix).get.return_value = {
        "Body": response_body_mock
    }
    config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
    expected_config = base_config_with_schema
    expected_config["SageMaker"] = valid_config_with_all_the_scopes
    assert expected_config == load_sagemaker_config(
        additional_config_paths=[config_file_s3_uri], s3_resource=s3_resource_mock
    )


def test_config_factory_when_default_s3_config_file_is_not_found(s3_resource_mock):
    config_file_bucket = "config-file-bucket"
    config_file_s3_prefix = "config/config.yaml"
    # Return empty list during list operation
    s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
        Prefix=config_file_s3_prefix
    ).all.return_value = []
    config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
    with pytest.raises(ValueError):
        load_sagemaker_config(
            additional_config_paths=[config_file_s3_uri], s3_resource=s3_resource_mock
        )


def test_s3_config_file_when_uri_provided_corresponds_to_a_path(
    config_file_as_yaml,
    valid_config_with_all_the_scopes,
    base_config_with_schema,
    s3_resource_mock,
):
    config_file_bucket = "config-file-bucket"
    config_file_s3_prefix = "config"
    list_of_files = ["/config.yaml", "/something.txt", "/README.MD"]
    list_s3_files_mock = []
    for file in list_of_files:
        entry_mock = Mock()
        entry_mock.key = config_file_s3_prefix + file
        list_s3_files_mock.append(entry_mock)
    s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
        Prefix=config_file_s3_prefix
    ).all.return_value = list_s3_files_mock
    response_body_mock = MagicMock()
    response_body_mock.read.return_value = config_file_as_yaml.encode("utf-8")
    s3_resource_mock.Object(config_file_bucket, config_file_s3_prefix).get.return_value = {
        "Body": response_body_mock
    }
    config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
    expected_config = base_config_with_schema
    expected_config["SageMaker"] = valid_config_with_all_the_scopes
    assert expected_config == load_sagemaker_config(
        additional_config_paths=[config_file_s3_uri], s3_resource=s3_resource_mock
    )


def test_merge_of_s3_default_config_file_and_regular_config_file(
    get_data_dir, expected_merged_config, s3_resource_mock
):
    config_file_content_path = os.path.join(get_data_dir, "sample_config_for_merge.yaml")
    config_file_as_yaml = open(config_file_content_path, "r").read()
    config_file_bucket = "config-file-bucket"
    config_file_s3_prefix = "config/config.yaml"
    config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
    list_file_entry_mock = Mock()
    list_file_entry_mock.key = config_file_s3_prefix
    s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
        Prefix=config_file_s3_prefix
    ).all.return_value = [list_file_entry_mock]
    response_body_mock = MagicMock()
    response_body_mock.read.return_value = config_file_as_yaml.encode("utf-8")
    s3_resource_mock.Object(config_file_bucket, config_file_s3_prefix).get.return_value = {
        "Body": response_body_mock
    }
    additional_override_config_file_path = os.path.join(
        get_data_dir, "sample_additional_config_for_merge.yaml"
    )
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = config_file_s3_uri
    assert expected_merged_config == load_sagemaker_config(
        additional_config_paths=[additional_override_config_file_path],
        s3_resource=s3_resource_mock,
    )
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]


def test_logging_when_overriden_admin_and_user_configs_are_found(get_data_dir, caplog):
    # Should log info message stating defaults were fetched since both exist
    logger.propagate = True

    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = get_data_dir
    load_sagemaker_config()
    assert "Fetched defaults config from location: {}".format(get_data_dir) in caplog.text
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_ADMIN_CONFIG_FILE_PATH)
        not in caplog.text
    )
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_USER_CONFIG_FILE_PATH)
        not in caplog.text
    )
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
    del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
    logger.propagate = False

def test_logging_when_overriden_admin_is_found_and_default_user_config_not_found(get_data_dir, caplog):
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=logger.name)
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
    load_sagemaker_config()
    assert "Fetched defaults config from location: {}".format(get_data_dir) in caplog.text
    assert (
        "Not applying SDK defaults from location: {}".format(
            _DEFAULT_USER_CONFIG_FILE_PATH)
        in caplog.text
    )
    assert "Unable to load the config file from the location: {}".format(
        _DEFAULT_USER_CONFIG_FILE_PATH
    ) in caplog.text
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
    logger.propagate = False


def test_logging_when_default_admin_not_found_and_overriden_user_config_is_found(get_data_dir, caplog):
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=logger.name)
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = get_data_dir
    load_sagemaker_config()
    assert "Fetched defaults config from location: {}".format(get_data_dir) in caplog.text
    assert (
        "Not applying SDK defaults from location: {}".format(
            _DEFAULT_ADMIN_CONFIG_FILE_PATH)
        in caplog.text
    )
    assert "Unable to load the config file from the location: {}".format(
        _DEFAULT_ADMIN_CONFIG_FILE_PATH
    ) in caplog.text
    del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
    logger.propagate = False


def test_logging_when_default_admin_and_default_user_config_not_found(caplog):
    # Should log info message stating sdk defaults were not applied
    # for admin and user config since both are missing from default location
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=logger.name)
    load_sagemaker_config()
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_ADMIN_CONFIG_FILE_PATH)
        in caplog.text
    )
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_USER_CONFIG_FILE_PATH)
        in caplog.text
    )
    assert "Unable to load the config file from the location: {}".format(_DEFAULT_ADMIN_CONFIG_FILE_PATH) in caplog.text
    assert "Unable to load the config file from the location: {}".format(_DEFAULT_USER_CONFIG_FILE_PATH) in caplog.text
    logger.propagate = False


def test_logging_when_default_admin_and_overriden_user_config_not_found(get_data_dir, caplog):
    # Should only log info message stating sdk defaults were not applied from default admin config.
    # Failing to load overriden user config should throw exception.
    logger.propagate = True
    fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = fake_config_file_path
    with pytest.raises(ValueError):
        load_sagemaker_config()
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_ADMIN_CONFIG_FILE_PATH)
        in caplog.text
    )
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_USER_CONFIG_FILE_PATH)
        not in caplog.text
    )
    del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
    logger.propagate = False


def test_logging_when_overriden_admin_and_overridden_user_config_not_found(get_data_dir, caplog):
    # Should not log any info messages since both config paths are overridden.
    # Should throw an exception on failure since both will fail to load.
    logger.propagate = True
    fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = fake_config_file_path
    os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = fake_config_file_path
    with pytest.raises(ValueError):
        load_sagemaker_config()
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_ADMIN_CONFIG_FILE_PATH)
        not in caplog.text
    )
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_USER_CONFIG_FILE_PATH)
        not in caplog.text
    )
    del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
    del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
    logger.propagate = False


def test_logging_with_additional_configs_and_none_are_found(caplog):
    # Should log info message stating sdk defaults were not applied
    # for admin and user config since both are missing from default location.
    # Should throw exception when config in additional_config_path is missing
    logger.propagate = True
    with pytest.raises(ValueError):
        load_sagemaker_config(additional_config_paths=["fake-path"])
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_ADMIN_CONFIG_FILE_PATH)
        in caplog.text
    )
    assert (
        "Not applying SDK defaults from location: {}".format(_DEFAULT_USER_CONFIG_FILE_PATH)
        in caplog.text
    )
    logger.propagate = False
