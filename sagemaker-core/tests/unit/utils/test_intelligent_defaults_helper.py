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
"""Unit tests for sagemaker.core.utils.intelligent_defaults_helper module."""
from __future__ import absolute_import

import pytest
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock, mock_open

from sagemaker.core.utils.intelligent_defaults_helper import (
    load_default_configs,
    validate_sagemaker_config,
    _load_config_from_s3,
    _get_inferred_s3_uri,
    _load_config_from_file,
    load_default_configs_for_resource_name,
    get_config_value,
)
from sagemaker.core.utils.exceptions import (
    LocalConfigNotFoundError,
    S3ConfigNotFoundError,
    ConfigSchemaValidationError,
)


class TestLoadDefaultConfigs:
    """Test load_default_configs function."""

    @patch("sagemaker.core.utils.intelligent_defaults_helper._load_config_from_file")
    @patch("sagemaker.core.utils.intelligent_defaults_helper.validate_sagemaker_config")
    def test_load_default_configs_basic(self, mock_validate, mock_load_file):
        """Test loading default configs."""
        mock_load_file.return_value = {"SageMaker": {"PythonSDK": {}}}
        
        result = load_default_configs()
        
        assert isinstance(result, dict)

    @patch("sagemaker.core.utils.intelligent_defaults_helper._load_config_from_file")
    @patch("sagemaker.core.utils.intelligent_defaults_helper.validate_sagemaker_config")
    def test_load_default_configs_with_additional_paths(self, mock_validate, mock_load_file):
        """Test loading configs with additional paths."""
        mock_load_file.return_value = {"key": "value"}
        
        result = load_default_configs(additional_config_paths=["/path/to/config.yaml"])
        
        assert isinstance(result, dict)

    @patch("sagemaker.core.utils.intelligent_defaults_helper._load_config_from_file")
    @patch("sagemaker.core.utils.intelligent_defaults_helper._load_config_from_s3")
    @patch("sagemaker.core.utils.intelligent_defaults_helper.validate_sagemaker_config")
    def test_load_default_configs_from_s3(self, mock_validate, mock_s3, mock_file):
        """Test loading configs from S3."""
        mock_file.side_effect = ValueError()
        mock_s3.return_value = {"key": "value"}
        
        result = load_default_configs(additional_config_paths=["s3://bucket/config.yaml"])
        
        assert isinstance(result, dict)
        mock_s3.assert_called_once()

    @patch("sagemaker.core.utils.intelligent_defaults_helper._load_config_from_file")
    def test_load_default_configs_validation_error(self, mock_load_file):
        """Test validation error handling."""
        mock_load_file.return_value = {"invalid": "config"}
        
        with pytest.raises(ConfigSchemaValidationError):
            load_default_configs(additional_config_paths=["/path/to/config.yaml"])


class TestValidateSagemakerConfig:
    """Test validate_sagemaker_config function."""

    def test_validate_sagemaker_config_valid(self):
        """Test validating valid config."""
        valid_config = {
            "SageMaker": {
                "PythonSDK": {
                    "Resources": {}
                }
            }
        }
        
        # Should not raise exception
        validate_sagemaker_config(valid_config)

    def test_validate_sagemaker_config_invalid(self):
        """Test validating invalid config."""
        invalid_config = {"invalid_key": "value"}
        
        with pytest.raises(Exception):
            validate_sagemaker_config(invalid_config)


class TestLoadConfigFromS3:
    """Test _load_config_from_s3 function."""

    @patch("sagemaker.core.utils.intelligent_defaults_helper._get_inferred_s3_uri")
    @patch("boto3.Session")
    def test_load_config_from_s3_basic(self, mock_session, mock_infer):
        """Test loading config from S3."""
        mock_infer.return_value = "s3://bucket/config.yaml"
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_object.get.return_value = {
            "Body": Mock(read=Mock(return_value=b"key: value"))
        }
        mock_s3_resource.Object.return_value = mock_s3_object
        
        result = _load_config_from_s3("s3://bucket/config.yaml", mock_s3_resource)
        
        assert isinstance(result, dict)


class TestGetInferredS3Uri:
    """Test _get_inferred_s3_uri function."""

    def test_get_inferred_s3_uri_single_file(self):
        """Test inferring S3 URI with single file."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_object = Mock()
        mock_object.key = "path/config.yaml"
        mock_bucket.objects.filter.return_value.all.return_value = [mock_object]
        mock_s3_resource.Bucket.return_value = mock_bucket
        
        result = _get_inferred_s3_uri("s3://bucket/path/config.yaml", mock_s3_resource)
        
        assert result == "s3://bucket/path/config.yaml"

    def test_get_inferred_s3_uri_directory(self):
        """Test inferring S3 URI with directory."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_object1 = Mock()
        mock_object1.key = "path/file1.yaml"
        mock_object2 = Mock()
        mock_object2.key = "path/config.yaml"
        mock_bucket.objects.filter.return_value.all.return_value = [mock_object1, mock_object2]
        mock_s3_resource.Bucket.return_value = mock_bucket
        
        result = _get_inferred_s3_uri("s3://bucket/path", mock_s3_resource)
        
        assert "config.yaml" in result

    def test_get_inferred_s3_uri_not_found(self):
        """Test inferring S3 URI when file not found."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.objects.filter.return_value.all.return_value = []
        mock_s3_resource.Bucket.return_value = mock_bucket
        
        with pytest.raises(S3ConfigNotFoundError):
            _get_inferred_s3_uri("s3://bucket/nonexistent", mock_s3_resource)


class TestLoadConfigFromFile:
    """Test _load_config_from_file function."""

    def test_load_config_from_file_basic(self, tmp_path):
        """Test loading config from file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")
        
        result = _load_config_from_file(str(config_file))
        
        assert result == {"key": "value"}

    def test_load_config_from_file_directory(self, tmp_path):
        """Test loading config from directory."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")
        
        result = _load_config_from_file(str(tmp_path))
        
        assert result == {"key": "value"}

    def test_load_config_from_file_not_found(self):
        """Test loading config from non-existent file."""
        with pytest.raises(ValueError):
            _load_config_from_file("/nonexistent/path/config.yaml")


class TestLoadDefaultConfigsForResourceName:
    """Test load_default_configs_for_resource_name function."""

    @patch("sagemaker.core.utils.intelligent_defaults_helper.load_default_configs")
    def test_load_default_configs_for_resource_name_found(self, mock_load):
        """Test loading configs for existing resource."""
        mock_load.return_value = {
            "SageMaker": {
                "PythonSDK": {
                    "Resources": {
                        "TrainingJob": {"InstanceType": "ml.m5.large"}
                    }
                }
            }
        }
        
        result = load_default_configs_for_resource_name("TrainingJob")
        
        assert result == {"InstanceType": "ml.m5.large"}

    @patch("sagemaker.core.utils.intelligent_defaults_helper.load_default_configs")
    def test_load_default_configs_for_resource_name_not_found(self, mock_load):
        """Test loading configs for non-existent resource."""
        mock_load.return_value = {
            "SageMaker": {
                "PythonSDK": {
                    "Resources": {}
                }
            }
        }
        
        result = load_default_configs_for_resource_name("NonExistentResource")
        
        assert result is None

    @patch("sagemaker.core.utils.intelligent_defaults_helper.load_default_configs")
    def test_load_default_configs_for_resource_name_no_config(self, mock_load):
        """Test loading configs when no config exists."""
        load_default_configs_for_resource_name.cache_clear()
        mock_load.return_value = {}
        
        result = load_default_configs_for_resource_name("TrainingJob")
        
        assert result == {}


class TestGetConfigValue:
    """Test get_config_value function."""

    def test_get_config_value_from_resource_defaults(self):
        """Test getting value from resource defaults."""
        resource_defaults = {"InstanceType": "ml.m5.large"}
        global_defaults = {"InstanceType": "ml.t2.medium"}
        
        result = get_config_value("InstanceType", resource_defaults, global_defaults)
        
        assert result == "ml.m5.large"

    def test_get_config_value_from_global_defaults(self):
        """Test getting value from global defaults."""
        resource_defaults = {}
        global_defaults = {"InstanceType": "ml.t2.medium"}
        
        result = get_config_value("InstanceType", resource_defaults, global_defaults)
        
        assert result == "ml.t2.medium"

    def test_get_config_value_not_found(self):
        """Test getting value when not found."""
        resource_defaults = {}
        global_defaults = {}
        
        result = get_config_value("InstanceType", resource_defaults, global_defaults)
        
        assert result is None

    def test_get_config_value_none_defaults(self):
        """Test getting value with None defaults."""
        result = get_config_value("InstanceType", None, None)
        
        assert result is None


class TestEnvironmentVariables:
    """Test environment variable handling."""

    @patch.dict(os.environ, {"SAGEMAKER_CORE_ADMIN_CONFIG_OVERRIDE": "/custom/admin/config.yaml"})
    @patch("sagemaker.core.utils.intelligent_defaults_helper._load_config_from_file")
    @patch("sagemaker.core.utils.intelligent_defaults_helper.validate_sagemaker_config")
    def test_load_default_configs_with_env_override(self, mock_validate, mock_load_file):
        """Test loading configs with environment variable override."""
        mock_load_file.return_value = {"key": "value"}
        
        result = load_default_configs()
        
        # Should attempt to load from custom path
        assert isinstance(result, dict)

    @patch.dict(os.environ, {"SAGEMAKER_CORE_USER_CONFIG_OVERRIDE": "/custom/user/config.yaml"})
    @patch("sagemaker.core.utils.intelligent_defaults_helper._load_config_from_file")
    @patch("sagemaker.core.utils.intelligent_defaults_helper.validate_sagemaker_config")
    def test_load_default_configs_with_user_env_override(self, mock_validate, mock_load_file):
        """Test loading configs with user environment variable override."""
        mock_load_file.return_value = {"key": "value"}
        
        result = load_default_configs()
        
        assert isinstance(result, dict)
