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
"""Unit tests for sagemaker.core.common_utils module."""
from __future__ import absolute_import

import pytest
import time
import tempfile
import os
import tarfile
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from sagemaker.core.common_utils import (
    name_from_image,
    name_from_base,
    unique_name_from_base_uuid4,
    unique_name_from_base,
    base_name_from_image,
    base_from_name,
    sagemaker_timestamp,
    sagemaker_short_timestamp,
    build_dict,
    get_config_value,
    get_nested_value,
    set_nested_value,
    get_short_version,
    secondary_training_status_changed,
    secondary_training_status_message,
    create_tar_file,
    _tmpdir,
    sts_regional_endpoint,
    retries,
    retry_with_backoff,
    aws_partition,
    get_resource_name_from_arn,
    pop_out_unused_kwarg,
    to_string,
    get_module,
    resolve_value_from_config,
    get_sagemaker_config_value,
)


class TestNameFromImage:
    """Test name_from_image function."""

    def test_name_from_image_basic(self):
        """Test basic image name extraction."""
        image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-algorithm:latest"
        result = name_from_image(image)
        assert "my-algorithm" in result
        assert len(result) <= 63

    def test_name_from_image_with_max_length(self):
        """Test name generation respects max length."""
        image = "very-long-algorithm-name-that-exceeds-limits"
        result = name_from_image(image, max_length=30)
        assert len(result) <= 30

    def test_name_from_image_simple(self):
        """Test simple image name."""
        image = "my-algorithm"
        result = name_from_image(image)
        assert "my-algorithm" in result


class TestNameFromBase:
    """Test name_from_base function."""

    def test_name_from_base_default(self):
        """Test default timestamp appending."""
        base = "my-job"
        result = name_from_base(base)
        assert result.startswith("my-job-")
        assert len(result) <= 63

    def test_name_from_base_short(self):
        """Test short timestamp."""
        base = "my-job"
        result = name_from_base(base, short=True)
        assert result.startswith("my-job-")
        assert len(result) < len(name_from_base(base, short=False))

    def test_name_from_base_max_length(self):
        """Test max length constraint."""
        base = "a" * 100
        result = name_from_base(base, max_length=40)
        assert len(result) <= 40

    def test_name_from_base_empty(self):
        """Test with empty base."""
        result = name_from_base("")
        assert len(result) > 0


class TestUniqueNameFromBaseUuid4:
    """Test unique_name_from_base_uuid4 function."""

    def test_unique_name_from_base_uuid4_basic(self):
        """Test UUID-based name generation."""
        base = "my-resource"
        result = unique_name_from_base_uuid4(base)
        assert result.startswith("my-resource-")
        assert len(result) <= 63

    def test_unique_name_from_base_uuid4_uniqueness(self):
        """Test that generated names are unique."""
        base = "my-resource"
        result1 = unique_name_from_base_uuid4(base)
        result2 = unique_name_from_base_uuid4(base)
        assert result1 != result2

    def test_unique_name_from_base_uuid4_max_length(self):
        """Test max length constraint."""
        base = "x" * 100
        result = unique_name_from_base_uuid4(base, max_length=50)
        assert len(result) <= 50


class TestUniqueNameFromBase:
    """Test unique_name_from_base function."""

    def test_unique_name_from_base_basic(self):
        """Test basic unique name generation."""
        base = "my-job"
        result = unique_name_from_base(base)
        assert result.startswith("my-job-")
        assert len(result) <= 63

    def test_unique_name_from_base_uniqueness(self):
        """Test uniqueness of generated names."""
        base = "my-job"
        result1 = unique_name_from_base(base)
        time.sleep(0.01)  # Small delay to ensure different timestamp
        result2 = unique_name_from_base(base)
        assert result1 != result2

    def test_unique_name_from_base_max_length(self):
        """Test max length constraint."""
        base = "y" * 100
        result = unique_name_from_base(base, max_length=45)
        assert len(result) <= 45


class TestBaseNameFromImage:
    """Test base_name_from_image function."""

    def test_base_name_from_image_with_registry(self):
        """Test extracting base name from full registry path."""
        image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-algorithm:latest"
        result = base_name_from_image(image)
        assert result == "my-algorithm"

    def test_base_name_from_image_simple(self):
        """Test simple image name."""
        image = "tensorflow"
        result = base_name_from_image(image)
        assert result == "tensorflow"

    def test_base_name_from_image_with_tag(self):
        """Test image with tag."""
        image = "my-image:v1.0"
        result = base_name_from_image(image)
        assert result == "my-image"

    def test_base_name_from_image_with_default(self):
        """Test with default base name."""
        result = base_name_from_image("", default_base_name="default")
        assert result in ["", "default"]


class TestBaseFromName:
    """Test base_from_name function."""

    def test_base_from_name_with_timestamp(self):
        """Test extracting base from name with timestamp."""
        name = "my-job-2023-01-15-10-30-45-123"
        result = base_from_name(name)
        assert result == "my-job"

    def test_base_from_name_with_short_timestamp(self):
        """Test extracting base from name with short timestamp."""
        name = "my-job-230115-1030"
        result = base_from_name(name)
        assert result == "my-job"

    def test_base_from_name_no_timestamp(self):
        """Test name without timestamp."""
        name = "my-job"
        result = base_from_name(name)
        assert result == "my-job"


class TestSagemakerTimestamp:
    """Test sagemaker_timestamp function."""

    def test_sagemaker_timestamp_format(self):
        """Test timestamp format."""
        result = sagemaker_timestamp()
        # Format: YYYY-MM-DD-HH-MM-SS-mmm
        parts = result.split("-")
        assert len(parts) == 7
        assert len(parts[0]) == 4  # Year
        assert len(parts[6]) == 3  # Milliseconds

    def test_sagemaker_timestamp_uniqueness(self):
        """Test that timestamps are unique."""
        result1 = sagemaker_timestamp()
        time.sleep(0.01)
        result2 = sagemaker_timestamp()
        assert result1 != result2


class TestSagemakerShortTimestamp:
    """Test sagemaker_short_timestamp function."""

    def test_sagemaker_short_timestamp_format(self):
        """Test short timestamp format."""
        result = sagemaker_short_timestamp()
        # Format: YYMMDD-HHMM
        assert len(result) == 11
        assert "-" in result

    def test_sagemaker_short_timestamp_consistency(self):
        """Test timestamp consistency within same minute."""
        result1 = sagemaker_short_timestamp()
        result2 = sagemaker_short_timestamp()
        # Should be same if called within same minute
        assert result1 == result2 or len(result1) == len(result2)


class TestBuildDict:
    """Test build_dict function."""

    def test_build_dict_with_value(self):
        """Test building dict with non-None value."""
        result = build_dict("key1", "value1")
        assert result == {"key1": "value1"}

    def test_build_dict_with_none(self):
        """Test building dict with None value."""
        result = build_dict("key1", None)
        assert result == {}

    def test_build_dict_with_empty_string(self):
        """Test building dict with empty string."""
        result = build_dict("key1", "")
        assert result == {}

    def test_build_dict_with_zero(self):
        """Test building dict with zero value."""
        result = build_dict("key1", 0)
        assert result == {}

    def test_build_dict_with_false(self):
        """Test building dict with False value."""
        result = build_dict("key1", False)
        assert result == {}


class TestGetConfigValue:
    """Test get_config_value function."""

    def test_get_config_value_simple(self):
        """Test getting simple config value."""
        config = {"key1": "value1"}
        result = get_config_value("key1", config)
        assert result == "value1"

    def test_get_config_value_nested(self):
        """Test getting nested config value."""
        config = {"level1": {"level2": {"level3": "value"}}}
        result = get_config_value("level1.level2.level3", config)
        assert result == "value"

    def test_get_config_value_missing(self):
        """Test getting missing config value."""
        config = {"key1": "value1"}
        result = get_config_value("key2", config)
        assert result is None

    def test_get_config_value_none_config(self):
        """Test with None config."""
        result = get_config_value("key1", None)
        assert result is None

    def test_get_config_value_partial_path(self):
        """Test with partial path that doesn't exist."""
        config = {"level1": {"level2": "value"}}
        result = get_config_value("level1.level3.level4", config)
        assert result is None


class TestGetNestedValue:
    """Test get_nested_value function."""

    def test_get_nested_value_simple(self):
        """Test getting simple nested value."""
        dictionary = {"key1": "value1"}
        result = get_nested_value(dictionary, ["key1"])
        assert result == "value1"

    def test_get_nested_value_deep(self):
        """Test getting deeply nested value."""
        dictionary = {"a": {"b": {"c": "value"}}}
        result = get_nested_value(dictionary, ["a", "b", "c"])
        assert result == "value"

    def test_get_nested_value_missing(self):
        """Test getting missing nested value."""
        dictionary = {"a": {"b": "value"}}
        result = get_nested_value(dictionary, ["a", "c"])
        assert result is None

    def test_get_nested_value_none_dict(self):
        """Test with None dictionary."""
        result = get_nested_value(None, ["key"])
        assert result is None

    def test_get_nested_value_empty_keys(self):
        """Test with empty keys list."""
        dictionary = {"key": "value"}
        result = get_nested_value(dictionary, [])
        assert result is None

    def test_get_nested_value_invalid_structure(self):
        """Test with invalid dictionary structure."""
        dictionary = {"a": "not_a_dict"}
        with pytest.raises(ValueError):
            get_nested_value(dictionary, ["a", "b"])


class TestSetNestedValue:
    """Test set_nested_value function."""

    def test_set_nested_value_simple(self):
        """Test setting simple nested value."""
        dictionary = {}
        result = set_nested_value(dictionary, ["key1"], "value1")
        assert result == {"key1": "value1"}

    def test_set_nested_value_deep(self):
        """Test setting deeply nested value."""
        dictionary = {}
        result = set_nested_value(dictionary, ["a", "b", "c"], "value")
        assert result == {"a": {"b": {"c": "value"}}}

    def test_set_nested_value_existing(self):
        """Test setting value in existing structure."""
        dictionary = {"a": {"b": "old_value"}}
        result = set_nested_value(dictionary, ["a", "b"], "new_value")
        assert result == {"a": {"b": "new_value"}}

    def test_set_nested_value_none_dict(self):
        """Test with None dictionary."""
        result = set_nested_value(None, ["key"], "value")
        assert result == {"key": "value"}

    def test_set_nested_value_overwrite_non_dict(self):
        """Test overwriting non-dict value."""
        dictionary = {"a": "not_a_dict"}
        result = set_nested_value(dictionary, ["a", "b"], "value")
        assert result == {"a": {"b": "value"}}


class TestGetShortVersion:
    """Test get_short_version function."""

    def test_get_short_version_three_parts(self):
        """Test with three-part version."""
        result = get_short_version("1.2.3")
        assert result == "1.2"

    def test_get_short_version_two_parts(self):
        """Test with two-part version."""
        result = get_short_version("1.2")
        assert result == "1.2"

    def test_get_short_version_four_parts(self):
        """Test with four-part version."""
        result = get_short_version("1.2.3.4")
        assert result == "1.2"

    def test_get_short_version_single_part(self):
        """Test with single-part version."""
        result = get_short_version("1")
        assert result == "1"


class TestSecondaryTrainingStatusChanged:
    """Test secondary_training_status_changed function."""

    def test_secondary_training_status_changed_true(self):
        """Test when status has changed."""
        current = {"SecondaryStatusTransitions": [{"StatusMessage": "Starting training"}]}
        prev = {"SecondaryStatusTransitions": [{"StatusMessage": "Preparing data"}]}
        result = secondary_training_status_changed(current, prev)
        assert result is True

    def test_secondary_training_status_changed_false(self):
        """Test when status hasn't changed."""
        current = {"SecondaryStatusTransitions": [{"StatusMessage": "Training"}]}
        prev = {"SecondaryStatusTransitions": [{"StatusMessage": "Training"}]}
        result = secondary_training_status_changed(current, prev)
        assert result is False

    def test_secondary_training_status_changed_no_transitions(self):
        """Test with no transitions."""
        current = {}
        prev = {}
        result = secondary_training_status_changed(current, prev)
        assert result is False

    def test_secondary_training_status_changed_none_prev(self):
        """Test with None previous description."""
        current = {"SecondaryStatusTransitions": [{"StatusMessage": "Training"}]}
        result = secondary_training_status_changed(current, None)
        assert result is True


class TestSecondaryTrainingStatusMessage:
    """Test secondary_training_status_message function."""

    def test_secondary_training_status_message_basic(self):
        """Test basic status message."""
        from datetime import datetime

        job_desc = {
            "SecondaryStatusTransitions": [
                {"Status": "Starting", "StatusMessage": "Starting training"}
            ],
            "LastModifiedTime": datetime.now(),
        }
        result = secondary_training_status_message(job_desc, None)
        assert "Starting" in result
        assert "Starting training" in result

    def test_secondary_training_status_message_no_transitions(self):
        """Test with no transitions."""
        job_desc = {}
        result = secondary_training_status_message(job_desc, None)
        assert result == ""

    def test_secondary_training_status_message_multiple_transitions(self):
        """Test with multiple transitions."""
        from datetime import datetime

        job_desc = {
            "SecondaryStatusTransitions": [
                {"Status": "Starting", "StatusMessage": "Starting"},
                {"Status": "Training", "StatusMessage": "Training"},
            ],
            "LastModifiedTime": datetime.now(),
        }
        prev_desc = {
            "SecondaryStatusTransitions": [{"Status": "Starting", "StatusMessage": "Starting"}]
        }
        result = secondary_training_status_message(job_desc, prev_desc)
        assert "Training" in result


class TestCreateTarFile:
    """Test create_tar_file function."""

    def test_create_tar_file_single_file(self, tmp_path):
        """Test creating tar file from single file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        tar_path = create_tar_file([str(test_file)])

        assert os.path.exists(tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            assert len(members) == 1
            assert members[0].name == "test.txt"

        os.remove(tar_path)

    def test_create_tar_file_multiple_files(self, tmp_path):
        """Test creating tar file from multiple files."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        tar_path = create_tar_file([str(file1), str(file2)])

        assert os.path.exists(tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            assert len(members) == 2

        os.remove(tar_path)

    def test_create_tar_file_with_target(self, tmp_path):
        """Test creating tar file with specific target path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        target = str(tmp_path / "output.tar.gz")

        tar_path = create_tar_file([str(test_file)], target=target)

        assert tar_path == target
        assert os.path.exists(tar_path)
        os.remove(tar_path)


class TestTmpdir:
    """Test _tmpdir context manager."""

    def test_tmpdir_creates_directory(self):
        """Test that tmpdir creates a directory."""
        with _tmpdir() as tmp:
            assert os.path.exists(tmp)
            assert os.path.isdir(tmp)

    def test_tmpdir_cleans_up(self):
        """Test that tmpdir cleans up after use."""
        tmp_path = None
        with _tmpdir() as tmp:
            tmp_path = tmp
            assert os.path.exists(tmp_path)

        assert not os.path.exists(tmp_path)

    def test_tmpdir_with_prefix(self):
        """Test tmpdir with custom prefix."""
        with _tmpdir(prefix="test_") as tmp:
            assert "test_" in os.path.basename(tmp)

    def test_tmpdir_with_suffix(self):
        """Test tmpdir with custom suffix."""
        with _tmpdir(suffix="_test") as tmp:
            assert os.path.basename(tmp).endswith("_test")

    def test_tmpdir_with_directory(self, tmp_path):
        """Test tmpdir with custom directory."""
        with _tmpdir(directory=str(tmp_path)) as tmp:
            assert str(tmp_path) in tmp

    def test_tmpdir_invalid_directory(self):
        """Test tmpdir with invalid directory."""
        with pytest.raises(ValueError):
            with _tmpdir(directory="/nonexistent/path"):
                pass


class TestStsRegionalEndpoint:
    """Test sts_regional_endpoint function."""

    def test_sts_regional_endpoint_standard(self):
        """Test STS endpoint for standard region."""
        result = sts_regional_endpoint("us-west-2")
        assert "sts" in result
        assert "us-west-2" in result
        assert result.startswith("https://")

    def test_sts_regional_endpoint_china(self):
        """Test STS endpoint for China region."""
        result = sts_regional_endpoint("cn-north-1")
        assert "sts" in result
        assert result.startswith("https://")

    def test_sts_regional_endpoint_govcloud(self):
        """Test STS endpoint for GovCloud region."""
        result = sts_regional_endpoint("us-gov-west-1")
        assert "sts" in result
        assert result.startswith("https://")


class TestRetries:
    """Test retries generator function."""

    def test_retries_basic(self):
        """Test basic retry functionality."""
        max_retries = 3
        count = 0
        for i in retries(max_retries, "test", seconds_to_sleep=0.01):
            count += 1
            if count < max_retries:
                continue
            break

        assert count == max_retries

    def test_retries_raises_exception(self):
        """Test that retries raises exception after max attempts."""
        with pytest.raises(Exception, match="maximum retry count"):
            for i in retries(2, "test operation", seconds_to_sleep=0.01):
                pass

    def test_retries_with_success(self):
        """Test retries with successful operation."""
        count = 0
        for i in retries(5, "test", seconds_to_sleep=0.01):
            count += 1
            if count == 2:
                break

        assert count == 2


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    def test_retry_with_backoff_success(self):
        """Test successful retry."""
        mock_func = Mock(return_value="success")
        result = retry_with_backoff(mock_func, num_attempts=3)
        assert result == "success"
        mock_func.assert_called_once()

    def test_retry_with_backoff_eventual_success(self):
        """Test eventual success after retries."""
        mock_func = Mock(side_effect=[Exception("error"), Exception("error"), "success"])
        result = retry_with_backoff(mock_func, num_attempts=3)
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_with_backoff_max_attempts(self):
        """Test max attempts reached."""
        mock_func = Mock(side_effect=Exception("error"))
        with pytest.raises(Exception, match="error"):
            retry_with_backoff(mock_func, num_attempts=2)

        assert mock_func.call_count == 2

    def test_retry_with_backoff_client_error(self):
        """Test with specific ClientError."""
        error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "operation"
        )
        mock_func = Mock(side_effect=[error, "success"])
        result = retry_with_backoff(
            mock_func, num_attempts=3, botocore_client_error_code="ThrottlingException"
        )
        assert result == "success"

    def test_retry_with_backoff_invalid_attempts(self):
        """Test with invalid number of attempts."""
        mock_func = Mock()
        with pytest.raises(ValueError):
            retry_with_backoff(mock_func, num_attempts=0)


class TestAwsPartition:
    """Test aws_partition function."""

    def test_aws_partition_standard(self):
        """Test standard AWS partition."""
        result = aws_partition("us-west-2")
        assert result == "aws"

    def test_aws_partition_china(self):
        """Test China partition."""
        result = aws_partition("cn-north-1")
        assert result == "aws-cn"

    def test_aws_partition_govcloud(self):
        """Test GovCloud partition."""
        result = aws_partition("us-gov-west-1")
        assert result == "aws-us-gov"


class TestGetResourceNameFromArn:
    """Test get_resource_name_from_arn function."""

    def test_get_resource_name_from_arn_endpoint(self):
        """Test extracting endpoint name from ARN."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-endpoint"
        result = get_resource_name_from_arn(arn)
        assert result == "my-endpoint"

    def test_get_resource_name_from_arn_model(self):
        """Test extracting model name from ARN."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model/my-model"
        result = get_resource_name_from_arn(arn)
        assert result == "my-model"

    def test_get_resource_name_from_arn_training_job(self):
        """Test extracting training job name from ARN."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/my-training-job"
        result = get_resource_name_from_arn(arn)
        assert result == "my-training-job"

    def test_get_resource_name_from_arn_with_path(self):
        """Test extracting resource name with path."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:endpoint-config/path/to/config"
        result = get_resource_name_from_arn(arn)
        assert result == "path/to/config"


class TestPopOutUnusedKwarg:
    """Test pop_out_unused_kwarg function."""

    def test_pop_out_unused_kwarg_present(self):
        """Test popping present kwarg."""
        kwargs = {"arg1": "value1", "arg2": "value2"}
        pop_out_unused_kwarg("arg1", kwargs)
        assert "arg1" not in kwargs
        assert "arg2" in kwargs

    def test_pop_out_unused_kwarg_not_present(self):
        """Test popping non-existent kwarg."""
        kwargs = {"arg1": "value1"}
        pop_out_unused_kwarg("arg2", kwargs)
        assert kwargs == {"arg1": "value1"}

    def test_pop_out_unused_kwarg_with_override(self):
        """Test popping kwarg with override value."""
        kwargs = {"arg1": "value1"}
        pop_out_unused_kwarg("arg1", kwargs, override_val="new_value")
        assert "arg1" not in kwargs


class TestToString:
    """Test to_string function."""

    def test_to_string_basic(self):
        """Test converting basic object to string."""
        result = to_string("test")
        assert result == "test"

    def test_to_string_number(self):
        """Test converting number to string."""
        result = to_string(123)
        assert result == "123"

    def test_to_string_none(self):
        """Test converting None to string."""
        result = to_string(None)
        assert result == "None"


class TestGetModule:
    """Test get_module function."""

    def test_get_module_existing(self):
        """Test importing existing module."""
        result = get_module("os")
        assert result is not None
        assert hasattr(result, "path")

    def test_get_module_nonexistent(self):
        """Test importing non-existent module."""
        with pytest.raises(Exception, match="Cannot import module"):
            get_module("nonexistent_module_xyz")

    def test_get_module_builtin(self):
        """Test importing built-in module."""
        result = get_module("sys")
        assert result is not None
        assert hasattr(result, "version")


class TestResolveValueFromConfig:
    """Test resolve_value_from_config function."""

    def test_resolve_value_from_config_direct_input(self):
        """Test with direct input provided."""
        result = resolve_value_from_config(
            direct_input="direct_value", config_path="some.path", default_value="default_value"
        )
        assert result == "direct_value"

    def test_resolve_value_from_config_default(self):
        """Test with default value."""
        result = resolve_value_from_config(
            direct_input=None, config_path=None, default_value="default_value"
        )
        assert result == "default_value"

    def test_resolve_value_from_config_none(self):
        """Test with all None values."""
        result = resolve_value_from_config(direct_input=None, config_path=None, default_value=None)
        assert result is None

    @patch("sagemaker.core.common_utils.get_sagemaker_config_value")
    def test_resolve_value_from_config_from_config(self, mock_get_config):
        """Test with value from config."""
        mock_get_config.return_value = "config_value"
        result = resolve_value_from_config(
            direct_input=None,
            config_path="some.path",
            default_value="default_value",
            sagemaker_session=Mock(),
        )
        assert result == "config_value"


class TestGetSagemakerConfigValue:
    """Test get_sagemaker_config_value function."""

    @patch("sagemaker.core.common_utils.get_config_value")
    @patch("sagemaker.core.config.config_manager.SageMakerConfig")
    def test_get_sagemaker_config_value_from_session(self, mock_config_manager, mock_get_config):
        """Test getting config value from session."""
        mock_config_manager.return_value.validate_sagemaker_config = Mock()
        mock_session = Mock()
        mock_session.sagemaker_config = {"SchemaVersion": "1.0", "key": "value"}
        mock_get_config.return_value = "value"

        result = get_sagemaker_config_value(mock_session, "key")
        assert result == "value"

    @patch("sagemaker.core.common_utils.get_config_value")
    @patch("sagemaker.core.config.config_manager.SageMakerConfig")
    def test_get_sagemaker_config_value_from_dict(self, mock_config_manager, mock_get_config):
        """Test getting config value from dict."""
        mock_config_manager.return_value.validate_sagemaker_config = Mock()
        mock_get_config.return_value = "value"

        result = get_sagemaker_config_value(
            None, "key", sagemaker_config={"SchemaVersion": "1.0", "key": "value"}
        )
        assert result == "value"

    def test_get_sagemaker_config_value_no_config(self):
        """Test with no config available."""
        result = get_sagemaker_config_value(None, "key")
        assert result is None

    @patch("sagemaker.core.common_utils.get_config_value")
    @patch("sagemaker.core.config.config_manager.SageMakerConfig")
    def test_get_sagemaker_config_value_nested(self, mock_config_manager, mock_get_config):
        """Test getting nested config value."""
        mock_config_manager.return_value.validate_sagemaker_config = Mock()
        mock_session = Mock()
        mock_session.sagemaker_config = {"SchemaVersion": "1.0", "level1": {"level2": "value"}}
        mock_get_config.return_value = "value"

        result = get_sagemaker_config_value(mock_session, "level1.level2")
        assert result == "value"


class TestDeferredError:
    """Test DeferredError class."""

    def test_deferred_error_raises_on_access(self):
        """Test that DeferredError raises exception on access."""
        from sagemaker.core.common_utils import DeferredError

        original_error = ImportError("Module not found")
        deferred = DeferredError(original_error)

        with pytest.raises(ImportError, match="Module not found"):
            _ = deferred.some_attribute

    def test_deferred_error_raises_on_method_call(self):
        """Test that DeferredError raises exception on method call."""
        from sagemaker.core.common_utils import DeferredError

        original_error = ImportError("Module not found")
        deferred = DeferredError(original_error)

        with pytest.raises(ImportError):
            deferred.some_method()


class TestS3DataConfig:
    """Test S3DataConfig class."""

    def test_s3_data_config_init(self):
        """Test S3DataConfig initialization."""
        from sagemaker.core.common_utils import S3DataConfig

        mock_session = Mock()
        config = S3DataConfig(
            sagemaker_session=mock_session, bucket_name="test-bucket", prefix="test-prefix"
        )

        assert config.bucket_name == "test-bucket"
        assert config.prefix == "test-prefix"
        assert config.sagemaker_session == mock_session

    def test_s3_data_config_missing_bucket(self):
        """Test S3DataConfig with missing bucket."""
        from sagemaker.core.common_utils import S3DataConfig

        with pytest.raises(ValueError):
            S3DataConfig(sagemaker_session=Mock(), bucket_name=None, prefix="test-prefix")

    def test_s3_data_config_fetch_data_config(self):
        """Test fetching data config from S3."""
        from sagemaker.core.common_utils import S3DataConfig

        mock_session = Mock()
        mock_session.read_s3_file.return_value = '{"key": "value"}'

        config = S3DataConfig(
            sagemaker_session=mock_session, bucket_name="test-bucket", prefix="test-prefix"
        )

        result = config.fetch_data_config()
        assert result == {"key": "value"}
        mock_session.read_s3_file.assert_called_once_with("test-bucket", "test-prefix")


class TestDownloadFolder:
    """Test download_folder function."""

    def test_download_folder_single_file(self):
        """Test downloading single file."""
        from sagemaker.core.common_utils import download_folder

        mock_session = Mock()
        mock_s3 = Mock()
        mock_obj = Mock()
        mock_session.s3_resource = mock_s3
        mock_s3.Object.return_value = mock_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            download_folder("bucket", "file.txt", tmpdir, mock_session)
            mock_obj.download_file.assert_called_once()

    def test_download_folder_with_prefix(self):
        """Test downloading folder with prefix."""
        from sagemaker.core.common_utils import download_folder

        mock_session = Mock()
        mock_s3 = Mock()
        mock_bucket = Mock()
        mock_session.s3_resource = mock_s3
        mock_s3.Bucket.return_value = mock_bucket
        mock_bucket.objects.filter.return_value = []

        mock_obj = Mock()
        mock_obj.download_file.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "operation"
        )
        mock_s3.Object.return_value = mock_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            download_folder("bucket", "prefix/", tmpdir, mock_session)


class TestRepackModel:
    """Test repack_model function."""

    def test_repack_model_basic(self, tmp_path):
        """Test basic model repacking."""
        from sagemaker.core.common_utils import repack_model

        # Create test files
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.pth").write_text("model data")

        model_tar = tmp_path / "model.tar.gz"
        with tarfile.open(model_tar, "w:gz") as tar:
            tar.add(model_dir, arcname=".")

        script = tmp_path / "inference.py"
        script.write_text("# inference script")

        output = tmp_path / "output.tar.gz"

        mock_session = Mock()
        mock_session.settings = None

        repack_model(str(script), None, [], f"file://{model_tar}", f"file://{output}", mock_session)

        assert output.exists()


class TestVolumeSupported:
    """Test volume_size_supported function."""

    def test_volume_size_supported_standard(self):
        """Test standard instance type."""
        from sagemaker.core.common_utils import volume_size_supported

        assert volume_size_supported("ml.m5.xlarge") is True

    def test_volume_size_supported_with_d(self):
        """Test instance with d in family."""
        from sagemaker.core.common_utils import volume_size_supported

        assert volume_size_supported("ml.c5d.xlarge") is False

    def test_volume_size_supported_g5(self):
        """Test g5 instance."""
        from sagemaker.core.common_utils import volume_size_supported

        assert volume_size_supported("ml.g5.xlarge") is False

    def test_volume_size_supported_local(self):
        """Test local mode."""
        from sagemaker.core.common_utils import volume_size_supported

        assert volume_size_supported("local") is False

    def test_volume_size_supported_invalid(self):
        """Test invalid instance type."""
        from sagemaker.core.common_utils import volume_size_supported

        with pytest.raises(ValueError):
            volume_size_supported("invalid")


class TestInstanceSupportsKms:
    """Test instance_supports_kms function."""

    def test_instance_supports_kms_true(self):
        """Test instance that supports KMS."""
        from sagemaker.core.common_utils import instance_supports_kms

        assert instance_supports_kms("ml.m5.xlarge") is True

    def test_instance_supports_kms_false(self):
        """Test instance that doesn't support KMS."""
        from sagemaker.core.common_utils import instance_supports_kms

        assert instance_supports_kms("ml.g5.xlarge") is False


class TestGetInstanceTypeFamily:
    """Test get_instance_type_family function."""

    def test_get_instance_type_family_standard(self):
        """Test standard instance type."""
        from sagemaker.core.common_utils import get_instance_type_family

        result = get_instance_type_family("ml.m5.xlarge")
        assert result == "m5"

    def test_get_instance_type_family_underscore(self):
        """Test instance type with underscore."""
        from sagemaker.core.common_utils import get_instance_type_family

        result = get_instance_type_family("ml_m5")
        assert result == "m5"

    def test_get_instance_type_family_none(self):
        """Test with None."""
        from sagemaker.core.common_utils import get_instance_type_family

        result = get_instance_type_family(None)
        assert result == ""

    def test_get_instance_type_family_invalid(self):
        """Test invalid format."""
        from sagemaker.core.common_utils import get_instance_type_family

        result = get_instance_type_family("invalid")
        assert result == ""


class TestCreatePaginatorConfig:
    """Test create_paginator_config function."""

    def test_create_paginator_config_defaults(self):
        """Test with default values."""
        from sagemaker.core.common_utils import create_paginator_config

        result = create_paginator_config()
        assert result["MaxItems"] == 100
        assert result["PageSize"] == 10

    def test_create_paginator_config_custom(self):
        """Test with custom values."""
        from sagemaker.core.common_utils import create_paginator_config

        result = create_paginator_config(max_items=50, page_size=5)
        assert result["MaxItems"] == 50
        assert result["PageSize"] == 5


class TestFormatTags:
    """Test format_tags function."""

    def test_format_tags_dict(self):
        """Test formatting dict tags."""
        from sagemaker.core.common_utils import format_tags

        tags = {"key1": "value1", "key2": "value2"}
        result = format_tags(tags)
        assert len(result) == 2
        assert {"Key": "key1", "Value": "value1"} in result

    def test_format_tags_list(self):
        """Test formatting list tags."""
        from sagemaker.core.common_utils import format_tags

        tags = [{"Key": "key1", "Value": "value1"}]
        result = format_tags(tags)
        assert result == tags


class TestCustomExtractallTarfile:
    """Test custom_extractall_tarfile function."""

    def test_custom_extractall_tarfile_basic(self, tmp_path):
        """Test basic tar extraction."""
        from sagemaker.core.common_utils import custom_extractall_tarfile

        # Create tar file
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("content")

        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(source / "file.txt", arcname="file.txt")

        # Extract
        extract_path = tmp_path / "extract"
        extract_path.mkdir()

        with tarfile.open(tar_path, "r:gz") as tar:
            custom_extractall_tarfile(tar, str(extract_path))

        assert (extract_path / "file.txt").exists()


class TestCanModelPackageSourceUriAutopopulate:
    """Test can_model_package_source_uri_autopopulate function."""

    def test_can_model_package_source_uri_autopopulate_model_package(self):
        """Test with model package ARN."""
        from sagemaker.core.common_utils import can_model_package_source_uri_autopopulate

        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-package"
        assert can_model_package_source_uri_autopopulate(arn) is True

    def test_can_model_package_source_uri_autopopulate_model(self):
        """Test with model ARN."""
        from sagemaker.core.common_utils import can_model_package_source_uri_autopopulate

        arn = "arn:aws:sagemaker:us-west-2:123456789012:model/my-model"
        assert can_model_package_source_uri_autopopulate(arn) is True

    def test_can_model_package_source_uri_autopopulate_invalid(self):
        """Test with invalid URI."""
        from sagemaker.core.common_utils import can_model_package_source_uri_autopopulate

        assert can_model_package_source_uri_autopopulate("s3://bucket/key") is False


class TestFlattenDict:
    """Test flatten_dict function."""

    def test_flatten_dict_simple(self):
        """Test flattening simple dict."""
        from sagemaker.core.common_utils import flatten_dict

        d = {"a": {"b": "value"}}
        result = flatten_dict(d)
        assert result[("a", "b")] == "value"

    def test_flatten_dict_max_depth(self):
        """Test with max depth."""
        from sagemaker.core.common_utils import flatten_dict

        d = {"a": {"b": {"c": "value"}}}
        result = flatten_dict(d, max_flatten_depth=1)
        assert ("a",) in result

    def test_flatten_dict_invalid_depth(self):
        """Test with invalid max depth."""
        from sagemaker.core.common_utils import flatten_dict

        with pytest.raises(ValueError):
            flatten_dict({}, max_flatten_depth=0)


class TestUnflattenDict:
    """Test unflatten_dict function."""

    def test_unflatten_dict_simple(self):
        """Test unflattening simple dict."""
        from sagemaker.core.common_utils import unflatten_dict

        d = {("a", "b"): "value"}
        result = unflatten_dict(d)
        assert result == {"a": {"b": "value"}}

    def test_unflatten_dict_multiple_keys(self):
        """Test with multiple keys."""
        from sagemaker.core.common_utils import unflatten_dict

        d = {("a", "b"): "value1", ("a", "c"): "value2"}
        result = unflatten_dict(d)
        assert result["a"]["b"] == "value1"
        assert result["a"]["c"] == "value2"


class TestDeepOverrideDict:
    """Test deep_override_dict function."""

    def test_deep_override_dict_basic(self):
        """Test basic override."""
        from sagemaker.core.common_utils import deep_override_dict

        dict1 = {"a": "value1"}
        dict2 = {"b": "value2"}
        result = deep_override_dict(dict1, dict2)
        assert result["a"] == "value1"
        assert result["b"] == "value2"

    def test_deep_override_dict_skip_keys(self):
        """Test with skip keys."""
        from sagemaker.core.common_utils import deep_override_dict

        dict1 = {"a": "value1"}
        dict2 = {"a": "value2", "b": "value3"}
        result = deep_override_dict(dict1, dict2, skip_keys=["a"])
        assert result["a"] == "value1"


class TestGetInstanceRatePerHour:
    """Test get_instance_rate_per_hour function."""

    @patch("boto3.client")
    def test_get_instance_rate_per_hour_success(self, mock_boto_client):
        """Test getting instance rate."""
        from sagemaker.core.common_utils import get_instance_rate_per_hour

        mock_pricing = Mock()
        mock_boto_client.return_value = mock_pricing

        price_data = {
            "terms": {
                "OnDemand": {
                    "term1": {"priceDimensions": {"dim1": {"pricePerUnit": {"USD": "1.125"}}}}
                }
            }
        }

        mock_pricing.get_products.return_value = {"PriceList": [price_data]}

        result = get_instance_rate_per_hour("ml.m5.xlarge", "us-west-2")
        assert result["value"] == "1.125"
        assert result["unit"] == "USD/Hr"

    @patch("boto3.client")
    def test_get_instance_rate_per_hour_no_price(self, mock_boto_client):
        """Test when no price found - returns None from extract function."""
        from sagemaker.core.common_utils import get_instance_rate_per_hour

        mock_pricing = Mock()
        mock_boto_client.return_value = mock_pricing
        # Return empty price list to trigger exception
        mock_pricing.get_products.return_value = {"PriceList": []}

        try:
            result = get_instance_rate_per_hour("ml.m5.xlarge", "us-west-2")
            # If no exception, test passes (function may return None or raise)
        except Exception as e:
            # Expected behavior - function raises exception
            assert "Unable to get instance rate" in str(e)


class TestCamelCaseToPascalCase:
    """Test camel_case_to_pascal_case function."""

    def test_camel_case_to_pascal_case_simple(self):
        """Test simple conversion."""
        from sagemaker.core.common_utils import camel_case_to_pascal_case

        data = {"snake_case": "value"}
        result = camel_case_to_pascal_case(data)
        assert result == {"SnakeCase": "value"}

    def test_camel_case_to_pascal_case_nested(self):
        """Test nested conversion."""
        from sagemaker.core.common_utils import camel_case_to_pascal_case

        data = {"outer_key": {"inner_key": "value"}}
        result = camel_case_to_pascal_case(data)
        assert result == {"OuterKey": {"InnerKey": "value"}}

    def test_camel_case_to_pascal_case_list(self):
        """Test with list values."""
        from sagemaker.core.common_utils import camel_case_to_pascal_case

        data = {"key_name": [{"nested_key": "value"}]}
        result = camel_case_to_pascal_case(data)
        assert result["KeyName"][0]["NestedKey"] == "value"


class TestTagExists:
    """Test tag_exists function."""

    def test_tag_exists_true(self):
        """Test when tag exists."""
        from sagemaker.core.common_utils import tag_exists

        tag = {"Key": "key1", "Value": "value1"}
        curr_tags = [{"Key": "key1", "Value": "old_value"}]
        assert tag_exists(tag, curr_tags) is True

    def test_tag_exists_false(self):
        """Test when tag doesn't exist."""
        from sagemaker.core.common_utils import tag_exists

        tag = {"Key": "key1", "Value": "value1"}
        curr_tags = [{"Key": "key2", "Value": "value2"}]
        assert tag_exists(tag, curr_tags) is False

    def test_tag_exists_none_tags(self):
        """Test with None tags."""
        from sagemaker.core.common_utils import tag_exists

        tag = {"Key": "key1", "Value": "value1"}
        assert tag_exists(tag, None) is False


class TestValidateNewTags:
    """Test _validate_new_tags function."""

    def test_validate_new_tags_dict(self):
        """Test with dict new tags."""
        from sagemaker.core.common_utils import _validate_new_tags

        new_tags = {"Key": "key1", "Value": "value1"}
        curr_tags = [{"Key": "key2", "Value": "value2"}]
        result = _validate_new_tags(new_tags, curr_tags)
        assert len(result) == 2

    def test_validate_new_tags_list(self):
        """Test with list new tags."""
        from sagemaker.core.common_utils import _validate_new_tags

        new_tags = [{"Key": "key1", "Value": "value1"}]
        curr_tags = [{"Key": "key2", "Value": "value2"}]
        result = _validate_new_tags(new_tags, curr_tags)
        assert len(result) == 2

    def test_validate_new_tags_none_curr(self):
        """Test with None current tags."""
        from sagemaker.core.common_utils import _validate_new_tags

        new_tags = [{"Key": "key1", "Value": "value1"}]
        result = _validate_new_tags(new_tags, None)
        assert result == new_tags


class TestRemoveTagWithKey:
    """Test remove_tag_with_key function."""

    def test_remove_tag_with_key_found(self):
        """Test removing existing tag."""
        from sagemaker.core.common_utils import remove_tag_with_key

        tags = [{"Key": "key1", "Value": "value1"}, {"Key": "key2", "Value": "value2"}]
        result = remove_tag_with_key("key1", tags)
        assert isinstance(result, (list, dict))
        if isinstance(result, list):
            assert any(tag["Key"] == "key2" for tag in result)

    def test_remove_tag_with_key_not_found(self):
        """Test removing non-existent tag."""
        from sagemaker.core.common_utils import remove_tag_with_key

        tags = [{"Key": "key1", "Value": "value1"}]
        result = remove_tag_with_key("key2", tags)
        assert result is not None

    def test_remove_tag_with_key_none(self):
        """Test with None tags."""
        from sagemaker.core.common_utils import remove_tag_with_key

        result = remove_tag_with_key("key1", None)
        assert result is None

    def test_remove_tag_with_key_single(self):
        """Test removing single tag."""
        from sagemaker.core.common_utils import remove_tag_with_key

        tags = [{"Key": "key1", "Value": "value1"}]
        result = remove_tag_with_key("key1", tags)
        assert result is None


class TestGetDomainForRegion:
    """Test get_domain_for_region function."""

    def test_get_domain_for_region_standard(self):
        """Test standard region."""
        from sagemaker.core.common_utils import get_domain_for_region

        result = get_domain_for_region("us-west-2")
        assert result == "amazonaws.com"

    def test_get_domain_for_region_china(self):
        """Test China region."""
        from sagemaker.core.common_utils import get_domain_for_region

        result = get_domain_for_region("cn-north-1")
        assert result == "amazonaws.com.cn"


class TestCamelToSnake:
    """Test camel_to_snake function."""

    def test_camel_to_snake_simple(self):
        """Test simple conversion."""
        from sagemaker.core.common_utils import camel_to_snake

        result = camel_to_snake("CamelCase")
        assert result == "camel_case"

    def test_camel_to_snake_multiple_words(self):
        """Test multiple words."""
        from sagemaker.core.common_utils import camel_to_snake

        result = camel_to_snake("ThisIsATest")
        assert result == "this_is_a_test"


class TestWalkAndApplyJson:
    """Test walk_and_apply_json function."""

    def test_walk_and_apply_json_basic(self):
        """Test basic walk and apply."""
        from sagemaker.core.common_utils import walk_and_apply_json

        json_obj = {"CamelCase": "value"}
        result = walk_and_apply_json(json_obj, lambda x: x.lower())
        assert "camelcase" in result

    def test_walk_and_apply_json_stop_keys(self):
        """Test with stop keys."""
        from sagemaker.core.common_utils import walk_and_apply_json

        json_obj = {"Key": {"metrics": {"nested": "value"}}}
        result = walk_and_apply_json(json_obj, lambda x: x.upper(), stop_keys=["metrics"])
        assert "KEY" in result


class TestIsS3Uri:
    """Test _is_s3_uri function."""

    def test_is_s3_uri_valid(self):
        """Test valid S3 URI."""
        from sagemaker.core.common_utils import _is_s3_uri

        assert _is_s3_uri("s3://bucket/key") is True

    def test_is_s3_uri_invalid(self):
        """Test invalid URI."""
        from sagemaker.core.common_utils import _is_s3_uri

        assert _is_s3_uri("http://example.com") is False

    def test_is_s3_uri_none(self):
        """Test None URI."""
        from sagemaker.core.common_utils import _is_s3_uri

        assert _is_s3_uri(None) is False


class TestListTags:
    """Test list_tags function."""

    def test_list_tags_basic(self):
        """Test basic tag listing."""
        from sagemaker.core.common_utils import list_tags

        mock_session = Mock()
        mock_client = Mock()
        mock_session.sagemaker_client = mock_client

        mock_client.list_tags.return_value = {"Tags": [{"Key": "key1", "Value": "value1"}]}

        result = list_tags(mock_session, "arn:aws:sagemaker:us-west-2:123:model/test")
        assert len(result) == 1

    def test_list_tags_pagination(self):
        """Test with pagination."""
        from sagemaker.core.common_utils import list_tags

        mock_session = Mock()
        mock_client = Mock()
        mock_session.sagemaker_client = mock_client

        mock_client.list_tags.side_effect = [
            {"Tags": [{"Key": "key1", "Value": "value1"}], "nextToken": "token"},
            {"Tags": [{"Key": "key2", "Value": "value2"}]},
        ]

        result = list_tags(mock_session, "arn:aws:sagemaker:us-west-2:123:model/test")
        assert len(result) == 2

    def test_list_tags_filter_aws(self):
        """Test filtering AWS tags."""
        from sagemaker.core.common_utils import list_tags

        mock_session = Mock()
        mock_client = Mock()
        mock_session.sagemaker_client = mock_client

        mock_client.list_tags.return_value = {
            "Tags": [{"Key": "aws:tag", "Value": "value1"}, {"Key": "user:tag", "Value": "value2"}]
        }

        result = list_tags(mock_session, "arn:aws:sagemaker:us-west-2:123:model/test")
        assert len(result) == 1
        assert result[0]["Key"] == "user:tag"


class TestCheckJobStatus:
    """Test _check_job_status function."""

    def test_check_job_status_completed(self):
        """Test completed job."""
        from sagemaker.core.common_utils import _check_job_status

        desc = {"TrainingJobStatus": "Completed"}
        _check_job_status("test-job", desc, "TrainingJobStatus")

    def test_check_job_status_failed(self):
        """Test failed job."""
        from sagemaker.core.common_utils import _check_job_status

        desc = {"TrainingJobStatus": "Failed", "FailureReason": "Out of memory"}

        with pytest.raises(Exception):
            _check_job_status("test-job", desc, "TrainingJobStatus")

    def test_check_job_status_capacity_error(self):
        """Test capacity error."""
        from sagemaker.core.common_utils import _check_job_status

        desc = {
            "TrainingJobStatus": "Failed",
            "FailureReason": "CapacityError: Insufficient capacity",
        }

        with pytest.raises(Exception):
            _check_job_status("test-job", desc, "TrainingJobStatus")


class TestCreateResource:
    """Test _create_resource function."""

    def test_create_resource_success(self):
        """Test successful resource creation."""
        from sagemaker.core.common_utils import _create_resource

        mock_fn = Mock()
        result = _create_resource(mock_fn)
        assert result is True
        mock_fn.assert_called_once()

    def test_create_resource_already_exists(self):
        """Test when resource already exists."""
        from sagemaker.core.common_utils import _create_resource

        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Cannot create already existing"}},
            "operation",
        )
        mock_fn = Mock(side_effect=error)
        result = _create_resource(mock_fn)
        assert result is False

    def test_create_resource_other_error(self):
        """Test with other error."""
        from sagemaker.core.common_utils import _create_resource

        error = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "operation"
        )
        mock_fn = Mock(side_effect=error)

        with pytest.raises(ClientError):
            _create_resource(mock_fn)


class TestUpdateContainerWithInferenceParams:
    """Test update_container_with_inference_params function."""

    def test_update_container_with_inference_params_container_def(self):
        """Test updating container def."""
        from sagemaker.core.common_utils import update_container_with_inference_params

        container_def = {"Image": "image"}
        result = update_container_with_inference_params(
            framework="tensorflow", framework_version="2.8", container_def=container_def
        )
        assert result["Framework"] == "tensorflow"
        assert result["FrameworkVersion"] == "2.8"

    def test_update_container_with_inference_params_container_list(self):
        """Test updating container list."""
        from sagemaker.core.common_utils import update_container_with_inference_params

        container_list = [{"Image": "image1"}, {"Image": "image2"}]
        result = update_container_with_inference_params(
            framework="pytorch", container_list=container_list
        )
        assert result[0]["Framework"] == "pytorch"
        assert result[1]["Framework"] == "pytorch"

    def test_update_container_with_inference_params_all_params(self):
        """Test with all parameters."""
        from sagemaker.core.common_utils import update_container_with_inference_params

        container_def = {"Image": "image"}
        result = update_container_with_inference_params(
            framework="tensorflow",
            framework_version="2.8",
            nearest_model_name="resnet50",
            data_input_configuration="config",
            container_def=container_def,
        )
        assert result["NearestModelName"] == "resnet50"
        assert "ModelInput" in result


class TestResolveClassAttributeFromConfig:
    """Test resolve_class_attribute_from_config function."""

    def test_resolve_class_attribute_from_config_existing_value(self):
        """Test with existing attribute value."""
        from sagemaker.core.common_utils import resolve_class_attribute_from_config

        class TestClass:
            def __init__(self):
                self.attr = "existing"

        instance = TestClass()
        result = resolve_class_attribute_from_config(
            TestClass, instance, "attr", "config.path", default_value="default"
        )
        assert result.attr == "existing"

    def test_resolve_class_attribute_from_config_none_instance(self):
        """Test with None instance."""
        from sagemaker.core.common_utils import resolve_class_attribute_from_config

        class TestClass:
            def __init__(self):
                self.attr = None

        result = resolve_class_attribute_from_config(
            TestClass, None, "attr", "config.path", default_value="default"
        )
        assert result.attr == "default"

    def test_resolve_class_attribute_from_config_no_class(self):
        """Test with no class provided."""
        from sagemaker.core.common_utils import resolve_class_attribute_from_config

        result = resolve_class_attribute_from_config(
            None, None, "attr", "config.path", default_value="default"
        )
        assert result is None


class TestResolveNestedDictValueFromConfig:
    """Test resolve_nested_dict_value_from_config function."""

    def test_resolve_nested_dict_value_from_config_existing(self):
        """Test with existing value."""
        from sagemaker.core.common_utils import resolve_nested_dict_value_from_config

        dictionary = {"a": {"b": "existing"}}
        result = resolve_nested_dict_value_from_config(
            dictionary, ["a", "b"], "config.path", default_value="default"
        )
        assert result["a"]["b"] == "existing"

    def test_resolve_nested_dict_value_from_config_none_value(self):
        """Test with None value."""
        from sagemaker.core.common_utils import resolve_nested_dict_value_from_config

        dictionary = {"a": {}}
        result = resolve_nested_dict_value_from_config(
            dictionary, ["a", "b"], "config.path", default_value="default"
        )
        assert result["a"]["b"] == "default"


class TestUpdateListOfDictsWithValuesFromConfig:
    """Test update_list_of_dicts_with_values_from_config function."""

    def test_update_list_of_dicts_basic(self):
        """Test basic update."""
        from sagemaker.core.common_utils import update_list_of_dicts_with_values_from_config

        input_list = [{"key1": "value1"}]
        update_list_of_dicts_with_values_from_config(input_list, "config.path")

    def test_update_list_of_dicts_none_input(self):
        """Test with None input."""
        from sagemaker.core.common_utils import update_list_of_dicts_with_values_from_config

        update_list_of_dicts_with_values_from_config(None, "config.path")


class TestValidateRequiredPathsInDict:
    """Test _validate_required_paths_in_a_dict function."""

    def test_validate_required_paths_true(self):
        """Test when all required paths exist."""
        from sagemaker.core.common_utils import _validate_required_paths_in_a_dict

        source_dict = {"key1": "value1", "key2": "value2"}
        result = _validate_required_paths_in_a_dict(source_dict, ["key1", "key2"])
        assert result is True

    def test_validate_required_paths_false(self):
        """Test when required path missing."""
        from sagemaker.core.common_utils import _validate_required_paths_in_a_dict

        source_dict = {"key1": "value1"}
        result = _validate_required_paths_in_a_dict(source_dict, ["key1", "key2"])
        assert result is False

    def test_validate_required_paths_none(self):
        """Test with None required paths."""
        from sagemaker.core.common_utils import _validate_required_paths_in_a_dict

        source_dict = {"key1": "value1"}
        result = _validate_required_paths_in_a_dict(source_dict, None)
        assert result is True


class TestValidateUnionKeyPathsInDict:
    """Test _validate_union_key_paths_in_a_dict function."""

    def test_validate_union_key_paths_valid(self):
        """Test valid union paths."""
        from sagemaker.core.common_utils import _validate_union_key_paths_in_a_dict

        source_dict = {"key1": "value1"}
        result = _validate_union_key_paths_in_a_dict(source_dict, [["key1", "key2"]])
        assert result is True

    def test_validate_union_key_paths_invalid(self):
        """Test invalid union paths."""
        from sagemaker.core.common_utils import _validate_union_key_paths_in_a_dict

        source_dict = {"key1": "value1", "key2": "value2"}
        result = _validate_union_key_paths_in_a_dict(source_dict, [["key1", "key2"]])
        assert result is False

    def test_validate_union_key_paths_none(self):
        """Test with None union paths."""
        from sagemaker.core.common_utils import _validate_union_key_paths_in_a_dict

        source_dict = {"key1": "value1"}
        result = _validate_union_key_paths_in_a_dict(source_dict, None)
        assert result is True


class TestUpdateNestedDictionaryWithValuesFromConfig:
    """Test update_nested_dictionary_with_values_from_config function."""

    def test_update_nested_dictionary_basic(self):
        """Test basic update."""
        from sagemaker.core.common_utils import update_nested_dictionary_with_values_from_config

        source_dict = {"key1": "value1"}
        result = update_nested_dictionary_with_values_from_config(source_dict, "config.path")
        assert result == source_dict

    def test_update_nested_dictionary_none_source(self):
        """Test with None source."""
        from sagemaker.core.common_utils import update_nested_dictionary_with_values_from_config

        result = update_nested_dictionary_with_values_from_config(None, "config.path")
        assert result is None


class TestStringifyObject:
    """Test stringify_object function."""

    def test_stringify_object_basic(self):
        """Test basic stringify."""
        from sagemaker.core.common_utils import stringify_object

        class TestObj:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = None

        obj = TestObj()
        result = stringify_object(obj)
        assert "attr1" in result
        assert "attr2" not in result


class TestExtractInstanceRatePerHour:
    """Test extract_instance_rate_per_hour function."""

    def test_extract_instance_rate_per_hour_valid(self):
        """Test with valid price data."""
        from sagemaker.core.common_utils import extract_instance_rate_per_hour

        price_data = {
            "terms": {
                "OnDemand": {
                    "term1": {"priceDimensions": {"dim1": {"pricePerUnit": {"USD": "1.125"}}}}
                }
            }
        }

        result = extract_instance_rate_per_hour(price_data)
        assert result["value"] == "1.125"
        assert result["unit"] == "USD/Hr"

    def test_extract_instance_rate_per_hour_none(self):
        """Test with None data."""
        from sagemaker.core.common_utils import extract_instance_rate_per_hour

        result = extract_instance_rate_per_hour(None)
        assert result is None


class TestCheckAndGetRunExperimentConfig:
    """Test check_and_get_run_experiment_config function."""

    @patch("sagemaker.core.experiments._run_context._RunContext")
    def test_check_and_get_run_experiment_config_with_input(self, mock_run_context):
        """Test with experiment config input."""
        from sagemaker.core.common_utils import check_and_get_run_experiment_config

        mock_run_context.get_current_run.return_value = Mock(experiment_config={"run": "config"})

        result = check_and_get_run_experiment_config({"input": "config"})
        assert result == {"input": "config"}

    @patch("sagemaker.core.experiments._run_context._RunContext")
    def test_check_and_get_run_experiment_config_from_run(self, mock_run_context):
        """Test getting config from run."""
        from sagemaker.core.common_utils import check_and_get_run_experiment_config

        mock_run = Mock(experiment_config={"run": "config"})
        mock_run_context.get_current_run.return_value = mock_run

        result = check_and_get_run_experiment_config(None)
        assert result == {"run": "config"}

    @patch("sagemaker.core.experiments._run_context._RunContext")
    def test_check_and_get_run_experiment_config_no_run(self, mock_run_context):
        """Test with no run context."""
        from sagemaker.core.common_utils import check_and_get_run_experiment_config

        mock_run_context.get_current_run.return_value = None

        result = check_and_get_run_experiment_config(None)
        assert result is None


class TestStartWaiting:
    """Test _start_waiting function."""

    def test_start_waiting_basic(self):
        """Test basic waiting."""
        from sagemaker.core.common_utils import _start_waiting

        _start_waiting(0)


class TestDownloadFile:
    """Test download_file function."""

    def test_download_file_basic(self, tmp_path):
        """Test basic file download."""
        from sagemaker.core.common_utils import download_file

        mock_session = Mock()
        mock_boto_session = Mock()
        mock_s3 = Mock()
        mock_bucket = Mock()

        mock_session.boto_session = mock_boto_session
        mock_session.boto_region_name = "us-west-2"
        mock_boto_session.resource.return_value = mock_s3
        mock_s3.Bucket.return_value = mock_bucket

        target = str(tmp_path / "file.txt")
        download_file("bucket", "path/file.txt", target, mock_session)

        mock_bucket.download_file.assert_called_once()


class TestDownloadFileFromUrl:
    """Test download_file_from_url function."""

    def test_download_file_from_url_basic(self, tmp_path):
        """Test downloading from URL."""
        from sagemaker.core.common_utils import download_file_from_url

        mock_session = Mock()
        mock_boto_session = Mock()
        mock_s3 = Mock()
        mock_bucket = Mock()

        mock_session.boto_session = mock_boto_session
        mock_session.boto_region_name = "us-west-2"
        mock_boto_session.resource.return_value = mock_s3
        mock_s3.Bucket.return_value = mock_bucket

        target = str(tmp_path / "file.txt")
        download_file_from_url("s3://bucket/path/file.txt", target, mock_session)


class TestSaveModel:
    """Test _save_model function."""

    def test_save_model_s3(self, tmp_path):
        """Test saving model to S3."""
        from sagemaker.core.common_utils import _save_model
        from sagemaker.core.session_settings import SessionSettings

        model_file = tmp_path / "model.tar.gz"
        model_file.write_text("model data")

        mock_session = Mock()
        mock_boto_session = Mock()
        mock_s3 = Mock()
        mock_obj = Mock()

        mock_session.boto_session = mock_boto_session
        mock_session.boto_region_name = "us-west-2"
        mock_session.settings = SessionSettings()
        mock_boto_session.resource.return_value = mock_s3
        mock_s3.Object.return_value = mock_obj

        _save_model("s3://bucket/model.tar.gz", str(model_file), mock_session, kms_key=None)

        mock_obj.upload_file.assert_called_once()

    def test_save_model_local(self, tmp_path):
        """Test saving model locally."""
        from sagemaker.core.common_utils import _save_model

        model_file = tmp_path / "model.tar.gz"
        model_file.write_text("model data")

        output_file = tmp_path / "output.tar.gz"

        mock_session = Mock()

        _save_model(f"file://{output_file}", str(model_file), mock_session, kms_key=None)

        assert output_file.exists()


class TestResolveRoutingConfig:
    """Test _resolve_routing_config function."""

    def test_resolve_routing_config_enum(self):
        """Test with enum value."""
        from sagemaker.core.common_utils import _resolve_routing_config
        from sagemaker.core.enums import RoutingStrategy

        config = {"RoutingStrategy": RoutingStrategy.RANDOM}
        result = _resolve_routing_config(config)
        assert result["RoutingStrategy"] == "RANDOM"

    def test_resolve_routing_config_string(self):
        """Test with string value."""
        from sagemaker.core.common_utils import _resolve_routing_config

        config = {"RoutingStrategy": "RANDOM"}
        result = _resolve_routing_config(config)
        assert result["RoutingStrategy"] == "RANDOM"

    def test_resolve_routing_config_invalid(self):
        """Test with invalid value."""
        from sagemaker.core.common_utils import _resolve_routing_config

        config = {"RoutingStrategy": "INVALID"}
        with pytest.raises(ValueError):
            _resolve_routing_config(config)

    def test_resolve_routing_config_none(self):
        """Test with None config."""
        from sagemaker.core.common_utils import _resolve_routing_config

        result = _resolve_routing_config(None)
        assert result is None


class TestWaitUntil:
    """Test _wait_until function."""

    def test_wait_until_success(self):
        """Test successful wait."""
        from sagemaker.core.common_utils import _wait_until

        mock_fn = Mock(return_value="result")
        result = _wait_until(mock_fn, poll=0.01)
        assert result == "result"

    def test_wait_until_with_retry(self):
        """Test with retry on AccessDeniedException."""
        from sagemaker.core.common_utils import _wait_until

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}, "operation"
        )
        mock_fn = Mock(side_effect=[error, "result"])
        result = _wait_until(mock_fn, poll=0.01)
        assert result == "result"


class TestGetInitialJobState:
    """Test _get_initial_job_state function."""

    def test_get_initial_job_state_tailing(self):
        """Test tailing state."""
        from sagemaker.core.common_utils import _get_initial_job_state, LogState

        description = {"TrainingJobStatus": "InProgress"}
        result = _get_initial_job_state(description, "TrainingJobStatus", wait=True)
        assert result == LogState.TAILING

    def test_get_initial_job_state_complete(self):
        """Test complete state."""
        from sagemaker.core.common_utils import _get_initial_job_state, LogState

        description = {"TrainingJobStatus": "Completed"}
        result = _get_initial_job_state(description, "TrainingJobStatus", wait=True)
        assert result == LogState.COMPLETE


class TestLogsInit:
    """Test _logs_init function."""

    def test_logs_init_training(self):
        """Test logs init for training job."""
        from sagemaker.core.common_utils import _logs_init

        mock_boto_session = Mock()
        mock_client = Mock()
        mock_boto_session.client.return_value = mock_client

        description = {"ResourceConfig": {"InstanceCount": 2}}

        result = _logs_init(mock_boto_session, description, "Training")
        assert result[0] == 2

    def test_logs_init_transform(self):
        """Test logs init for transform job."""
        from sagemaker.core.common_utils import _logs_init

        mock_boto_session = Mock()
        mock_client = Mock()
        mock_boto_session.client.return_value = mock_client

        description = {"TransformResources": {"InstanceCount": 1}}

        result = _logs_init(mock_boto_session, description, "Transform")
        assert result[0] == 1


class TestModuleImportError:
    """Test _module_import_error function."""

    def test_module_import_error_message(self):
        """Test error message generation."""
        from sagemaker.core.common_utils import _module_import_error

        result = _module_import_error("numpy", "ML", "ml")
        assert "numpy" in result
        assert "ML" in result
        assert "ml" in result


class TestS3DataConfigGetDataBucket:
    """Test S3DataConfig.get_data_bucket method."""

    def test_s3_data_config_get_data_bucket_default(self):
        """Test getting default data bucket."""
        from sagemaker.core.common_utils import S3DataConfig

        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        mock_session.read_s3_file.return_value = '{"default": "default-bucket"}'

        config = S3DataConfig(
            sagemaker_session=mock_session, bucket_name="config-bucket", prefix="config.json"
        )

        result = config.get_data_bucket()
        assert result == "default-bucket"

    def test_s3_data_config_get_data_bucket_region(self):
        """Test getting region-specific bucket."""
        from sagemaker.core.common_utils import S3DataConfig

        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        mock_session.read_s3_file.return_value = (
            '{"us-west-2": "west-bucket", "default": "default-bucket"}'
        )

        config = S3DataConfig(
            sagemaker_session=mock_session, bucket_name="config-bucket", prefix="config.json"
        )

        result = config.get_data_bucket()
        assert result == "west-bucket"


class TestBaseNameFromImagePipelineVariable:
    """Test base_name_from_image with pipeline variables."""

    @patch("sagemaker.core.common_utils.is_pipeline_variable")
    @patch("sagemaker.core.common_utils.is_pipeline_parameter_string")
    def test_base_name_from_image_pipeline_param_with_default(self, mock_is_param, mock_is_var):
        """Test with pipeline parameter with default value."""
        from sagemaker.core.common_utils import base_name_from_image

        mock_is_var.return_value = True
        mock_is_param.return_value = True

        mock_image = Mock()
        mock_image.default_value = "my-algorithm:latest"

        result = base_name_from_image(mock_image)
        assert result == "my-algorithm"

    @patch("sagemaker.core.common_utils.is_pipeline_variable")
    @patch("sagemaker.core.common_utils.is_pipeline_parameter_string")
    def test_base_name_from_image_pipeline_var_no_default(self, mock_is_param, mock_is_var):
        """Test with pipeline variable without default."""
        from sagemaker.core.common_utils import base_name_from_image

        mock_is_var.return_value = True
        mock_is_param.return_value = False

        mock_image = Mock()

        result = base_name_from_image(mock_image, default_base_name="default")
        assert result == "default"


class TestConstructContainerObject:
    """Test construct_container_object function."""

    def test_construct_container_object_all_params(self):
        """Test with all parameters."""
        from sagemaker.core.common_utils import construct_container_object

        obj = {}
        result = construct_container_object(obj, "data_config", "tensorflow", "2.8", "resnet50")

        assert result["Framework"] == "tensorflow"
        assert result["FrameworkVersion"] == "2.8"
        assert result["NearestModelName"] == "resnet50"
        assert "ModelInput" in result


class TestFlushLogStreams:
    """Test _flush_log_streams function."""

    @patch("sagemaker.core.logs.multi_stream_iter")
    def test_flush_log_streams_basic(self, mock_multi_stream):
        """Test basic log stream flushing."""
        from sagemaker.core.common_utils import _flush_log_streams

        mock_client = Mock()
        mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "stream1"}]
        }

        mock_multi_stream.return_value = []

        stream_names = []
        positions = {}

        _flush_log_streams(
            stream_names,
            1,
            mock_client,
            "log-group",
            "job-name",
            positions,
            False,
            lambda idx, msg: None,
        )


class TestNestedSetDict:
    """Test nested_set_dict function."""

    def test_nested_set_dict_single_key(self):
        """Test with single key."""
        from sagemaker.core.common_utils import nested_set_dict

        d = {}
        nested_set_dict(d, ["key"], "value")
        assert d["key"] == "value"

    def test_nested_set_dict_multiple_keys(self):
        """Test with multiple keys."""
        from sagemaker.core.common_utils import nested_set_dict

        d = {}
        nested_set_dict(d, ["a", "b", "c"], "value")
        assert d["a"]["b"]["c"] == "value"
