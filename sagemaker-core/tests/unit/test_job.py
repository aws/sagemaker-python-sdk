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
"""Unit tests for sagemaker.core.job module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.job import _Job
from sagemaker.core.inputs import TrainingInput, FileSystemInput


class TestJob:
    """Test _Job class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session = Mock()
        self.job = _Job(self.session, "test-job")

    def test_init(self):
        """Test _Job initialization."""
        assert self.job.sagemaker_session == self.session
        assert self.job.job_name == "test-job"

    def test_name_property(self):
        """Test name property."""
        assert self.job.name == "test-job"

    def test_prepare_output_config_basic(self):
        """Test _prepare_output_config with basic parameters."""
        config = _Job._prepare_output_config("s3://bucket/output", None)

        assert config["S3OutputPath"] == "s3://bucket/output"
        assert "KmsKeyId" not in config
        assert "CompressionType" not in config

    def test_prepare_output_config_with_kms(self):
        """Test _prepare_output_config with KMS key."""
        config = _Job._prepare_output_config(
            "s3://bucket/output", "arn:aws:kms:us-west-2:123456789:key/abc"
        )

        assert config["KmsKeyId"] == "arn:aws:kms:us-west-2:123456789:key/abc"

    def test_prepare_output_config_with_compression_disabled(self):
        """Test _prepare_output_config with compression disabled."""
        config = _Job._prepare_output_config(
            "s3://bucket/output", None, disable_output_compression=True
        )

        assert config["CompressionType"] == "NONE"

    def test_prepare_resource_config_basic(self):
        """Test _prepare_resource_config with basic parameters."""
        config = _Job._prepare_resource_config(
            instance_count=2,
            instance_type="ml.m5.xlarge",
            instance_groups=None,
            volume_size=30,
            volume_kms_key=None,
            keep_alive_period_in_seconds=None,
            training_plan=None,
        )

        assert config["InstanceCount"] == 2
        assert config["InstanceType"] == "ml.m5.xlarge"
        assert config["VolumeSizeInGB"] == 30

    def test_prepare_resource_config_with_volume_kms(self):
        """Test _prepare_resource_config with volume KMS key."""
        config = _Job._prepare_resource_config(
            instance_count=1,
            instance_type="ml.p3.2xlarge",
            instance_groups=None,
            volume_size=50,
            volume_kms_key="arn:aws:kms:us-west-2:123456789:key/xyz",
            keep_alive_period_in_seconds=None,
            training_plan=None,
        )

        assert config["VolumeKmsKeyId"] == "arn:aws:kms:us-west-2:123456789:key/xyz"

    def test_prepare_resource_config_with_keep_alive(self):
        """Test _prepare_resource_config with keep alive period."""
        config = _Job._prepare_resource_config(
            instance_count=1,
            instance_type="ml.m5.xlarge",
            instance_groups=None,
            volume_size=30,
            volume_kms_key=None,
            keep_alive_period_in_seconds=3600,
            training_plan=None,
        )

        assert config["KeepAlivePeriodInSeconds"] == 3600

    def test_prepare_resource_config_with_training_plan(self):
        """Test _prepare_resource_config with training plan."""
        config = _Job._prepare_resource_config(
            instance_count=1,
            instance_type="ml.m5.xlarge",
            instance_groups=None,
            volume_size=30,
            volume_kms_key=None,
            keep_alive_period_in_seconds=None,
            training_plan="arn:aws:sagemaker:us-west-2:123456789:training-plan/plan-1",
        )

        assert (
            config["TrainingPlanArn"]
            == "arn:aws:sagemaker:us-west-2:123456789:training-plan/plan-1"
        )

    def test_prepare_resource_config_with_instance_groups(self):
        """Test _prepare_resource_config with instance groups."""
        mock_group = Mock()
        mock_group._to_request_dict.return_value = {"InstanceType": "ml.m5.xlarge"}

        config = _Job._prepare_resource_config(
            instance_count=None,
            instance_type=None,
            instance_groups=[mock_group],
            volume_size=30,
            volume_kms_key=None,
            keep_alive_period_in_seconds=None,
            training_plan=None,
        )

        assert "InstanceGroups" in config
        assert len(config["InstanceGroups"]) == 1

    def test_prepare_resource_config_instance_groups_with_instance_count_raises_error(self):
        """Test _prepare_resource_config with both instance_groups and instance_count."""
        mock_group = Mock()

        with pytest.raises(ValueError, match="instance_count and instance_type cannot be set"):
            _Job._prepare_resource_config(
                instance_count=1,
                instance_type="ml.m5.xlarge",
                instance_groups=[mock_group],
                volume_size=30,
                volume_kms_key=None,
                keep_alive_period_in_seconds=None,
                training_plan=None,
            )

    def test_prepare_resource_config_without_instance_params_raises_error(self):
        """Test _prepare_resource_config without required instance parameters."""
        with pytest.raises(ValueError, match="instance_count and instance_type must be set"):
            _Job._prepare_resource_config(
                instance_count=None,
                instance_type=None,
                instance_groups=None,
                volume_size=30,
                volume_kms_key=None,
                keep_alive_period_in_seconds=None,
                training_plan=None,
            )

    def test_prepare_stop_condition_basic(self):
        """Test _prepare_stop_condition with basic parameters."""
        config = _Job._prepare_stop_condition(3600, None)

        assert config["MaxRuntimeInSeconds"] == 3600
        assert "MaxWaitTimeInSeconds" not in config

    def test_prepare_stop_condition_with_max_wait(self):
        """Test _prepare_stop_condition with max wait time."""
        config = _Job._prepare_stop_condition(3600, 7200)

        assert config["MaxRuntimeInSeconds"] == 3600
        assert config["MaxWaitTimeInSeconds"] == 7200

    def test_format_string_uri_input_s3(self):
        """Test _format_string_uri_input with S3 URI."""
        result = _Job._format_string_uri_input("s3://bucket/data", validate_uri=True)

        assert isinstance(result, TrainingInput)

    def test_format_string_uri_input_invalid_raises_error(self):
        """Test _format_string_uri_input with invalid URI."""
        with pytest.raises(ValueError, match="must be a valid S3 or FILE URI"):
            _Job._format_string_uri_input("invalid://bucket/data", validate_uri=True)

    def test_format_string_uri_input_training_input(self):
        """Test _format_string_uri_input with TrainingInput."""
        training_input = TrainingInput("s3://bucket/data")
        result = _Job._format_string_uri_input(training_input, validate_uri=True)

        assert result == training_input

    def test_format_string_uri_input_file_system_input(self):
        """Test _format_string_uri_input with FileSystemInput."""
        fs_input = FileSystemInput(
            file_system_id="fs-123", file_system_type="EFS", directory_path="/data"
        )
        result = _Job._format_string_uri_input(fs_input, validate_uri=True)

        assert result == fs_input

    @pytest.mark.skip(reason="Requires sagemaker.core.amazon module which is not available")
    def test_format_inputs_to_input_config_string(self):
        """Test _format_inputs_to_input_config with string input."""
        channels = _Job._format_inputs_to_input_config("s3://bucket/data")

        assert len(channels) == 1
        assert channels[0]["ChannelName"] == "training"

    @pytest.mark.skip(reason="Requires sagemaker.core.amazon module which is not available")
    def test_format_inputs_to_input_config_dict(self):
        """Test _format_inputs_to_input_config with dict input."""
        inputs = {"train": "s3://bucket/train", "validation": "s3://bucket/val"}
        channels = _Job._format_inputs_to_input_config(inputs)

        assert len(channels) == 2
        channel_names = [ch["ChannelName"] for ch in channels]
        assert "train" in channel_names
        assert "validation" in channel_names

    @pytest.mark.skip(reason="Requires sagemaker.core.amazon module which is not available")
    def test_format_inputs_to_input_config_training_input(self):
        """Test _format_inputs_to_input_config with TrainingInput."""
        training_input = TrainingInput("s3://bucket/data")
        channels = _Job._format_inputs_to_input_config(training_input)

        assert len(channels) == 1
        assert channels[0]["ChannelName"] == "training"

    def test_format_inputs_to_input_config_none(self):
        """Test _format_inputs_to_input_config with None."""
        result = _Job._format_inputs_to_input_config(None)

        assert result is None

    @pytest.mark.skip(reason="Requires sagemaker.core.amazon module which is not available")
    def test_format_inputs_to_input_config_invalid_raises_error(self):
        """Test _format_inputs_to_input_config with invalid input."""
        with pytest.raises(ValueError, match="Cannot format input"):
            _Job._format_inputs_to_input_config(12345)

    def test_prepare_channel_valid(self):
        """Test _prepare_channel with valid parameters."""
        channel = _Job._prepare_channel(
            input_config=None,
            channel_uri="s3://bucket/model",
            channel_name="model",
            validate_uri=True,
        )

        assert channel is not None
        assert channel["ChannelName"] == "model"

    def test_prepare_channel_no_uri(self):
        """Test _prepare_channel without URI."""
        channel = _Job._prepare_channel(
            input_config=None, channel_uri=None, channel_name="model", validate_uri=True
        )

        assert channel is None

    def test_prepare_channel_no_name_raises_error(self):
        """Test _prepare_channel without channel name."""
        with pytest.raises(ValueError, match="Expected a channel name"):
            _Job._prepare_channel(
                input_config=None,
                channel_uri="s3://bucket/model",
                channel_name=None,
                validate_uri=True,
            )

    def test_prepare_channel_duplicate_raises_error(self):
        """Test _prepare_channel with duplicate channel name."""
        input_config = [{"ChannelName": "model"}]

        with pytest.raises(ValueError, match="Duplicate channel"):
            _Job._prepare_channel(
                input_config=input_config,
                channel_uri="s3://bucket/model",
                channel_name="model",
                validate_uri=True,
            )

    def test_convert_input_to_channel(self):
        """Test _convert_input_to_channel."""
        training_input = TrainingInput("s3://bucket/data")
        channel = _Job._convert_input_to_channel("training", training_input)

        assert channel["ChannelName"] == "training"

    def test_get_access_configs_with_configs(self):
        """Test _get_access_configs with access configs."""
        estimator = Mock()
        estimator.model_access_config = {"key": "value"}
        estimator.hub_access_config = {"hub": "config"}

        model_config, hub_config = _Job._get_access_configs(estimator)

        assert model_config == {"key": "value"}
        assert hub_config == {"hub": "config"}

    def test_get_access_configs_without_configs(self):
        """Test _get_access_configs without access configs."""
        estimator = Mock(spec=[])

        model_config, hub_config = _Job._get_access_configs(estimator)

        assert model_config is None
        assert hub_config is None

    def test_prepare_output_config_with_all_params(self):
        """Test _prepare_output_config with all parameters."""
        config = _Job._prepare_output_config(
            "s3://bucket/output",
            "arn:aws:kms:us-west-2:123456789:key/abc",
            disable_output_compression=True,
        )

        assert config["S3OutputPath"] == "s3://bucket/output"
        assert config["KmsKeyId"] == "arn:aws:kms:us-west-2:123456789:key/abc"
        assert config["CompressionType"] == "NONE"

    def test_prepare_resource_config_with_all_optional_params(self):
        """Test _prepare_resource_config with all optional parameters."""
        config = _Job._prepare_resource_config(
            instance_count=2,
            instance_type="ml.p3.8xlarge",
            instance_groups=None,
            volume_size=100,
            volume_kms_key="arn:aws:kms:us-west-2:123456789:key/xyz",
            keep_alive_period_in_seconds=7200,
            training_plan="arn:aws:sagemaker:us-west-2:123456789:training-plan/plan-1",
        )

        assert config["InstanceCount"] == 2
        assert config["InstanceType"] == "ml.p3.8xlarge"
        assert config["VolumeSizeInGB"] == 100
        assert config["VolumeKmsKeyId"] == "arn:aws:kms:us-west-2:123456789:key/xyz"
        assert config["KeepAlivePeriodInSeconds"] == 7200
        assert (
            config["TrainingPlanArn"]
            == "arn:aws:sagemaker:us-west-2:123456789:training-plan/plan-1"
        )

    def test_prepare_stop_condition_with_zero_values(self):
        """Test _prepare_stop_condition with zero values."""
        config = _Job._prepare_stop_condition(0, 0)

        assert config["MaxRuntimeInSeconds"] == 0
        assert "MaxWaitTimeInSeconds" not in config

    def test_format_string_uri_input_file_uri(self):
        """Test _format_string_uri_input with FILE URI."""
        result = _Job._format_string_uri_input("file:///local/data", validate_uri=True)

        from sagemaker.core.local.local_session import FileInput

        assert isinstance(result, FileInput)

    def test_format_string_uri_input_no_validation(self):
        """Test _format_string_uri_input without validation."""
        result = _Job._format_string_uri_input("invalid://bucket/data", validate_uri=False)

        # Should not raise error when validation is disabled
        assert result is not None

    def test_prepare_channel_with_input_config(self):
        """Test _prepare_channel with existing input_config."""
        input_config = [{"ChannelName": "training"}]
        channel = _Job._prepare_channel(
            input_config=input_config,
            channel_uri="s3://bucket/validation",
            channel_name="validation",
            validate_uri=True,
        )

        assert channel is not None
        assert channel["ChannelName"] == "validation"

    def test_convert_input_to_channel_with_file_system_input(self):
        """Test _convert_input_to_channel with FileSystemInput."""
        fs_input = FileSystemInput(
            file_system_id="fs-123", file_system_type="EFS", directory_path="/data"
        )
        channel = _Job._convert_input_to_channel("training", fs_input)

        assert channel["ChannelName"] == "training"
        assert "FileSystemDataSource" in channel["DataSource"]
