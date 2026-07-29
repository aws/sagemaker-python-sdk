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
"""Unit tests for KMS key propagation in ClarifyCheckStep."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock

from sagemaker.mlops.workflow.clarify_check_step import (
    ClarifyCheckStep,
    DataBiasCheckConfig,
    ModelBiasCheckConfig,
    ModelExplainabilityCheckConfig,
)
from sagemaker.mlops.workflow.check_job_config import CheckJobConfig


_OUTPUT_KMS_KEY = "arn:aws:kms:us-east-1:123456789012:key/output-key-id"
_VOLUME_KMS_KEY = "arn:aws:kms:us-east-1:123456789012:key/volume-key-id"

# Patch trim_request_dict to be a no-op since it only trims job names
# and is not relevant to KMS key propagation testing.
_TRIM_PATCH = "sagemaker.mlops.workflow.clarify_check_step.trim_request_dict"


def _noop_trim(request_dict, *args, **kwargs):
    """No-op replacement for trim_request_dict in tests."""
    return request_dict


def _create_mock_clarify_check_step(output_kms_key=None, volume_kms_key=None):
    """Create a ClarifyCheckStep with mocked internals for testing arguments."""
    step = object.__new__(ClarifyCheckStep)

    # Mock check_job_config
    step.check_job_config = Mock()
    step.check_job_config.output_kms_key = output_kms_key
    step.check_job_config.volume_kms_key = volume_kms_key

    # Mock processing params (config_input, data_input, result_output)
    config_input = Mock()
    config_input.input_name = "analysis_config"
    config_input.s3_input = Mock()
    config_input.s3_input.s3_uri = "s3://bucket/config/analysis_config.json"
    config_input.s3_input.local_path = "/opt/ml/processing/input/config"
    config_input.s3_input.s3_data_type = "S3Prefix"
    config_input.s3_input.s3_input_mode = "File"

    data_input = Mock()
    data_input.input_name = "dataset"
    data_input.s3_input = Mock()
    data_input.s3_input.s3_uri = "s3://bucket/input/data.csv"
    data_input.s3_input.local_path = "/opt/ml/processing/input/data"
    data_input.s3_input.s3_data_type = "S3Prefix"
    data_input.s3_input.s3_input_mode = "File"

    result_output = Mock()
    result_output.output_name = "analysis_result"
    result_output.s3_output = Mock()
    result_output.s3_output.s3_uri = "s3://bucket/output/results"
    result_output.s3_output.local_path = "/opt/ml/processing/output"
    result_output.s3_output.s3_upload_mode = "EndOfJob"

    step._processing_params = {
        "config_input": config_input,
        "data_input": data_input,
        "result_output": result_output,
    }

    # Mock baselining processor
    step._baselining_processor = Mock()
    step._baselining_processor._current_job_name = "clarify-check-job"
    step._baselining_processor.instance_count = 1
    step._baselining_processor.instance_type = "ml.m5.xlarge"
    step._baselining_processor.volume_size_in_gb = 30
    step._baselining_processor.image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/clarify:latest"
    step._baselining_processor.role = "arn:aws:iam::123456789012:role/SageMakerRole"
    step._baselining_processor.max_runtime_in_seconds = 3600
    step._baselining_processor.env = None
    step._baselining_processor.network_config = None
    step._baselining_processor.entrypoint = None
    step._baselining_processor.arguments = None

    return step


class TestClarifyCheckStepKmsKeyPropagation:
    """Tests for KMS key propagation in ClarifyCheckStep.arguments."""

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_output_kms_key_propagated_in_s3_output(self, mock_trim):
        """Test that output_kms_key from CheckJobConfig is included in S3Output."""
        step = _create_mock_clarify_check_step(output_kms_key=_OUTPUT_KMS_KEY)

        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert "KmsKeyId" in s3_output
        assert s3_output["KmsKeyId"] == _OUTPUT_KMS_KEY

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_volume_kms_key_propagated_in_cluster_config(self, mock_trim):
        """Test that volume_kms_key from CheckJobConfig is included in ClusterConfig."""
        step = _create_mock_clarify_check_step(volume_kms_key=_VOLUME_KMS_KEY)

        args = step.arguments

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert "VolumeKmsKeyId" in cluster_config
        assert cluster_config["VolumeKmsKeyId"] == _VOLUME_KMS_KEY

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_both_kms_keys_propagated(self, mock_trim):
        """Test that both output_kms_key and volume_kms_key are propagated together."""
        step = _create_mock_clarify_check_step(
            output_kms_key=_OUTPUT_KMS_KEY,
            volume_kms_key=_VOLUME_KMS_KEY,
        )

        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert s3_output["KmsKeyId"] == _OUTPUT_KMS_KEY

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert cluster_config["VolumeKmsKeyId"] == _VOLUME_KMS_KEY

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_no_kms_keys_when_not_set(self, mock_trim):
        """Test that KmsKeyId and VolumeKmsKeyId are absent when not configured."""
        step = _create_mock_clarify_check_step(output_kms_key=None, volume_kms_key=None)

        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert "KmsKeyId" not in s3_output

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert "VolumeKmsKeyId" not in cluster_config

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_s3_output_retains_other_fields_with_kms(self, mock_trim):
        """Test that S3Output still contains S3Uri, LocalPath, S3UploadMode alongside KmsKeyId."""
        step = _create_mock_clarify_check_step(output_kms_key=_OUTPUT_KMS_KEY)

        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert s3_output["S3Uri"] == "s3://bucket/output/results"
        assert s3_output["LocalPath"] == "/opt/ml/processing/output"
        assert s3_output["S3UploadMode"] == "EndOfJob"
        assert s3_output["KmsKeyId"] == _OUTPUT_KMS_KEY

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_cluster_config_retains_other_fields_with_kms(self, mock_trim):
        """Test that ClusterConfig still contains instance fields alongside VolumeKmsKeyId."""
        step = _create_mock_clarify_check_step(volume_kms_key=_VOLUME_KMS_KEY)

        args = step.arguments

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert cluster_config["InstanceCount"] == 1
        assert cluster_config["InstanceType"] == "ml.m5.xlarge"
        assert cluster_config["VolumeSizeInGB"] == 30
        assert cluster_config["VolumeKmsKeyId"] == _VOLUME_KMS_KEY

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_output_kms_key_only_without_volume_kms(self, mock_trim):
        """Test output_kms_key set but volume_kms_key not set."""
        step = _create_mock_clarify_check_step(
            output_kms_key=_OUTPUT_KMS_KEY, volume_kms_key=None
        )

        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert s3_output["KmsKeyId"] == _OUTPUT_KMS_KEY

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert "VolumeKmsKeyId" not in cluster_config

    @patch(_TRIM_PATCH, side_effect=_noop_trim)
    def test_volume_kms_key_only_without_output_kms(self, mock_trim):
        """Test volume_kms_key set but output_kms_key not set."""
        step = _create_mock_clarify_check_step(
            output_kms_key=None, volume_kms_key=_VOLUME_KMS_KEY
        )

        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert "KmsKeyId" not in s3_output

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert cluster_config["VolumeKmsKeyId"] == _VOLUME_KMS_KEY
