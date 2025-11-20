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
"""Comprehensive unit tests for uncovered lines in sagemaker.core.remote_function.job module."""
from __future__ import absolute_import

import json
import os
import pytest
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import BytesIO

from sagemaker.core.remote_function.job import (
    _JobSettings,
    _Job,
    _update_job_request_with_checkpoint_config,
    _convert_run_to_json,
    _upload_spark_submit_deps,
    _upload_serialized_spark_configuration,
    _extend_mpirun_to_request,
    _extend_torchrun_to_request,
    _check_job_status,
    _rule_statuses_changed,
    _logs_init,
    _get_initial_job_state,
    LogState,
    _RunInfo,
)
from sagemaker.core.remote_function.checkpoint_location import CheckpointLocation


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_region_name = "us-west-2"
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = "prefix"
    session.sagemaker_client = Mock()
    session.boto_session = Mock()
    session.sagemaker_config = {}
    return session


class TestJobSettingsValidation:
    """Test _JobSettings validation logic for uncovered lines."""

    def test_spark_config_with_image_uri_raises_error(self, mock_session):
        """Test lines 619-620: spark_config and image_uri validation."""
        from sagemaker.core.remote_function.spark_config import SparkConfig
        spark_config = SparkConfig()
        with pytest.raises(ValueError, match="spark_config and image_uri cannot be specified"):
            _JobSettings(
                sagemaker_session=mock_session,
                spark_config=spark_config,
                image_uri="test-image",
                instance_type="ml.m5.xlarge"
            )

    def test_spark_config_with_conda_env_raises_error(self, mock_session):
        """Test lines 622-623: spark_config and job_conda_env validation."""
        from sagemaker.core.remote_function.spark_config import SparkConfig
        spark_config = SparkConfig()
        with pytest.raises(ValueError, match="Remote Spark jobs do not support job_conda_env"):
            _JobSettings(
                sagemaker_session=mock_session,
                spark_config=spark_config,
                job_conda_env="test-env",
                instance_type="ml.m5.xlarge"
            )

    def test_spark_config_with_auto_capture_raises_error(self, mock_session):
        """Test lines 625-628: spark_config and auto_capture validation."""
        from sagemaker.core.remote_function.spark_config import SparkConfig
        spark_config = SparkConfig()
        with pytest.raises(ValueError, match="Remote Spark jobs do not support automatically"):
            _JobSettings(
                sagemaker_session=mock_session,
                spark_config=spark_config,
                dependencies="auto_capture",
                instance_type="ml.m5.xlarge"
            )

    def test_pre_execution_commands_and_script_raises_error(self, mock_session):
        """Test lines 651-653: pre_execution validation."""
        with pytest.raises(ValueError, match="Only one of pre_execution_commands or pre_execution_script"):
            _JobSettings(
                sagemaker_session=mock_session,
                pre_execution_commands=["echo test"],
                pre_execution_script="/path/to/script.sh",
                instance_type="ml.m5.xlarge",
                image_uri="test-image"
            )

    def test_instance_type_required(self, mock_session):
        """Test lines 665-666: instance_type validation."""
        with pytest.raises(ValueError, match="instance_type is a required parameter"):
            _JobSettings(sagemaker_session=mock_session, image_uri="test-image")

    @patch.dict(os.environ, {"SAGEMAKER_INTERNAL_IMAGE_URI": "custom-image"})
    def test_get_default_image_from_env(self, mock_session):
        """Test lines 785-788: get default image from environment."""
        image = _JobSettings._get_default_image(mock_session)
        assert image == "custom-image"

    def test_get_default_image_unsupported_python(self, mock_session):
        """Test lines 792-795: unsupported Python version."""
        with patch.object(sys, "version_info", (3, 7, 0)):
            with pytest.raises(ValueError, match="Default image is supported only for Python versions"):
                _JobSettings._get_default_image(mock_session)

    def test_get_default_spark_image_unsupported_python(self, mock_session):
        """Test lines 815-817: unsupported Python for Spark."""
        with patch.object(sys, "version_info", (3, 8, 0)):
            with pytest.raises(ValueError, match="SageMaker Spark image for remote job only supports Python version 3.9"):
                _JobSettings._get_default_spark_image(mock_session)


class TestJobMethods:
    """Test _Job class methods for uncovered lines."""

    def test_from_describe_response(self, mock_session):
        """Test lines 848-852: from_describe_response method."""
        response = {
            "TrainingJobName": "test-job",
            "OutputDataConfig": {"S3OutputPath": "s3://bucket/output"},
            "Environment": {"REMOTE_FUNCTION_SECRET_KEY": "test-key"}
        }
        job = _Job.from_describe_response(response, mock_session)
        assert job.job_name == "test-job"
        assert job.s3_uri == "s3://bucket/output"
        assert job.hmac_key == "test-key"
        assert job._last_describe_response == response

    def test_describe_cached_completed(self, mock_session):
        """Test lines 865-871: describe with cached completed job."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        job._last_describe_response = {"TrainingJobStatus": "Completed"}
        
        result = job.describe()
        assert result["TrainingJobStatus"] == "Completed"
        mock_session.sagemaker_client.describe_training_job.assert_not_called()

    def test_describe_cached_failed(self, mock_session):
        """Test lines 865-871: describe with cached failed job."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        job._last_describe_response = {"TrainingJobStatus": "Failed"}
        
        result = job.describe()
        assert result["TrainingJobStatus"] == "Failed"
        mock_session.sagemaker_client.describe_training_job.assert_not_called()

    def test_describe_cached_stopped(self, mock_session):
        """Test lines 865-871: describe with cached stopped job."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        job._last_describe_response = {"TrainingJobStatus": "Stopped"}
        
        result = job.describe()
        assert result["TrainingJobStatus"] == "Stopped"
        mock_session.sagemaker_client.describe_training_job.assert_not_called()

    def test_stop(self, mock_session):
        """Test lines 886-887: stop method."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        job.stop()
        mock_session.sagemaker_client.stop_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )

    @patch("sagemaker.core.remote_function.job._logs_for_job")
    def test_wait(self, mock_logs, mock_session):
        """Test lines 889-903: wait method."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        mock_logs.return_value = {"TrainingJobStatus": "Completed"}
        
        job.wait(timeout=100)
        mock_logs.assert_called_once_with(
            sagemaker_session=mock_session,
            job_name="test-job",
            wait=True,
            timeout=100
        )
        assert job._last_describe_response["TrainingJobStatus"] == "Completed"


class TestCheckpointConfig:
    """Test checkpoint configuration for uncovered lines."""

    def test_checkpoint_in_args(self):
        """Test lines 1219-1227: checkpoint in positional args."""
        checkpoint = CheckpointLocation(s3_uri="s3://bucket/checkpoint")
        args = (checkpoint,)
        kwargs = {}
        request_dict = {}
        
        _update_job_request_with_checkpoint_config(args, kwargs, request_dict)
        
        assert "CheckpointConfig" in request_dict
        assert request_dict["CheckpointConfig"]["S3Uri"] == "s3://bucket/checkpoint"
        assert request_dict["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints/"

    def test_checkpoint_in_kwargs(self):
        """Test lines 1228-1230: checkpoint in keyword args."""
        checkpoint = CheckpointLocation(s3_uri="s3://bucket/checkpoint")
        args = ()
        kwargs = {"checkpoint": checkpoint}
        request_dict = {}
        
        _update_job_request_with_checkpoint_config(args, kwargs, request_dict)
        
        assert "CheckpointConfig" in request_dict
        assert request_dict["CheckpointConfig"]["S3Uri"] == "s3://bucket/checkpoint"

    def test_multiple_checkpoints_raises_error(self):
        """Test lines 1237-1239: multiple checkpoints error."""
        checkpoint1 = CheckpointLocation(s3_uri="s3://bucket/checkpoint1")
        checkpoint2 = CheckpointLocation(s3_uri="s3://bucket/checkpoint2")
        args = (checkpoint1,)
        kwargs = {"checkpoint": checkpoint2}
        request_dict = {}
        
        with pytest.raises(ValueError, match="cannot have more than one argument of type CheckpointLocation"):
            _update_job_request_with_checkpoint_config(args, kwargs, request_dict)

    def test_no_checkpoint(self):
        """Test lines 1232-1233: no checkpoint location."""
        args = ("arg1", "arg2")
        kwargs = {"key": "value"}
        request_dict = {}
        
        _update_job_request_with_checkpoint_config(args, kwargs, request_dict)
        
        assert "CheckpointConfig" not in request_dict


class TestConvertRunToJson:
    """Test _convert_run_to_json for uncovered lines."""

    def test_convert_run(self):
        """Test lines 1276-1278: convert run to JSON."""
        mock_run = Mock()
        mock_run.experiment_name = "test-experiment"
        mock_run.run_name = "test-run"
        
        result = _convert_run_to_json(mock_run)
        data = json.loads(result)
        
        assert data["experiment_name"] == "test-experiment"
        assert data["run_name"] == "test-run"


class TestSparkDependencies:
    """Test Spark dependency functions for uncovered lines."""

    def test_upload_spark_config_none(self, mock_session):
        """Test lines 1356: upload None Spark configuration."""
        result = _upload_serialized_spark_configuration(
            "s3://bucket/base",
            "kms-key",
            None,
            mock_session
        )
        assert result is None

    @patch("sagemaker.core.remote_function.job.S3Uploader")
    def test_upload_spark_config(self, mock_uploader, mock_session):
        """Test lines 1339-1356: upload Spark configuration."""
        config = {"spark.executor.memory": "4g"}
        mock_uploader.upload_string_as_file_body = Mock()
        
        _upload_serialized_spark_configuration(
            "s3://bucket/base",
            "kms-key",
            config,
            mock_session
        )
        
        mock_uploader.upload_string_as_file_body.assert_called_once()

    def test_upload_spark_deps_none(self, mock_session):
        """Test lines 1379-1380: None dependencies."""
        result = _upload_spark_submit_deps(None, "workspace", "s3://bucket", "kms-key", mock_session)
        assert result is None

    def test_upload_spark_deps_s3_uri(self, mock_session):
        """Test lines 1388-1389: S3 URI dependency."""
        deps = ["s3://bucket/dep.jar"]
        result = _upload_spark_submit_deps(deps, "workspace", "s3://bucket", "kms-key", mock_session)
        assert "s3://bucket/dep.jar" in result

    def test_upload_spark_deps_s3a_uri(self, mock_session):
        """Test lines 1388-1389: S3A URI dependency."""
        deps = ["s3a://bucket/dep.jar"]
        result = _upload_spark_submit_deps(deps, "workspace", "s3://bucket", "kms-key", mock_session)
        assert "s3a://bucket/dep.jar" in result

    def test_upload_spark_deps_empty_workspace_raises_error(self, mock_session):
        """Test lines 1382-1383: empty workspace validation."""
        deps = ["s3://bucket/dep.jar"]
        with pytest.raises(ValueError, match="workspace_name or s3_base_uri may not be empty"):
            _upload_spark_submit_deps(deps, "", "s3://bucket", "kms-key", mock_session)

    @patch("os.path.isfile", return_value=False)
    def test_upload_spark_deps_invalid_file_raises_error(self, mock_isfile, mock_session):
        """Test lines 1391-1392: invalid local file."""
        deps = ["/invalid/path.jar"]
        with pytest.raises(ValueError, match="is not a valid local file"):
            _upload_spark_submit_deps(deps, "workspace", "s3://bucket", "kms-key", mock_session)


class TestDistributedTraining:
    """Test distributed training functions for uncovered lines."""

    def test_extend_mpirun_no_mpirun(self, mock_session):
        """Test lines 1441-1442: mpirun disabled."""
        job_settings = Mock()
        job_settings.use_mpirun = False
        request_dict = {"InputDataConfig": []}
        
        result = _extend_mpirun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_extend_mpirun_single_instance(self, mock_session):
        """Test lines 1444-1445: single instance."""
        job_settings = Mock()
        job_settings.use_mpirun = True
        job_settings.instance_count = 1
        request_dict = {"InputDataConfig": []}
        
        result = _extend_mpirun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_extend_mpirun_multiple_instances(self, mock_session):
        """Test lines 1447-1453: multiple instances."""
        job_settings = Mock()
        job_settings.use_mpirun = True
        job_settings.instance_count = 2
        request_dict = {
            "InputDataConfig": [
                {"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/data"}}}
            ]
        }
        
        result = _extend_mpirun_to_request(request_dict, job_settings)
        assert result["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3DataDistributionType"] == "FullyReplicated"

    def test_extend_torchrun_no_torchrun(self, mock_session):
        """Test lines 1506-1507: torchrun disabled."""
        job_settings = Mock()
        job_settings.use_torchrun = False
        request_dict = {"InputDataConfig": []}
        
        result = _extend_torchrun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_extend_torchrun_single_instance(self, mock_session):
        """Test lines 1524-1525: single instance."""
        job_settings = Mock()
        job_settings.use_torchrun = True
        job_settings.instance_count = 1
        request_dict = {"InputDataConfig": []}
        
        result = _extend_torchrun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_extend_torchrun_multiple_instances(self, mock_session):
        """Test lines 1527-1533: multiple instances."""
        job_settings = Mock()
        job_settings.use_torchrun = True
        job_settings.instance_count = 2
        request_dict = {
            "InputDataConfig": [
                {"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/data"}}}
            ]
        }
        
        result = _extend_torchrun_to_request(request_dict, job_settings)
        assert result["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3DataDistributionType"] == "FullyReplicated"


class TestJobStatus:
    """Test job status functions for uncovered lines."""

    def test_check_job_status_completed(self):
        """Test lines 1978-1979: completed status."""
        desc = {"TrainingJobStatus": "Completed"}
        _check_job_status("test-job", desc, "TrainingJobStatus")

    def test_check_job_status_stopped(self):
        """Test lines 1978-1986: stopped status."""
        desc = {"TrainingJobStatus": "Stopped"}
        with patch("sagemaker.core.remote_function.job.logger") as mock_logger:
            _check_job_status("test-job", desc, "TrainingJobStatus")
            mock_logger.warning.assert_called_once()

    def test_check_job_status_failed(self):
        """Test lines 1987-2011: failed status."""
        desc = {
            "TrainingJobStatus": "Failed",
            "FailureReason": "Test failure"
        }
        from sagemaker.core import exceptions
        with pytest.raises(exceptions.UnexpectedStatusException):
            _check_job_status("test-job", desc, "TrainingJobStatus")

    def test_check_job_status_capacity_error(self):
        """Test lines 2002-2007: CapacityError."""
        desc = {
            "TrainingJobStatus": "Failed",
            "FailureReason": "CapacityError: Insufficient capacity"
        }
        from sagemaker.core import exceptions
        with pytest.raises(exceptions.CapacityError):
            _check_job_status("test-job", desc, "TrainingJobStatus")


class TestRuleStatuses:
    """Test rule status functions for uncovered lines."""

    def test_rule_statuses_no_last(self):
        """Test lines 2092-2093: no last statuses."""
        current = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        result = _rule_statuses_changed(current, None)
        assert result is True

    def test_rule_statuses_changed(self):
        """Test lines 2095-2098: changed status."""
        current = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "Completed"}]
        last = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        result = _rule_statuses_changed(current, last)
        assert result is True

    def test_rule_statuses_unchanged(self):
        """Test lines 2100: unchanged status."""
        current = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        last = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        result = _rule_statuses_changed(current, last)
        assert result is False


class TestLogsInit:
    """Test _logs_init function for uncovered lines."""

    def test_logs_init_training_job(self, mock_session):
        """Test lines 2098-2105: training job."""
        description = {
            "ResourceConfig": {"InstanceCount": 2}
        }
        result = _logs_init(mock_session.boto_session, description, "Training")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 2
        assert log_group == "/aws/sagemaker/TrainingJobs"

    def test_logs_init_training_job_instance_groups(self, mock_session):
        """Test lines 2098-2103: training job with instance groups."""
        description = {
            "ResourceConfig": {
                "InstanceGroups": [
                    {"InstanceCount": 2},
                    {"InstanceCount": 3}
                ]
            }
        }
        result = _logs_init(mock_session.boto_session, description, "Training")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 5

    def test_logs_init_transform_job(self, mock_session):
        """Test lines 2106-2107: transform job."""
        description = {
            "TransformResources": {"InstanceCount": 1}
        }
        result = _logs_init(mock_session.boto_session, description, "Transform")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 1
        assert log_group == "/aws/sagemaker/TransformJobs"

    def test_logs_init_processing_job(self, mock_session):
        """Test lines 2108-2109: processing job."""
        description = {
            "ProcessingResources": {"ClusterConfig": {"InstanceCount": 3}}
        }
        result = _logs_init(mock_session.boto_session, description, "Processing")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 3
        assert log_group == "/aws/sagemaker/ProcessingJobs"

    def test_logs_init_automl_job(self, mock_session):
        """Test lines 2110-2111: AutoML job."""
        description = {}
        result = _logs_init(mock_session.boto_session, description, "AutoML")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 0
        assert log_group == "/aws/sagemaker/AutoMLJobs"


class TestGetInitialJobState:
    """Test _get_initial_job_state for uncovered lines."""

    def test_completed_with_wait(self):
        """Test lines 2021-2023: completed job with wait."""
        description = {"TrainingJobStatus": "Completed"}
        state = _get_initial_job_state(description, "TrainingJobStatus", True)
        assert state == LogState.COMPLETE

    def test_failed_with_wait(self):
        """Test lines 2021-2023: failed job with wait."""
        description = {"TrainingJobStatus": "Failed"}
        state = _get_initial_job_state(description, "TrainingJobStatus", True)
        assert state == LogState.COMPLETE

    def test_stopped_with_wait(self):
        """Test lines 2021-2023: stopped job with wait."""
        description = {"TrainingJobStatus": "Stopped"}
        state = _get_initial_job_state(description, "TrainingJobStatus", True)
        assert state == LogState.COMPLETE

    def test_in_progress_with_wait(self):
        """Test lines 2022: in-progress job with wait."""
        description = {"TrainingJobStatus": "InProgress"}
        state = _get_initial_job_state(description, "TrainingJobStatus", True)
        assert state == LogState.TAILING

    def test_in_progress_without_wait(self):
        """Test lines 2022: in-progress job without wait."""
        description = {"TrainingJobStatus": "InProgress"}
        state = _get_initial_job_state(description, "TrainingJobStatus", False)
        assert state == LogState.COMPLETE
