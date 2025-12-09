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
"""Unit tests for sagemaker.core.remote_function.job module."""
from __future__ import absolute_import

import json
import os
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from io import BytesIO

from sagemaker.core.remote_function.job import (
    _JobSettings,
    _Job,
    _prepare_and_upload_runtime_scripts,
    _generate_input_data_config,
    _prepare_dependencies_and_pre_execution_scripts,
    _prepare_and_upload_workspace,
    _convert_run_to_json,
    _prepare_and_upload_spark_dependent_files,
    _upload_spark_submit_deps,
    _upload_serialized_spark_configuration,
    _extend_mpirun_to_request,
    _extend_torchrun_to_request,
    _extend_spark_config_to_request,
    _update_job_request_with_checkpoint_config,
    _RunInfo,
    _get_initial_job_state,
    _logs_for_job,
    _check_job_status,
    _flush_log_streams,
    _rule_statuses_changed,
    _logs_init,
    LogState,
)
from sagemaker.core.remote_function.spark_config import SparkConfig
from sagemaker.core.remote_function.checkpoint_location import CheckpointLocation


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_region_name = "us-west-2"
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = "prefix"
    session.sagemaker_client = Mock()
    session.boto_session = Mock()
    session.sagemaker_config = None
    return session


class TestJobSettings:
    """Test _JobSettings class."""

    def test_init_with_spark_and_image_raises_error(self, mock_session):
        """Test that spark_config and image_uri cannot be set together."""
        spark_config = SparkConfig()
        with pytest.raises(ValueError, match="spark_config and image_uri cannot be specified"):
            _JobSettings(
                sagemaker_session=mock_session,
                spark_config=spark_config,
                image_uri="test-image",
                instance_type="ml.m5.xlarge",
            )

    def test_init_with_spark_and_conda_env_raises_error(self, mock_session):
        """Test that spark_config and job_conda_env cannot be set together."""
        spark_config = SparkConfig()
        with pytest.raises(ValueError, match="Remote Spark jobs do not support job_conda_env"):
            _JobSettings(
                sagemaker_session=mock_session,
                spark_config=spark_config,
                job_conda_env="test-env",
                instance_type="ml.m5.xlarge",
            )

    def test_init_with_spark_and_auto_capture_raises_error(self, mock_session):
        """Test that spark_config and auto_capture dependencies cannot be set together."""
        spark_config = SparkConfig()
        with pytest.raises(ValueError, match="Remote Spark jobs do not support automatically"):
            _JobSettings(
                sagemaker_session=mock_session,
                spark_config=spark_config,
                dependencies="auto_capture",
                instance_type="ml.m5.xlarge",
            )

    def test_init_with_pre_execution_commands_and_script_raises_error(self, mock_session):
        """Test that pre_execution_commands and pre_execution_script cannot be set together."""
        with pytest.raises(
            ValueError, match="Only one of pre_execution_commands or pre_execution_script"
        ):
            _JobSettings(
                sagemaker_session=mock_session,
                pre_execution_commands=["echo test"],
                pre_execution_script="/path/to/script.sh",
                instance_type="ml.m5.xlarge",
                image_uri="test-image",
            )

    def test_init_without_instance_type_raises_error(self, mock_session):
        """Test that instance_type is required."""
        with pytest.raises(ValueError, match="instance_type is a required parameter"):
            _JobSettings(sagemaker_session=mock_session, image_uri="test-image")

    @patch.dict(os.environ, {"SAGEMAKER_INTERNAL_IMAGE_URI": "custom-image"})
    def test_get_default_image_from_env(self, mock_session):
        """Test getting default image from environment variable."""
        image = _JobSettings._get_default_image(mock_session)
        assert image == "custom-image"

    def test_get_default_image_unsupported_python_raises_error(self, mock_session):
        """Test that unsupported Python version raises error."""
        with patch.object(sys, "version_info", (3, 7, 0)):
            with pytest.raises(
                ValueError, match="Default image is supported only for Python versions"
            ):
                _JobSettings._get_default_image(mock_session)

    def test_get_default_spark_image_unsupported_python_raises_error(self, mock_session):
        """Test that unsupported Python version for Spark raises error."""
        with patch.object(sys, "version_info", (3, 8, 0)):
            with pytest.raises(
                ValueError,
                match="SageMaker Spark image for remote job only supports Python version 3.9",
            ):
                _JobSettings._get_default_spark_image(mock_session)


class TestJob:
    """Test _Job class."""

    def test_init(self, mock_session):
        """Test _Job initialization."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        assert job.job_name == "test-job"
        assert job.s3_uri == "s3://bucket/output"
        assert job.hmac_key == "test-key"

    def test_from_describe_response(self, mock_session):
        """Test creating _Job from describe response."""
        response = {
            "TrainingJobName": "test-job",
            "OutputDataConfig": {"S3OutputPath": "s3://bucket/output"},
            "Environment": {"REMOTE_FUNCTION_SECRET_KEY": "test-key"},
        }
        job = _Job.from_describe_response(response, mock_session)
        assert job.job_name == "test-job"
        assert job.s3_uri == "s3://bucket/output"
        assert job.hmac_key == "test-key"

    def test_describe_returns_cached_response(self, mock_session):
        """Test that describe returns cached response for completed jobs."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        job._last_describe_response = {"TrainingJobStatus": "Completed"}

        result = job.describe()
        assert result["TrainingJobStatus"] == "Completed"
        mock_session.sagemaker_client.describe_training_job.assert_not_called()

    def test_describe_calls_api_for_in_progress_jobs(self, mock_session):
        """Test that describe calls API for in-progress jobs."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        mock_session.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        result = job.describe()
        assert result["TrainingJobStatus"] == "InProgress"
        mock_session.sagemaker_client.describe_training_job.assert_called_once()

    def test_stop(self, mock_session):
        """Test stopping a job."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        job.stop()
        mock_session.sagemaker_client.stop_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )

    @patch("sagemaker.core.remote_function.job._logs_for_job")
    def test_wait(self, mock_logs, mock_session):
        """Test waiting for job completion."""
        job = _Job("test-job", "s3://bucket/output", mock_session, "test-key")
        mock_logs.return_value = {"TrainingJobStatus": "Completed"}

        job.wait(timeout=100)
        mock_logs.assert_called_once_with(
            sagemaker_session=mock_session, job_name="test-job", wait=True, timeout=100
        )


class TestUpdateJobRequestWithCheckpointConfig:
    """Test _update_job_request_with_checkpoint_config function."""

    def test_with_checkpoint_in_args(self):
        """Test checkpoint config in positional args."""
        checkpoint = CheckpointLocation(s3_uri="s3://bucket/checkpoint")
        args = (checkpoint,)
        kwargs = {}
        request_dict = {}

        _update_job_request_with_checkpoint_config(args, kwargs, request_dict)

        assert "CheckpointConfig" in request_dict
        assert request_dict["CheckpointConfig"]["S3Uri"] == "s3://bucket/checkpoint"
        assert request_dict["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints/"

    def test_with_checkpoint_in_kwargs(self):
        """Test checkpoint config in keyword args."""
        checkpoint = CheckpointLocation(s3_uri="s3://bucket/checkpoint")
        args = ()
        kwargs = {"checkpoint": checkpoint}
        request_dict = {}

        _update_job_request_with_checkpoint_config(args, kwargs, request_dict)

        assert "CheckpointConfig" in request_dict

    def test_with_multiple_checkpoints_raises_error(self):
        """Test that multiple checkpoints raise error."""
        checkpoint1 = CheckpointLocation(s3_uri="s3://bucket/checkpoint1")
        checkpoint2 = CheckpointLocation(s3_uri="s3://bucket/checkpoint2")
        args = (checkpoint1,)
        kwargs = {"checkpoint": checkpoint2}
        request_dict = {}

        with pytest.raises(
            ValueError, match="cannot have more than one argument of type CheckpointLocation"
        ):
            _update_job_request_with_checkpoint_config(args, kwargs, request_dict)

    def test_without_checkpoint(self):
        """Test without checkpoint location."""
        args = ("arg1", "arg2")
        kwargs = {"key": "value"}
        request_dict = {}

        _update_job_request_with_checkpoint_config(args, kwargs, request_dict)

        assert "CheckpointConfig" not in request_dict


class TestConvertRunToJson:
    """Test _convert_run_to_json function."""

    def test_convert_run_to_json(self):
        """Test converting run to JSON."""
        mock_run = Mock()
        mock_run.experiment_name = "test-experiment"
        mock_run.run_name = "test-run"

        result = _convert_run_to_json(mock_run)
        data = json.loads(result)

        assert data["experiment_name"] == "test-experiment"
        assert data["run_name"] == "test-run"


class TestUploadSerializedSparkConfiguration:
    """Test _upload_serialized_spark_configuration function."""

    @patch("sagemaker.core.remote_function.job.S3Uploader")
    def test_upload_spark_config(self, mock_uploader, mock_session):
        """Test uploading Spark configuration."""
        config = {"spark.executor.memory": "4g"}

        _upload_serialized_spark_configuration("s3://bucket/base", "kms-key", config, mock_session)

        mock_uploader.upload_string_as_file_body.assert_called_once()

    def test_upload_spark_config_none(self, mock_session):
        """Test uploading None Spark configuration."""
        result = _upload_serialized_spark_configuration(
            "s3://bucket/base", "kms-key", None, mock_session
        )

        assert result is None


class TestUploadSparkSubmitDeps:
    """Test _upload_spark_submit_deps function."""

    def test_with_none_deps(self, mock_session):
        """Test with None dependencies."""
        result = _upload_spark_submit_deps(
            None, "workspace", "s3://bucket", "kms-key", mock_session
        )
        assert result is None

    def test_with_s3_uri(self, mock_session):
        """Test with S3 URI."""
        deps = ["s3://bucket/dep.jar"]
        result = _upload_spark_submit_deps(
            deps, "workspace", "s3://bucket", "kms-key", mock_session
        )
        assert "s3://bucket/dep.jar" in result

    def test_with_empty_workspace_raises_error(self, mock_session):
        """Test with empty workspace name."""
        deps = ["s3://bucket/dep.jar"]
        with pytest.raises(ValueError, match="workspace_name or s3_base_uri may not be empty"):
            _upload_spark_submit_deps(deps, "", "s3://bucket", "kms-key", mock_session)

    @patch("os.path.isfile", return_value=False)
    def test_with_invalid_local_file_raises_error(self, mock_isfile, mock_session):
        """Test with invalid local file."""
        deps = ["/invalid/path.jar"]
        with pytest.raises(ValueError, match="is not a valid local file"):
            _upload_spark_submit_deps(deps, "workspace", "s3://bucket", "kms-key", mock_session)


class TestExtendMpirunToRequest:
    """Test _extend_mpirun_to_request function."""

    def test_without_mpirun(self, mock_session):
        """Test without mpirun enabled."""
        job_settings = Mock()
        job_settings.use_mpirun = False
        request_dict = {"InputDataConfig": []}

        result = _extend_mpirun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_with_single_instance(self, mock_session):
        """Test with single instance."""
        job_settings = Mock()
        job_settings.use_mpirun = True
        job_settings.instance_count = 1
        request_dict = {"InputDataConfig": []}

        result = _extend_mpirun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_with_multiple_instances(self, mock_session):
        """Test with multiple instances."""
        job_settings = Mock()
        job_settings.use_mpirun = True
        job_settings.instance_count = 2
        request_dict = {
            "InputDataConfig": [{"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/data"}}}]
        }

        result = _extend_mpirun_to_request(request_dict, job_settings)
        assert (
            result["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3DataDistributionType"]
            == "FullyReplicated"
        )


class TestExtendTorchrunToRequest:
    """Test _extend_torchrun_to_request function."""

    def test_without_torchrun(self, mock_session):
        """Test without torchrun enabled."""
        job_settings = Mock()
        job_settings.use_torchrun = False
        request_dict = {"InputDataConfig": []}

        result = _extend_torchrun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_with_single_instance(self, mock_session):
        """Test with single instance."""
        job_settings = Mock()
        job_settings.use_torchrun = True
        job_settings.instance_count = 1
        request_dict = {"InputDataConfig": []}

        result = _extend_torchrun_to_request(request_dict, job_settings)
        assert result == request_dict

    def test_with_multiple_instances(self, mock_session):
        """Test with multiple instances."""
        job_settings = Mock()
        job_settings.use_torchrun = True
        job_settings.instance_count = 2
        request_dict = {
            "InputDataConfig": [{"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/data"}}}]
        }

        result = _extend_torchrun_to_request(request_dict, job_settings)
        assert (
            result["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3DataDistributionType"]
            == "FullyReplicated"
        )


class TestExtendSparkConfigToRequest:
    """Test _extend_spark_config_to_request function."""

    def test_without_spark_config(self, mock_session):
        """Test without spark config."""
        job_settings = Mock()
        job_settings.spark_config = None
        request_dict = {"AlgorithmSpecification": {"ContainerEntrypoint": []}}

        result = _extend_spark_config_to_request(request_dict, job_settings, "s3://bucket")
        assert result == request_dict

    @patch("sagemaker.core.remote_function.job._prepare_and_upload_spark_dependent_files")
    def test_with_spark_config(self, mock_upload, mock_session):
        """Test with spark config."""
        mock_upload.return_value = (None, None, None, "s3://bucket/config.json")

        job_settings = Mock()
        spark_config = SparkConfig(spark_event_logs_uri="s3://bucket/logs")
        job_settings.spark_config = spark_config
        job_settings.s3_kms_key = None
        job_settings.sagemaker_session = mock_session

        request_dict = {
            "AlgorithmSpecification": {"ContainerEntrypoint": []},
            "InputDataConfig": [{"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/data"}}}],
        }

        result = _extend_spark_config_to_request(request_dict, job_settings, "s3://bucket")
        assert (
            "--spark-event-logs-s3-uri" in result["AlgorithmSpecification"]["ContainerEntrypoint"]
        )


class TestGetInitialJobState:
    """Test _get_initial_job_state function."""

    def test_with_completed_job_and_wait(self):
        """Test with completed job and wait=True."""
        description = {"TrainingJobStatus": "Completed"}
        state = _get_initial_job_state(description, "TrainingJobStatus", True)
        assert state == LogState.COMPLETE

    def test_with_in_progress_job_and_wait(self):
        """Test with in-progress job and wait=True."""
        description = {"TrainingJobStatus": "InProgress"}
        state = _get_initial_job_state(description, "TrainingJobStatus", True)
        assert state == LogState.TAILING

    def test_with_in_progress_job_and_no_wait(self):
        """Test with in-progress job and wait=False."""
        description = {"TrainingJobStatus": "InProgress"}
        state = _get_initial_job_state(description, "TrainingJobStatus", False)
        assert state == LogState.COMPLETE


class TestCheckJobStatus:
    """Test _check_job_status function."""

    def test_with_completed_status(self):
        """Test with completed status."""
        desc = {"TrainingJobStatus": "Completed"}
        _check_job_status("test-job", desc, "TrainingJobStatus")

    def test_with_stopped_status(self):
        """Test with stopped status."""
        desc = {"TrainingJobStatus": "Stopped"}
        with patch("sagemaker.core.remote_function.job.logger") as mock_logger:
            _check_job_status("test-job", desc, "TrainingJobStatus")
            mock_logger.warning.assert_called_once()

    def test_with_failed_status_raises_error(self):
        """Test with failed status."""
        desc = {"TrainingJobStatus": "Failed", "FailureReason": "Test failure"}
        with pytest.raises(Exception):
            _check_job_status("test-job", desc, "TrainingJobStatus")

    def test_with_capacity_error_raises_capacity_error(self):
        """Test with CapacityError."""
        desc = {
            "TrainingJobStatus": "Failed",
            "FailureReason": "CapacityError: Insufficient capacity",
        }
        from sagemaker.core import exceptions

        with pytest.raises(exceptions.CapacityError):
            _check_job_status("test-job", desc, "TrainingJobStatus")


class TestRuleStatusesChanged:
    """Test _rule_statuses_changed function."""

    def test_with_no_last_statuses(self):
        """Test with no last statuses."""
        current = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        result = _rule_statuses_changed(current, None)
        assert result is True

    def test_with_changed_status(self):
        """Test with changed status."""
        current = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "Completed"}]
        last = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        result = _rule_statuses_changed(current, last)
        assert result is True

    def test_with_unchanged_status(self):
        """Test with unchanged status."""
        current = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        last = [{"RuleConfigurationName": "rule1", "RuleEvaluationStatus": "InProgress"}]
        result = _rule_statuses_changed(current, last)
        assert result is False


class TestLogsInit:
    """Test _logs_init function."""

    def test_with_training_job(self, mock_session):
        """Test with training job."""
        description = {"ResourceConfig": {"InstanceCount": 2}}
        result = _logs_init(mock_session.boto_session, description, "Training")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 2
        assert log_group == "/aws/sagemaker/TrainingJobs"

    def test_with_training_job_instance_groups(self, mock_session):
        """Test with training job using instance groups."""
        description = {
            "ResourceConfig": {"InstanceGroups": [{"InstanceCount": 2}, {"InstanceCount": 3}]}
        }
        result = _logs_init(mock_session.boto_session, description, "Training")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 5

    def test_with_transform_job(self, mock_session):
        """Test with transform job."""
        description = {"TransformResources": {"InstanceCount": 1}}
        result = _logs_init(mock_session.boto_session, description, "Transform")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 1
        assert log_group == "/aws/sagemaker/TransformJobs"

    def test_with_processing_job(self, mock_session):
        """Test with processing job."""
        description = {"ProcessingResources": {"ClusterConfig": {"InstanceCount": 3}}}
        result = _logs_init(mock_session.boto_session, description, "Processing")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 3
        assert log_group == "/aws/sagemaker/ProcessingJobs"

    def test_with_automl_job(self, mock_session):
        """Test with AutoML job."""
        description = {}
        result = _logs_init(mock_session.boto_session, description, "AutoML")
        instance_count, stream_names, positions, client, log_group, dot, color_wrap = result
        assert instance_count == 0
        assert log_group == "/aws/sagemaker/AutoMLJobs"


class TestFlushLogStreams:
    """Test _flush_log_streams function."""

    @patch("sagemaker.core.remote_function.job.sagemaker_logs")
    def test_with_no_streams(self, mock_logs, mock_session):
        """Test with no log streams."""
        stream_names = []
        positions = {}
        client = Mock()
        client.describe_log_streams.return_value = {"logStreams": []}

        _flush_log_streams(
            stream_names,
            1,
            client,
            "/aws/sagemaker/TrainingJobs",
            "test-job",
            positions,
            False,
            lambda x, y: None,
        )

    @patch("sagemaker.core.remote_function.job.sagemaker_logs")
    def test_with_client_error_resource_not_found(self, mock_logs, mock_session):
        """Test with ResourceNotFoundException."""
        from botocore.exceptions import ClientError

        stream_names = []
        positions = {}
        client = Mock()
        error_response = {"Error": {"Code": "ResourceNotFoundException"}}
        client.describe_log_streams.side_effect = ClientError(
            error_response, "describe_log_streams"
        )

        _flush_log_streams(
            stream_names,
            1,
            client,
            "/aws/sagemaker/TrainingJobs",
            "test-job",
            positions,
            False,
            lambda x, y: None,
        )

    @patch("sagemaker.core.remote_function.job.sagemaker_logs")
    def test_with_client_error_other(self, mock_logs, mock_session):
        """Test with other ClientError."""
        from botocore.exceptions import ClientError

        stream_names = []
        positions = {}
        client = Mock()
        error_response = {"Error": {"Code": "OtherError"}}
        client.describe_log_streams.side_effect = ClientError(
            error_response, "describe_log_streams"
        )

        with pytest.raises(ClientError):
            _flush_log_streams(
                stream_names,
                1,
                client,
                "/aws/sagemaker/TrainingJobs",
                "test-job",
                positions,
                False,
                lambda x, y: None,
            )


class TestPrepareAndUploadRuntimeScripts:
    """Test _prepare_and_upload_runtime_scripts function."""

    @patch("sagemaker.core.remote_function.job.S3Uploader")
    @patch("sagemaker.core.remote_function.job._tmpdir")
    @patch("sagemaker.core.remote_function.job.shutil")
    @patch("builtins.open", new_callable=mock_open)
    def test_without_spark_or_distributed(
        self, mock_file, mock_shutil, mock_tmpdir, mock_uploader, mock_session
    ):
        """Test without Spark or distributed training."""
        mock_tmpdir.return_value.__enter__ = Mock(return_value="/tmp/test")
        mock_tmpdir.return_value.__exit__ = Mock(return_value=False)
        mock_uploader.upload.return_value = "s3://bucket/scripts"

        result = _prepare_and_upload_runtime_scripts(
            None, "s3://bucket", "kms-key", mock_session, False, False
        )

        assert result == "s3://bucket/scripts"

    @patch("sagemaker.core.remote_function.job.S3Uploader")
    @patch("sagemaker.core.remote_function.job._tmpdir")
    @patch("sagemaker.core.remote_function.job.shutil")
    @patch("builtins.open", new_callable=mock_open)
    def test_with_spark(self, mock_file, mock_shutil, mock_tmpdir, mock_uploader, mock_session):
        """Test with Spark config."""
        mock_tmpdir.return_value.__enter__ = Mock(return_value="/tmp/test")
        mock_tmpdir.return_value.__exit__ = Mock(return_value=False)
        mock_uploader.upload.return_value = "s3://bucket/scripts"

        spark_config = SparkConfig()
        result = _prepare_and_upload_runtime_scripts(
            spark_config, "s3://bucket", "kms-key", mock_session, False, False
        )

        assert result == "s3://bucket/scripts"

    @patch("sagemaker.core.remote_function.job.S3Uploader")
    @patch("sagemaker.core.remote_function.job._tmpdir")
    @patch("sagemaker.core.remote_function.job.shutil")
    @patch("builtins.open", new_callable=mock_open)
    def test_with_torchrun(self, mock_file, mock_shutil, mock_tmpdir, mock_uploader, mock_session):
        """Test with torchrun."""
        mock_tmpdir.return_value.__enter__ = Mock(return_value="/tmp/test")
        mock_tmpdir.return_value.__exit__ = Mock(return_value=False)
        mock_uploader.upload.return_value = "s3://bucket/scripts"

        result = _prepare_and_upload_runtime_scripts(
            None, "s3://bucket", "kms-key", mock_session, True, False
        )

        assert result == "s3://bucket/scripts"

    @patch("sagemaker.core.remote_function.job.S3Uploader")
    @patch("sagemaker.core.remote_function.job._tmpdir")
    @patch("sagemaker.core.remote_function.job.shutil")
    @patch("builtins.open", new_callable=mock_open)
    def test_with_mpirun(self, mock_file, mock_shutil, mock_tmpdir, mock_uploader, mock_session):
        """Test with mpirun."""
        mock_tmpdir.return_value.__enter__ = Mock(return_value="/tmp/test")
        mock_tmpdir.return_value.__exit__ = Mock(return_value=False)
        mock_uploader.upload.return_value = "s3://bucket/scripts"

        result = _prepare_and_upload_runtime_scripts(
            None, "s3://bucket", "kms-key", mock_session, False, True
        )

        assert result == "s3://bucket/scripts"


class TestPrepareAndUploadWorkspace:
    """Test _prepare_and_upload_workspace function."""

    def test_without_dependencies_or_workdir(self, mock_session):
        """Test without dependencies or workdir."""
        result = _prepare_and_upload_workspace(
            None, False, None, None, "s3://bucket", "kms-key", mock_session, None
        )
        assert result is None

    @patch("sagemaker.core.remote_function.job.S3Uploader")
    @patch("sagemaker.core.remote_function.job._tmpdir")
    @patch("sagemaker.core.remote_function.job.shutil")
    @patch("sagemaker.core.remote_function.job.copy_workdir")
    @patch("os.mkdir")
    @patch("os.path.isdir", return_value=False)
    def test_with_workdir(
        self,
        mock_isdir,
        mock_mkdir,
        mock_copy,
        mock_shutil,
        mock_tmpdir,
        mock_uploader,
        mock_session,
    ):
        """Test with workdir."""
        mock_tmpdir.return_value.__enter__ = Mock(return_value="/tmp/test")
        mock_tmpdir.return_value.__exit__ = Mock(return_value=False)
        mock_shutil.make_archive.return_value = "/tmp/test/workspace.zip"
        mock_uploader.upload.return_value = "s3://bucket/workspace.zip"

        result = _prepare_and_upload_workspace(
            None, True, None, None, "s3://bucket", "kms-key", mock_session, None
        )

        assert result == "s3://bucket/workspace.zip"


class TestPrepareDependenciesAndPreExecutionScripts:
    """Test _prepare_dependencies_and_pre_execution_scripts function."""

    def test_without_dependencies_or_scripts(self, mock_session):
        """Test without dependencies or scripts."""
        result = _prepare_dependencies_and_pre_execution_scripts(
            None, None, None, "s3://bucket", "kms-key", mock_session, "/tmp"
        )
        assert result is None

    @patch("sagemaker.core.workflow.utilities.load_step_compilation_context")
    @patch("sagemaker.core.remote_function.job.shutil")
    @patch("sagemaker.core.remote_function.job.S3Uploader")
    def test_with_dependencies(self, mock_uploader, mock_shutil, mock_context, mock_session):
        """Test with dependencies file."""
        mock_shutil.copy2.return_value = "/tmp/requirements.txt"
        mock_uploader.upload.return_value = "s3://bucket/deps"
        mock_context.return_value = Mock(step_name="step", pipeline_build_time="123")

        result = _prepare_dependencies_and_pre_execution_scripts(
            "/path/to/requirements.txt", None, None, "s3://bucket", "kms-key", mock_session, "/tmp"
        )

        assert result == "s3://bucket/deps"

    @patch("sagemaker.core.workflow.utilities.load_step_compilation_context")
    @patch("builtins.open", create=True)
    @patch("sagemaker.core.remote_function.job.S3Uploader")
    def test_with_pre_execution_commands(
        self, mock_uploader, mock_open, mock_context, mock_session
    ):
        """Test with pre-execution commands."""
        mock_uploader.upload.return_value = "s3://bucket/scripts"
        mock_context.return_value = Mock(step_name="step", pipeline_build_time="123")

        result = _prepare_dependencies_and_pre_execution_scripts(
            None, ["echo test"], None, "s3://bucket", "kms-key", mock_session, "/tmp"
        )

        assert result == "s3://bucket/scripts"

    @patch("sagemaker.core.workflow.utilities.load_step_compilation_context")
    @patch("sagemaker.core.remote_function.job.shutil")
    @patch("sagemaker.core.remote_function.job.S3Uploader")
    def test_with_pre_execution_script(
        self, mock_uploader, mock_shutil, mock_context, mock_session
    ):
        """Test with pre-execution script."""
        mock_shutil.copy2.return_value = "/tmp/pre_exec.sh"
        mock_uploader.upload.return_value = "s3://bucket/scripts"
        mock_context.return_value = Mock(step_name="step", pipeline_build_time="123")

        result = _prepare_dependencies_and_pre_execution_scripts(
            None, None, "/path/to/script.sh", "s3://bucket", "kms-key", mock_session, "/tmp"
        )

        assert result == "s3://bucket/scripts"


class TestPrepareAndUploadSparkDependentFiles:
    """Test _prepare_and_upload_spark_dependent_files function."""

    def test_without_spark_config(self, mock_session):
        """Test without Spark config."""
        result = _prepare_and_upload_spark_dependent_files(
            None, "s3://bucket", "kms-key", mock_session
        )
        assert result == (None, None, None, None)

    @patch("sagemaker.core.remote_function.job._upload_spark_submit_deps")
    @patch("sagemaker.core.remote_function.job._upload_serialized_spark_configuration")
    def test_with_spark_config(self, mock_upload_config, mock_upload_deps, mock_session):
        """Test with Spark config."""
        mock_upload_deps.return_value = "s3://bucket/deps"
        mock_upload_config.return_value = "s3://bucket/config.json"

        spark_config = SparkConfig(
            submit_jars=["test.jar"],
            submit_py_files=["test.py"],
            submit_files=["test.txt"],
            configuration={"Classification": "spark-defaults", "Properties": {"key": "value"}},
        )

        result = _prepare_and_upload_spark_dependent_files(
            spark_config, "s3://bucket", "kms-key", mock_session
        )

        assert len(result) == 4


class TestJobCompile:
    """Test _Job.compile method."""

    @patch("sagemaker.core.remote_function.job.StoredFunction")
    @patch("sagemaker.core.remote_function.job._generate_input_data_config")
    def test_compile_basic(self, mock_input_config, mock_stored_func, mock_session):
        """Test basic compile."""
        mock_input_config.return_value = []
        mock_stored_func.return_value.save = Mock()

        job_settings = Mock()
        job_settings.max_runtime_in_seconds = 3600
        job_settings.max_wait_time_in_seconds = None
        job_settings.max_retry_attempts = 1
        job_settings.role = "arn:aws:iam::123456789012:role/test"
        job_settings.tags = None
        job_settings.s3_kms_key = None
        job_settings.disable_output_compression = False
        job_settings.volume_size = 30
        job_settings.instance_count = 1
        job_settings.instance_type = "ml.m5.xlarge"
        job_settings.volume_kms_key = None
        job_settings.keep_alive_period_in_seconds = None
        job_settings.enable_network_isolation = False
        job_settings.encrypt_inter_container_traffic = False
        job_settings.vpc_config = None
        job_settings.use_spot_instances = False
        job_settings.environment_variables = {}
        job_settings.image_uri = "test-image"
        job_settings.sagemaker_session = mock_session
        job_settings.use_torchrun = False
        job_settings.use_mpirun = False
        job_settings.nproc_per_node = None
        job_settings.job_conda_env = None
        job_settings.spark_config = None
        job_settings.dependencies = None

        def test_func():
            pass

        result = _Job.compile(job_settings, "test-job", "s3://bucket", test_func, (), {})

        assert result["TrainingJobName"] == "test-job"
        assert result["RoleArn"] == "arn:aws:iam::123456789012:role/test"


class TestJobStart:
    """Test _Job.start method."""

    @patch("sagemaker.core.remote_function.job._Job.compile")
    @patch("sagemaker.core.remote_function.job._Job._get_job_name")
    def test_start(self, mock_get_name, mock_compile, mock_session):
        """Test starting a job."""
        mock_get_name.return_value = "test-job"
        mock_compile.return_value = {
            "TrainingJobName": "test-job",
            "Environment": {"REMOTE_FUNCTION_SECRET_KEY": "test-key"},
        }

        job_settings = Mock()
        job_settings.s3_root_uri = "s3://bucket"
        job_settings.sagemaker_session = mock_session

        def test_func():
            pass

        job = _Job.start(job_settings, test_func, (), {})

        assert job.job_name == "test-job"
        mock_session.sagemaker_client.create_training_job.assert_called_once()


class TestJobGetJobName:
    """Test _Job._get_job_name method."""

    def test_with_job_name_prefix(self, mock_session):
        """Test with job_name_prefix."""
        job_settings = Mock()
        job_settings.job_name_prefix = "my-job"

        def test_func():
            pass

        result = _Job._get_job_name(job_settings, test_func)
        assert "my-job" in result

    def test_without_job_name_prefix(self, mock_session):
        """Test without job_name_prefix."""
        job_settings = Mock()
        job_settings.job_name_prefix = None

        def test_func():
            pass

        result = _Job._get_job_name(job_settings, test_func)
        assert "test-func" in result

    def test_with_special_characters_in_func_name(self, mock_session):
        """Test with special characters in function name."""
        job_settings = Mock()
        job_settings.job_name_prefix = None

        def _test_func():
            pass

        result = _Job._get_job_name(job_settings, _test_func)
        assert result.startswith("test-func")
