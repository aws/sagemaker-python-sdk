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

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json

from sagemaker.core.model_monitor.clarify_model_monitoring import (
    ClarifyModelMonitor,
    ModelBiasMonitor,
    ClarifyMonitoringExecution,
    ClarifyBaseliningJob,
    ClarifyBaseliningConfig,
    BiasAnalysisConfig,
)
from sagemaker.core.model_monitor.model_monitoring import EndpointInput
from sagemaker.core.clarify import BiasConfig, DataConfig, ModelConfig, ModelPredictedLabelConfig
from sagemaker.core.exceptions import UnexpectedStatusException


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session"""
    session = Mock()
    session.boto_session.region_name = "us-west-2"
    session.boto_region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.s3_resource = Mock()
    session.s3_resource.meta.client._endpoint.host = "https://s3.us-west-2.amazonaws.com"
    session.sagemaker_config = {"SchemaVersion": "1.0"}
    return session


class TestClarifyModelMonitor:
    """Test cases for ClarifyModelMonitor class"""

    def test_init_raises_error_for_abstract_class(self, mock_session):
        """Test that ClarifyModelMonitor cannot be instantiated directly"""
        with pytest.raises(TypeError, match="is abstract"):
            ClarifyModelMonitor(
                role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
            )

    def test_run_baseline_not_implemented(self, mock_session):
        """Test that run_baseline raises NotImplementedError"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        with pytest.raises(NotImplementedError, match="only allowed for ModelMonitor"):
            monitor.run_baseline()

    def test_latest_monitoring_statistics_not_implemented(self, mock_session):
        """Test that latest_monitoring_statistics raises NotImplementedError"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        with pytest.raises(NotImplementedError, match="doesn't support statistics"):
            monitor.latest_monitoring_statistics()

    @patch("sagemaker.core.model_monitor.model_monitoring.boto_list_monitoring_executions")
    @patch("sagemaker.core.model_monitor.model_monitoring.MonitoringExecution.from_processing_arn")
    def test_list_executions(self, mock_from_arn, mock_list_executions, mock_session):
        """Test list_executions returns ClarifyMonitoringExecution objects"""
        from sagemaker.core.processing import ProcessingOutput
        from sagemaker.core.shapes import ProcessingS3Output

        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )
        monitor.monitoring_schedule_name = "test-schedule"

        mock_list_executions.return_value = {
            "MonitoringExecutionSummaries": [
                {
                    "MonitoringExecutionStatus": "Completed",
                    "ProcessingJobArn": "arn:aws:sagemaker:us-west-2:123456789012:processing-job/test-job",
                    "ScheduledTime": "2023-01-01T00:00:00Z",
                }
            ]
        }

        mock_execution = Mock()
        mock_execution.sagemaker_session = mock_session
        mock_execution.job_name = "test-job"
        mock_execution.inputs = []
        mock_execution.output = ProcessingOutput(
            output_name="output",
            s3_output=ProcessingS3Output(
                s3_uri="s3://bucket/output",
                local_path="/opt/ml/processing/output",
                s3_upload_mode="EndOfJob",
            ),
        )
        mock_execution.output_kms_key = None
        mock_from_arn.return_value = mock_execution

        executions = monitor.list_executions()

        assert len(executions) == 1
        assert isinstance(executions[0], ClarifyMonitoringExecution)

    @patch("sagemaker.core.model_monitor.clarify_model_monitoring.boto_list_monitoring_executions")
    @patch("sagemaker.core.model_monitor.clarify_model_monitoring.logs_for_processing_job")
    def test_get_latest_execution_logs_success(self, mock_logs, mock_list_executions, mock_session):
        """Test get_latest_execution_logs with successful execution"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )
        monitor.monitoring_schedule_name = "test-schedule"

        mock_list_executions.return_value = {
            "MonitoringExecutionSummaries": [
                {
                    "ProcessingJobArn": "arn:aws:sagemaker:us-west-2:123456789012:processing-job/test-job"
                }
            ]
        }

        monitor.get_latest_execution_logs(wait=False)

        mock_logs.assert_called_once_with(mock_session, job_name="test-job", wait=False)

    @patch("sagemaker.core.model_monitor.clarify_model_monitoring.boto_list_monitoring_executions")
    def test_get_latest_execution_logs_no_executions(self, mock_list_executions, mock_session):
        """Test get_latest_execution_logs raises error when no executions"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )
        monitor.monitoring_schedule_name = "test-schedule"

        mock_list_executions.return_value = {"MonitoringExecutionSummaries": []}

        with pytest.raises(ValueError, match="No execution jobs were kicked off"):
            monitor.get_latest_execution_logs()

    @patch("sagemaker.core.model_monitor.clarify_model_monitoring.boto_list_monitoring_executions")
    def test_get_latest_execution_logs_no_processing_job(self, mock_list_executions, mock_session):
        """Test get_latest_execution_logs raises error when no processing job"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )
        monitor.monitoring_schedule_name = "test-schedule"

        mock_list_executions.return_value = {"MonitoringExecutionSummaries": [{}]}

        with pytest.raises(ValueError, match="Processing Job did not run"):
            monitor.get_latest_execution_logs()


class TestModelBiasMonitor:
    """Test cases for ModelBiasMonitor class"""

    def test_init_with_minimal_params(self, mock_session):
        """Test initialization with minimal parameters"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        assert monitor.role == "arn:aws:iam::123456789012:role/SageMakerRole"
        assert monitor.instance_count == 1
        assert monitor.instance_type == "ml.m5.xlarge"
        assert monitor.volume_size_in_gb == 30
        assert monitor.sagemaker_session == mock_session

    def test_init_with_all_params(self, mock_session):
        """Test initialization with all parameters"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_count=2,
            instance_type="ml.m5.2xlarge",
            volume_size_in_gb=50,
            volume_kms_key="kms-key-id",
            output_kms_key="output-kms-key",
            max_runtime_in_seconds=7200,
            base_job_name="bias-monitor",
            sagemaker_session=mock_session,
            env={"KEY": "VALUE"},
            tags=[("Project", "ML")],
        )

        assert monitor.instance_count == 2
        assert monitor.instance_type == "ml.m5.2xlarge"
        assert monitor.volume_size_in_gb == 50
        assert monitor.volume_kms_key == "kms-key-id"
        assert monitor.output_kms_key == "output-kms-key"
        assert monitor.max_runtime_in_seconds == 7200
        assert monitor.base_job_name == "bias-monitor"
        assert monitor.env == {"KEY": "VALUE"}

    def test_monitoring_type(self):
        """Test monitoring_type class method"""
        assert ModelBiasMonitor.monitoring_type() == "ModelBias"

    @patch("sagemaker.core.model_monitor.clarify_model_monitoring.SageMakerClarifyProcessor")
    def test_suggest_baseline(self, mock_processor_class, mock_session):
        """Test suggest_baseline method"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        mock_processor = Mock()
        mock_processor.latest_job = Mock()
        mock_processor_class.return_value = mock_processor

        data_config = DataConfig(
            s3_data_input_path="s3://bucket/data",
            s3_output_path="s3://bucket/output",
            label="target",
            headers=["feature1", "feature2", "target"],
            dataset_type="text/csv",
        )

        bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="feature1")

        model_config = ModelConfig(
            model_name="test-model", instance_type="ml.m5.xlarge", instance_count=1
        )

        model_predicted_label_config = ModelPredictedLabelConfig(label=0)

        monitor.suggest_baseline(
            data_config=data_config,
            bias_config=bias_config,
            model_config=model_config,
            model_predicted_label_config=model_predicted_label_config,
            wait=False,
            logs=False,
        )

        assert monitor.latest_baselining_job_config is not None
        assert monitor.latest_baselining_job_name is not None
        assert monitor.latest_baselining_job is not None

    def test_create_monitoring_schedule_without_ground_truth(self, mock_session):
        """Test create_monitoring_schedule raises error without ground_truth_input"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        with pytest.raises(ValueError, match="ground_truth_input can not be None"):
            monitor.create_monitoring_schedule(
                endpoint_input="test-endpoint", output_s3_uri="s3://bucket/output"
            )

    @patch("sagemaker.core.s3.S3Uploader.upload_string_as_file_body")
    @patch("sagemaker.core.common_utils.name_from_base")
    def test_create_monitoring_schedule_success(self, mock_name, mock_upload, mock_session):
        """Test successful create_monitoring_schedule"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        mock_name.side_effect = ["test-schedule-name", "test-job-def-name"]
        mock_upload.return_value = "s3://bucket/analysis_config.json"

        # Set up baselining config
        bias_config_mock = Mock()
        bias_config_mock.get_config.return_value = {"facet_name": "f1"}

        monitor.latest_baselining_job_name = "baseline-job"
        monitor.latest_baselining_job_config = ClarifyBaseliningConfig(
            analysis_config=BiasAnalysisConfig(
                bias_config=bias_config_mock, headers=["f1", "f2"], label="target"
            ),
            features_attribute="features",
        )

        with patch.object(mock_session.sagemaker_client, "create_model_bias_job_definition"):
            with patch.object(mock_session.sagemaker_client, "create_monitoring_schedule"):
                monitor.create_monitoring_schedule(
                    endpoint_input="test-endpoint",
                    ground_truth_input="s3://bucket/ground-truth",
                    output_s3_uri="s3://bucket/output",
                    schedule_cron_expression="cron(0 * * * ? *)",
                )

        assert monitor.monitoring_schedule_name is not None


class TestClarifyBaseliningConfig:
    """Test cases for ClarifyBaseliningConfig"""

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters"""
        config = ClarifyBaseliningConfig(analysis_config=Mock())

        assert config.analysis_config is not None
        assert config.features_attribute is None
        assert config.inference_attribute is None

    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        analysis_config = Mock()
        config = ClarifyBaseliningConfig(
            analysis_config=analysis_config,
            features_attribute="features",
            inference_attribute="prediction",
            probability_attribute="prob",
            probability_threshold_attribute=0.5,
        )

        assert config.analysis_config == analysis_config
        assert config.features_attribute == "features"
        assert config.inference_attribute == "prediction"
        assert config.probability_attribute == "prob"
        assert config.probability_threshold_attribute == 0.5


class TestBiasAnalysisConfig:
    """Test cases for BiasAnalysisConfig"""

    def test_to_dict(self):
        """Test _to_dict method"""
        bias_config = Mock()
        bias_config.get_config.return_value = {"facet_name": "feature1"}

        config = BiasAnalysisConfig(
            bias_config=bias_config, headers=["f1", "f2", "target"], label="target"
        )

        result = config._to_dict()

        assert "facet_name" in result
        assert result["headers"] == ["f1", "f2", "target"]
        assert result["label"] == "target"

    def test_to_dict_with_minimal_params(self):
        """Test _to_dict with minimal parameters"""
        bias_config = Mock()
        bias_config.get_config.return_value = {"facet_name": "age"}

        config = BiasAnalysisConfig(bias_config=bias_config, headers=None, label=None)

        result = config._to_dict()

        assert "facet_name" in result


class TestClarifyMonitoringExecution:
    """Test cases for ClarifyMonitoringExecution"""

    def test_init(self, mock_session):
        """Test ClarifyMonitoringExecution initialization"""
        from sagemaker.core.processing import ProcessingOutput
        from sagemaker.core.shapes import ProcessingS3Output

        output = ProcessingOutput(
            output_name="output",
            s3_output=ProcessingS3Output(
                s3_uri="s3://bucket/output",
                local_path="/opt/ml/processing/output",
                s3_upload_mode="EndOfJob",
            ),
        )

        execution = ClarifyMonitoringExecution(
            sagemaker_session=mock_session,
            job_name="test-job",
            inputs=[],
            output=output,
            output_kms_key="kms-key",
        )

        assert execution.processing_job_name == "test-job"
        assert execution.processing_output_config.kms_key_id == "kms-key"


class TestModelBiasMonitorAdvanced:
    """Advanced test cases for ModelBiasMonitor"""

    def test_suggest_baseline_with_all_params(self, mock_session):
        """Test suggest_baseline with all parameters"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_count=2,
            instance_type="ml.m5.2xlarge",
            volume_size_in_gb=50,
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.core.model_monitor.clarify_model_monitoring.SageMakerClarifyProcessor"
        ) as mock_processor_class:
            mock_processor = Mock()
            mock_processor.latest_job = Mock()
            mock_processor_class.return_value = mock_processor

            data_config = DataConfig(
                s3_data_input_path="s3://bucket/data",
                s3_output_path="s3://bucket/output",
                label="target",
                headers=["f1", "f2", "target"],
                dataset_type="text/csv",
            )

            bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="f1")

            model_config = ModelConfig(
                model_name="test-model", instance_type="ml.m5.xlarge", instance_count=1
            )

            model_predicted_label_config = ModelPredictedLabelConfig(
                label=0, probability=0.8, probability_threshold=0.5
            )

            monitor.suggest_baseline(
                data_config=data_config,
                bias_config=bias_config,
                model_config=model_config,
                model_predicted_label_config=model_predicted_label_config,
                wait=True,
                logs=True,
                job_name="custom-baseline-job",
                kms_key="kms-key-123",
            )

            assert monitor.latest_baselining_job_config is not None
            assert monitor.latest_baselining_job_config.inference_attribute == "0"
            assert monitor.latest_baselining_job_config.probability_attribute == "0.8"
            assert monitor.latest_baselining_job_config.probability_threshold_attribute == 0.5

    @pytest.mark.skip(reason="BatchTransformInput has initialization issues in the source code")
    @patch("sagemaker.core.s3.S3Uploader.upload_string_as_file_body")
    @patch("sagemaker.core.common_utils.name_from_base")
    def test_create_monitoring_schedule_with_batch_transform(
        self, mock_name, mock_upload, mock_session
    ):
        """Test create_monitoring_schedule with batch transform input"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        mock_name.side_effect = ["test-schedule-name", "test-job-def-name"]
        mock_upload.return_value = "s3://bucket/analysis_config.json"

        bias_config_mock = Mock()
        bias_config_mock.get_config.return_value = {"facet_name": "f1"}

        monitor.latest_baselining_job_name = "baseline-job"
        monitor.latest_baselining_job_config = ClarifyBaseliningConfig(
            analysis_config=BiasAnalysisConfig(
                bias_config=bias_config_mock, headers=["f1", "f2"], label="target"
            ),
            features_attribute="features",
        )

        from sagemaker.core.model_monitor.model_monitoring import BatchTransformInput
        from sagemaker.core.model_monitor.dataset_format import MonitoringDatasetFormat

        batch_input = BatchTransformInput(
            data_captured_destination_s3_uri="s3://bucket/batch-data",
            destination="/opt/ml/processing/input",
            dataset_format=MonitoringDatasetFormat.csv(),
        )

        with patch.object(mock_session.sagemaker_client, "create_model_bias_job_definition"):
            with patch.object(mock_session.sagemaker_client, "create_monitoring_schedule"):
                monitor.create_monitoring_schedule(
                    batch_transform_input=batch_input,
                    ground_truth_input="s3://bucket/ground-truth",
                    output_s3_uri="s3://bucket/output",
                    schedule_cron_expression="cron(0 * * * ? *)",
                )

        assert monitor.monitoring_schedule_name is not None

    @patch("sagemaker.core.s3.S3Uploader.upload_string_as_file_body")
    @patch("sagemaker.core.common_utils.name_from_base")
    def test_create_monitoring_schedule_with_data_analysis_time(
        self, mock_name, mock_upload, mock_session
    ):
        """Test create_monitoring_schedule with data analysis time window"""
        monitor = ModelBiasMonitor(
            role="arn:aws:iam::123456789012:role/SageMakerRole", sagemaker_session=mock_session
        )

        bias_config_mock = Mock()
        bias_config_mock.get_config.return_value = {"facet_name": "f1"}

        monitor.latest_baselining_job_name = "baseline-job"
        monitor.latest_baselining_job_config = ClarifyBaseliningConfig(
            analysis_config=BiasAnalysisConfig(
                bias_config=bias_config_mock, headers=["f1"], label="target"
            )
        )

        mock_name.side_effect = ["test-schedule", "test-job-def"]
        mock_upload.return_value = "s3://bucket/analysis_config.json"

        with patch.object(mock_session.sagemaker_client, "create_model_bias_job_definition"):
            with patch.object(mock_session.sagemaker_client, "create_monitoring_schedule"):
                monitor.create_monitoring_schedule(
                    endpoint_input="test-endpoint",
                    ground_truth_input="s3://bucket/ground-truth",
                    output_s3_uri="s3://bucket/output",
                    data_analysis_start_time="-PT1H",
                    data_analysis_end_time="-PT0H",
                )

                assert monitor.monitoring_schedule_name is not None
