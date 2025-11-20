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
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.model_monitor.utils import (
    boto_create_monitoring_schedule,
    boto_update_monitoring_schedule,
    boto_start_monitoring_schedule,
    boto_stop_monitoring_schedule,
    boto_delete_monitoring_schedule,
    boto_describe_monitoring_schedule,
    boto_list_monitoring_executions,
    boto_list_monitoring_schedules,
    boto_update_monitoring_alert,
    boto_list_monitoring_alerts,
    boto_list_monitoring_alert_history,
    MODEL_MONITOR_ONE_TIME_SCHEDULE,
)


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session"""
    session = Mock()
    session.boto_session.region_name = "us-west-2"
    session.boto_region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.sagemaker_config = None
    return session


class TestModelMonitorUtils:
    """Test cases for model monitor utility functions"""

    def test_boto_create_monitoring_schedule_minimal(self, mock_session):
        """Test boto_create_monitoring_schedule with minimal parameters"""
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 * * * ? *)",
            statistics_s3_uri=None,
            constraints_s3_uri=None,
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": []},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test"
        )
        
        assert mock_session.sagemaker_client.create_monitoring_schedule.called
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        assert call_args["MonitoringScheduleName"] == "test-schedule"
        assert call_args["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"] == "cron(0 * * * ? *)"

    def test_boto_create_monitoring_schedule_with_baseline(self, mock_session):
        """Test boto_create_monitoring_schedule with baseline config"""
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 * * * ? *)",
            statistics_s3_uri="s3://bucket/statistics.json",
            constraints_s3_uri="s3://bucket/constraints.json",
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": []},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test"
        )
        
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        assert "BaselineConfig" in call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        baseline = call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["BaselineConfig"]
        assert baseline["StatisticsResource"]["S3Uri"] == "s3://bucket/statistics.json"
        assert baseline["ConstraintsResource"]["S3Uri"] == "s3://bucket/constraints.json"

    def test_boto_create_monitoring_schedule_with_encryption(self, mock_session):
        """Test boto_create_monitoring_schedule with encryption keys"""
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 * * * ? *)",
            statistics_s3_uri=None,
            constraints_s3_uri=None,
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": [], "KmsKeyId": "output-key"},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            volume_kms_key="volume-key",
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test"
        )
        
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        resources = call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["MonitoringResources"]
        assert resources["ClusterConfig"]["VolumeKmsKeyId"] == "volume-key"

    def test_boto_create_monitoring_schedule_with_custom_scripts(self, mock_session):
        """Test boto_create_monitoring_schedule with custom preprocessing scripts"""
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 * * * ? *)",
            statistics_s3_uri=None,
            constraints_s3_uri=None,
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": []},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test",
            record_preprocessor_source_uri="s3://bucket/preprocess.py",
            post_analytics_processor_source_uri="s3://bucket/postprocess.py"
        )
        
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        app_spec = call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["MonitoringAppSpecification"]
        assert app_spec["RecordPreprocessorSourceUri"] == "s3://bucket/preprocess.py"
        assert app_spec["PostAnalyticsProcessorSourceUri"] == "s3://bucket/postprocess.py"

    def test_boto_create_monitoring_schedule_with_entrypoint(self, mock_session):
        """Test boto_create_monitoring_schedule with custom entrypoint and arguments"""
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 * * * ? *)",
            statistics_s3_uri=None,
            constraints_s3_uri=None,
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": []},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test",
            entrypoint=["/bin/bash", "run.sh"],
            arguments=["--arg1", "value1"]
        )
        
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        app_spec = call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["MonitoringAppSpecification"]
        assert app_spec["ContainerEntrypoint"] == ["/bin/bash", "run.sh"]
        assert app_spec["ContainerArguments"] == ["--arg1", "value1"]

    def test_boto_create_monitoring_schedule_with_network_config(self, mock_session):
        """Test boto_create_monitoring_schedule with network configuration"""
        network_config = {
            "EnableNetworkIsolation": True,
            "VpcConfig": {
                "SecurityGroupIds": ["sg-123"],
                "Subnets": ["subnet-123"]
            }
        }
        
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 * * * ? *)",
            statistics_s3_uri=None,
            constraints_s3_uri=None,
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": []},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test",
            network_config=network_config
        )
        
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        assert "NetworkConfig" in call_args["MonitoringScheduleConfig"]["MonitoringJobDefinition"]

    def test_boto_create_monitoring_schedule_with_tags(self, mock_session):
        """Test boto_create_monitoring_schedule with tags"""
        tags = [{"Key": "Environment", "Value": "Test"}]
        
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 * * * ? *)",
            statistics_s3_uri=None,
            constraints_s3_uri=None,
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": []},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test",
            tags=tags
        )
        
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        assert "Tags" in call_args

    def test_boto_create_monitoring_schedule_one_time(self, mock_session):
        """Test boto_create_monitoring_schedule with one-time schedule"""
        boto_create_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="NOW",
            statistics_s3_uri=None,
            constraints_s3_uri=None,
            monitoring_inputs=[{"EndpointInput": {"EndpointName": "test-endpoint"}}],
            monitoring_output_config={"MonitoringOutputs": []},
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_size_in_gb=30,
            image_uri="test-image:latest",
            role_arn="arn:aws:iam::123:role/test",
            data_analysis_start_time="-PT1H",
            data_analysis_end_time="-PT0H"
        )
        
        call_args = mock_session.sagemaker_client.create_monitoring_schedule.call_args[1]
        schedule_config = call_args["MonitoringScheduleConfig"]["ScheduleConfig"]
        assert schedule_config["ScheduleExpression"] == "NOW"
        assert schedule_config["DataAnalysisStartTime"] == "-PT1H"
        assert schedule_config["DataAnalysisEndTime"] == "-PT0H"

    def test_boto_update_monitoring_schedule_minimal(self, mock_session):
        """Test boto_update_monitoring_schedule with minimal parameters"""
        mock_session.sagemaker_client.describe_monitoring_schedule.return_value = {
            "MonitoringScheduleConfig": {
                "ScheduleConfig": {"ScheduleExpression": "cron(0 * * * ? *)"},
                "MonitoringJobDefinition": {
                    "MonitoringInputs": [{"EndpointInput": {"EndpointName": "test-endpoint"}}],
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.xlarge",
                            "VolumeSizeInGB": 30
                        }
                    },
                    "MonitoringAppSpecification": {"ImageUri": "test-image:latest"},
                    "RoleArn": "arn:aws:iam::123:role/test"
                }
            }
        }
        
        boto_update_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        assert mock_session.sagemaker_client.update_monitoring_schedule.called

    def test_boto_update_monitoring_schedule_with_new_schedule(self, mock_session):
        """Test boto_update_monitoring_schedule with new schedule expression"""
        mock_session.sagemaker_client.describe_monitoring_schedule.return_value = {
            "MonitoringScheduleConfig": {
                "ScheduleConfig": {"ScheduleExpression": "cron(0 * * * ? *)"},
                "MonitoringJobDefinition": {
                    "MonitoringInputs": [{"EndpointInput": {"EndpointName": "test-endpoint"}}],
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.xlarge",
                            "VolumeSizeInGB": 30
                        }
                    },
                    "MonitoringAppSpecification": {"ImageUri": "test-image:latest"},
                    "RoleArn": "arn:aws:iam::123:role/test"
                }
            }
        }
        
        boto_update_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            schedule_expression="cron(0 0 * * ? *)"
        )
        
        call_args = mock_session.sagemaker_client.update_monitoring_schedule.call_args[1]
        assert call_args["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"] == "cron(0 0 * * ? *)"

    def test_boto_update_monitoring_schedule_one_time_missing_times(self, mock_session):
        """Test boto_update_monitoring_schedule raises error for one-time schedule without times"""
        mock_session.sagemaker_client.describe_monitoring_schedule.return_value = {
            "MonitoringScheduleConfig": {
                "ScheduleConfig": {"ScheduleExpression": "cron(0 * * * ? *)"},
                "MonitoringJobDefinition": {
                    "MonitoringInputs": [{"EndpointInput": {"EndpointName": "test-endpoint"}}],
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.xlarge",
                            "VolumeSizeInGB": 30
                        }
                    },
                    "MonitoringAppSpecification": {"ImageUri": "test-image:latest"},
                    "RoleArn": "arn:aws:iam::123:role/test"
                }
            }
        }
        
        with pytest.raises(ValueError, match="Both data_analysis_start_time and data_analysis_end_time are required"):
            boto_update_monitoring_schedule(
                sagemaker_session=mock_session,
                monitoring_schedule_name="test-schedule",
                schedule_expression="NOW"
            )

    def test_boto_start_monitoring_schedule(self, mock_session):
        """Test boto_start_monitoring_schedule"""
        boto_start_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        mock_session.sagemaker_client.start_monitoring_schedule.assert_called_once_with(
            MonitoringScheduleName="test-schedule"
        )

    def test_boto_stop_monitoring_schedule(self, mock_session):
        """Test boto_stop_monitoring_schedule"""
        boto_stop_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        mock_session.sagemaker_client.stop_monitoring_schedule.assert_called_once_with(
            MonitoringScheduleName="test-schedule"
        )

    def test_boto_delete_monitoring_schedule(self, mock_session):
        """Test boto_delete_monitoring_schedule"""
        boto_delete_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        mock_session.sagemaker_client.delete_monitoring_schedule.assert_called_once_with(
            MonitoringScheduleName="test-schedule"
        )

    def test_boto_describe_monitoring_schedule(self, mock_session):
        """Test boto_describe_monitoring_schedule"""
        mock_session.sagemaker_client.describe_monitoring_schedule.return_value = {
            "MonitoringScheduleName": "test-schedule"
        }
        
        result = boto_describe_monitoring_schedule(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        assert result["MonitoringScheduleName"] == "test-schedule"

    def test_boto_list_monitoring_executions(self, mock_session):
        """Test boto_list_monitoring_executions"""
        mock_session.sagemaker_client.list_monitoring_executions.return_value = {
            "MonitoringExecutionSummaries": []
        }
        
        result = boto_list_monitoring_executions(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        assert "MonitoringExecutionSummaries" in result
        mock_session.sagemaker_client.list_monitoring_executions.assert_called_once()

    def test_boto_list_monitoring_executions_with_params(self, mock_session):
        """Test boto_list_monitoring_executions with custom parameters"""
        mock_session.sagemaker_client.list_monitoring_executions.return_value = {
            "MonitoringExecutionSummaries": []
        }
        
        result = boto_list_monitoring_executions(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            sort_by="CreationTime",
            sort_order="Ascending",
            max_results=50
        )
        
        call_args = mock_session.sagemaker_client.list_monitoring_executions.call_args[1]
        assert call_args["SortBy"] == "CreationTime"
        assert call_args["SortOrder"] == "Ascending"
        assert call_args["MaxResults"] == 50

    def test_boto_list_monitoring_schedules(self, mock_session):
        """Test boto_list_monitoring_schedules"""
        mock_session.sagemaker_client.list_monitoring_schedules.return_value = {
            "MonitoringScheduleSummaries": []
        }
        
        result = boto_list_monitoring_schedules(
            sagemaker_session=mock_session
        )
        
        assert "MonitoringScheduleSummaries" in result

    def test_boto_list_monitoring_schedules_with_endpoint(self, mock_session):
        """Test boto_list_monitoring_schedules with endpoint filter"""
        mock_session.sagemaker_client.list_monitoring_schedules.return_value = {
            "MonitoringScheduleSummaries": []
        }
        
        result = boto_list_monitoring_schedules(
            sagemaker_session=mock_session,
            endpoint_name="test-endpoint"
        )
        
        call_args = mock_session.sagemaker_client.list_monitoring_schedules.call_args[1]
        assert call_args["EndpointName"] == "test-endpoint"

    def test_boto_update_monitoring_alert(self, mock_session):
        """Test boto_update_monitoring_alert"""
        mock_session.sagemaker_client.update_monitoring_alert.return_value = {
            "MonitoringScheduleArn": "arn:aws:sagemaker:us-west-2:123:monitoring-schedule/test"
        }
        
        result = boto_update_monitoring_alert(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            monitoring_alert_name="test-alert",
            data_points_to_alert=3,
            evaluation_period=5
        )
        
        assert "MonitoringScheduleArn" in result
        mock_session.sagemaker_client.update_monitoring_alert.assert_called_once()

    def test_boto_list_monitoring_alerts(self, mock_session):
        """Test boto_list_monitoring_alerts"""
        mock_session.sagemaker_client.list_monitoring_alerts.return_value = {
            "MonitoringAlertSummaries": []
        }
        
        result = boto_list_monitoring_alerts(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        assert "MonitoringAlertSummaries" in result

    def test_boto_list_monitoring_alerts_with_pagination(self, mock_session):
        """Test boto_list_monitoring_alerts with pagination"""
        mock_session.sagemaker_client.list_monitoring_alerts.return_value = {
            "MonitoringAlertSummaries": [],
            "NextToken": "token123"
        }
        
        result = boto_list_monitoring_alerts(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            next_token="prev_token",
            max_results=20
        )
        
        call_args = mock_session.sagemaker_client.list_monitoring_alerts.call_args[1]
        assert call_args["NextToken"] == "prev_token"
        assert call_args["MaxResults"] == 20

    def test_boto_list_monitoring_alert_history(self, mock_session):
        """Test boto_list_monitoring_alert_history"""
        mock_session.sagemaker_client.list_monitoring_alert_history.return_value = {
            "MonitoringAlertHistory": []
        }
        
        result = boto_list_monitoring_alert_history(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule"
        )
        
        assert "MonitoringAlertHistory" in result

    def test_boto_list_monitoring_alert_history_with_filters(self, mock_session):
        """Test boto_list_monitoring_alert_history with filters"""
        mock_session.sagemaker_client.list_monitoring_alert_history.return_value = {
            "MonitoringAlertHistory": []
        }
        
        result = boto_list_monitoring_alert_history(
            sagemaker_session=mock_session,
            monitoring_schedule_name="test-schedule",
            monitoring_alert_name="test-alert",
            creation_time_before="2024-01-01T00:00:00Z",
            creation_time_after="2023-01-01T00:00:00Z",
            status_equals="InAlert"
        )
        
        call_args = mock_session.sagemaker_client.list_monitoring_alert_history.call_args[1]
        assert call_args["MonitoringAlertName"] == "test-alert"
        assert call_args["CreationTimeBefore"] == "2024-01-01T00:00:00Z"
        assert call_args["CreationTimeAfter"] == "2023-01-01T00:00:00Z"
        assert call_args["StatusEquals"] == "InAlert"
