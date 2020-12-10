# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import copy
import time

import pytest
from mock import Mock, MagicMock

from sagemaker.model_monitor import (
    Constraints,
    CronExpressionGenerator,
    DefaultModelMonitor,
    EndpointInput,
    ModelMonitor,
    ModelQualityMonitor,
    MonitoringOutput,
    Statistics,
)

from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.network import NetworkConfig

REGION = "us-west-2"
BUCKET_NAME = "mybucket"

ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.m5.10xlarge"
VOLUME_SIZE_IN_GB = 2
VOLUME_KMS_KEY = "volume-kms-key"
OUTPUT_KMS_KEY = "output-kms-key"
MAX_RUNTIME_IN_SECONDS = 3
BASE_JOB_NAME = "base-job-name"
ENV_KEY_1 = "env_key_1"
ENV_VALUE_1 = "env_key_1"
ENVIRONMENT = {
    ENV_KEY_1: ENV_VALUE_1,
    "publish_cloudwatch_metrics": "Enabled",
}
TAG_KEY_1 = "tag_key_1"
TAG_VALUE_1 = "tag_value_1"
TAGS = [{"Key": TAG_KEY_1, "Value": TAG_VALUE_1}]
NETWORK_CONFIG = NetworkConfig(enable_network_isolation=False)
ENABLE_CLOUDWATCH_METRICS = True
PROBLEM_TYPE = "Regression"
GROUND_TRUTH_ATTRIBUTE = "TestAttribute"


BASELINING_JOB_NAME = "baselining-job"
BASELINE_DATASET_PATH = "/my/local/path/baseline.csv"
PREPROCESSOR_PATH = "/my/local/path/preprocessor.py"
POSTPROCESSOR_PATH = "/my/local/path/postprocessor.py"
OUTPUT_S3_URI = "s3://output-s3-uri/"


CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"
DEFAULT_IMAGE_URI = "159807026194.dkr.ecr.us-west-2.amazonaws.com/sagemaker-model-monitor-analyzer"

INTER_CONTAINER_ENCRYPTION_EXCEPTION_MSG = (
    "EnableInterContainerTrafficEncryption is not supported in Model Monitor. Please ensure that "
)
"encrypt_inter_container_traffic=None when creating your NetworkConfig object."

MONITORING_SCHEDULE_DESC = {
    "MonitoringScheduleArn": "arn:aws:monitoring-schedule",
    "MonitoringScheduleName": "my-monitoring-schedule",
    "MonitoringScheduleConfig": {
        "MonitoringJobDefinition": {
            "MonitoringOutputConfig": {},
            "MonitoringResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.t3.medium",
                    "VolumeSizeInGB": 8,
                }
            },
            "MonitoringAppSpecification": {
                "ImageUri": "image-uri",
                "ContainerEntrypoint": [
                    "entrypoint.py",
                ],
            },
            "RoleArn": ROLE,
        }
    },
    "EndpointName": "my-endpoint",
}

JOB_DEFINITION_NAME = "job-definition"
SCHEDULE_NAME = "schedule"
SCHEDULE_ARN = "arn:aws:sagemaker:us-west-2:012345678901:monitoring-schedule/" + SCHEDULE_NAME
OUTPUT_LOCAL_PATH = "/opt/ml/processing/output"
ENDPOINT_INPUT_LOCAL_PATH = "/opt/ml/processing/input/endpoint"
SCHEDULE_NAME = "schedule"
CRON_HOURLY = CronExpressionGenerator.hourly()
S3_INPUT_MODE = "File"
S3_DATA_DISTRIBUTION_TYPE = "FullyReplicated"
S3_UPLOAD_MODE = "Continuous"
ENDPOINT_NAME = "endpoint"
GROUND_TRUTH_S3_URI = "s3://bucket/monitoring_captured/actuals"
ANALYSIS_CONFIG_S3_URI = "s3://bucket/analysis_config.json"
START_TIME_OFFSET = "-PT1H"
END_TIME_OFFSET = "-PT0H"
CONSTRAINTS = Constraints("", "s3://bucket/constraints.json")
STATISTICS = Statistics("", "s3://bucket/statistics.json")
FEATURES_ATTRIBUTE = "features"
INFERENCE_ATTRIBUTE = "predicted_label"
PROBABILITY_ATTRIBUTE = "probabilities"
PROBABILITY_THRESHOLD_ATTRIBUTE = 0.6
PREPROCESSOR_URI = "s3://my_bucket/preprocessor.py"
POSTPROCESSOR_URI = "s3://my_bucket/postprocessor.py"
JOB_OUTPUT_CONFIG = {
    "MonitoringOutputs": [
        {
            "S3Output": {
                "S3Uri": OUTPUT_S3_URI,
                "LocalPath": OUTPUT_LOCAL_PATH,
                "S3UploadMode": S3_UPLOAD_MODE,
            }
        }
    ],
    "KmsKeyId": OUTPUT_KMS_KEY,
}
JOB_RESOURCES = {
    "ClusterConfig": {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": VOLUME_SIZE_IN_GB,
        "VolumeKmsKeyId": VOLUME_KMS_KEY,
    }
}
STOP_CONDITION = {"MaxRuntimeInSeconds": MAX_RUNTIME_IN_SECONDS}
DATA_QUALITY_JOB_INPUT = {
    "EndpointInput": {
        "EndpointName": ENDPOINT_NAME,
        "LocalPath": ENDPOINT_INPUT_LOCAL_PATH,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
    },
}
DATA_QUALITY_APP_SPECIFICATION = {
    "ImageUri": DEFAULT_IMAGE_URI,
    "Environment": ENVIRONMENT,
    "RecordPreprocessorSourceUri": PREPROCESSOR_URI,
    "PostAnalyticsProcessorSourceUri": POSTPROCESSOR_URI,
}
DATA_QUALITY_BASELINE_CONFIG = {
    "ConstraintsResource": {"S3Uri": CONSTRAINTS.file_s3_uri},
    "StatisticsResource": {"S3Uri": STATISTICS.file_s3_uri},
}
DATA_QUALITY_JOB_DEFINITION = {
    "DataQualityAppSpecification": DATA_QUALITY_APP_SPECIFICATION,
    "DataQualityBaselineConfig": DATA_QUALITY_BASELINE_CONFIG,
    "DataQualityJobInput": DATA_QUALITY_JOB_INPUT,
    "DataQualityJobOutputConfig": JOB_OUTPUT_CONFIG,
    "JobResources": JOB_RESOURCES,
    "RoleArn": ROLE,
    "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": STOP_CONDITION,
}
MODEL_QUALITY_APP_SPECIFICATION = {
    "ImageUri": DEFAULT_IMAGE_URI,
    "ProblemType": PROBLEM_TYPE,
    "Environment": ENVIRONMENT,
    "RecordPreprocessorSourceUri": PREPROCESSOR_URI,
    "PostAnalyticsProcessorSourceUri": POSTPROCESSOR_URI,
}
MODEL_QUALITY_BASELINE_CONFIG = {"BaseliningJobName": BASELINING_JOB_NAME}
MODEL_QUALITY_JOB_INPUT = {
    "EndpointInput": {
        "EndpointName": ENDPOINT_NAME,
        "LocalPath": ENDPOINT_INPUT_LOCAL_PATH,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
        "StartTimeOffset": START_TIME_OFFSET,
        "EndTimeOffset": END_TIME_OFFSET,
        "FeaturesAttribute": FEATURES_ATTRIBUTE,
        "InferenceAttribute": INFERENCE_ATTRIBUTE,
        "ProbabilityAttribute": PROBABILITY_ATTRIBUTE,
        "ProbabilityThresholdAttribute": PROBABILITY_THRESHOLD_ATTRIBUTE,
    },
    "GroundTruthS3Input": {"S3Uri": GROUND_TRUTH_S3_URI},
}
MODEL_QUALITY_JOB_DEFINITION = {
    "ModelQualityAppSpecification": MODEL_QUALITY_APP_SPECIFICATION,
    "ModelQualityJobInput": MODEL_QUALITY_JOB_INPUT,
    "ModelQualityJobOutputConfig": JOB_OUTPUT_CONFIG,
    "JobResources": JOB_RESOURCES,
    "RoleArn": ROLE,
    "ModelQualityBaselineConfig": MODEL_QUALITY_BASELINE_CONFIG,
    "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": STOP_CONDITION,
}

# For update API
NEW_ROLE_ARN = "arn:aws:iam::012345678902:role/{}".format(ROLE)
NEW_INSTANCE_COUNT = 2
NEW_INSTANCE_TYPE = "ml.m4.xlarge"
NEW_VOLUME_SIZE_IN_GB = 20
NEW_VOLUME_KMS_KEY = "new-volume-kms-key"
NEW_OUTPUT_KMS_KEY = "new-output-kms-key"
NEW_MAX_RUNTIME_IN_SECONDS = 60 * 60
NEW_ENVIRONMENT = {
    "new_env_key_1": "new_env_key_1",
    "publish_cloudwatch_metrics": "Disabled",
}
NEW_SECURITY_GROUP_IDS = ["new_test_security_group_ids"]
NEW_SUBNETS = ["new_test_subnets"]
NEW_NETWORK_CONFIG = NetworkConfig(
    enable_network_isolation=False,
    security_group_ids=NEW_SECURITY_GROUP_IDS,
    subnets=NEW_SUBNETS,
)
CRON_DAILY = CronExpressionGenerator.daily()
NEW_ENDPOINT_NAME = "new-endpoint"
NEW_GROUND_TRUTH_S3_URI = "s3://bucket/monitoring_captured/groundtruth"
NEW_START_TIME_OFFSET = "-PT2H"
NEW_END_TIME_OFFSET = "-PT1H"
NEW_OUTPUT_S3_URI = "s3://bucket/new/output"
NEW_CONSTRAINTS = Constraints("", "s3://new_bucket/constraints.json")
NEW_STATISTICS = Statistics("", "s3://new_bucket/statistics.json")
NEW_FEATURES_ATTRIBUTE = "new_features"
NEW_INFERENCE_ATTRIBUTE = "new_predicted_label"
NEW_PROBABILITY_ATTRIBUTE = "new_probabilities"
NEW_PROBABILITY_THRESHOLD_ATTRIBUTE = 0.4
NEW_PROBLEM_TYPE = "BinaryClassification"
NEW_PREPROCESSOR_URI = "s3://my_new_bucket/preprocessor.py"
NEW_POSTPROCESSOR_URI = "s3://my_new_bucket/postprocessor.py"
NEW_JOB_OUTPUT_CONFIG = {
    "MonitoringOutputs": [
        {
            "S3Output": {
                "S3Uri": NEW_OUTPUT_S3_URI,
                "LocalPath": OUTPUT_LOCAL_PATH,
                "S3UploadMode": S3_UPLOAD_MODE,
            }
        }
    ],
    "KmsKeyId": NEW_OUTPUT_KMS_KEY,
}
NEW_JOB_RESOURCES = {
    "ClusterConfig": {
        "InstanceCount": NEW_INSTANCE_COUNT,
        "InstanceType": NEW_INSTANCE_TYPE,
        "VolumeSizeInGB": NEW_VOLUME_SIZE_IN_GB,
        "VolumeKmsKeyId": NEW_VOLUME_KMS_KEY,
    }
}
NEW_STOP_CONDITION = {"MaxRuntimeInSeconds": NEW_MAX_RUNTIME_IN_SECONDS}
NEW_DATA_QUALITY_JOB_INPUT = {
    "EndpointInput": {
        "EndpointName": NEW_ENDPOINT_NAME,
        "LocalPath": ENDPOINT_INPUT_LOCAL_PATH,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
    },
}
NEW_DATA_QUALITY_APP_SPECIFICATION = {
    "ImageUri": DEFAULT_IMAGE_URI,
    "Environment": NEW_ENVIRONMENT,
    "RecordPreprocessorSourceUri": NEW_PREPROCESSOR_URI,
    "PostAnalyticsProcessorSourceUri": NEW_POSTPROCESSOR_URI,
}
NEW_DATA_QUALITY_BASELINE_CONFIG = {
    "ConstraintsResource": {"S3Uri": NEW_CONSTRAINTS.file_s3_uri},
    "StatisticsResource": {"S3Uri": NEW_STATISTICS.file_s3_uri},
}
NEW_DATA_QUALITY_JOB_DEFINITION = {
    "DataQualityAppSpecification": NEW_DATA_QUALITY_APP_SPECIFICATION,
    "DataQualityJobInput": NEW_DATA_QUALITY_JOB_INPUT,
    "DataQualityJobOutputConfig": NEW_JOB_OUTPUT_CONFIG,
    "JobResources": NEW_JOB_RESOURCES,
    "RoleArn": NEW_ROLE_ARN,
    "DataQualityBaselineConfig": NEW_DATA_QUALITY_BASELINE_CONFIG,
    "NetworkConfig": NEW_NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": NEW_STOP_CONDITION,
}
NEW_MODEL_QUALITY_APP_SPECIFICATION = {
    "ImageUri": DEFAULT_IMAGE_URI,
    "ProblemType": NEW_PROBLEM_TYPE,
    "Environment": NEW_ENVIRONMENT,
    "RecordPreprocessorSourceUri": NEW_PREPROCESSOR_URI,
    "PostAnalyticsProcessorSourceUri": NEW_POSTPROCESSOR_URI,
}
NEW_MODEL_QUALITY_BASELINE_CONFIG = {
    "ConstraintsResource": {"S3Uri": NEW_CONSTRAINTS.file_s3_uri},
    "BaseliningJobName": BASELINING_JOB_NAME,
}
NEW_MODEL_QUALITY_JOB_INPUT = {
    "EndpointInput": {
        "EndpointName": NEW_ENDPOINT_NAME,
        "LocalPath": ENDPOINT_INPUT_LOCAL_PATH,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
        "StartTimeOffset": NEW_START_TIME_OFFSET,
        "EndTimeOffset": NEW_END_TIME_OFFSET,
        "FeaturesAttribute": NEW_FEATURES_ATTRIBUTE,
        "InferenceAttribute": NEW_INFERENCE_ATTRIBUTE,
        "ProbabilityAttribute": NEW_PROBABILITY_ATTRIBUTE,
        "ProbabilityThresholdAttribute": NEW_PROBABILITY_THRESHOLD_ATTRIBUTE,
    },
    "GroundTruthS3Input": {"S3Uri": NEW_GROUND_TRUTH_S3_URI},
}
NEW_MODEL_QUALITY_JOB_DEFINITION = {
    "ModelQualityAppSpecification": NEW_MODEL_QUALITY_APP_SPECIFICATION,
    "ModelQualityJobInput": NEW_MODEL_QUALITY_JOB_INPUT,
    "ModelQualityJobOutputConfig": NEW_JOB_OUTPUT_CONFIG,
    "JobResources": NEW_JOB_RESOURCES,
    "RoleArn": NEW_ROLE_ARN,
    "ModelQualityBaselineConfig": NEW_MODEL_QUALITY_BASELINE_CONFIG,
    "NetworkConfig": NEW_NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": NEW_STOP_CONDITION,
}


# TODO-reinvent-2019: Continue to flesh these out.
@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session_mock.upload_data = Mock(
        name="upload_data", return_value="mocked_s3_uri_from_upload_data"
    )
    session_mock.download_data = Mock(name="download_data")
    session_mock.describe_monitoring_schedule = Mock(
        name="describe_monitoring_schedule", return_value=MONITORING_SCHEDULE_DESC
    )
    session_mock.expand_role.return_value = ROLE
    return session_mock


@pytest.fixture()
def model_quality_monitor(sagemaker_session):
    return ModelQualityMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=VOLUME_KMS_KEY,
        output_kms_key=OUTPUT_KMS_KEY,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        base_job_name=BASE_JOB_NAME,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )


@pytest.fixture()
def data_quality_monitor(sagemaker_session):
    return DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=VOLUME_KMS_KEY,
        output_kms_key=OUTPUT_KMS_KEY,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        base_job_name=BASE_JOB_NAME,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )


def test_default_model_monitor_suggest_baseline(sagemaker_session):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=VOLUME_KMS_KEY,
        output_kms_key=OUTPUT_KMS_KEY,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        base_job_name=BASE_JOB_NAME,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    my_default_monitor.suggest_baseline(
        baseline_dataset=BASELINE_DATASET_PATH,
        dataset_format=DatasetFormat.csv(header=False),
        record_preprocessor_script=PREPROCESSOR_PATH,
        post_analytics_processor_script=POSTPROCESSOR_PATH,
        output_s3_uri=OUTPUT_S3_URI,
        wait=False,
        logs=False,
    )

    assert my_default_monitor.role == ROLE
    assert my_default_monitor.instance_count == INSTANCE_COUNT
    assert my_default_monitor.instance_type == INSTANCE_TYPE
    assert my_default_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert my_default_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert my_default_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert my_default_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert my_default_monitor.base_job_name == BASE_JOB_NAME
    assert my_default_monitor.sagemaker_session == sagemaker_session
    assert my_default_monitor.tags == TAGS
    assert my_default_monitor.network_config == NETWORK_CONFIG
    assert my_default_monitor.image_uri == DEFAULT_IMAGE_URI

    assert BASE_JOB_NAME in my_default_monitor.latest_baselining_job_name
    assert my_default_monitor.latest_baselining_job_name != BASE_JOB_NAME

    assert my_default_monitor.env[ENV_KEY_1] == ENV_VALUE_1


def test_default_model_monitor_with_invalid_network_config(sagemaker_session):
    invalid_network_config = NetworkConfig(encrypt_inter_container_traffic=False)
    my_default_monitor = DefaultModelMonitor(
        role=ROLE, sagemaker_session=sagemaker_session, network_config=invalid_network_config
    )
    with pytest.raises(ValueError) as exception:
        my_default_monitor.create_monitoring_schedule(endpoint_input="test_endpoint")
    assert INTER_CONTAINER_ENCRYPTION_EXCEPTION_MSG in str(exception.value)

    with pytest.raises(ValueError) as exception:
        my_default_monitor.update_monitoring_schedule()
    assert INTER_CONTAINER_ENCRYPTION_EXCEPTION_MSG in str(exception.value)


def test_model_monitor_without_network_config(sagemaker_session):
    my_model_monitor = ModelMonitor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        sagemaker_session=sagemaker_session,
    )
    model_monitor_schedule_name = "model-monitoring-without-network-config"
    attached = my_model_monitor.attach(model_monitor_schedule_name, sagemaker_session)
    assert attached.network_config is None


def test_model_monitor_with_invalid_network_config(sagemaker_session):
    invalid_network_config = NetworkConfig(encrypt_inter_container_traffic=False)
    my_model_monitor = ModelMonitor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        sagemaker_session=sagemaker_session,
        network_config=invalid_network_config,
    )
    with pytest.raises(ValueError) as exception:
        my_model_monitor.create_monitoring_schedule(
            endpoint_input="test_endpoint",
            output=MonitoringOutput(
                source="/opt/ml/processing/output", destination="/opt/ml/processing/output"
            ),
        )
    assert INTER_CONTAINER_ENCRYPTION_EXCEPTION_MSG in str(exception.value)

    with pytest.raises(ValueError) as exception:
        my_model_monitor.update_monitoring_schedule()
    assert INTER_CONTAINER_ENCRYPTION_EXCEPTION_MSG in str(exception.value)


def test_data_quality_monitor_suggest_baseline(sagemaker_session, data_quality_monitor):
    data_quality_monitor.suggest_baseline(
        baseline_dataset=BASELINE_DATASET_PATH,
        dataset_format=DatasetFormat.csv(header=False),
        record_preprocessor_script=PREPROCESSOR_PATH,
        post_analytics_processor_script=POSTPROCESSOR_PATH,
        output_s3_uri=OUTPUT_S3_URI,
        job_name=BASELINING_JOB_NAME,
        wait=False,
        logs=False,
    )

    _test_data_quality_monitor_create_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
        constraints=CONSTRAINTS,
        statistics=STATISTICS,
        baseline_job_name=data_quality_monitor.latest_baselining_job_name,
    )

    _test_data_quality_monitor_update_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
    )

    _test_data_quality_monitor_delete_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_data_quality_monitor(data_quality_monitor, sagemaker_session):
    # create schedule
    _test_data_quality_monitor_create_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
        constraints=CONSTRAINTS,
        statistics=STATISTICS,
    )

    # update schedule
    _test_data_quality_monitor_update_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_data_quality_monitor_delete_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_data_quality_monitor_created_by_attach(sagemaker_session):
    # attach and validate
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition = MagicMock()
    data_quality_monitor = _test_data_quality_monitor_created_by_attach(
        sagemaker_session=sagemaker_session,
        model_monitor_cls=DefaultModelMonitor,
        describe_job_definition=sagemaker_session.sagemaker_client.describe_data_quality_job_definition,
    )

    # update schedule
    _test_data_quality_monitor_update_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_data_quality_monitor_delete_schedule(
        data_quality_monitor=data_quality_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_data_quality_monitor_invalid_create(data_quality_monitor, sagemaker_session):
    # invalid: can not create new job definition if one already exists
    data_quality_monitor.job_definition_name = JOB_DEFINITION_NAME
    with pytest.raises(ValueError):
        _test_data_quality_monitor_create_schedule(
            data_quality_monitor=data_quality_monitor,
            sagemaker_session=sagemaker_session,
            constraints=CONSTRAINTS,
            statistics=STATISTICS,
        )

    # invalid: can not create new schedule if one already exists
    data_quality_monitor.job_definition_name = None
    data_quality_monitor.monitoring_schedule_name = SCHEDULE_NAME
    with pytest.raises(ValueError):
        _test_data_quality_monitor_create_schedule(
            data_quality_monitor=data_quality_monitor,
            sagemaker_session=sagemaker_session,
            constraints=CONSTRAINTS,
            statistics=STATISTICS,
        )


def test_data_quality_monitor_creation_failure(data_quality_monitor, sagemaker_session):
    sagemaker_session.sagemaker_client.create_monitoring_schedule = Mock(
        side_effect=Exception("400")
    )
    with pytest.raises(Exception):
        _test_data_quality_monitor_create_schedule(
            data_quality_monitor=data_quality_monitor,
            sagemaker_session=sagemaker_session,
            constraints=CONSTRAINTS,
            statistics=STATISTICS,
        )
    assert data_quality_monitor.job_definition_name is None
    assert data_quality_monitor.monitoring_schedule_name is None
    sagemaker_session.sagemaker_client.create_data_quality_job_definition.assert_called_once()
    sagemaker_session.sagemaker_client.delete_data_quality_job_definition.assert_called_once()


def test_data_quality_monitor_invalid_attach(data_quality_monitor, sagemaker_session):
    # mock
    sagemaker_session.describe_monitoring_schedule = MagicMock(
        return_value={
            "MonitoringScheduleArn": SCHEDULE_ARN,
            "MonitoringScheduleName": SCHEDULE_NAME,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinitionName": JOB_DEFINITION_NAME,
                "MonitoringType": "UnknownType",
            },
        }
    )
    with pytest.raises(TypeError):
        data_quality_monitor.attach(
            monitor_schedule_name=SCHEDULE_NAME, sagemaker_session=sagemaker_session
        )


def test_data_quality_monitor_update_failure(data_quality_monitor, sagemaker_session):
    data_quality_monitor.create_monitoring_schedule(
        endpoint_input=ENDPOINT_NAME,
    )
    old_job_definition_name = data_quality_monitor.job_definition_name
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition = Mock(
        return_value=copy.deepcopy(DATA_QUALITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock(
        side_effect=ConnectionError("400")
    )
    with pytest.raises(ConnectionError):
        data_quality_monitor.update_monitoring_schedule(
            max_runtime_in_seconds=NEW_MAX_RUNTIME_IN_SECONDS,
        )
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_data_quality_job_definition.assert_called_once()
    assert (
        sagemaker_session.sagemaker_client.delete_data_quality_job_definition.call_args[1][
            "JobDefinitionName"
        ]
        != old_job_definition_name
    )

    # no effect
    data_quality_monitor.update_monitoring_schedule()


def test_data_quality_monitor_with_invalid_network_config(sagemaker_session):
    invalid_network_config = NetworkConfig(encrypt_inter_container_traffic=False)
    data_quality_monitor = DefaultModelMonitor(
        role=ROLE,
        sagemaker_session=sagemaker_session,
        network_config=invalid_network_config,
    )
    with pytest.raises(ValueError) as exception:
        data_quality_monitor.create_monitoring_schedule(
            endpoint_input="test_endpoint",
        )
    assert INTER_CONTAINER_ENCRYPTION_EXCEPTION_MSG in str(exception.value)


def _test_data_quality_monitor_create_schedule(
    data_quality_monitor,
    sagemaker_session,
    constraints=None,
    statistics=None,
    baseline_job_name=None,
    endpoint_input=EndpointInput(
        endpoint_name=ENDPOINT_NAME, destination=ENDPOINT_INPUT_LOCAL_PATH
    ),
):
    data_quality_monitor.create_monitoring_schedule(
        endpoint_input=endpoint_input,
        record_preprocessor_script=PREPROCESSOR_URI,
        post_analytics_processor_script=POSTPROCESSOR_URI,
        output_s3_uri=OUTPUT_S3_URI,
        constraints=constraints,
        statistics=statistics,
        monitor_schedule_name=SCHEDULE_NAME,
        schedule_cron_expression=CRON_HOURLY,
    )

    # validation
    expected_arguments = {
        "JobDefinitionName": data_quality_monitor.job_definition_name,
        **copy.deepcopy(DATA_QUALITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    if baseline_job_name:
        baseline_config = expected_arguments.get("DataQualityBaselineConfig", {})
        baseline_config["BaseliningJobName"] = baseline_job_name

    sagemaker_session.sagemaker_client.create_data_quality_job_definition.assert_called_with(
        **expected_arguments
    )

    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_with(
        MonitoringScheduleName=SCHEDULE_NAME,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": data_quality_monitor.job_definition_name,
            "MonitoringType": "DataQuality",
            "ScheduleConfig": {"ScheduleExpression": CRON_HOURLY},
        },
        Tags=TAGS,
    )


def _test_data_quality_monitor_update_schedule(data_quality_monitor, sagemaker_session):
    # update schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition = MagicMock()
    sagemaker_session.sagemaker_client.create_data_quality_job_definition = MagicMock()
    data_quality_monitor.update_monitoring_schedule(schedule_cron_expression=CRON_DAILY)
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=data_quality_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": data_quality_monitor.job_definition_name,
            "MonitoringType": DefaultModelMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.delete_data_quality_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.create_data_quality_job_definition.assert_not_called()

    # update one property of job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition = Mock(
        return_value=copy.deepcopy(DATA_QUALITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_data_quality_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = data_quality_monitor.job_definition_name
    data_quality_monitor.update_monitoring_schedule(role=NEW_ROLE_ARN)
    expected_arguments = {
        "JobDefinitionName": data_quality_monitor.job_definition_name,
        **copy.deepcopy(DATA_QUALITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    assert old_job_definition_name != data_quality_monitor.job_definition_name
    assert data_quality_monitor.role == NEW_ROLE_ARN
    assert data_quality_monitor.instance_count == INSTANCE_COUNT
    assert data_quality_monitor.instance_type == INSTANCE_TYPE
    assert data_quality_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert data_quality_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert data_quality_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert data_quality_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert data_quality_monitor.env == ENVIRONMENT
    assert data_quality_monitor.network_config == NETWORK_CONFIG
    expected_arguments[
        "RoleArn"
    ] = NEW_ROLE_ARN  # all but role arn are from existing job definition
    sagemaker_session.sagemaker_client.create_data_quality_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=data_quality_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": data_quality_monitor.job_definition_name,
            "MonitoringType": DefaultModelMonitor.monitoring_type(),
        },
    )
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_data_quality_job_definition.assert_not_called()

    # update full job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition = Mock(
        return_value=copy.deepcopy(DATA_QUALITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_data_quality_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = data_quality_monitor.job_definition_name
    data_quality_monitor.role = ROLE
    data_quality_monitor.update_monitoring_schedule(
        endpoint_input=NEW_ENDPOINT_NAME,
        record_preprocessor_script=NEW_PREPROCESSOR_URI,
        post_analytics_processor_script=NEW_POSTPROCESSOR_URI,
        output_s3_uri=NEW_OUTPUT_S3_URI,
        constraints=NEW_CONSTRAINTS,
        statistics=NEW_STATISTICS,
        enable_cloudwatch_metrics=False,
        role=NEW_ROLE_ARN,
        instance_count=NEW_INSTANCE_COUNT,
        instance_type=NEW_INSTANCE_TYPE,
        volume_size_in_gb=NEW_VOLUME_SIZE_IN_GB,
        volume_kms_key=NEW_VOLUME_KMS_KEY,
        output_kms_key=NEW_OUTPUT_KMS_KEY,
        max_runtime_in_seconds=NEW_MAX_RUNTIME_IN_SECONDS,
        env=NEW_ENVIRONMENT,
        network_config=NEW_NETWORK_CONFIG,
        schedule_cron_expression=CRON_DAILY,
    )
    assert old_job_definition_name != data_quality_monitor.job_definition_name
    assert data_quality_monitor.role == NEW_ROLE_ARN
    assert data_quality_monitor.instance_count == NEW_INSTANCE_COUNT
    assert data_quality_monitor.instance_type == NEW_INSTANCE_TYPE
    assert data_quality_monitor.volume_size_in_gb == NEW_VOLUME_SIZE_IN_GB
    assert data_quality_monitor.volume_kms_key == NEW_VOLUME_KMS_KEY
    assert data_quality_monitor.output_kms_key == NEW_OUTPUT_KMS_KEY
    assert data_quality_monitor.max_runtime_in_seconds == NEW_MAX_RUNTIME_IN_SECONDS
    assert data_quality_monitor.env == NEW_ENVIRONMENT
    assert data_quality_monitor.network_config == NEW_NETWORK_CONFIG
    expected_arguments = {  # all from new job definition
        "JobDefinitionName": data_quality_monitor.job_definition_name,
        **copy.deepcopy(NEW_DATA_QUALITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    sagemaker_session.sagemaker_client.create_data_quality_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=data_quality_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": data_quality_monitor.job_definition_name,
            "MonitoringType": DefaultModelMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_data_quality_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_data_quality_job_definition.assert_not_called()


def _test_data_quality_monitor_created_by_attach(
    sagemaker_session, model_monitor_cls, describe_job_definition
):
    # mock
    sagemaker_session.describe_monitoring_schedule = MagicMock(
        return_value={
            "MonitoringScheduleArn": SCHEDULE_ARN,
            "MonitoringScheduleName": SCHEDULE_NAME,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinitionName": JOB_DEFINITION_NAME,
                "MonitoringType": model_monitor_cls.monitoring_type(),
            },
        }
    )
    sagemaker_session.list_tags = MagicMock(return_value=TAGS)
    describe_job_definition.return_value = {
        "RoleArn": ROLE,
        "JobResources": JOB_RESOURCES,
        "{}JobOutputConfig".format(model_monitor_cls.monitoring_type()): {
            "KmsKeyId": OUTPUT_KMS_KEY,
        },
        "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
        "StoppingCondition": STOP_CONDITION,
        "{}AppSpecification".format(model_monitor_cls.monitoring_type()): {
            "Environment": ENVIRONMENT
        },
    }

    # attach
    data_quality_monitor = model_monitor_cls.attach(SCHEDULE_NAME, sagemaker_session)

    # validation
    sagemaker_session.describe_monitoring_schedule.assert_called_once_with(
        monitoring_schedule_name=SCHEDULE_NAME
    )
    sagemaker_session.list_tags.assert_called_once_with(resource_arn=SCHEDULE_ARN)
    describe_job_definition.assert_called_once_with(JobDefinitionName=JOB_DEFINITION_NAME)
    assert data_quality_monitor.monitoring_schedule_name == SCHEDULE_NAME
    assert data_quality_monitor.job_definition_name == JOB_DEFINITION_NAME
    assert data_quality_monitor.env == ENVIRONMENT
    assert data_quality_monitor.instance_count == INSTANCE_COUNT
    assert data_quality_monitor.instance_type == INSTANCE_TYPE
    assert data_quality_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert data_quality_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert data_quality_monitor.role == ROLE
    assert data_quality_monitor.tags == TAGS
    assert data_quality_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert data_quality_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert not data_quality_monitor.network_config.enable_network_isolation
    data_quality_monitor.network_config = NETWORK_CONFIG  # Restore the object for validation
    return data_quality_monitor


def _test_data_quality_monitor_delete_schedule(data_quality_monitor, sagemaker_session):
    # delete schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    job_definition_name = data_quality_monitor.job_definition_name
    data_quality_monitor.delete_monitoring_schedule()
    sagemaker_session.delete_monitoring_schedule.assert_called_once_with(
        monitoring_schedule_name=SCHEDULE_NAME
    )
    sagemaker_session.sagemaker_client.delete_data_quality_job_definition.assert_called_once_with(
        JobDefinitionName=job_definition_name
    )


def test_model_quality_monitor_suggest_baseline(sagemaker_session, model_quality_monitor):
    model_quality_monitor.suggest_baseline(
        baseline_dataset=BASELINE_DATASET_PATH,
        dataset_format=DatasetFormat.csv(header=False),
        post_analytics_processor_script=POSTPROCESSOR_PATH,
        problem_type=PROBLEM_TYPE,
        inference_attribute=INFERENCE_ATTRIBUTE,
        probability_attribute=PROBABILITY_ATTRIBUTE,
        ground_truth_attribute=GROUND_TRUTH_ATTRIBUTE,
        probability_threshold_attribute=PROBABILITY_THRESHOLD_ATTRIBUTE,
        output_s3_uri=OUTPUT_S3_URI,
        job_name=BASELINING_JOB_NAME,
        wait=False,
        logs=False,
    )

    _test_model_quality_monitor_create_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
    )

    _test_model_quality_monitor_update_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
    )

    _test_model_quality_monitor_delete_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_quality_monitor(model_quality_monitor, sagemaker_session):
    # create schedule
    _test_model_quality_monitor_create_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
        constraints=CONSTRAINTS,
    )

    # update schedule
    _test_model_quality_monitor_update_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_quality_monitor_delete_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_quality_monitor_created_by_attach(sagemaker_session):
    # attach and validate
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition = MagicMock()
    model_quality_monitor = _test_model_quality_monitor_created_by_attach(
        sagemaker_session=sagemaker_session,
        model_monitor_cls=ModelQualityMonitor,
        describe_job_definition=sagemaker_session.sagemaker_client.describe_model_quality_job_definition,
    )

    # update schedule
    _test_model_quality_monitor_update_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_quality_monitor_delete_schedule(
        model_quality_monitor=model_quality_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_quality_monitor_invalid_create(model_quality_monitor, sagemaker_session):
    # invalid: can not create new job definition if one already exists
    model_quality_monitor.job_definition_name = JOB_DEFINITION_NAME
    with pytest.raises(ValueError):
        _test_model_quality_monitor_create_schedule(
            model_quality_monitor=model_quality_monitor,
            sagemaker_session=sagemaker_session,
            constraints=CONSTRAINTS,
        )

    # invalid: can not create new schedule if one already exists
    model_quality_monitor.job_definition_name = None
    model_quality_monitor.monitoring_schedule_name = SCHEDULE_NAME
    with pytest.raises(ValueError):
        _test_model_quality_monitor_create_schedule(
            model_quality_monitor=model_quality_monitor,
            sagemaker_session=sagemaker_session,
            constraints=CONSTRAINTS,
        )


def test_model_quality_monitor_creation_failure(model_quality_monitor, sagemaker_session):
    sagemaker_session.sagemaker_client.create_monitoring_schedule = Mock(
        side_effect=Exception("400")
    )
    with pytest.raises(Exception):
        _test_model_quality_monitor_create_schedule(
            model_quality_monitor=model_quality_monitor,
            sagemaker_session=sagemaker_session,
            constraints=CONSTRAINTS,
        )
    assert model_quality_monitor.job_definition_name is None
    assert model_quality_monitor.monitoring_schedule_name is None
    sagemaker_session.sagemaker_client.create_model_quality_job_definition.assert_called_once()
    sagemaker_session.sagemaker_client.delete_model_quality_job_definition.assert_called_once()


def test_model_quality_monitor_invalid_attach(model_quality_monitor, sagemaker_session):
    # mock
    sagemaker_session.describe_monitoring_schedule = MagicMock(
        return_value={
            "MonitoringScheduleArn": SCHEDULE_ARN,
            "MonitoringScheduleName": SCHEDULE_NAME,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinitionName": JOB_DEFINITION_NAME,
                "MonitoringType": "UnknownType",
            },
        }
    )
    with pytest.raises(TypeError):
        model_quality_monitor.attach(
            monitor_schedule_name=SCHEDULE_NAME, sagemaker_session=sagemaker_session
        )


def test_model_quality_monitor_update_failure(model_quality_monitor, sagemaker_session):
    model_quality_monitor.create_monitoring_schedule(
        endpoint_input=ENDPOINT_NAME,
        ground_truth_input=GROUND_TRUTH_S3_URI,
        problem_type=PROBLEM_TYPE,
    )
    old_job_definition_name = model_quality_monitor.job_definition_name
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition = Mock(
        return_value=copy.deepcopy(MODEL_QUALITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock(
        side_effect=ConnectionError("400")
    )
    with pytest.raises(ConnectionError):
        model_quality_monitor.update_monitoring_schedule(
            max_runtime_in_seconds=NEW_MAX_RUNTIME_IN_SECONDS,
        )
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_quality_job_definition.assert_called_once()
    assert (
        sagemaker_session.sagemaker_client.delete_model_quality_job_definition.call_args[1][
            "JobDefinitionName"
        ]
        != old_job_definition_name
    )

    # no effect
    model_quality_monitor.update_monitoring_schedule()


def test_model_quality_monitor_with_invalid_network_config(sagemaker_session):
    invalid_network_config = NetworkConfig(encrypt_inter_container_traffic=False)
    model_quality_monitor = ModelQualityMonitor(
        role=ROLE,
        sagemaker_session=sagemaker_session,
        network_config=invalid_network_config,
    )
    with pytest.raises(ValueError) as exception:
        model_quality_monitor.create_monitoring_schedule(
            endpoint_input="test_endpoint",
            problem_type=PROBLEM_TYPE,
            ground_truth_input=GROUND_TRUTH_S3_URI,
        )
    assert INTER_CONTAINER_ENCRYPTION_EXCEPTION_MSG in str(exception.value)


def _test_model_quality_monitor_create_schedule(
    model_quality_monitor,
    sagemaker_session,
    constraints=None,
    baseline_job_name=None,
    endpoint_input=EndpointInput(
        endpoint_name=ENDPOINT_NAME,
        destination=ENDPOINT_INPUT_LOCAL_PATH,
        start_time_offset=START_TIME_OFFSET,
        end_time_offset=END_TIME_OFFSET,
        features_attribute=FEATURES_ATTRIBUTE,
        inference_attribute=INFERENCE_ATTRIBUTE,
        probability_attribute=PROBABILITY_ATTRIBUTE,
        probability_threshold_attribute=PROBABILITY_THRESHOLD_ATTRIBUTE,
    ),
):
    model_quality_monitor.create_monitoring_schedule(
        endpoint_input=endpoint_input,
        ground_truth_input=GROUND_TRUTH_S3_URI,
        problem_type=PROBLEM_TYPE,
        record_preprocessor_script=PREPROCESSOR_URI,
        post_analytics_processor_script=POSTPROCESSOR_URI,
        output_s3_uri=OUTPUT_S3_URI,
        constraints=constraints,
        monitor_schedule_name=SCHEDULE_NAME,
        schedule_cron_expression=CRON_HOURLY,
    )

    # validation
    expected_arguments = {
        "JobDefinitionName": model_quality_monitor.job_definition_name,
        **copy.deepcopy(MODEL_QUALITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    if constraints:
        expected_arguments["ModelQualityBaselineConfig"] = {
            "ConstraintsResource": {"S3Uri": constraints.file_s3_uri}
        }
    if baseline_job_name:
        expected_arguments["ModelQualityBaselineConfig"] = {
            "BaseliningJobName": baseline_job_name,
        }

    sagemaker_session.sagemaker_client.create_model_quality_job_definition.assert_called_with(
        **expected_arguments
    )

    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_with(
        MonitoringScheduleName=SCHEDULE_NAME,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_quality_monitor.job_definition_name,
            "MonitoringType": "ModelQuality",
            "ScheduleConfig": {"ScheduleExpression": CRON_HOURLY},
        },
        Tags=TAGS,
    )


def _test_model_quality_monitor_update_schedule(model_quality_monitor, sagemaker_session):
    # update schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition = MagicMock()
    sagemaker_session.sagemaker_client.create_model_quality_job_definition = MagicMock()
    model_quality_monitor.update_monitoring_schedule(schedule_cron_expression=CRON_DAILY)
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_quality_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_quality_monitor.job_definition_name,
            "MonitoringType": ModelQualityMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.delete_model_quality_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.create_model_quality_job_definition.assert_not_called()

    # update one property of job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition = Mock(
        return_value=copy.deepcopy(MODEL_QUALITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_model_quality_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = model_quality_monitor.job_definition_name
    model_quality_monitor.update_monitoring_schedule(role=NEW_ROLE_ARN)
    expected_arguments = {
        "JobDefinitionName": model_quality_monitor.job_definition_name,
        **copy.deepcopy(MODEL_QUALITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    assert old_job_definition_name != model_quality_monitor.job_definition_name
    assert model_quality_monitor.role == NEW_ROLE_ARN
    assert model_quality_monitor.instance_count == INSTANCE_COUNT
    assert model_quality_monitor.instance_type == INSTANCE_TYPE
    assert model_quality_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert model_quality_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert model_quality_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert model_quality_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert model_quality_monitor.env == ENVIRONMENT
    assert model_quality_monitor.network_config == NETWORK_CONFIG
    expected_arguments[
        "RoleArn"
    ] = NEW_ROLE_ARN  # all but role arn are from existing job definition
    sagemaker_session.sagemaker_client.create_model_quality_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_quality_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_quality_monitor.job_definition_name,
            "MonitoringType": ModelQualityMonitor.monitoring_type(),
        },
    )
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_quality_job_definition.assert_not_called()

    # update full job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition = Mock(
        return_value=copy.deepcopy(MODEL_QUALITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_model_quality_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = model_quality_monitor.job_definition_name
    model_quality_monitor.role = ROLE
    model_quality_monitor.update_monitoring_schedule(
        endpoint_input=EndpointInput(
            endpoint_name=NEW_ENDPOINT_NAME,
            destination=ENDPOINT_INPUT_LOCAL_PATH,
            start_time_offset=NEW_START_TIME_OFFSET,
            end_time_offset=NEW_END_TIME_OFFSET,
            features_attribute=NEW_FEATURES_ATTRIBUTE,
            inference_attribute=NEW_INFERENCE_ATTRIBUTE,
            probability_attribute=NEW_PROBABILITY_ATTRIBUTE,
            probability_threshold_attribute=NEW_PROBABILITY_THRESHOLD_ATTRIBUTE,
        ),
        ground_truth_input=NEW_GROUND_TRUTH_S3_URI,
        problem_type=NEW_PROBLEM_TYPE,
        record_preprocessor_script=NEW_PREPROCESSOR_URI,
        post_analytics_processor_script=NEW_POSTPROCESSOR_URI,
        output_s3_uri=NEW_OUTPUT_S3_URI,
        constraints=NEW_CONSTRAINTS,
        enable_cloudwatch_metrics=False,
        role=NEW_ROLE_ARN,
        instance_count=NEW_INSTANCE_COUNT,
        instance_type=NEW_INSTANCE_TYPE,
        volume_size_in_gb=NEW_VOLUME_SIZE_IN_GB,
        volume_kms_key=NEW_VOLUME_KMS_KEY,
        output_kms_key=NEW_OUTPUT_KMS_KEY,
        max_runtime_in_seconds=NEW_MAX_RUNTIME_IN_SECONDS,
        env=NEW_ENVIRONMENT,
        network_config=NEW_NETWORK_CONFIG,
        schedule_cron_expression=CRON_DAILY,
    )
    assert old_job_definition_name != model_quality_monitor.job_definition_name
    assert model_quality_monitor.role == NEW_ROLE_ARN
    assert model_quality_monitor.instance_count == NEW_INSTANCE_COUNT
    assert model_quality_monitor.instance_type == NEW_INSTANCE_TYPE
    assert model_quality_monitor.volume_size_in_gb == NEW_VOLUME_SIZE_IN_GB
    assert model_quality_monitor.volume_kms_key == NEW_VOLUME_KMS_KEY
    assert model_quality_monitor.output_kms_key == NEW_OUTPUT_KMS_KEY
    assert model_quality_monitor.max_runtime_in_seconds == NEW_MAX_RUNTIME_IN_SECONDS
    assert model_quality_monitor.env == NEW_ENVIRONMENT
    assert model_quality_monitor.network_config == NEW_NETWORK_CONFIG
    expected_arguments = {  # all from new job definition
        "JobDefinitionName": model_quality_monitor.job_definition_name,
        **copy.deepcopy(NEW_MODEL_QUALITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    sagemaker_session.sagemaker_client.create_model_quality_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_quality_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_quality_monitor.job_definition_name,
            "MonitoringType": ModelQualityMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_model_quality_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_quality_job_definition.assert_not_called()


def _test_model_quality_monitor_created_by_attach(
    sagemaker_session, model_monitor_cls, describe_job_definition
):
    # mock
    sagemaker_session.describe_monitoring_schedule = MagicMock(
        return_value={
            "MonitoringScheduleArn": SCHEDULE_ARN,
            "MonitoringScheduleName": SCHEDULE_NAME,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinitionName": JOB_DEFINITION_NAME,
                "MonitoringType": model_monitor_cls.monitoring_type(),
            },
        }
    )
    sagemaker_session.list_tags = MagicMock(return_value=TAGS)
    describe_job_definition.return_value = {
        "RoleArn": ROLE,
        "JobResources": JOB_RESOURCES,
        "{}JobOutputConfig".format(model_monitor_cls.monitoring_type()): {
            "KmsKeyId": OUTPUT_KMS_KEY,
        },
        "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
        "StoppingCondition": STOP_CONDITION,
        "{}AppSpecification".format(model_monitor_cls.monitoring_type()): {
            "Environment": ENVIRONMENT
        },
    }

    # attach
    model_quality_monitor = model_monitor_cls.attach(SCHEDULE_NAME, sagemaker_session)

    # validation
    sagemaker_session.describe_monitoring_schedule.assert_called_once_with(
        monitoring_schedule_name=SCHEDULE_NAME
    )
    sagemaker_session.list_tags.assert_called_once_with(resource_arn=SCHEDULE_ARN)
    describe_job_definition.assert_called_once_with(JobDefinitionName=JOB_DEFINITION_NAME)
    assert model_quality_monitor.monitoring_schedule_name == SCHEDULE_NAME
    assert model_quality_monitor.job_definition_name == JOB_DEFINITION_NAME
    assert model_quality_monitor.env == ENVIRONMENT
    assert model_quality_monitor.instance_count == INSTANCE_COUNT
    assert model_quality_monitor.instance_type == INSTANCE_TYPE
    assert model_quality_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert model_quality_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert model_quality_monitor.role == ROLE
    assert model_quality_monitor.tags == TAGS
    assert model_quality_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert model_quality_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert not model_quality_monitor.network_config.enable_network_isolation
    model_quality_monitor.network_config = NETWORK_CONFIG  # Restore the object for validation
    return model_quality_monitor


def _test_model_quality_monitor_delete_schedule(model_quality_monitor, sagemaker_session):
    # delete schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    job_definition_name = model_quality_monitor.job_definition_name
    model_quality_monitor.delete_monitoring_schedule()
    sagemaker_session.delete_monitoring_schedule.assert_called_once_with(
        monitoring_schedule_name=SCHEDULE_NAME
    )
    sagemaker_session.sagemaker_client.delete_model_quality_job_definition.assert_called_once_with(
        JobDefinitionName=job_definition_name
    )
