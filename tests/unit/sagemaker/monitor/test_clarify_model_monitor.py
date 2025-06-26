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

from __future__ import print_function, absolute_import

import copy
import json

# noinspection PyPackageRequirements
import time

from mock import patch, Mock, MagicMock

# noinspection PyPackageRequirements
import pytest

from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    SHAPConfig,
    ModelPredictedLabelConfig,
)
from sagemaker.model_monitor import (
    BiasAnalysisConfig,
    Constraints,
    CronExpressionGenerator,
    EndpointInput,
    BatchTransformInput,
    ExplainabilityAnalysisConfig,
    ModelBiasMonitor,
    ModelExplainabilityMonitor,
    MonitoringExecution,
    NetworkConfig,
)

from sagemaker.model_monitor.clarify_model_monitoring import (
    ClarifyModelMonitor,
    ClarifyBaseliningJob,
    ClarifyMonitoringExecution,
)
from sagemaker.model_monitor.dataset_format import MonitoringDatasetFormat

# shared
CLARIFY_IMAGE_URI = "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0"
ENDPOINT_INPUT_LOCAL_PATH = "/opt/ml/processing/input/endpoint"
OUTPUT_LOCAL_PATH = "/opt/ml/processing/output"
MONITORING_JOB_NAME = "monitoring-job"
JOB_DEFINITION_NAME = "job-definition"
SCHEDULE_NAME = "schedule"
SCHEDULE_ARN = "arn:aws:sagemaker:us-west-2:012345678901:monitoring-schedule/" + SCHEDULE_NAME
PROCESSING_INPUTS = Mock()
PROCESSING_OUTPUT = Mock()
S3_INPUT_MODE = "File"
S3_DATA_DISTRIBUTION_TYPE = "FullyReplicated"
S3_UPLOAD_MODE = "Continuous"
DATASET_FORMAT = MonitoringDatasetFormat.csv(header=False)

# For create API
ROLE = "SageMakerRole"
ROLE_ARN = "arn:aws:iam::012345678901:role/{}".format(ROLE)
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.m5.xlarge"
VOLUME_SIZE_IN_GB = 40
VOLUME_KMS_KEY = "volume-kms-key"
OUTPUT_KMS_KEY = "output-kms-key"
MAX_RUNTIME_IN_SECONDS = 45 * 60
ENVIRONMENT = {
    "env_key_1": "env_key_1",
    "publish_cloudwatch_metrics": "Enabled",
}
TAGS = [{"Key": "tag_key_1", "Value": "tag_value_1"}]
SECURITY_GROUP_IDS = ["test_security_group_ids"]
SUBNETS = ["test_subnets"]
NETWORK_CONFIG = NetworkConfig(
    enable_network_isolation=False,
    encrypt_inter_container_traffic=False,
    security_group_ids=SECURITY_GROUP_IDS,
    subnets=SUBNETS,
)
CRON_HOURLY = CronExpressionGenerator.hourly()
CRON_NOW = CronExpressionGenerator.now()
ENDPOINT_NAME = "endpoint"
GROUND_TRUTH_S3_URI = "s3://bucket/monitoring_captured/actuals"
ANALYSIS_CONFIG_S3_URI = "s3://bucket/analysis_config.json"
START_TIME_OFFSET = "-PT1H"
END_TIME_OFFSET = "-PT0H"
DATA_CAPTURED_S3_URI = "s3://my-bucket/batch-fraud-detection/on-schedule-monitoring/in/"
SCHEDULE_DESTINATION = "/opt/ml/processing/data"
OUTPUT_S3_URI = "s3://bucket/output"
CONSTRAINTS = Constraints("", "s3://bucket/analysis.json")
FEATURES_ATTRIBUTE = "features"
INFERENCE_ATTRIBUTE = 0
PROBABILITY_ATTRIBUTE = 1
PROBABILITY_THRESHOLD_ATTRIBUTE = 0.6
APP_SPECIFICATION = {
    "ConfigUri": ANALYSIS_CONFIG_S3_URI,
    "ImageUri": CLARIFY_IMAGE_URI,
    "Environment": ENVIRONMENT,
}
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
BASELINE_CONFIG = {"ConstraintsResource": {"S3Uri": CONSTRAINTS.file_s3_uri}}
BIAS_JOB_INPUT = {
    "EndpointInput": {
        "EndpointName": ENDPOINT_NAME,
        "LocalPath": ENDPOINT_INPUT_LOCAL_PATH,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
        "StartTimeOffset": START_TIME_OFFSET,
        "EndTimeOffset": END_TIME_OFFSET,
        "FeaturesAttribute": FEATURES_ATTRIBUTE,
        "InferenceAttribute": str(INFERENCE_ATTRIBUTE),
        "ProbabilityAttribute": str(PROBABILITY_ATTRIBUTE),
        "ProbabilityThresholdAttribute": PROBABILITY_THRESHOLD_ATTRIBUTE,
    },
    "GroundTruthS3Input": {"S3Uri": GROUND_TRUTH_S3_URI},
}
BIAS_BATCH_TRANSFORM_JOB_INPUT = {
    "BatchTransformInput": {
        "DataCapturedDestinationS3Uri": DATA_CAPTURED_S3_URI,
        "LocalPath": SCHEDULE_DESTINATION,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
        "StartTimeOffset": START_TIME_OFFSET,
        "EndTimeOffset": END_TIME_OFFSET,
        "FeaturesAttribute": FEATURES_ATTRIBUTE,
        "InferenceAttribute": str(INFERENCE_ATTRIBUTE),
        "ProbabilityAttribute": str(PROBABILITY_ATTRIBUTE),
        "ProbabilityThresholdAttribute": PROBABILITY_THRESHOLD_ATTRIBUTE,
        "DatasetFormat": DATASET_FORMAT,
    },
    "GroundTruthS3Input": {"S3Uri": GROUND_TRUTH_S3_URI},
}
STOP_CONDITION = {"MaxRuntimeInSeconds": MAX_RUNTIME_IN_SECONDS}
BIAS_JOB_DEFINITION = {
    "ModelBiasAppSpecification": APP_SPECIFICATION,
    "ModelBiasJobInput": BIAS_JOB_INPUT,
    "ModelBiasJobOutputConfig": JOB_OUTPUT_CONFIG,
    "JobResources": JOB_RESOURCES,
    "RoleArn": ROLE_ARN,
    "ModelBiasBaselineConfig": BASELINE_CONFIG,
    "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": STOP_CONDITION,
}
BIAS_BATCH_TRANSFORM_JOB_DEFINITION = {
    "ModelBiasAppSpecification": APP_SPECIFICATION,
    "ModelBiasJobInput": BIAS_BATCH_TRANSFORM_JOB_INPUT,
    "ModelBiasJobOutputConfig": JOB_OUTPUT_CONFIG,
    "JobResources": JOB_RESOURCES,
    "RoleArn": ROLE_ARN,
    "ModelBiasBaselineConfig": BASELINE_CONFIG,
    "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": STOP_CONDITION,
}

EXPLAINABILITY_JOB_INPUT = {
    "EndpointInput": {
        "EndpointName": ENDPOINT_NAME,
        "LocalPath": ENDPOINT_INPUT_LOCAL_PATH,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
        "FeaturesAttribute": FEATURES_ATTRIBUTE,
        "InferenceAttribute": str(INFERENCE_ATTRIBUTE),
    }
}
EXPLAINABILITY_BATCH_TRANSFORM_JOB_INPUT = {
    "BatchTransformInput": {
        "DataCapturedDestinationS3Uri": DATA_CAPTURED_S3_URI,
        "LocalPath": SCHEDULE_DESTINATION,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
        "FeaturesAttribute": FEATURES_ATTRIBUTE,
        "InferenceAttribute": str(INFERENCE_ATTRIBUTE),
        "DatasetFormat": DATASET_FORMAT,
    }
}
EXPLAINABILITY_JOB_DEFINITION = {
    "ModelExplainabilityAppSpecification": APP_SPECIFICATION,
    "ModelExplainabilityJobInput": EXPLAINABILITY_JOB_INPUT,
    "ModelExplainabilityJobOutputConfig": JOB_OUTPUT_CONFIG,
    "JobResources": JOB_RESOURCES,
    "StoppingCondition": STOP_CONDITION,
    "RoleArn": ROLE_ARN,
    "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
}
EXPLAINABILITY__BATCH_TRANSFORM_JOB_DEFINITION = {
    "ModelExplainabilityAppSpecification": APP_SPECIFICATION,
    "ModelExplainabilityJobInput": EXPLAINABILITY_BATCH_TRANSFORM_JOB_INPUT,
    "ModelExplainabilityJobOutputConfig": JOB_OUTPUT_CONFIG,
    "JobResources": JOB_RESOURCES,
    "StoppingCondition": STOP_CONDITION,
    "RoleArn": ROLE_ARN,
    "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
}

MONITORING_EXECUTIONS_EMPTY = {
    "MonitoringExecutionSummaries": [],
}

MONITORING_EXECUTIONS_NO_PROCESSING_JOB = {
    "MonitoringExecutionSummaries": [{"MonitoringSchedule": "MonitoringSchedule"}],
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
NEW_ANALYSIS_CONFIG_S3_URI = "s3://bucket/new/analysis_config.json"
NEW_START_TIME_OFFSET = "-PT2H"
NEW_END_TIME_OFFSET = "-PT1H"
NEW_OUTPUT_S3_URI = "s3://bucket/new/output"
NEW_CONSTRAINTS = Constraints("", "s3://bucket/new/analysis.json")
NEW_FEATURES_ATTRIBUTE = "new_features"
NEW_INFERENCE_ATTRIBUTE = "new_predicted_label"
NEW_PROBABILITY_ATTRIBUTE = "new_probabilities"
NEW_PROBABILITY_THRESHOLD_ATTRIBUTE = 0.4
NEW_APP_SPECIFICATION = {
    "ConfigUri": NEW_ANALYSIS_CONFIG_S3_URI,
    "ImageUri": CLARIFY_IMAGE_URI,
    "Environment": NEW_ENVIRONMENT,
}
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
NEW_BASELINE_CONFIG = {"ConstraintsResource": {"S3Uri": NEW_CONSTRAINTS.file_s3_uri}}
NEW_BIAS_JOB_INPUT = {
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
NEW_STOP_CONDITION = {"MaxRuntimeInSeconds": NEW_MAX_RUNTIME_IN_SECONDS}
NEW_BIAS_JOB_DEFINITION = {
    "ModelBiasAppSpecification": NEW_APP_SPECIFICATION,
    "ModelBiasJobInput": NEW_BIAS_JOB_INPUT,
    "ModelBiasJobOutputConfig": NEW_JOB_OUTPUT_CONFIG,
    "JobResources": NEW_JOB_RESOURCES,
    "RoleArn": NEW_ROLE_ARN,
    "ModelBiasBaselineConfig": NEW_BASELINE_CONFIG,
    "NetworkConfig": NEW_NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": NEW_STOP_CONDITION,
}
NEW_EXPLAINABILITY_JOB_INPUT = {
    "EndpointInput": {
        "EndpointName": NEW_ENDPOINT_NAME,
        "LocalPath": ENDPOINT_INPUT_LOCAL_PATH,
        "S3InputMode": S3_INPUT_MODE,
        "S3DataDistributionType": S3_DATA_DISTRIBUTION_TYPE,
        "FeaturesAttribute": NEW_FEATURES_ATTRIBUTE,
        "InferenceAttribute": NEW_INFERENCE_ATTRIBUTE,
    }
}
NEW_EXPLAINABILITY_JOB_DEFINITION = {
    "ModelExplainabilityAppSpecification": NEW_APP_SPECIFICATION,
    "ModelExplainabilityJobInput": NEW_EXPLAINABILITY_JOB_INPUT,
    "ModelExplainabilityJobOutputConfig": NEW_JOB_OUTPUT_CONFIG,
    "JobResources": NEW_JOB_RESOURCES,
    "RoleArn": NEW_ROLE_ARN,
    "ModelExplainabilityBaselineConfig": NEW_BASELINE_CONFIG,
    "NetworkConfig": NEW_NETWORK_CONFIG._to_request_dict(),
    "StoppingCondition": NEW_STOP_CONDITION,
}

# for baselining
BASELINING_JOB_NAME = "baselining-job"
BASELINING_DATASET_S3_URI = "s3://bucket/dataset"

# for bias
ANALYSIS_CONFIG_LABEL = "Label"
ANALYSIS_CONFIG_HEADERS_OF_FEATURES = ["F1", "F2", "F3"]
ANALYSIS_CONFIG_LABEL_HEADERS = ["Decision"]
ANALYSIS_CONFIG_ALL_HEADERS = [*ANALYSIS_CONFIG_HEADERS_OF_FEATURES, ANALYSIS_CONFIG_LABEL]
ANALYSIS_CONFIG_LABEL_VALUES = [1]
ANALYSIS_CONFIG_FACET_NAME = "F1"
ANALYSIS_CONFIG_FACET_VALUE = [0.3]
ANALYSIS_CONFIG_GROUP_VARIABLE = "F2"
BIAS_ANALYSIS_CONFIG = {
    "label_values_or_threshold": ANALYSIS_CONFIG_LABEL_VALUES,
    "facet": [
        {
            "name_or_index": ANALYSIS_CONFIG_FACET_NAME,
            "value_or_threshold": ANALYSIS_CONFIG_FACET_VALUE,
        }
    ],
    "group_variable": ANALYSIS_CONFIG_GROUP_VARIABLE,
    "headers": ANALYSIS_CONFIG_ALL_HEADERS,
    "label": ANALYSIS_CONFIG_LABEL,
}

# for explainability
SHAP_BASELINE = [
    [
        0.26124998927116394,
        0.2824999988079071,
        0.06875000149011612,
    ]
]
SHAP_NUM_SAMPLES = 100
SHAP_AGG_METHOD = "mean_sq"
SHAP_USE_LOGIT = True
MODEL_NAME = "xgboost-model"
ACCEPT_TYPE = "text/csv"
CONTENT_TYPE = "application/jsonlines"
JSONLINES_CONTENT_TEMPLATE = '{"instances":$features}'
EXPLAINABILITY_ANALYSIS_CONFIG = {
    "headers": ANALYSIS_CONFIG_HEADERS_OF_FEATURES,
    "methods": {
        "shap": {
            "baseline": SHAP_BASELINE,
            "num_samples": SHAP_NUM_SAMPLES,
            "agg_method": SHAP_AGG_METHOD,
            "use_logit": SHAP_USE_LOGIT,
            "save_local_shap_values": True,
        },
    },
    "predictor": {
        "model_name": MODEL_NAME,
        "instance_type": INSTANCE_TYPE,
        "initial_instance_count": INSTANCE_COUNT,
        "accept_type": ACCEPT_TYPE,
        "content_type": CONTENT_TYPE,
        "content_template": JSONLINES_CONTENT_TEMPLATE,
    },
}
EXPLAINABILITY_ANALYSIS_CONFIG_WITH_LABEL_HEADERS = copy.deepcopy(EXPLAINABILITY_ANALYSIS_CONFIG)
# noinspection PyTypeChecker
EXPLAINABILITY_ANALYSIS_CONFIG_WITH_LABEL_HEADERS["predictor"][
    "label_headers"
] = ANALYSIS_CONFIG_LABEL_HEADERS


@pytest.fixture()
def sagemaker_client():
    return MagicMock()


@pytest.fixture()
def sagemaker_session(sagemaker_client):
    boto_mock = Mock(name="boto_session", region_name="us-west-2")
    session_mock = MagicMock(
        name="sagemaker_session",
        sagemaker_client=sagemaker_client,
        boto_session=boto_mock,
        boto_region_name="us-west-2",
        config=None,
        local_mode=False,
        default_bucket_prefix=None,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value="mybucket")
    session_mock.upload_data = Mock(
        name="upload_data", return_value="mocked_s3_uri_from_upload_data"
    )
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = ROLE_ARN
    session_mock._append_sagemaker_config_tags = Mock(
        name="_append_sagemaker_config_tags", side_effect=lambda tags, config_path_to_tags: tags
    )
    # For tests which doesn't verify config file injection, operate with empty config
    session_mock.sagemaker_config = {}
    return session_mock


@pytest.fixture()
def model_bias_monitor(sagemaker_session):
    return ModelBiasMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=VOLUME_KMS_KEY,
        output_kms_key=OUTPUT_KMS_KEY,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )


@pytest.fixture()
def model_explainability_monitor(sagemaker_session):
    return ModelExplainabilityMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=VOLUME_KMS_KEY,
        output_kms_key=OUTPUT_KMS_KEY,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )


@pytest.fixture()
def clarify_model_monitors(model_bias_monitor, model_explainability_monitor):
    return [model_bias_monitor, model_explainability_monitor]


@pytest.fixture(scope="module")
def data_config():
    return DataConfig(
        s3_data_input_path=BASELINING_DATASET_S3_URI,
        s3_output_path=OUTPUT_S3_URI,
        label=ANALYSIS_CONFIG_LABEL,
        headers=ANALYSIS_CONFIG_ALL_HEADERS,
        features=FEATURES_ATTRIBUTE,
    )


@pytest.fixture(scope="module")
def bias_config():
    return BiasConfig(
        label_values_or_threshold=ANALYSIS_CONFIG_LABEL_VALUES,
        facet_name=ANALYSIS_CONFIG_FACET_NAME,
        facet_values_or_threshold=ANALYSIS_CONFIG_FACET_VALUE,
        group_name=ANALYSIS_CONFIG_GROUP_VARIABLE,
    )


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(
        model_name=MODEL_NAME,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        content_type=CONTENT_TYPE,
        accept_type=ACCEPT_TYPE,
        content_template=JSONLINES_CONTENT_TEMPLATE,
    )


@pytest.fixture(scope="module")
def model_predicted_label_config():
    return ModelPredictedLabelConfig(
        label=INFERENCE_ATTRIBUTE,
        probability=PROBABILITY_ATTRIBUTE,
        probability_threshold=PROBABILITY_THRESHOLD_ATTRIBUTE,
    )


@pytest.fixture(scope="module")
def shap_config():
    return SHAPConfig(
        baseline=SHAP_BASELINE,
        num_samples=SHAP_NUM_SAMPLES,
        agg_method=SHAP_AGG_METHOD,
        use_logit=SHAP_USE_LOGIT,
    )


def test_clarify_baselining_job():
    processing_job = MagicMock()
    baselining_job = ClarifyBaseliningJob(processing_job=processing_job)

    with pytest.raises(NotImplementedError):
        baselining_job.baseline_statistics()

    with patch(
        "sagemaker.model_monitor.BaseliningJob.suggested_constraints"
    ) as suggested_constraints:
        baselining_job.suggested_constraints(kms_key=VOLUME_KMS_KEY)
        suggested_constraints.assert_called_with("analysis.json", VOLUME_KMS_KEY)


def test_clarify_monitoring_execution(sagemaker_session):
    execution = ClarifyMonitoringExecution(
        sagemaker_session=sagemaker_session,
        job_name=MONITORING_JOB_NAME,
        inputs=PROCESSING_INPUTS,
        output=PROCESSING_OUTPUT,
        output_kms_key=OUTPUT_KMS_KEY,
    )
    with pytest.raises(NotImplementedError):
        execution.statistics()


def test_clarify_model_monitor():
    # The base class is not supposed to be instantiated
    with pytest.raises(TypeError):
        ClarifyModelMonitor(
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            volume_size_in_gb=VOLUME_SIZE_IN_GB,
            volume_kms_key=VOLUME_KMS_KEY,
            output_kms_key=OUTPUT_KMS_KEY,
            max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
            sagemaker_session=sagemaker_session,
            env=ENVIRONMENT,
            tags=TAGS,
            network_config=NETWORK_CONFIG,
        )

    # The subclass should has monitoring_type() defined
    # noinspection PyAbstractClass
    class DummyClarifyModelMonitor(ClarifyModelMonitor):
        _TEST_CLASS = True
        pass

    with pytest.raises(TypeError):
        DummyClarifyModelMonitor.monitoring_type()


def test_clarify_model_monitor_invalid_update(clarify_model_monitors):
    # invalid: no schedule yet
    for clarify_model_monitor in clarify_model_monitors:
        with pytest.raises(ValueError):
            clarify_model_monitor.update_monitoring_schedule(schedule_cron_expression=CRON_DAILY)


def test_clarify_model_monitor_invalid_attach(sagemaker_session):
    # mock
    sagemaker_session.describe_monitoring_schedule = MagicMock(
        return_value={
            "MonitoringScheduleConfig": {
                "MonitoringType": "NotModelBias",
            },
        }
    )
    # attach, invalid monitoring type
    for clarify_model_monitor_cls in ClarifyModelMonitor.__subclasses__():
        if hasattr(clarify_model_monitor_cls, "_TEST_CLASS"):
            continue
        with pytest.raises(TypeError):
            clarify_model_monitor_cls.attach(SCHEDULE_NAME, sagemaker_session)


def test_clarify_model_monitor_unsupported_methods(clarify_model_monitors, sagemaker_session):
    for clarify_model_monitor in clarify_model_monitors:
        with pytest.raises(NotImplementedError):
            clarify_model_monitor.run_baseline()

        with pytest.raises(NotImplementedError):
            clarify_model_monitor.latest_monitoring_statistics()

        executions = [
            MonitoringExecution(
                sagemaker_session=sagemaker_session,
                job_name=MONITORING_JOB_NAME,
                inputs=PROCESSING_INPUTS,
                output=PROCESSING_OUTPUT,
                output_kms_key=OUTPUT_KMS_KEY,
            )
        ]
        with patch("sagemaker.model_monitor.ModelMonitor.list_executions", return_value=executions):
            with pytest.raises(NotImplementedError):
                clarify_model_monitor.latest_monitoring_statistics()


def test_clarify_model_monitor_list_executions(clarify_model_monitors):
    for clarify_model_monitor in clarify_model_monitors:
        # list executions
        executions = [
            MonitoringExecution(
                sagemaker_session=sagemaker_session,
                job_name=MONITORING_JOB_NAME,
                inputs=PROCESSING_INPUTS,
                output=PROCESSING_OUTPUT,
                output_kms_key=OUTPUT_KMS_KEY,
            )
        ]
        with patch(
            "sagemaker.model_monitor.ModelMonitor.list_executions", return_value=executions
        ) as list_executions:
            clarify_executions = clarify_model_monitor.list_executions()
            list_executions.assert_called_once()
            assert len(clarify_executions) == len(executions) == 1
            clarify_execution = clarify_executions[0]
            execution = executions[0]
            assert isinstance(clarify_execution, ClarifyMonitoringExecution)
            assert clarify_execution.sagemaker_session == execution.sagemaker_session
            assert clarify_execution.job_name == execution.job_name
            assert clarify_execution.inputs == execution.inputs
            assert clarify_execution.output == execution.output
            assert clarify_execution.output_kms_key == execution.output_kms_key


def _test_clarify_model_monitor_created_by_attach(
    sagemaker_session, clarify_model_monitor_cls, describe_job_definition
):
    # mock
    sagemaker_session.describe_monitoring_schedule = MagicMock(
        return_value={
            "MonitoringScheduleArn": SCHEDULE_ARN,
            "MonitoringScheduleName": SCHEDULE_NAME,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinitionName": JOB_DEFINITION_NAME,
                "MonitoringType": clarify_model_monitor_cls.monitoring_type(),
            },
        }
    )
    sagemaker_session.list_tags = MagicMock(return_value=TAGS)
    describe_job_definition.return_value = {
        "RoleArn": ROLE_ARN,
        "JobResources": JOB_RESOURCES,
        "{}JobOutputConfig".format(clarify_model_monitor_cls.monitoring_type()): {
            "KmsKeyId": OUTPUT_KMS_KEY,
        },
        "NetworkConfig": NETWORK_CONFIG._to_request_dict(),
        "StoppingCondition": STOP_CONDITION,
        "{}AppSpecification".format(clarify_model_monitor_cls.monitoring_type()): {
            "Environment": ENVIRONMENT
        },
    }

    # attach
    clarify_model_monitor = clarify_model_monitor_cls.attach(SCHEDULE_NAME, sagemaker_session)

    # validation
    sagemaker_session.describe_monitoring_schedule.assert_called_once_with(
        monitoring_schedule_name=SCHEDULE_NAME
    )
    sagemaker_session.list_tags.assert_called_once_with(resource_arn=SCHEDULE_ARN)
    describe_job_definition.assert_called_once_with(JobDefinitionName=JOB_DEFINITION_NAME)
    assert clarify_model_monitor.monitoring_schedule_name == SCHEDULE_NAME
    assert clarify_model_monitor.job_definition_name == JOB_DEFINITION_NAME
    assert clarify_model_monitor.env == ENVIRONMENT
    assert clarify_model_monitor.image_uri == CLARIFY_IMAGE_URI
    assert clarify_model_monitor.instance_count == INSTANCE_COUNT
    assert clarify_model_monitor.instance_type == INSTANCE_TYPE
    assert clarify_model_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert clarify_model_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert clarify_model_monitor.role == ROLE_ARN
    assert clarify_model_monitor.tags == TAGS
    assert clarify_model_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert clarify_model_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert not clarify_model_monitor.network_config.enable_network_isolation
    clarify_model_monitor.network_config = NETWORK_CONFIG  # Restore the object for validation
    return clarify_model_monitor


def test_bias_analysis_config(bias_config):
    config = BiasAnalysisConfig(
        bias_config=bias_config,
        headers=ANALYSIS_CONFIG_ALL_HEADERS,
        label=ANALYSIS_CONFIG_LABEL,
    )
    assert BIAS_ANALYSIS_CONFIG == config._to_dict()


def test_model_bias_monitor_suggest_baseline(
    model_bias_monitor,
    sagemaker_session,
    data_config,
    bias_config,
    model_config,
    model_predicted_label_config,
):
    # suggest baseline
    model_bias_monitor.suggest_baseline(
        data_config=data_config,
        bias_config=bias_config,
        model_config=model_config,
        model_predicted_label_config=model_predicted_label_config,
        job_name=BASELINING_JOB_NAME,
    )
    assert isinstance(model_bias_monitor.latest_baselining_job, ClarifyBaseliningJob)
    assert (
        BIAS_ANALYSIS_CONFIG
        == model_bias_monitor.latest_baselining_job_config.analysis_config._to_dict()
    )
    clarify_baselining_job = model_bias_monitor.latest_baselining_job
    assert data_config.s3_data_input_path == clarify_baselining_job.inputs[0].source
    assert data_config.s3_output_path == clarify_baselining_job.outputs[0].destination

    # create schedule
    _test_model_bias_monitor_create_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=None,  # will pick up config from baselining job
        baseline_job_name=BASELINING_JOB_NAME,
        endpoint_input=EndpointInput(
            endpoint_name=ENDPOINT_NAME,
            destination=ENDPOINT_INPUT_LOCAL_PATH,
            start_time_offset=START_TIME_OFFSET,
            end_time_offset=END_TIME_OFFSET,
            #  will pick up attributes from baselining job
        ),
    )

    # update schedule
    _test_model_bias_monitor_update_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_bias_monitor_delete_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_bias_monitor(model_bias_monitor, sagemaker_session):
    # create schedule
    _test_model_bias_monitor_create_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=ANALYSIS_CONFIG_S3_URI,
        constraints=CONSTRAINTS,
    )

    # update schedule
    _test_model_bias_monitor_update_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_bias_monitor_delete_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_batch_transform_bias_monitor(model_bias_monitor, sagemaker_session):
    # create schedule
    _test_model_bias_monitor_batch_transform_create_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=ANALYSIS_CONFIG_S3_URI,
        constraints=CONSTRAINTS,
    )

    # update schedule
    _test_model_bias_monitor_update_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_bias_monitor_delete_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_bias_monitor_created_with_config(model_bias_monitor, sagemaker_session, bias_config):
    # create schedule
    analysis_config = BiasAnalysisConfig(
        bias_config=bias_config, headers=ANALYSIS_CONFIG_ALL_HEADERS, label=ANALYSIS_CONFIG_LABEL
    )
    _test_model_bias_monitor_create_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=analysis_config,
        constraints=CONSTRAINTS,
    )

    # update schedule
    _test_model_bias_monitor_update_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_bias_monitor_delete_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_bias_monitor_created_by_attach(sagemaker_session):
    # attach and validate
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition = MagicMock()
    model_bias_monitor = _test_clarify_model_monitor_created_by_attach(
        sagemaker_session=sagemaker_session,
        clarify_model_monitor_cls=ModelBiasMonitor,
        describe_job_definition=sagemaker_session.sagemaker_client.describe_model_bias_job_definition,
    )

    # update schedule
    _test_model_bias_monitor_update_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_bias_monitor_delete_schedule(
        model_bias_monitor=model_bias_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_bias_monitor_invalid_create(model_bias_monitor, sagemaker_session):
    # invalid: analysis config is missing
    with pytest.raises(ValueError):
        _test_model_bias_monitor_create_schedule(
            model_bias_monitor=model_bias_monitor,
            sagemaker_session=sagemaker_session,
        )

    # invalid: can not create new job definition if one already exists
    model_bias_monitor.job_definition_name = JOB_DEFINITION_NAME
    with pytest.raises(ValueError):
        _test_model_bias_monitor_create_schedule(
            model_bias_monitor=model_bias_monitor,
            sagemaker_session=sagemaker_session,
            analysis_config=ANALYSIS_CONFIG_S3_URI,
            constraints=CONSTRAINTS,
        )

    # invalid: can not create new schedule if one already exists
    model_bias_monitor.job_definition_name = None
    model_bias_monitor.monitoring_schedule_name = SCHEDULE_NAME
    with pytest.raises(ValueError):
        _test_model_bias_monitor_create_schedule(
            model_bias_monitor=model_bias_monitor,
            sagemaker_session=sagemaker_session,
            analysis_config=ANALYSIS_CONFIG_S3_URI,
            constraints=CONSTRAINTS,
        )


def test_model_bias_monitor_creation_failure(model_bias_monitor, sagemaker_session):
    sagemaker_session.sagemaker_client.create_monitoring_schedule = Mock(
        side_effect=Exception("400")
    )
    with pytest.raises(Exception):
        _test_model_bias_monitor_create_schedule(
            model_bias_monitor=model_bias_monitor,
            sagemaker_session=sagemaker_session,
            analysis_config=ANALYSIS_CONFIG_S3_URI,
            constraints=CONSTRAINTS,
        )
    assert model_bias_monitor.job_definition_name is None
    assert model_bias_monitor.monitoring_schedule_name is None
    sagemaker_session.sagemaker_client.create_model_bias_job_definition.assert_called_once()
    sagemaker_session.sagemaker_client.delete_model_bias_job_definition.assert_called_once()


def test_model_bias_monitor_update_failure(model_bias_monitor, sagemaker_session):
    model_bias_monitor.create_monitoring_schedule(
        endpoint_input=ENDPOINT_NAME,
        ground_truth_input=GROUND_TRUTH_S3_URI,
        analysis_config=ANALYSIS_CONFIG_S3_URI,
    )
    old_job_definition_name = model_bias_monitor.job_definition_name
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition = Mock(
        return_value=copy.deepcopy(BIAS_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock(
        side_effect=ConnectionError("400")
    )
    with pytest.raises(ConnectionError):
        model_bias_monitor.update_monitoring_schedule(
            max_runtime_in_seconds=NEW_MAX_RUNTIME_IN_SECONDS,
        )
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_bias_job_definition.assert_called_once()
    assert (
        sagemaker_session.sagemaker_client.delete_model_bias_job_definition.call_args[1][
            "JobDefinitionName"
        ]
        != old_job_definition_name
    )


def _test_model_bias_monitor_create_schedule(
    model_bias_monitor,
    sagemaker_session,
    analysis_config=None,
    constraints=None,
    baseline_job_name=None,
    endpoint_input=EndpointInput(
        endpoint_name=ENDPOINT_NAME,
        destination=ENDPOINT_INPUT_LOCAL_PATH,
        start_time_offset=START_TIME_OFFSET,
        end_time_offset=END_TIME_OFFSET,
        features_attribute=FEATURES_ATTRIBUTE,
        inference_attribute=str(INFERENCE_ATTRIBUTE),
        probability_attribute=str(PROBABILITY_ATTRIBUTE),
        probability_threshold_attribute=PROBABILITY_THRESHOLD_ATTRIBUTE,
    ),
):
    # create schedule
    with patch(
        "sagemaker.s3.S3Uploader.upload_string_as_file_body", return_value=ANALYSIS_CONFIG_S3_URI
    ) as upload:
        model_bias_monitor.create_monitoring_schedule(
            endpoint_input=endpoint_input,
            ground_truth_input=GROUND_TRUTH_S3_URI,
            analysis_config=analysis_config,
            output_s3_uri=OUTPUT_S3_URI,
            constraints=constraints,
            monitor_schedule_name=SCHEDULE_NAME,
            schedule_cron_expression=CRON_HOURLY,
        )
        if not isinstance(analysis_config, str):
            upload.assert_called_once()
            assert json.loads(upload.call_args[0][0]) == BIAS_ANALYSIS_CONFIG

    # validation
    expected_arguments = {
        "JobDefinitionName": model_bias_monitor.job_definition_name,
        **copy.deepcopy(BIAS_JOB_DEFINITION),
        "Tags": TAGS,
    }
    if constraints:
        expected_arguments["ModelBiasBaselineConfig"] = {
            "ConstraintsResource": {"S3Uri": constraints.file_s3_uri}
        }
    elif baseline_job_name:
        expected_arguments["ModelBiasBaselineConfig"] = {
            "BaseliningJobName": baseline_job_name,
        }

    sagemaker_session.sagemaker_client.create_model_bias_job_definition.assert_called_with(
        **expected_arguments
    )

    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_with(
        MonitoringScheduleName=SCHEDULE_NAME,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_bias_monitor.job_definition_name,
            "MonitoringType": "ModelBias",
            "ScheduleConfig": {"ScheduleExpression": CRON_HOURLY},
        },
        Tags=TAGS,
    )


def _test_model_bias_monitor_batch_transform_create_schedule(
    model_bias_monitor,
    sagemaker_session,
    analysis_config=None,
    constraints=None,
    baseline_job_name=None,
    batch_transform_input=BatchTransformInput(
        data_captured_destination_s3_uri=DATA_CAPTURED_S3_URI,
        destination=SCHEDULE_DESTINATION,
        start_time_offset=START_TIME_OFFSET,
        end_time_offset=END_TIME_OFFSET,
        features_attribute=FEATURES_ATTRIBUTE,
        inference_attribute=str(INFERENCE_ATTRIBUTE),
        probability_attribute=str(PROBABILITY_ATTRIBUTE),
        probability_threshold_attribute=PROBABILITY_THRESHOLD_ATTRIBUTE,
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    ),
):
    # create schedule
    with patch(
        "sagemaker.s3.S3Uploader.upload_string_as_file_body", return_value=ANALYSIS_CONFIG_S3_URI
    ) as upload:
        model_bias_monitor.create_monitoring_schedule(
            batch_transform_input=batch_transform_input,
            ground_truth_input=GROUND_TRUTH_S3_URI,
            analysis_config=analysis_config,
            output_s3_uri=OUTPUT_S3_URI,
            constraints=constraints,
            monitor_schedule_name=SCHEDULE_NAME,
            schedule_cron_expression=CRON_HOURLY,
        )
        if not isinstance(analysis_config, str):
            upload.assert_called_once()
            assert json.loads(upload.call_args[0][0]) == BIAS_ANALYSIS_CONFIG

    # validation
    expected_arguments = {
        "JobDefinitionName": model_bias_monitor.job_definition_name,
        **copy.deepcopy(BIAS_BATCH_TRANSFORM_JOB_DEFINITION),
        "Tags": TAGS,
    }
    if constraints:
        expected_arguments["ModelBiasBaselineConfig"] = {
            "ConstraintsResource": {"S3Uri": constraints.file_s3_uri}
        }
    elif baseline_job_name:
        expected_arguments["ModelBiasBaselineConfig"] = {
            "BaseliningJobName": baseline_job_name,
        }

    sagemaker_session.sagemaker_client.create_model_bias_job_definition.assert_called_with(
        **expected_arguments
    )

    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_with(
        MonitoringScheduleName=SCHEDULE_NAME,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_bias_monitor.job_definition_name,
            "MonitoringType": "ModelBias",
            "ScheduleConfig": {"ScheduleExpression": CRON_HOURLY},
        },
        Tags=TAGS,
    )


def _test_model_bias_monitor_update_schedule(model_bias_monitor, sagemaker_session):
    # update schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition = MagicMock()
    sagemaker_session.sagemaker_client.create_model_bias_job_definition = MagicMock()
    model_bias_monitor.update_monitoring_schedule(schedule_cron_expression=CRON_DAILY)
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_bias_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_bias_monitor.job_definition_name,
            "MonitoringType": ModelBiasMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.delete_model_bias_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.create_model_bias_job_definition.assert_not_called()

    # update one property of job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition = Mock(
        return_value=copy.deepcopy(BIAS_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_model_bias_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = model_bias_monitor.job_definition_name
    model_bias_monitor.update_monitoring_schedule(role=NEW_ROLE_ARN)
    expected_arguments = {
        "JobDefinitionName": model_bias_monitor.job_definition_name,
        **copy.deepcopy(BIAS_JOB_DEFINITION),
        "Tags": TAGS,
    }
    assert old_job_definition_name != model_bias_monitor.job_definition_name
    assert model_bias_monitor.role == NEW_ROLE_ARN
    assert model_bias_monitor.instance_count == INSTANCE_COUNT
    assert model_bias_monitor.instance_type == INSTANCE_TYPE
    assert model_bias_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert model_bias_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert model_bias_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert model_bias_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert model_bias_monitor.env == ENVIRONMENT
    assert model_bias_monitor.network_config == NETWORK_CONFIG
    expected_arguments["RoleArn"] = (
        NEW_ROLE_ARN  # all but role arn are from existing job definition
    )
    sagemaker_session.sagemaker_client.create_model_bias_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_bias_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_bias_monitor.job_definition_name,
            "MonitoringType": ModelBiasMonitor.monitoring_type(),
        },
    )
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_bias_job_definition.assert_not_called()

    # update full job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition = Mock(
        return_value=copy.deepcopy(BIAS_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_model_bias_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = model_bias_monitor.job_definition_name
    model_bias_monitor.role = ROLE_ARN
    model_bias_monitor.update_monitoring_schedule(
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
        analysis_config=NEW_ANALYSIS_CONFIG_S3_URI,
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
    assert old_job_definition_name != model_bias_monitor.job_definition_name
    assert model_bias_monitor.role == NEW_ROLE_ARN
    assert model_bias_monitor.instance_count == NEW_INSTANCE_COUNT
    assert model_bias_monitor.instance_type == NEW_INSTANCE_TYPE
    assert model_bias_monitor.volume_size_in_gb == NEW_VOLUME_SIZE_IN_GB
    assert model_bias_monitor.volume_kms_key == NEW_VOLUME_KMS_KEY
    assert model_bias_monitor.output_kms_key == NEW_OUTPUT_KMS_KEY
    assert model_bias_monitor.max_runtime_in_seconds == NEW_MAX_RUNTIME_IN_SECONDS
    assert model_bias_monitor.env == NEW_ENVIRONMENT
    assert model_bias_monitor.network_config == NEW_NETWORK_CONFIG
    expected_arguments = {  # all from new job definition
        "JobDefinitionName": model_bias_monitor.job_definition_name,
        **copy.deepcopy(NEW_BIAS_JOB_DEFINITION),
        "Tags": TAGS,
    }
    sagemaker_session.sagemaker_client.create_model_bias_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_bias_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_bias_monitor.job_definition_name,
            "MonitoringType": ModelBiasMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_model_bias_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_bias_job_definition.assert_not_called()


def _test_model_bias_monitor_delete_schedule(model_bias_monitor, sagemaker_session):
    # delete schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    job_definition_name = model_bias_monitor.job_definition_name
    model_bias_monitor.delete_monitoring_schedule()
    sagemaker_session.delete_monitoring_schedule.assert_called_once_with(
        monitoring_schedule_name=SCHEDULE_NAME
    )
    sagemaker_session.sagemaker_client.delete_model_bias_job_definition.assert_called_once_with(
        JobDefinitionName=job_definition_name
    )


def test_explainability_analysis_config(shap_config, model_config):
    config = ExplainabilityAnalysisConfig(
        explainability_config=shap_config,
        model_config=model_config,
        headers=ANALYSIS_CONFIG_HEADERS_OF_FEATURES,
        label_headers=ANALYSIS_CONFIG_LABEL_HEADERS,
    )
    assert EXPLAINABILITY_ANALYSIS_CONFIG_WITH_LABEL_HEADERS == config._to_dict()


@pytest.mark.parametrize(
    "model_scores,explainability_analysis_config",
    [
        (INFERENCE_ATTRIBUTE, EXPLAINABILITY_ANALYSIS_CONFIG),
        (
            ModelPredictedLabelConfig(
                label=INFERENCE_ATTRIBUTE, label_headers=ANALYSIS_CONFIG_LABEL_HEADERS
            ),
            EXPLAINABILITY_ANALYSIS_CONFIG_WITH_LABEL_HEADERS,
        ),
    ],
)
def test_model_explainability_monitor_suggest_baseline(
    model_explainability_monitor,
    sagemaker_session,
    data_config,
    shap_config,
    model_config,
    model_scores,
    explainability_analysis_config,
):
    clarify_model_monitor = model_explainability_monitor
    # suggest baseline
    clarify_model_monitor.suggest_baseline(
        data_config=data_config,
        explainability_config=shap_config,
        model_config=model_config,
        model_scores=model_scores,
        job_name=BASELINING_JOB_NAME,
    )
    assert isinstance(clarify_model_monitor.latest_baselining_job, ClarifyBaseliningJob)
    assert (
        explainability_analysis_config
        == clarify_model_monitor.latest_baselining_job_config.analysis_config._to_dict()
    )
    clarify_baselining_job = clarify_model_monitor.latest_baselining_job
    assert data_config.s3_data_input_path == clarify_baselining_job.inputs[0].source
    assert data_config.s3_output_path == clarify_baselining_job.outputs[0].destination

    # create schedule
    # noinspection PyTypeChecker
    _test_model_explainability_monitor_create_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=None,  # will pick up config from baselining job
        baseline_job_name=BASELINING_JOB_NAME,
        endpoint_input=ENDPOINT_NAME,
        explainability_analysis_config=explainability_analysis_config,
        #  will pick up attributes from baselining job
    )

    # update schedule
    _test_model_explainability_monitor_update_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_explainability_monitor_delete_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_explainability_monitor(model_explainability_monitor, sagemaker_session):
    # create schedule
    _test_model_explainability_monitor_create_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=ANALYSIS_CONFIG_S3_URI,
        constraints=CONSTRAINTS,
    )

    # update schedule
    _test_model_explainability_monitor_update_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_explainability_monitor_delete_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_explainability_create_one_time_schedule(
    model_explainability_monitor, sagemaker_session
):
    endpoint_input = EndpointInput(
        endpoint_name=ENDPOINT_NAME,
        destination=ENDPOINT_INPUT_LOCAL_PATH,
        features_attribute=FEATURES_ATTRIBUTE,
        inference_attribute=str(INFERENCE_ATTRIBUTE),
    )

    # Create one-time schedule
    with patch(
        "sagemaker.s3.S3Uploader.upload_string_as_file_body", return_value=ANALYSIS_CONFIG_S3_URI
    ) as _:
        model_explainability_monitor.create_monitoring_schedule(
            endpoint_input=endpoint_input,
            analysis_config=ANALYSIS_CONFIG_S3_URI,
            output_s3_uri=OUTPUT_S3_URI,
            monitor_schedule_name=SCHEDULE_NAME,
            schedule_cron_expression=CRON_NOW,
            data_analysis_start_time=START_TIME_OFFSET,
            data_analysis_end_time=END_TIME_OFFSET,
        )

    # Validate job definition creation
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition.assert_called_once()
    job_definition_args = (
        sagemaker_session.sagemaker_client.create_model_explainability_job_definition.call_args[1]
    )
    assert (
        job_definition_args["JobDefinitionName"] == model_explainability_monitor.job_definition_name
    )
    assert job_definition_args == {
        "JobDefinitionName": model_explainability_monitor.job_definition_name,
        **EXPLAINABILITY_JOB_DEFINITION,
        "Tags": TAGS,
    }

    # Validate monitoring schedule creation
    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_once()
    schedule_args = sagemaker_session.sagemaker_client.create_monitoring_schedule.call_args[1]
    assert schedule_args == {
        "MonitoringScheduleName": SCHEDULE_NAME,
        "MonitoringScheduleConfig": {
            "MonitoringJobDefinitionName": model_explainability_monitor.job_definition_name,
            "MonitoringType": "ModelExplainability",
            "ScheduleConfig": {
                "ScheduleExpression": CRON_NOW,
                "DataAnalysisStartTime": START_TIME_OFFSET,
                "DataAnalysisEndTime": END_TIME_OFFSET,
            },
        },
        "Tags": TAGS,
    }

    # Check if the monitoring schedule is stored in the monitor object
    assert model_explainability_monitor.monitoring_schedule_name == SCHEDULE_NAME
    assert model_explainability_monitor.job_definition_name is not None


def test_model_explainability_batch_transform_monitor(
    model_explainability_monitor, sagemaker_session
):
    # create schedule
    _test_model_explainability_batch_transform_monitor_create_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=ANALYSIS_CONFIG_S3_URI,
        constraints=CONSTRAINTS,
    )

    # update schedule
    _test_model_explainability_monitor_update_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_explainability_monitor_delete_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_explainability_monitor_created_with_config(
    model_explainability_monitor, sagemaker_session, shap_config, model_config
):
    # create schedule
    analysis_config = ExplainabilityAnalysisConfig(
        explainability_config=shap_config,
        model_config=model_config,
        headers=ANALYSIS_CONFIG_HEADERS_OF_FEATURES,
    )
    _test_model_explainability_monitor_create_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
        analysis_config=analysis_config,
        constraints=CONSTRAINTS,
        explainability_analysis_config=EXPLAINABILITY_ANALYSIS_CONFIG,
    )

    # update schedule
    _test_model_explainability_monitor_update_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_explainability_monitor_delete_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_explainability_monitor_created_by_attach(sagemaker_session):
    # attach and validate
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition = MagicMock()
    model_explainability_monitor = _test_clarify_model_monitor_created_by_attach(
        sagemaker_session=sagemaker_session,
        clarify_model_monitor_cls=ModelExplainabilityMonitor,
        describe_job_definition=sagemaker_session.sagemaker_client.describe_model_explainability_job_definition,
    )

    # update schedule
    _test_model_explainability_monitor_update_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )

    # delete schedule
    _test_model_explainability_monitor_delete_schedule(
        model_explainability_monitor=model_explainability_monitor,
        sagemaker_session=sagemaker_session,
    )


def test_model_explainability_monitor_invalid_create(
    model_explainability_monitor, sagemaker_session
):
    # invalid: analysis config is missing
    with pytest.raises(ValueError):
        _test_model_explainability_monitor_create_schedule(
            model_explainability_monitor=model_explainability_monitor,
            sagemaker_session=sagemaker_session,
        )

    # invalid: can not create new job definition if one already exists
    model_explainability_monitor.job_definition_name = JOB_DEFINITION_NAME
    with pytest.raises(ValueError):
        _test_model_explainability_monitor_create_schedule(
            model_explainability_monitor=model_explainability_monitor,
            sagemaker_session=sagemaker_session,
            analysis_config=ANALYSIS_CONFIG_S3_URI,
            constraints=CONSTRAINTS,
        )

    # invalid: can not create new schedule if one already exists
    model_explainability_monitor.job_definition_name = None
    model_explainability_monitor.monitoring_schedule_name = SCHEDULE_NAME
    with pytest.raises(ValueError):
        _test_model_explainability_monitor_create_schedule(
            model_explainability_monitor=model_explainability_monitor,
            sagemaker_session=sagemaker_session,
            analysis_config=ANALYSIS_CONFIG_S3_URI,
            constraints=CONSTRAINTS,
        )


def test_model_explainability_monitor_creation_failure(
    model_explainability_monitor, sagemaker_session
):
    sagemaker_session.sagemaker_client.create_monitoring_schedule = Mock(
        side_effect=Exception("400")
    )
    with pytest.raises(Exception):
        _test_model_explainability_monitor_create_schedule(
            model_explainability_monitor=model_explainability_monitor,
            sagemaker_session=sagemaker_session,
            analysis_config=ANALYSIS_CONFIG_S3_URI,
            constraints=CONSTRAINTS,
        )
    assert model_explainability_monitor.job_definition_name is None
    assert model_explainability_monitor.monitoring_schedule_name is None
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition.assert_called_once()
    sagemaker_session.sagemaker_client.delete_model_explainability_job_definition.assert_called_once()


def test_model_explainability_monitor_update_failure(
    model_explainability_monitor, sagemaker_session
):
    model_explainability_monitor.create_monitoring_schedule(
        endpoint_input=ENDPOINT_NAME,
        analysis_config=ANALYSIS_CONFIG_S3_URI,
    )
    old_job_definition_name = model_explainability_monitor.job_definition_name
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition = Mock(
        return_value=copy.deepcopy(EXPLAINABILITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock(
        side_effect=ConnectionError("400")
    )
    with pytest.raises(ConnectionError):
        model_explainability_monitor.update_monitoring_schedule(
            max_runtime_in_seconds=NEW_MAX_RUNTIME_IN_SECONDS,
        )
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_explainability_job_definition.assert_called_once()
    assert (
        sagemaker_session.sagemaker_client.delete_model_explainability_job_definition.call_args[1][
            "JobDefinitionName"
        ]
        != old_job_definition_name
    )


def _test_model_explainability_monitor_create_schedule(
    model_explainability_monitor,
    sagemaker_session,
    analysis_config=None,
    constraints=None,
    baseline_job_name=None,
    endpoint_input=EndpointInput(
        endpoint_name=ENDPOINT_NAME,
        destination=ENDPOINT_INPUT_LOCAL_PATH,
        features_attribute=FEATURES_ATTRIBUTE,
        inference_attribute=str(INFERENCE_ATTRIBUTE),
    ),
    explainability_analysis_config=None,
):
    # create schedule
    with patch(
        "sagemaker.s3.S3Uploader.upload_string_as_file_body", return_value=ANALYSIS_CONFIG_S3_URI
    ) as upload:
        model_explainability_monitor.create_monitoring_schedule(
            endpoint_input=endpoint_input,
            analysis_config=analysis_config,
            output_s3_uri=OUTPUT_S3_URI,
            constraints=constraints,
            monitor_schedule_name=SCHEDULE_NAME,
            schedule_cron_expression=CRON_HOURLY,
        )
        if not isinstance(analysis_config, str):
            upload.assert_called_once()
            assert json.loads(upload.call_args[0][0]) == explainability_analysis_config

    # validation
    expected_arguments = {
        "JobDefinitionName": model_explainability_monitor.job_definition_name,
        **copy.deepcopy(EXPLAINABILITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    if constraints:
        expected_arguments["ModelExplainabilityBaselineConfig"] = {
            "ConstraintsResource": {"S3Uri": constraints.file_s3_uri}
        }
    elif baseline_job_name:
        expected_arguments["ModelExplainabilityBaselineConfig"] = {
            "BaseliningJobName": baseline_job_name,
        }

    sagemaker_session.sagemaker_client.create_model_explainability_job_definition.assert_called_with(
        **expected_arguments
    )

    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_with(
        MonitoringScheduleName=SCHEDULE_NAME,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_explainability_monitor.job_definition_name,
            "MonitoringType": "ModelExplainability",
            "ScheduleConfig": {"ScheduleExpression": CRON_HOURLY},
        },
        Tags=TAGS,
    )


def _test_model_explainability_batch_transform_monitor_create_schedule(
    model_explainability_monitor,
    sagemaker_session,
    analysis_config=None,
    constraints=None,
    baseline_job_name=None,
    batch_transform_input=BatchTransformInput(
        data_captured_destination_s3_uri=DATA_CAPTURED_S3_URI,
        destination=SCHEDULE_DESTINATION,
        features_attribute=FEATURES_ATTRIBUTE,
        inference_attribute=str(INFERENCE_ATTRIBUTE),
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    ),
    explainability_analysis_config=None,
):
    # create schedule
    with patch(
        "sagemaker.s3.S3Uploader.upload_string_as_file_body", return_value=ANALYSIS_CONFIG_S3_URI
    ) as upload:
        model_explainability_monitor.create_monitoring_schedule(
            batch_transform_input=batch_transform_input,
            analysis_config=analysis_config,
            output_s3_uri=OUTPUT_S3_URI,
            constraints=constraints,
            monitor_schedule_name=SCHEDULE_NAME,
            schedule_cron_expression=CRON_HOURLY,
        )
        if not isinstance(analysis_config, str):
            upload.assert_called_once()
            assert json.loads(upload.call_args[0][0]) == explainability_analysis_config

    # validation
    expected_arguments = {
        "JobDefinitionName": model_explainability_monitor.job_definition_name,
        **copy.deepcopy(EXPLAINABILITY__BATCH_TRANSFORM_JOB_DEFINITION),
        "Tags": TAGS,
    }
    if constraints:
        expected_arguments["ModelExplainabilityBaselineConfig"] = {
            "ConstraintsResource": {"S3Uri": constraints.file_s3_uri}
        }
    elif baseline_job_name:
        expected_arguments["ModelExplainabilityBaselineConfig"] = {
            "BaseliningJobName": baseline_job_name,
        }

    sagemaker_session.sagemaker_client.create_model_explainability_job_definition.assert_called_with(
        **expected_arguments
    )

    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_with(
        MonitoringScheduleName=SCHEDULE_NAME,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_explainability_monitor.job_definition_name,
            "MonitoringType": "ModelExplainability",
            "ScheduleConfig": {"ScheduleExpression": CRON_HOURLY},
        },
        Tags=TAGS,
    )


def _test_model_explainability_monitor_update_schedule(
    model_explainability_monitor, sagemaker_session
):
    # update schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition = MagicMock()
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition = MagicMock()
    model_explainability_monitor.update_monitoring_schedule(schedule_cron_expression=CRON_DAILY)
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_explainability_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_explainability_monitor.job_definition_name,
            "MonitoringType": ModelExplainabilityMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.delete_model_explainability_job_definition.assert_not_called()
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition.assert_not_called()

    # update one property of job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition = Mock(
        return_value=copy.deepcopy(EXPLAINABILITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = model_explainability_monitor.job_definition_name
    model_explainability_monitor.update_monitoring_schedule(role=NEW_ROLE_ARN)
    expected_arguments = {
        "JobDefinitionName": model_explainability_monitor.job_definition_name,
        **copy.deepcopy(EXPLAINABILITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    assert old_job_definition_name != model_explainability_monitor.job_definition_name
    assert model_explainability_monitor.role == NEW_ROLE_ARN
    assert model_explainability_monitor.instance_count == INSTANCE_COUNT
    assert model_explainability_monitor.instance_type == INSTANCE_TYPE
    assert model_explainability_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert model_explainability_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert model_explainability_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert model_explainability_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert model_explainability_monitor.env == ENVIRONMENT
    assert model_explainability_monitor.network_config == NETWORK_CONFIG
    expected_arguments["RoleArn"] = (
        NEW_ROLE_ARN  # all but role arn are from existing job definition
    )
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_explainability_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_explainability_monitor.job_definition_name,
            "MonitoringType": ModelExplainabilityMonitor.monitoring_type(),
        },
    )
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_explainability_job_definition.assert_not_called()

    # update full job definition
    time.sleep(
        0.001
    )  # Make sure timestamp changed so that a different job definition name is generated
    sagemaker_session.sagemaker_client.update_monitoring_schedule = Mock()
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition = Mock(
        return_value=copy.deepcopy(EXPLAINABILITY_JOB_DEFINITION)
    )
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition = Mock()
    sagemaker_session.expand_role = Mock(return_value=NEW_ROLE_ARN)
    old_job_definition_name = model_explainability_monitor.job_definition_name
    model_explainability_monitor.role = ROLE_ARN
    model_explainability_monitor.update_monitoring_schedule(
        endpoint_input=EndpointInput(
            endpoint_name=NEW_ENDPOINT_NAME,
            destination=ENDPOINT_INPUT_LOCAL_PATH,
            features_attribute=NEW_FEATURES_ATTRIBUTE,
            inference_attribute=NEW_INFERENCE_ATTRIBUTE,
        ),
        analysis_config=NEW_ANALYSIS_CONFIG_S3_URI,
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
    assert old_job_definition_name != model_explainability_monitor.job_definition_name
    assert model_explainability_monitor.role == NEW_ROLE_ARN
    assert model_explainability_monitor.instance_count == NEW_INSTANCE_COUNT
    assert model_explainability_monitor.instance_type == NEW_INSTANCE_TYPE
    assert model_explainability_monitor.volume_size_in_gb == NEW_VOLUME_SIZE_IN_GB
    assert model_explainability_monitor.volume_kms_key == NEW_VOLUME_KMS_KEY
    assert model_explainability_monitor.output_kms_key == NEW_OUTPUT_KMS_KEY
    assert model_explainability_monitor.max_runtime_in_seconds == NEW_MAX_RUNTIME_IN_SECONDS
    assert model_explainability_monitor.env == NEW_ENVIRONMENT
    assert model_explainability_monitor.network_config == NEW_NETWORK_CONFIG
    expected_arguments = {  # all from new job definition
        "JobDefinitionName": model_explainability_monitor.job_definition_name,
        **copy.deepcopy(NEW_EXPLAINABILITY_JOB_DEFINITION),
        "Tags": TAGS,
    }
    sagemaker_session.sagemaker_client.create_model_explainability_job_definition.assert_called_once_with(
        **expected_arguments
    )
    sagemaker_session.sagemaker_client.update_monitoring_schedule.assert_called_once_with(
        MonitoringScheduleName=model_explainability_monitor.monitoring_schedule_name,
        MonitoringScheduleConfig={
            "MonitoringJobDefinitionName": model_explainability_monitor.job_definition_name,
            "MonitoringType": ModelExplainabilityMonitor.monitoring_type(),
            "ScheduleConfig": {"ScheduleExpression": CRON_DAILY},
        },
    )
    sagemaker_session.sagemaker_client.describe_model_explainability_job_definition.assert_called_once_with(
        JobDefinitionName=old_job_definition_name
    )
    sagemaker_session.sagemaker_client.delete_model_explainability_job_definition.assert_not_called()


def _test_model_explainability_monitor_delete_schedule(
    model_explainability_monitor, sagemaker_session
):
    # delete schedule
    sagemaker_session.describe_monitoring_schedule = MagicMock()
    job_definition_name = model_explainability_monitor.job_definition_name
    model_explainability_monitor.delete_monitoring_schedule()
    sagemaker_session.delete_monitoring_schedule.assert_called_once_with(
        monitoring_schedule_name=SCHEDULE_NAME
    )
    sagemaker_session.sagemaker_client.delete_model_explainability_job_definition.assert_called_once_with(
        JobDefinitionName=job_definition_name
    )


def test_model_explainability_monitor_logs_failure(model_explainability_monitor, sagemaker_session):
    sagemaker_session.list_monitoring_executions = MagicMock(
        return_value=MONITORING_EXECUTIONS_EMPTY
    )
    try:
        model_explainability_monitor.get_latest_execution_logs()
    except ValueError as ve:
        assert "No execution jobs were kicked off." in str(ve)
    sagemaker_session.list_monitoring_executions = MagicMock(
        return_value=MONITORING_EXECUTIONS_NO_PROCESSING_JOB
    )
    try:
        model_explainability_monitor.get_latest_execution_logs()
    except ValueError as ve:
        assert "Processing Job did not run for the last execution" in str(ve)
