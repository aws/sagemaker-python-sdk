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
from __future__ import absolute_import

import json
import os
import uuid
import time
import pytest

import tests.integ
import tests.integ.timeout

from sagemaker.s3 import S3Uploader
from datetime import datetime, timedelta

from tests.integ import DATA_DIR
from sagemaker.model_monitor import DatasetFormat
from sagemaker.model_monitor import MonitoringDatasetFormat
from sagemaker.model_monitor import NetworkConfig, Statistics, Constraints
from sagemaker.model_monitor import ModelMonitor
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor import MonitoringOutput
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.model_monitor import BatchTransformInput
from sagemaker.model_monitor import DataQualityMonitoringConfig
from sagemaker.model_monitor import DataQualityDistributionConstraints
from sagemaker.model_monitor.data_capture_config import _MODEL_MONITOR_S3_PATH
from sagemaker.model_monitor.data_capture_config import _DATA_CAPTURE_S3_PATH
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor.monitoring_alert import MonitoringAlertSummary
from sagemaker.processing import ProcessingInput
from sagemaker.processing import ProcessingOutput
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.utils import unique_name_from_base

from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ.retry import retries

ROLE = "SageMakerRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.m5.xlarge"
VOLUME_SIZE_IN_GB = 30
MAX_RUNTIME_IN_SECONDS = 60 * 60
ENV_KEY_1 = "env_key_1"
ENV_VALUE_1 = "env_key_1"
ENVIRONMENT = {ENV_KEY_1: ENV_VALUE_1}
TAG_KEY_1 = "tag_key_1"
TAG_VALUE_1 = "tag_value_1"
TAGS = [{"Key": TAG_KEY_1, "Value": TAG_VALUE_1}]
NETWORK_CONFIG = NetworkConfig(
    enable_network_isolation=True,
    encrypt_inter_container_traffic=True,
)
ENABLE_CLOUDWATCH_METRICS = True
ENABLE_AUTOMATIC_DASHBOARD = True
DASHBOARD_NAME = "DASHBOARD NAME"

DEFAULT_BASELINING_MAX_RUNTIME_IN_SECONDS = 86400
DEFAULT_EXECUTION_MAX_RUNTIME_IN_SECONDS = 3600
DEFAULT_IMAGE_SUFFIX = "/sagemaker-model-monitor-analyzer"

UPDATED_ROLE = "SageMakerRole"
UPDATED_INSTANCE_COUNT = 2
UPDATED_INSTANCE_TYPE = "ml.m5.2xlarge"
UPDATED_VOLUME_SIZE_IN_GB = 50
UPDATED_MAX_RUNTIME_IN_SECONDS = 46 * 2
UPDATED_ENV_KEY_1 = "env_key_2"
UPDATED_ENV_VALUE_1 = "env_key_2"
UPDATED_ENVIRONMENT = {UPDATED_ENV_KEY_1: UPDATED_ENV_VALUE_1}
UPDATED_TAG_KEY_1 = "tag_key_2"
UPDATED_TAG_VALUE_1 = "tag_value_2"
UPDATED_TAGS = [{"Key": TAG_KEY_1, "Value": TAG_VALUE_1}]
UPDATED_NETWORK_CONFIG = NetworkConfig(enable_network_isolation=False)
DISABLE_CLOUDWATCH_METRICS = False

CUSTOM_SAMPLING_PERCENTAGE = 10
CUSTOM_CAPTURE_OPTIONS = ["REQUEST"]
CUSTOM_CSV_CONTENT_TYPES = ["text/csvtype1", "text/csvtype2"]
CUSTOM_JSON_CONTENT_TYPES = ["application/jsontype1", "application/jsontype2"]

INTEG_TEST_MONITORING_OUTPUT_BUCKET = "integ-test-monitoring-output-bucket"

HOURLY_CRON_EXPRESSION = "cron(0 * ? * * *)"


@pytest.fixture(scope="module")
def predictor(sagemaker_session, tensorflow_inference_latest_version):
    endpoint_name = unique_name_from_base("sagemaker-tensorflow-serving")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        hours=2,
        sleep_between_cleanup_attempts=20,
        exponential_sleep=True,
    ):
        model = TensorFlowModel(
            model_data=model_data,
            role=ROLE,
            framework_version=tensorflow_inference_latest_version,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(
            INSTANCE_COUNT,
            INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            data_capture_config=DataCaptureConfig(True, sagemaker_session=sagemaker_session),
        )
        yield predictor


@pytest.fixture(scope="module")
def default_monitoring_schedule_name(sagemaker_session, output_kms_key, volume_kms_key, predictor):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=HOURLY_CRON_EXPRESSION,
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    _upload_captured_data_to_endpoint(predictor=predictor, sagemaker_session=sagemaker_session)

    _predict_while_waiting_for_first_monitoring_job_to_complete(predictor, my_default_monitor)

    return my_default_monitor.monitoring_schedule_name


@pytest.fixture(scope="module")
def byoc_monitoring_schedule_name(sagemaker_session, output_kms_key, volume_kms_key, predictor):
    byoc_env = ENVIRONMENT.copy()
    byoc_env["dataset_format"] = json.dumps(DatasetFormat.csv(header=False))
    byoc_env["dataset_source"] = "/opt/ml/processing/input/baseline_dataset_input"
    byoc_env["output_path"] = os.path.join("/opt/ml/processing/output")
    byoc_env["publish_cloudwatch_metrics"] = "Disabled"

    my_byoc_monitor = ModelMonitor(
        role=ROLE,
        image_uri=DefaultModelMonitor._get_default_image_uri(
            sagemaker_session.boto_session.region_name
        ),
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=byoc_env,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_byoc_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output=MonitoringOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=HOURLY_CRON_EXPRESSION,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    _upload_captured_data_to_endpoint(predictor=predictor, sagemaker_session=sagemaker_session)

    _predict_while_waiting_for_first_monitoring_job_to_complete(predictor, my_byoc_monitor)

    return my_byoc_monitor.monitoring_schedule_name


@pytest.fixture(scope="module")
def volume_kms_key(sagemaker_session):
    role_arn = sagemaker_session.expand_role(ROLE)
    return get_or_create_kms_key(
        sagemaker_session=sagemaker_session,
        role_arn=role_arn,
        alias="integ-test-processing-volume-kms-key-{}".format(
            sagemaker_session.boto_session.region_name
        ),
    )


@pytest.fixture(scope="module")
def output_kms_key(sagemaker_session):
    role_arn = sagemaker_session.expand_role(ROLE)
    return get_or_create_kms_key(
        sagemaker_session=sagemaker_session,
        role_arn=role_arn,
        alias="integ-test-processing-output-kms-key-{}".format(
            sagemaker_session.boto_session.region_name
        ),
    )


@pytest.fixture(scope="module")
def updated_volume_kms_key(sagemaker_session):
    role_arn = sagemaker_session.expand_role(ROLE)
    return get_or_create_kms_key(
        sagemaker_session=sagemaker_session,
        role_arn=role_arn,
        alias="integ-test-processing-volume-kms-key-updated-{}".format(
            sagemaker_session.boto_session.region_name
        ),
    )


@pytest.fixture(scope="module")
def updated_output_kms_key(sagemaker_session):
    role_arn = sagemaker_session.expand_role(ROLE)
    return get_or_create_kms_key(
        sagemaker_session=sagemaker_session,
        role_arn=role_arn,
        alias="integ-test-processing-output-kms-key-updated-{}".format(
            sagemaker_session.boto_session.region_name
        ),
    )


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitoring_batch_transform_schedule_name(
    sagemaker_session, output_kms_key, volume_kms_key
):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-tensorflow-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_default_monitor.create_monitoring_schedule(
        batch_transform_input=batch_transform_input,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=HOURLY_CRON_EXPRESSION,
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule_with_batch_transform(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        cron_expression=HOURLY_CRON_EXPRESSION,
        statistics=statistics,
        constraints=constraints,
        output_kms_key=output_kms_key,
        volume_kms_key=volume_kms_key,
        network_config=NETWORK_CONFIG,
    )

    my_default_monitor.stop_monitoring_schedule()


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_suggest_baseline_and_create_monitoring_schedule_with_customizations(
    sagemaker_session, output_kms_key, volume_kms_key, predictor
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )

    my_default_monitor.suggest_baseline(
        baseline_dataset=baseline_dataset,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=output_s3_uri,
        wait=True,
        logs=False,
    )

    baselining_job_description = my_default_monitor.latest_baselining_job.describe()

    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        == INSTANCE_TYPE
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
        == INSTANCE_COUNT
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"]
        == VOLUME_SIZE_IN_GB
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"]
        == volume_kms_key
    )
    assert DEFAULT_IMAGE_SUFFIX in baselining_job_description["AppSpecification"]["ImageUri"]
    assert ROLE in baselining_job_description["RoleArn"]
    assert (
        baselining_job_description["ProcessingInputs"][0]["InputName"] == "baseline_dataset_input"
    )
    assert (
        baselining_job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
        == "monitoring_output"
    )
    assert baselining_job_description["ProcessingOutputConfig"]["KmsKeyId"] == output_kms_key
    assert baselining_job_description["Environment"][ENV_KEY_1] == ENV_VALUE_1
    assert baselining_job_description["Environment"]["output_path"] == "/opt/ml/processing/output"
    assert "categorical_drift_method" not in baselining_job_description["Environment"]
    assert (
        baselining_job_description["Environment"]["dataset_source"]
        == "/opt/ml/processing/input/baseline_dataset_input"
    )
    assert (
        baselining_job_description["StoppingCondition"]["MaxRuntimeInSeconds"]
        == MAX_RUNTIME_IN_SECONDS
    )
    assert (
        baselining_job_description["NetworkConfig"]["EnableNetworkIsolation"]
        == NETWORK_CONFIG.enable_network_isolation
    )

    statistics = my_default_monitor.baseline_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    constraints = my_default_monitor.suggested_constraints()
    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"
    assert (
        constraints.body_dict["monitoring_config"]["distribution_constraints"][
            "categorical_drift_method"
        ]
        == "LInfinity"
    )
    assert (
        constraints.body_dict["monitoring_config"]["distribution_constraints"][
            "categorical_comparison_threshold"
        ]
        == 0.1
    )

    constraints.set_monitoring(enable_monitoring=False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.save()

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output_s3_uri=output_s3_uri,
        statistics=my_default_monitor.baseline_statistics(),
        constraints=my_default_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=my_default_monitor.baseline_statistics(),
        constraints=my_default_monitor.suggested_constraints(),
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        network_config=NETWORK_CONFIG,
    )

    summary = sagemaker_session.list_monitoring_schedules()
    assert len(summary["MonitoringScheduleSummaries"]) > 0


def test_default_monitor_display_logs_errors(sagemaker_session):
    my_default_monitor = DefaultModelMonitor(role=ROLE, sagemaker_session=sagemaker_session)

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    my_default_monitor.create_monitoring_schedule(
        batch_transform_input=batch_transform_input,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    try:
        my_default_monitor.get_latest_execution_logs(wait=False)
    except ValueError as ve:
        assert "No execution jobs were kicked off." in str(ve)

    my_default_monitor.stop_monitoring_schedule()
    my_default_monitor.delete_monitoring_schedule()


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_suggest_baseline_with_chisquared_categorical_drift_method(
    sagemaker_session, output_kms_key, volume_kms_key
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )

    distribution_constraints_with_chisquared_categorical_drift_method = (
        DataQualityDistributionConstraints(categorical_drift_method="ChiSquared")
    )
    monitoring_config_with_chisquared_categorical_drift_method = DataQualityMonitoringConfig(
        distribution_constraints=distribution_constraints_with_chisquared_categorical_drift_method
    )

    my_default_monitor.suggest_baseline(
        baseline_dataset=baseline_dataset,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=output_s3_uri,
        wait=True,
        logs=False,
        monitoring_config_override=monitoring_config_with_chisquared_categorical_drift_method,
    )

    baselining_job_description = my_default_monitor.latest_baselining_job.describe()

    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        == INSTANCE_TYPE
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
        == INSTANCE_COUNT
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"]
        == VOLUME_SIZE_IN_GB
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"]
        == volume_kms_key
    )
    assert DEFAULT_IMAGE_SUFFIX in baselining_job_description["AppSpecification"]["ImageUri"]
    assert ROLE in baselining_job_description["RoleArn"]
    assert (
        baselining_job_description["ProcessingInputs"][0]["InputName"] == "baseline_dataset_input"
    )
    assert (
        baselining_job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
        == "monitoring_output"
    )
    assert baselining_job_description["ProcessingOutputConfig"]["KmsKeyId"] == output_kms_key
    assert baselining_job_description["Environment"][ENV_KEY_1] == ENV_VALUE_1
    assert baselining_job_description["Environment"]["output_path"] == "/opt/ml/processing/output"
    assert baselining_job_description["Environment"]["categorical_drift_method"] == "ChiSquared"
    assert (
        baselining_job_description["Environment"]["dataset_source"]
        == "/opt/ml/processing/input/baseline_dataset_input"
    )
    assert (
        baselining_job_description["StoppingCondition"]["MaxRuntimeInSeconds"]
        == MAX_RUNTIME_IN_SECONDS
    )
    assert (
        baselining_job_description["NetworkConfig"]["EnableNetworkIsolation"]
        == NETWORK_CONFIG.enable_network_isolation
    )

    statistics = my_default_monitor.baseline_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    constraints = my_default_monitor.suggested_constraints()
    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"
    assert (
        constraints.body_dict["monitoring_config"]["distribution_constraints"][
            "categorical_drift_method"
        ]
        == "ChiSquared"
    )
    assert (
        constraints.body_dict["monitoring_config"]["distribution_constraints"][
            "categorical_comparison_threshold"
        ]
        == 0.1
    )

    constraints.set_monitoring(enable_monitoring=False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.save()


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_suggest_baseline_with_linfinity_categorical_drift_method(
    sagemaker_session, output_kms_key, volume_kms_key
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )
    monitoring_config_with_linfinity_categorical_drift_method = DataQualityMonitoringConfig(
        distribution_constraints=DataQualityDistributionConstraints(
            categorical_drift_method="LInfinity"
        )
    )
    my_default_monitor.suggest_baseline(
        baseline_dataset=baseline_dataset,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=output_s3_uri,
        wait=True,
        logs=False,
        monitoring_config_override=monitoring_config_with_linfinity_categorical_drift_method,
    )

    baselining_job_description = my_default_monitor.latest_baselining_job.describe()

    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        == INSTANCE_TYPE
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
        == INSTANCE_COUNT
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"]
        == VOLUME_SIZE_IN_GB
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"]
        == volume_kms_key
    )
    assert DEFAULT_IMAGE_SUFFIX in baselining_job_description["AppSpecification"]["ImageUri"]
    assert ROLE in baselining_job_description["RoleArn"]
    assert (
        baselining_job_description["ProcessingInputs"][0]["InputName"] == "baseline_dataset_input"
    )
    assert (
        baselining_job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
        == "monitoring_output"
    )
    assert baselining_job_description["ProcessingOutputConfig"]["KmsKeyId"] == output_kms_key
    assert baselining_job_description["Environment"][ENV_KEY_1] == ENV_VALUE_1
    assert baselining_job_description["Environment"]["output_path"] == "/opt/ml/processing/output"
    assert baselining_job_description["Environment"]["categorical_drift_method"] == "LInfinity"
    assert (
        baselining_job_description["Environment"]["dataset_source"]
        == "/opt/ml/processing/input/baseline_dataset_input"
    )
    assert (
        baselining_job_description["StoppingCondition"]["MaxRuntimeInSeconds"]
        == MAX_RUNTIME_IN_SECONDS
    )
    assert (
        baselining_job_description["NetworkConfig"]["EnableNetworkIsolation"]
        == NETWORK_CONFIG.enable_network_isolation
    )

    statistics = my_default_monitor.baseline_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    constraints = my_default_monitor.suggested_constraints()
    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"
    assert (
        constraints.body_dict["monitoring_config"]["distribution_constraints"][
            "categorical_drift_method"
        ]
        == "LInfinity"
    )
    assert (
        constraints.body_dict["monitoring_config"]["distribution_constraints"][
            "categorical_comparison_threshold"
        ]
        == 0.1
    )

    constraints.set_monitoring(enable_monitoring=False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.save()


def test_default_monitor_suggest_baseline_with_invalid_categorical_drift_method(
    sagemaker_session, output_kms_key, volume_kms_key
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )

    distribution_constraints_with_invalid_categorical_drift_method = (
        DataQualityDistributionConstraints(
            categorical_drift_method="INVALID_CATEGORICAL_DRIFT_METHOD"
        )
    )
    monitoring_config_with_invalid_categorical_drift_method = DataQualityMonitoringConfig(
        distribution_constraints=distribution_constraints_with_invalid_categorical_drift_method
    )

    with pytest.raises(RuntimeError) as error:
        my_default_monitor.suggest_baseline(
            baseline_dataset=baseline_dataset,
            dataset_format=DatasetFormat.csv(header=False),
            output_s3_uri=output_s3_uri,
            wait=True,
            logs=False,
            monitoring_config_override=monitoring_config_with_invalid_categorical_drift_method,
        )
    assert "Invalid value for monitoring_config_override." in str(error)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_suggest_baseline_and_create_monitoring_schedule_without_customizations(
    sagemaker_session, predictor
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    my_default_monitor = DefaultModelMonitor(role=ROLE, sagemaker_session=sagemaker_session)

    my_default_monitor.suggest_baseline(
        baseline_dataset=baseline_dataset,
        dataset_format=DatasetFormat.csv(header=False),
        logs=False,
    )

    baselining_job_description = my_default_monitor.latest_baselining_job.describe()

    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        == INSTANCE_TYPE
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
        == INSTANCE_COUNT
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"]
        == VOLUME_SIZE_IN_GB
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"].get("VolumeKmsKeyId")
        is None
    )
    assert DEFAULT_IMAGE_SUFFIX in baselining_job_description["AppSpecification"]["ImageUri"]
    assert ROLE in baselining_job_description["RoleArn"]
    assert (
        baselining_job_description["ProcessingInputs"][0]["InputName"] == "baseline_dataset_input"
    )
    assert len(baselining_job_description["ProcessingInputs"]) == 1
    assert (
        baselining_job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
        == "monitoring_output"
    )
    assert baselining_job_description["ProcessingOutputConfig"].get("KmsKeyId") is None
    assert baselining_job_description["Environment"].get(ENV_KEY_1) is None
    assert baselining_job_description["Environment"]["output_path"] == "/opt/ml/processing/output"
    assert baselining_job_description["Environment"].get("record_preprocessor_script") is None
    assert baselining_job_description["Environment"].get("post_analytics_processor_script") is None
    assert (
        baselining_job_description["Environment"]["dataset_source"]
        == "/opt/ml/processing/input/baseline_dataset_input"
    )
    assert (
        baselining_job_description["StoppingCondition"]["MaxRuntimeInSeconds"]
        == DEFAULT_BASELINING_MAX_RUNTIME_IN_SECONDS
    )
    assert baselining_job_description.get("NetworkConfig") is None

    statistics = my_default_monitor.baseline_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    constraints = my_default_monitor.suggested_constraints()
    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"

    constraints.set_monitoring(enable_monitoring=False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.save()

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        schedule_cron_expression=CronExpressionGenerator.daily(),
    )
    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=statistics,
        constraints=constraints,
    )

    summary = sagemaker_session.list_monitoring_schedules()
    assert len(summary["MonitoringScheduleSummaries"]) > 0


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_create_stop_and_start_monitoring_schedule_with_customizations(
    sagemaker_session, output_kms_key, volume_kms_key, predictor
):

    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=statistics,
        constraints=constraints,
        output_kms_key=output_kms_key,
        volume_kms_key=volume_kms_key,
        network_config=NETWORK_CONFIG,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    stopped_schedule_description = my_default_monitor.describe_schedule()
    assert stopped_schedule_description["MonitoringScheduleStatus"] == "Stopped"

    my_default_monitor.start_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    started_schedule_description = my_default_monitor.describe_schedule()
    assert started_schedule_description["MonitoringScheduleStatus"] == "Scheduled"

    my_default_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_create_and_update_schedule_config_with_customizations(
    sagemaker_session,
    predictor,
    volume_kms_key,
    output_kms_key,
    updated_volume_kms_key,
    updated_output_kms_key,
):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=statistics,
        constraints=constraints,
        output_kms_key=output_kms_key,
        volume_kms_key=volume_kms_key,
        network_config=NETWORK_CONFIG,
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.update_monitoring_schedule(
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        instance_count=UPDATED_INSTANCE_COUNT,
        instance_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env=UPDATED_ENVIRONMENT,
        network_config=UPDATED_NETWORK_CONFIG,
        enable_cloudwatch_metrics=DISABLE_CLOUDWATCH_METRICS,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(my_default_monitor)

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=statistics,
        constraints=constraints,
        output_kms_key=updated_output_kms_key,
        volume_kms_key=updated_volume_kms_key,
        cron_expression=CronExpressionGenerator.hourly(),
        instant_count=UPDATED_INSTANCE_COUNT,
        instant_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        network_config=UPDATED_NETWORK_CONFIG,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        publish_cloudwatch_metrics="Disabled",
        env_key=UPDATED_ENV_KEY_1,
        env_value=UPDATED_ENV_VALUE_1,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    assert len(predictor.list_monitors()) > 0


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_create_and_update_schedule_config_with_dashboards(
    sagemaker_session,
    predictor,
    volume_kms_key,
    output_kms_key,
    updated_volume_kms_key,
    updated_output_kms_key,
):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
        enable_automatic_dashboard=ENABLE_AUTOMATIC_DASHBOARD,
        dashboard_name=DASHBOARD_NAME,
    )

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=statistics,
        constraints=constraints,
        output_kms_key=output_kms_key,
        volume_kms_key=volume_kms_key,
        network_config=NETWORK_CONFIG,
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.update_monitoring_schedule(
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        instance_count=UPDATED_INSTANCE_COUNT,
        instance_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env=UPDATED_ENVIRONMENT,
        network_config=UPDATED_NETWORK_CONFIG,
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
        enable_automatic_dashboard=ENABLE_AUTOMATIC_DASHBOARD,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(my_default_monitor)

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=statistics,
        constraints=constraints,
        output_kms_key=updated_output_kms_key,
        volume_kms_key=updated_volume_kms_key,
        cron_expression=CronExpressionGenerator.hourly(),
        instant_count=UPDATED_INSTANCE_COUNT,
        instant_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        network_config=UPDATED_NETWORK_CONFIG,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        publish_cloudwatch_metrics="Disabled",
        env_key=UPDATED_ENV_KEY_1,
        env_value=UPDATED_ENV_VALUE_1,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    assert len(predictor.list_monitors()) > 0


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_create_and_update_schedule_config_without_customizations(
    sagemaker_session, predictor
):
    my_default_monitor = DefaultModelMonitor(role=ROLE, sagemaker_session=sagemaker_session)

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        schedule_cron_expression=CronExpressionGenerator.daily(),
    )

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
    )

    _wait_for_schedule_changes_to_apply(my_default_monitor)

    my_default_monitor.update_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(my_default_monitor)

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_create_and_simple_update_schedule_config(sagemaker_session, predictor):
    my_default_monitor = DefaultModelMonitor(role=ROLE, sagemaker_session=sagemaker_session)

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )

    schedule_description = my_default_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        cron_expression=CronExpressionGenerator.hourly(),
    )

    _wait_for_schedule_changes_to_apply(my_default_monitor)

    my_default_monitor.update_monitoring_schedule(
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS
    )

    _wait_for_schedule_changes_to_apply(my_default_monitor)

    schedule_description = my_default_monitor.describe_schedule()

    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        cron_expression=CronExpressionGenerator.hourly(),
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
@pytest.mark.cron
def test_default_monitor_attach_followed_by_baseline_and_update_monitoring_schedule(
    sagemaker_session,
    default_monitoring_schedule_name,
    updated_volume_kms_key,
    updated_output_kms_key,
):
    my_attached_monitor = DefaultModelMonitor.attach(
        monitor_schedule_name=default_monitoring_schedule_name, sagemaker_session=sagemaker_session
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    _wait_for_schedule_changes_to_apply(my_attached_monitor)

    my_attached_monitor.update_monitoring_schedule(
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        instance_count=UPDATED_INSTANCE_COUNT,
        instance_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env=UPDATED_ENVIRONMENT,
        network_config=UPDATED_NETWORK_CONFIG,
        enable_cloudwatch_metrics=DISABLE_CLOUDWATCH_METRICS,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(my_attached_monitor)

    schedule_description = my_attached_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        cron_expression=CronExpressionGenerator.hourly(),
        statistics=statistics,
        constraints=constraints,
        instant_count=UPDATED_INSTANCE_COUNT,
        instant_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env_key=UPDATED_ENV_KEY_1,
        env_value=UPDATED_ENV_VALUE_1,
        publish_cloudwatch_metrics="Disabled",
        network_config=UPDATED_NETWORK_CONFIG,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_attached_monitor)

    my_attached_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_attached_monitor)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
@pytest.mark.cron
def test_default_monitor_monitoring_execution_interactions(
    sagemaker_session, default_monitoring_schedule_name
):
    my_attached_monitor = DefaultModelMonitor.attach(
        monitor_schedule_name=default_monitoring_schedule_name, sagemaker_session=sagemaker_session
    )
    description = my_attached_monitor.describe_schedule()
    assert description["MonitoringScheduleName"] == default_monitoring_schedule_name

    executions = my_attached_monitor.list_executions()
    assert len(executions) > 0

    with open(os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"), "r") as f:
        file_body = f.read()

    file_name = "statistics.json"
    desired_s3_uri = os.path.join(executions[-1].output.destination, file_name)

    S3Uploader.upload_string_as_file_body(
        body=file_body, desired_s3_uri=desired_s3_uri, sagemaker_session=sagemaker_session
    )

    statistics = my_attached_monitor.latest_monitoring_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraint_violations.json"), "r") as f:
        file_body = f.read()

    file_name = "constraint_violations.json"
    desired_s3_uri = os.path.join(executions[-1].output.destination, file_name)

    S3Uploader.upload_string_as_file_body(
        body=file_body, desired_s3_uri=desired_s3_uri, sagemaker_session=sagemaker_session
    )

    constraint_violations = my_attached_monitor.latest_monitoring_constraint_violations()
    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_byoc_monitor_suggest_baseline_and_create_monitoring_schedule_with_customizations(
    sagemaker_session, output_kms_key, volume_kms_key, predictor
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    byoc_env = ENVIRONMENT.copy()
    byoc_env["dataset_format"] = json.dumps(DatasetFormat.csv(header=False))
    byoc_env["dataset_source"] = "/opt/ml/processing/input/baseline_dataset_input"
    byoc_env["output_path"] = os.path.join("/opt/ml/processing/output")
    byoc_env["publish_cloudwatch_metrics"] = "Disabled"

    my_byoc_monitor = ModelMonitor(
        role=ROLE,
        image_uri=DefaultModelMonitor._get_default_image_uri(
            sagemaker_session.boto_session.region_name
        ),
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=byoc_env,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )

    my_byoc_monitor.run_baseline(
        baseline_inputs=[
            ProcessingInput(
                source=baseline_dataset,
                destination="/opt/ml/processing/input/baseline_dataset_input",
            )
        ],
        output=ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        wait=True,
        logs=False,
    )

    baselining_job_description = my_byoc_monitor.latest_baselining_job.describe()

    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        == INSTANCE_TYPE
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
        == INSTANCE_COUNT
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"]
        == VOLUME_SIZE_IN_GB
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"]
        == volume_kms_key
    )
    assert DEFAULT_IMAGE_SUFFIX in baselining_job_description["AppSpecification"]["ImageUri"]
    assert ROLE in baselining_job_description["RoleArn"]
    assert baselining_job_description["ProcessingInputs"][0]["InputName"] == "input-1"
    assert (
        baselining_job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
        == "output-1"
    )
    assert baselining_job_description["ProcessingOutputConfig"]["KmsKeyId"] == output_kms_key
    assert baselining_job_description["Environment"][ENV_KEY_1] == ENV_VALUE_1
    assert baselining_job_description["Environment"]["output_path"] == "/opt/ml/processing/output"
    assert (
        baselining_job_description["Environment"]["dataset_source"]
        == "/opt/ml/processing/input/baseline_dataset_input"
    )
    assert (
        baselining_job_description["StoppingCondition"]["MaxRuntimeInSeconds"]
        == MAX_RUNTIME_IN_SECONDS
    )
    assert (
        baselining_job_description["NetworkConfig"]["EnableNetworkIsolation"]
        == NETWORK_CONFIG.enable_network_isolation
    )

    statistics = my_byoc_monitor.baseline_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    constraints = my_byoc_monitor.suggested_constraints()
    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"

    constraints.set_monitoring(enable_monitoring=False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.save()

    my_byoc_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output=MonitoringOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        statistics=my_byoc_monitor.baseline_statistics(),
        constraints=my_byoc_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
    )

    schedule_description = my_byoc_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=my_byoc_monitor.baseline_statistics(),
        constraints=my_byoc_monitor.suggested_constraints(),
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        publish_cloudwatch_metrics="Disabled",
        network_config=NETWORK_CONFIG,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    my_byoc_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    summary = sagemaker_session.list_monitoring_schedules()
    assert len(summary["MonitoringScheduleSummaries"]) > 0


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_byoc_monitor_suggest_baseline_and_create_monitoring_schedule_without_customizations(
    sagemaker_session, predictor
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    byoc_env = ENVIRONMENT.copy()
    byoc_env["dataset_format"] = json.dumps(DatasetFormat.csv(header=False))
    byoc_env["dataset_source"] = "/opt/ml/processing/input/baseline_dataset_input"
    byoc_env["output_path"] = os.path.join("/opt/ml/processing/output")
    byoc_env["publish_cloudwatch_metrics"] = "Disabled"

    my_byoc_monitor = ModelMonitor(
        role=ROLE,
        image_uri=DefaultModelMonitor._get_default_image_uri(
            sagemaker_session.boto_session.region_name
        ),
        sagemaker_session=sagemaker_session,
        env=byoc_env,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )

    my_byoc_monitor.run_baseline(
        baseline_inputs=[
            ProcessingInput(
                source=baseline_dataset,
                destination="/opt/ml/processing/input/baseline_dataset_input",
            )
        ],
        output=ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        logs=False,
    )

    baselining_job_description = my_byoc_monitor.latest_baselining_job.describe()

    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
        == INSTANCE_COUNT
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        == INSTANCE_TYPE
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"]
        == VOLUME_SIZE_IN_GB
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"].get("VolumeKmsKeyId")
        is None
    )
    assert DEFAULT_IMAGE_SUFFIX in baselining_job_description["AppSpecification"]["ImageUri"]
    assert ROLE in baselining_job_description["RoleArn"]
    assert baselining_job_description["ProcessingInputs"][0]["InputName"] == "input-1"
    assert (
        baselining_job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
        == "output-1"
    )
    assert baselining_job_description["ProcessingOutputConfig"].get("KmsKeyId") is None
    assert baselining_job_description["Environment"][ENV_KEY_1] == ENV_VALUE_1
    assert baselining_job_description["Environment"]["output_path"] == "/opt/ml/processing/output"
    assert (
        baselining_job_description["Environment"]["dataset_source"]
        == "/opt/ml/processing/input/baseline_dataset_input"
    )
    assert (
        baselining_job_description["StoppingCondition"]["MaxRuntimeInSeconds"]
        == DEFAULT_BASELINING_MAX_RUNTIME_IN_SECONDS
    )
    assert baselining_job_description.get("NetworkConfig") is None

    statistics = my_byoc_monitor.baseline_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    constraints = my_byoc_monitor.suggested_constraints()
    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"

    constraints.set_monitoring(enable_monitoring=False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.save()

    my_byoc_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output=MonitoringOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        schedule_cron_expression=CronExpressionGenerator.daily(),
    )

    schedule_description = my_byoc_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        max_runtime_in_seconds=DEFAULT_EXECUTION_MAX_RUNTIME_IN_SECONDS,
        publish_cloudwatch_metrics="Disabled",
    )

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    my_byoc_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    summary = sagemaker_session.list_monitoring_schedules()
    assert len(summary["MonitoringScheduleSummaries"]) > 0


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_byoc_monitor_create_and_update_schedule_config_with_customizations(
    sagemaker_session,
    predictor,
    volume_kms_key,
    output_kms_key,
    updated_volume_kms_key,
    updated_output_kms_key,
):
    byoc_env = ENVIRONMENT.copy()
    byoc_env["dataset_format"] = json.dumps(DatasetFormat.csv(header=False))
    byoc_env["dataset_source"] = "/opt/ml/processing/input/baseline_dataset_input"
    byoc_env["output_path"] = os.path.join("/opt/ml/processing/output")
    byoc_env["publish_cloudwatch_metrics"] = "Disabled"

    my_byoc_monitor = ModelMonitor(
        role=ROLE,
        image_uri=DefaultModelMonitor._get_default_image_uri(
            sagemaker_session.boto_session.region_name
        ),
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=byoc_env,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_byoc_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output=MonitoringOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.daily(),
    )

    schedule_description = my_byoc_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        statistics=statistics,
        constraints=constraints,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        publish_cloudwatch_metrics="Disabled",
        network_config=NETWORK_CONFIG,
    )

    _wait_for_schedule_changes_to_apply(my_byoc_monitor)

    byoc_env.update(UPDATED_ENVIRONMENT)

    my_byoc_monitor.update_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output=MonitoringOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        instance_count=UPDATED_INSTANCE_COUNT,
        instance_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env=byoc_env,
        network_config=UPDATED_NETWORK_CONFIG,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(my_byoc_monitor)

    schedule_description = my_byoc_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        cron_expression=CronExpressionGenerator.hourly(),
        statistics=statistics,
        constraints=constraints,
        instant_count=UPDATED_INSTANCE_COUNT,
        instant_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        publish_cloudwatch_metrics="Disabled",
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env_key=UPDATED_ENV_KEY_1,
        env_value=UPDATED_ENV_VALUE_1,
        network_config=UPDATED_NETWORK_CONFIG,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    my_byoc_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    assert len(predictor.list_monitors()) > 0


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
@pytest.mark.cron
def test_byoc_monitor_attach_followed_by_baseline_and_update_monitoring_schedule(
    sagemaker_session,
    predictor,
    byoc_monitoring_schedule_name,
    volume_kms_key,
    output_kms_key,
    updated_volume_kms_key,
    updated_output_kms_key,
    capfd,
):
    baseline_dataset = os.path.join(DATA_DIR, "monitor/baseline_dataset.csv")

    byoc_env = ENVIRONMENT.copy()
    byoc_env["dataset_format"] = json.dumps(DatasetFormat.csv(header=False))
    byoc_env["dataset_source"] = "/opt/ml/processing/input/baseline_dataset_input"
    byoc_env["output_path"] = os.path.join("/opt/ml/processing/output")
    byoc_env["publish_cloudwatch_metrics"] = "Disabled"

    my_attached_monitor = ModelMonitor.attach(
        monitor_schedule_name=byoc_monitoring_schedule_name, sagemaker_session=sagemaker_session
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        INTEG_TEST_MONITORING_OUTPUT_BUCKET,
        str(uuid.uuid4()),
    )

    my_attached_monitor.run_baseline(
        baseline_inputs=[
            ProcessingInput(
                source=baseline_dataset,
                destination="/opt/ml/processing/input/baseline_dataset_input",
            )
        ],
        output=ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        wait=True,
        logs=False,
    )

    baselining_job_description = my_attached_monitor.latest_baselining_job.describe()

    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        == INSTANCE_TYPE
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
        == INSTANCE_COUNT
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"]
        == VOLUME_SIZE_IN_GB
    )
    assert (
        baselining_job_description["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"]
        == volume_kms_key
    )
    assert DEFAULT_IMAGE_SUFFIX in baselining_job_description["AppSpecification"]["ImageUri"]
    assert ROLE in baselining_job_description["RoleArn"]
    assert baselining_job_description["ProcessingInputs"][0]["InputName"] == "input-1"
    assert (
        baselining_job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
        == "output-1"
    )
    assert baselining_job_description["ProcessingOutputConfig"]["KmsKeyId"] == output_kms_key
    assert baselining_job_description["Environment"][ENV_KEY_1] == ENV_VALUE_1
    assert baselining_job_description["Environment"]["output_path"] == "/opt/ml/processing/output"
    assert (
        baselining_job_description["Environment"]["dataset_source"]
        == "/opt/ml/processing/input/baseline_dataset_input"
    )
    assert (
        baselining_job_description["StoppingCondition"]["MaxRuntimeInSeconds"]
        == MAX_RUNTIME_IN_SECONDS
    )
    assert (
        baselining_job_description["NetworkConfig"]["EnableNetworkIsolation"]
        == NETWORK_CONFIG.enable_network_isolation
    )

    statistics = my_attached_monitor.baseline_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    constraints = my_attached_monitor.suggested_constraints()
    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Enabled"

    constraints.set_monitoring(enable_monitoring=False)

    assert constraints.body_dict["monitoring_config"]["evaluate_constraints"] == "Disabled"

    constraints.save()

    byoc_env.update(UPDATED_ENVIRONMENT)

    my_attached_monitor.update_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output=MonitoringOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        instance_count=UPDATED_INSTANCE_COUNT,
        instance_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env=byoc_env,
        network_config=UPDATED_NETWORK_CONFIG,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(my_attached_monitor)

    schedule_description = my_attached_monitor.describe_schedule()
    _verify_default_monitoring_schedule(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        cron_expression=CronExpressionGenerator.hourly(),
        statistics=statistics,
        constraints=constraints,
        instant_count=UPDATED_INSTANCE_COUNT,
        instant_type=UPDATED_INSTANCE_TYPE,
        volume_size_in_gb=UPDATED_VOLUME_SIZE_IN_GB,
        volume_kms_key=updated_volume_kms_key,
        output_kms_key=updated_output_kms_key,
        publish_cloudwatch_metrics="Disabled",
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
        env_key=UPDATED_ENV_KEY_1,
        env_value=UPDATED_ENV_VALUE_1,
        network_config=UPDATED_NETWORK_CONFIG,
        role=UPDATED_ROLE,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_attached_monitor)

    _check_processing_logs_generated(
        monitor=my_attached_monitor, schedule_description=schedule_description, capfd=capfd
    )

    my_attached_monitor.stop_monitoring_schedule()

    _wait_for_schedule_changes_to_apply(monitor=my_attached_monitor)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
@pytest.mark.cron
def test_byoc_monitor_monitoring_execution_interactions(
    sagemaker_session, byoc_monitoring_schedule_name
):
    my_attached_monitor = ModelMonitor.attach(
        monitor_schedule_name=byoc_monitoring_schedule_name, sagemaker_session=sagemaker_session
    )
    description = my_attached_monitor.describe_schedule()
    assert description["MonitoringScheduleName"] == byoc_monitoring_schedule_name

    executions = my_attached_monitor.list_executions()
    assert len(executions) > 0

    with open(os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"), "r") as f:
        file_body = f.read()

    file_name = "statistics.json"
    desired_s3_uri = os.path.join(executions[-1].output.destination, file_name)

    S3Uploader.upload_string_as_file_body(
        body=file_body, desired_s3_uri=desired_s3_uri, sagemaker_session=sagemaker_session
    )

    statistics = my_attached_monitor.latest_monitoring_statistics()
    assert statistics.body_dict["dataset"]["item_count"] == 418

    with open(os.path.join(tests.integ.DATA_DIR, "monitor/constraint_violations.json"), "r") as f:
        file_body = f.read()

    file_name = "constraint_violations.json"
    desired_s3_uri = os.path.join(executions[-1].output.destination, file_name)

    S3Uploader.upload_string_as_file_body(
        body=file_body, desired_s3_uri=desired_s3_uri, sagemaker_session=sagemaker_session
    )

    constraint_violations = my_attached_monitor.latest_monitoring_constraint_violations()
    assert constraint_violations.body_dict["violations"][0]["feature_name"] == "store_and_fwd_flag"


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_default_monitor_monitoring_alerts(sagemaker_session, predictor):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    my_default_monitor.create_monitoring_schedule(
        endpoint_input=predictor.endpoint_name,
        output_s3_uri=output_s3_uri,
        schedule_cron_expression=HOURLY_CRON_EXPRESSION,
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )
    # No need to wait for execution
    description = my_default_monitor.describe_schedule()
    assert description["MonitoringScheduleName"] == my_default_monitor.monitoring_schedule_name
    # list alerts
    alerts, _ = my_default_monitor.list_monitoring_alerts(max_results=10)
    # we know each monitoring schedule will have a default alert
    assert len(alerts) > 0
    alert: MonitoringAlertSummary = alerts[0]
    alert_name = alert.alert_name
    alert_status = alert.alert_status

    my_default_monitor.update_monitoring_alert(
        alert_name, data_points_to_alert=3, evaluation_period=5
    )
    alerts, _ = my_default_monitor.list_monitoring_alerts(max_results=10)
    alert: MonitoringAlertSummary = alerts[0]
    assert len(alerts) > 0
    assert alert.alert_name == alert_name
    assert alert.alert_status == alert_status
    assert alert.data_points_to_alert == 3
    assert alert.evaluation_period == 5

    my_default_monitor.list_monitoring_alert_history(monitoring_alert_name=alert.alert_name)

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    my_default_monitor.stop_monitoring_schedule()
    my_default_monitor.delete_monitoring_schedule()


def _check_processing_logs_generated(monitor, schedule_description, capfd):
    monitor.get_latest_execution_logs(wait=False)
    out, _ = capfd.readouterr()
    assert len(out) > 0
    assert schedule_description.get("LastMonitoringExecutionSummary")["ProcessingJobArn"] in out


def _wait_for_schedule_changes_to_apply(monitor):
    """Waits for the monitor to no longer be in the 'Pending' state. Updates take under a minute
    to apply.

    Args:
        monitor (sagemaker.model_monitor.ModelMonitor): The monitor to watch.

    """
    for _ in retries(
        max_retry_count=100,
        exception_message_prefix="Waiting for schedule to leave 'Pending' status",
        seconds_to_sleep=5,
    ):
        schedule_desc = monitor.describe_schedule()
        if schedule_desc["MonitoringScheduleStatus"] != "Pending":
            break


def _predict_while_waiting_for_first_monitoring_job_to_complete(predictor, monitor):
    """Waits for the schedule to have an execution in a terminal status.

    Args:
        monitor (sagemaker.model_monitor.ModelMonitor): The monitor to watch.

    """
    for _ in retries(
        max_retry_count=200,
        exception_message_prefix="Waiting for the latest execution to be in a terminal status.",
        seconds_to_sleep=50,
    ):
        predictor.predict({"instances": [1.0, 2.0, 5.0]})
        schedule_desc = monitor.describe_schedule()
        execution_summary = schedule_desc.get("LastMonitoringExecutionSummary")
        last_execution_status = None

        # Once there is an execution, get its status
        if execution_summary is not None:
            last_execution_status = execution_summary["MonitoringExecutionStatus"]
            # Stop the schedule as soon as it's kicked off the execution that we need from it.
            if schedule_desc["MonitoringScheduleStatus"] not in ["Pending", "Stopped"]:
                monitor.stop_monitoring_schedule()
        # End this loop once the execution has reached a terminal state.
        if last_execution_status in ["Completed", "CompletedWithViolations", "Failed", "Stopped"]:
            break


def _upload_captured_data_to_endpoint(sagemaker_session, predictor):
    current_hour_date_time = datetime.now()
    previous_hour_date_time = datetime.now() - timedelta(hours=1)
    current_hour_folder_structure = current_hour_date_time.strftime("%Y/%m/%d/%H")
    previous_hour_folder_structure = previous_hour_date_time.strftime("%Y/%m/%d/%H")
    s3_uri_base = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        _MODEL_MONITOR_S3_PATH,
        _DATA_CAPTURE_S3_PATH,
        predictor.endpoint_name,
        "AllTraffic",
    )
    s3_uri_previous_hour = os.path.join(s3_uri_base, previous_hour_folder_structure)
    s3_uri_current_hour = os.path.join(s3_uri_base, current_hour_folder_structure)
    S3Uploader.upload(
        local_path=os.path.join(DATA_DIR, "monitor/captured-data.jsonl"),
        desired_s3_uri=s3_uri_previous_hour,
        sagemaker_session=sagemaker_session,
    )
    S3Uploader.upload(
        local_path=os.path.join(DATA_DIR, "monitor/captured-data.jsonl"),
        desired_s3_uri=s3_uri_current_hour,
        sagemaker_session=sagemaker_session,
    )


def _verify_default_monitoring_schedule(
    sagemaker_session,
    schedule_description,
    cron_expression=CronExpressionGenerator.daily(),
    statistics=None,
    constraints=None,
    output_kms_key=None,
    volume_kms_key=None,
    instant_count=INSTANCE_COUNT,
    instant_type=INSTANCE_TYPE,
    volume_size_in_gb=VOLUME_SIZE_IN_GB,
    network_config=None,
    max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
    publish_cloudwatch_metrics="Enabled",
    env_key=ENV_KEY_1,
    env_value=ENV_VALUE_1,
    preprocessor=None,
    postprocessor=None,
    role=ROLE,
):
    assert (
        schedule_description["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
        == cron_expression
    )
    if schedule_description.get("MonitoringType") is not None:
        assert schedule_description["MonitoringType"] == "DataQuality"
    else:
        assert schedule_description["MonitoringScheduleConfig"]["MonitoringType"] == "DataQuality"
    job_definition_name = schedule_description["MonitoringScheduleConfig"].get(
        "MonitoringJobDefinitionName"
    )
    if job_definition_name:
        job_desc = sagemaker_session.sagemaker_client.describe_data_quality_job_definition(
            JobDefinitionName=job_definition_name,
        )
        # app specification
        app_specification = job_desc["DataQualityAppSpecification"]
        env = app_specification["Environment"]
        baseline_config = job_desc.get("DataQualityBaselineConfig")
        job_input = job_desc["DataQualityJobInput"]
        job_output_config = job_desc["DataQualityJobOutputConfig"]
        client_config = job_desc["JobResources"]["ClusterConfig"]
    else:
        job_desc = schedule_description["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        app_specification = job_desc["MonitoringAppSpecification"]
        env = job_desc["Environment"]
        baseline_config = job_desc.get("BaselineConfig")
        job_input = job_desc["MonitoringInputs"][0]
        job_output_config = job_desc["MonitoringOutputConfig"]
        client_config = job_desc["MonitoringResources"]["ClusterConfig"]

    assert DEFAULT_IMAGE_SUFFIX in app_specification["ImageUri"]
    if env.get(env_key):
        assert env[env_key] == env_value
    assert env["publish_cloudwatch_metrics"] == publish_cloudwatch_metrics
    assert app_specification.get("RecordPreprocessorSourceUri") == preprocessor
    assert app_specification.get("PostAnalyticsProcessorSourceUri") == postprocessor

    # baseline
    if baseline_config:
        if baseline_config["StatisticsResource"]:
            assert baseline_config["StatisticsResource"]["S3Uri"] == statistics.file_s3_uri
        else:
            assert statistics is None
        if baseline_config["ConstraintsResource"]:
            assert baseline_config["ConstraintsResource"]["S3Uri"] == constraints.file_s3_uri
        else:
            assert constraints is None
    else:
        assert statistics is None
        assert constraints is None
    # job input
    assert "sagemaker-tensorflow-serving" in job_input["EndpointInput"]["EndpointName"]
    # job output config
    assert len(job_output_config["MonitoringOutputs"]) == 1
    assert job_output_config.get("KmsKeyId") == output_kms_key
    # job resources
    assert client_config["InstanceCount"] == instant_count
    assert client_config["InstanceType"] == instant_type
    assert client_config["VolumeSizeInGB"] == volume_size_in_gb
    assert client_config.get("VolumeKmsKeyId") == volume_kms_key
    # role
    assert role in job_desc["RoleArn"]
    # stop condition
    assert job_desc["StoppingCondition"]["MaxRuntimeInSeconds"] == max_runtime_in_seconds
    # network config
    if job_desc.get("NetworkConfig"):
        assert (
            job_desc["NetworkConfig"].get("EnableNetworkIsolation")
            == network_config.enable_network_isolation
        )
    else:
        assert network_config is None


def _verify_default_monitoring_schedule_with_batch_transform(
    sagemaker_session,
    schedule_description,
    cron_expression=CronExpressionGenerator.daily(),
    statistics=None,
    constraints=None,
    output_kms_key=None,
    volume_kms_key=None,
    instant_count=INSTANCE_COUNT,
    instant_type=INSTANCE_TYPE,
    volume_size_in_gb=VOLUME_SIZE_IN_GB,
    network_config=None,
    max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
    publish_cloudwatch_metrics="Enabled",
    env_key=ENV_KEY_1,
    env_value=ENV_VALUE_1,
    preprocessor=None,
    postprocessor=None,
    role=ROLE,
):
    assert (
        schedule_description["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
        == cron_expression
    )
    assert schedule_description["MonitoringScheduleConfig"]["MonitoringType"] == "DataQuality"

    job_definition_name = schedule_description["MonitoringScheduleConfig"].get(
        "MonitoringJobDefinitionName"
    )
    if job_definition_name:
        job_desc = sagemaker_session.sagemaker_client.describe_data_quality_job_definition(
            JobDefinitionName=job_definition_name,
        )
        # app specification
        app_specification = job_desc["DataQualityAppSpecification"]
        env = app_specification["Environment"]
        baseline_config = job_desc.get("DataQualityBaselineConfig")
        job_input = job_desc["DataQualityJobInput"]
        job_output_config = job_desc["DataQualityJobOutputConfig"]
        client_config = job_desc["JobResources"]["ClusterConfig"]
    else:
        job_desc = schedule_description["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        app_specification = job_desc["MonitoringAppSpecification"]
        env = job_desc["Environment"]
        baseline_config = job_desc.get("BaselineConfig")
        job_input = job_desc["MonitoringInputs"][0]
        job_output_config = job_desc["MonitoringOutputConfig"]
        client_config = job_desc["MonitoringResources"]["ClusterConfig"]

    assert DEFAULT_IMAGE_SUFFIX in app_specification["ImageUri"]
    if env.get(env_key):
        assert env[env_key] == env_value
    assert env["publish_cloudwatch_metrics"] == publish_cloudwatch_metrics
    assert app_specification.get("RecordPreprocessorSourceUri") == preprocessor
    assert app_specification.get("PostAnalyticsProcessorSourceUri") == postprocessor

    # baseline
    if baseline_config:
        if baseline_config["StatisticsResource"]:
            assert baseline_config["StatisticsResource"]["S3Uri"] == statistics.file_s3_uri
        else:
            assert statistics is None
        if baseline_config["ConstraintsResource"]:
            assert baseline_config["ConstraintsResource"]["S3Uri"] == constraints.file_s3_uri
        else:
            assert constraints is None
    else:
        assert statistics is None
        assert constraints is None
    # job input
    assert (
        "sagemaker-tensorflow-serving"
        in job_input["BatchTransformInput"]["DataCapturedDestinationS3Uri"]
    )
    # job output config
    assert len(job_output_config["MonitoringOutputs"]) == 1
    assert job_output_config.get("KmsKeyId") == output_kms_key
    # job resources
    assert client_config["InstanceCount"] == instant_count
    assert client_config["InstanceType"] == instant_type
    assert client_config["VolumeSizeInGB"] == volume_size_in_gb
    assert client_config.get("VolumeKmsKeyId") == volume_kms_key
    # role
    assert role in job_desc["RoleArn"]
    # stop condition
    assert job_desc["StoppingCondition"]["MaxRuntimeInSeconds"] == max_runtime_in_seconds
    # network config
    if job_desc.get("NetworkConfig"):
        assert (
            job_desc["NetworkConfig"].get("EnableNetworkIsolation")
            == network_config.enable_network_isolation
        )
    else:
        assert network_config is None


def test_default_update_monitoring_batch_transform(
    sagemaker_session, output_kms_key, volume_kms_key
):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    my_default_monitor.create_monitoring_schedule(
        batch_transform_input=batch_transform_input,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=HOURLY_CRON_EXPRESSION,
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-tensorflow-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    my_default_monitor.update_monitoring_schedule(
        batch_transform_input=batch_transform_input,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

    schedule_description = my_default_monitor.describe_schedule()

    _verify_default_monitoring_schedule_with_batch_transform(
        sagemaker_session=sagemaker_session,
        schedule_description=schedule_description,
        cron_expression=HOURLY_CRON_EXPRESSION,
        statistics=statistics,
        constraints=constraints,
        output_kms_key=output_kms_key,
        volume_kms_key=volume_kms_key,
        network_config=NETWORK_CONFIG,
    )

    my_default_monitor.stop_monitoring_schedule()
    my_default_monitor.delete_monitoring_schedule()


def test_byoc_monitoring_schedule_name_update_batch(
    sagemaker_session, output_kms_key, volume_kms_key
):
    byoc_env = ENVIRONMENT.copy()
    byoc_env["dataset_format"] = json.dumps(DatasetFormat.csv(header=False))
    byoc_env["dataset_source"] = "/opt/ml/processing/input/baseline_dataset_input"
    byoc_env["output_path"] = os.path.join("/opt/ml/processing/output")
    byoc_env["publish_cloudwatch_metrics"] = "Disabled"

    my_byoc_monitor = ModelMonitor(
        role=ROLE,
        image_uri=DefaultModelMonitor._get_default_image_uri(
            sagemaker_session.boto_session.region_name
        ),
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=byoc_env,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    my_byoc_monitor.create_monitoring_schedule(
        endpoint_input=batch_transform_input,
        output=MonitoringOutput(source="/opt/ml/processing/output", destination=output_s3_uri),
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=HOURLY_CRON_EXPRESSION,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-tensorflow-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    my_byoc_monitor.update_monitoring_schedule(
        batch_transform_input=batch_transform_input,
    )

    _wait_for_schedule_changes_to_apply(monitor=my_byoc_monitor)

    schedule_description = my_byoc_monitor.describe_schedule()

    assert (
        data_captured_destination_s3_uri
        == schedule_description["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringInputs"
        ][0]["BatchTransformInput"]["DataCapturedDestinationS3Uri"]
    )

    my_byoc_monitor.stop_monitoring_schedule()
    my_byoc_monitor.delete_monitoring_schedule()


def _wait_for_first_execution_to_complete(monitor):
    for _ in retries(
        max_retry_count=30,
        exception_message_prefix="Waiting for first execution to reach terminal state",
        seconds_to_sleep=60,
    ):
        schedule_desc = monitor.describe_schedule()
        execution_summary = schedule_desc.get("LastMonitoringExecutionSummary")

        # Once there is an execution, we can break
        if execution_summary is not None:
            monitor.stop_monitoring_schedule()
            last_execution_status = execution_summary["MonitoringExecutionStatus"]

            if last_execution_status not in ["Pending", "InProgess"]:
                # to prevent delete as it takes sometime for schedule to update out of InProgress
                time.sleep(120)
                break


def _wait_for_first_execution_to_start(monitor):
    for _ in retries(
        max_retry_count=30,
        exception_message_prefix="Waiting for an execution for the schedule to start",
        seconds_to_sleep=60,
    ):
        schedule_desc = monitor.describe_schedule()
        execution_summary = schedule_desc.get("LastMonitoringExecutionSummary")

        # Once there is an execution, we can break
        if execution_summary is not None:
            monitor.stop_monitoring_schedule()
            break


def test_one_time_monitoring_schedule(sagemaker_session):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    # storing the time before creation of monitoring schedule
    start = datetime.now()

    my_default_monitor.create_monitoring_schedule(
        batch_transform_input=batch_transform_input,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.now(),
        data_analysis_start_time="-PT1H",
        data_analysis_end_time="-PT0H",
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )

    try:

        _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)
        _wait_for_first_execution_to_start(monitor=my_default_monitor)

        # storing the start time as soon as an execution starts
        end = datetime.now()

        _wait_for_first_execution_to_complete(monitor=my_default_monitor)
        my_default_monitor.stop_monitoring_schedule()
        my_default_monitor.delete_monitoring_schedule()

        # the execution should start within 10 minutes
        assert (end - start).total_seconds() < 10 * 60

    except Exception as e:
        my_default_monitor.stop_monitoring_schedule()
        my_default_monitor.delete_monitoring_schedule()
        raise e


def test_create_monitoring_schedule_with_validation_error(sagemaker_session):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    output_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-monitoring-output-bucket",
        str(uuid.uuid4()),
    )

    data_captured_destination_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "sagemaker-serving-batch-transform",
        str(uuid.uuid4()),
    )

    batch_transform_input = BatchTransformInput(
        data_captured_destination_s3_uri=data_captured_destination_s3_uri,
        destination="/opt/ml/processing/output",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    )

    statistics = Statistics.from_file_path(
        statistics_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/statistics.json"),
        sagemaker_session=sagemaker_session,
    )

    constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(tests.integ.DATA_DIR, "monitor/constraints.json"),
        sagemaker_session=sagemaker_session,
    )

    try:
        my_default_monitor.create_monitoring_schedule(
            monitor_schedule_name="schedule-name-more-than-63-characters-to-get-a-validation-exception",
            batch_transform_input=batch_transform_input,
            output_s3_uri=output_s3_uri,
            statistics=statistics,
            constraints=constraints,
            schedule_cron_expression=CronExpressionGenerator.now(),
            data_analysis_start_time="-PT1H",
            data_analysis_end_time="-PT0H",
            enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
        )
    except Exception as e:
        assert "ValidationException" in str(e)

    my_default_monitor.create_monitoring_schedule(
        monitor_schedule_name=unique_name_from_base("valid-schedule-name"),
        batch_transform_input=batch_transform_input,
        output_s3_uri=output_s3_uri,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.now(),
        data_analysis_start_time="-PT1H",
        data_analysis_end_time="-PT0H",
        enable_cloudwatch_metrics=ENABLE_CLOUDWATCH_METRICS,
    )
    try:

        _wait_for_schedule_changes_to_apply(monitor=my_default_monitor)

        my_default_monitor.stop_monitoring_schedule()
        my_default_monitor.delete_monitoring_schedule()

    except Exception as e:
        my_default_monitor.stop_monitoring_schedule()
        my_default_monitor.delete_monitoring_schedule()
        raise e
