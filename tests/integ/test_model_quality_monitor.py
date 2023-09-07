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

import tempfile
from datetime import datetime, timedelta
import json
import os
import numpy as np
import pytest
import pandas as pd

import tests.integ
import tests.integ.timeout

from sagemaker import image_uris, utils
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DataCaptureConfig,
    EndpointInput,
    ModelQualityMonitor,
    DatasetFormat,
)
from sagemaker.s3 import S3Uploader
from sagemaker.utils import unique_name_from_base
from sagemaker.model import Model

from tests.integ import DATA_DIR
from tests.integ.retry import retries

XGBOOST_DATA_PATH = os.path.join(DATA_DIR, "xgboost_model")
ENDPOINT_INPUT_LOCAL_PATH = "/opt/ml/processing/input/endpoint"
PROBABILITY_THRESHOLD = 0.5005
PROBLEM_TYPE = "Regression"
INFERENCE_ATTRIBUTE = "0"
HEADER_OF_LABEL = "Label"
HEADER_OF_PREDICTED_LABEL = "Prediction"
HEADERS_OF_FEATURES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
ALL_HEADERS = [*HEADERS_OF_FEATURES, HEADER_OF_LABEL, HEADER_OF_PREDICTED_LABEL]

CRON = "cron(0 * * * ? *)"
UPDATED_CRON = CronExpressionGenerator.daily()
MAX_RUNTIME_IN_SECONDS = 30 * 60
UPDATED_MAX_RUNTIME_IN_SECONDS = 25 * 60
ROLE = "SageMakerRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c5.xlarge"
VOLUME_SIZE_IN_GB = 100
START_TIME_OFFSET = "-PT1H"
END_TIME_OFFSET = "-PT0H"
TEST_TAGS = [{"Key": "integration", "Value": "test"}]
# TODO: Remove this workaround once the API service fix is deployed to Prod
TEST_ENV = {"problem_type": PROBLEM_TYPE}


@pytest.yield_fixture(scope="module")
def endpoint_name(sagemaker_session):
    endpoint_name = unique_name_from_base("model-quality-monitor-integ")
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(XGBOOST_DATA_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    xgb_image = image_uris.retrieve(
        "xgboost", sagemaker_session.boto_region_name, version="1", image_scope="inference"
    )

    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=endpoint_name, sagemaker_session=sagemaker_session, hours=2
    ):
        xgb_model = Model(
            model_data=xgb_model_data,
            image_uri=xgb_image,
            name=endpoint_name,  # model name
            role=ROLE,
            sagemaker_session=sagemaker_session,
        )
        xgb_model.deploy(
            INSTANCE_COUNT,
            INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            data_capture_config=DataCaptureConfig(True, sagemaker_session=sagemaker_session),
        )
        yield endpoint_name


@pytest.fixture
def upload_actual_data(sagemaker_session, endpoint_name, ground_truth_input):
    _upload_actual_data(sagemaker_session, endpoint_name, ground_truth_input)


@pytest.yield_fixture(scope="module")
def data_path():
    # Generate 100 samples, each has 7 features and 1 label
    # feature value in range [0,50)
    # label value in range [0,2)
    # predicted label, float value in range [0,1)
    features = np.random.randint(50, size=(100, 7))
    label = np.random.randint(2, size=100)
    predicted_label = np.random.rand(100)
    data = pd.concat(
        [pd.DataFrame(features), pd.DataFrame(label), pd.DataFrame(predicted_label)],
        axis=1,
        sort=False,
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train.csv")
        data.to_csv(filename, index=False, header=ALL_HEADERS)
        yield filename


@pytest.fixture(scope="module")
def ground_truth_input(sagemaker_session, endpoint_name):
    return os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        endpoint_name,
        "IntegTestGroundtruth",
    )


@pytest.fixture
def model_quality_monitor(sagemaker_session):
    monitor = ModelQualityMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=TEST_ENV,
        tags=TEST_TAGS,
    )
    return monitor


@pytest.fixture
def scheduled_model_quality_monitor(
    sagemaker_session, model_quality_monitor, endpoint_name, ground_truth_input
):
    monitor_schedule_name = utils.unique_name_from_base("model-quality-monitor")
    s3_uri_monitoring_output = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        endpoint_name,
        monitor_schedule_name,
        "monitor_output",
    )
    # To include attributes
    endpoint_input = EndpointInput(
        endpoint_name=endpoint_name,
        destination=ENDPOINT_INPUT_LOCAL_PATH,
        start_time_offset=START_TIME_OFFSET,
        end_time_offset=END_TIME_OFFSET,
        inference_attribute=INFERENCE_ATTRIBUTE,
    )
    model_quality_monitor.create_monitoring_schedule(
        monitor_schedule_name=monitor_schedule_name,
        endpoint_input=endpoint_input,
        ground_truth_input=ground_truth_input,
        problem_type=PROBLEM_TYPE,
        output_s3_uri=s3_uri_monitoring_output,
        schedule_cron_expression=CRON,
    )
    return model_quality_monitor


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
def test_model_quality_monitor(
    sagemaker_session, scheduled_model_quality_monitor, endpoint_name, ground_truth_input
):
    monitor = scheduled_model_quality_monitor
    monitor._wait_for_schedule_changes_to_apply()

    # stop it as soon as possible to avoid any execution
    monitor.stop_monitoring_schedule()
    _verify_monitoring_schedule(
        monitor=monitor,
        schedule_status="Stopped",
    )
    _verify_model_quality_job_description(
        sagemaker_session=sagemaker_session,
        monitor=monitor,
        endpoint_name=endpoint_name,
        ground_truth_input=ground_truth_input,
    )

    # attach to schedule
    monitoring_schedule_name = monitor.monitoring_schedule_name
    job_definition_name = monitor.job_definition_name
    monitor = ModelQualityMonitor.attach(
        monitor_schedule_name=monitor.monitoring_schedule_name,
        sagemaker_session=sagemaker_session,
    )
    assert monitor.monitoring_schedule_name == monitoring_schedule_name
    assert monitor.job_definition_name == job_definition_name

    # update schedule
    monitor.update_monitoring_schedule(
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS, schedule_cron_expression=UPDATED_CRON
    )
    assert monitor.monitoring_schedule_name == monitoring_schedule_name
    assert monitor.job_definition_name != job_definition_name
    _verify_monitoring_schedule(
        monitor=monitor, schedule_status="Scheduled", schedule_cron_expression=UPDATED_CRON
    )
    _verify_model_quality_job_description(
        sagemaker_session=sagemaker_session,
        monitor=monitor,
        endpoint_name=endpoint_name,
        ground_truth_input=ground_truth_input,
        max_runtime_in_seconds=UPDATED_MAX_RUNTIME_IN_SECONDS,
    )

    # delete schedule
    monitor.delete_monitoring_schedule()


@pytest.mark.slow_test
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_run_model_quality_monitor(
    scheduled_model_quality_monitor,
    sagemaker_session,
    endpoint_name,
    ground_truth_input,
    upload_actual_data,
):
    monitor = scheduled_model_quality_monitor
    _verify_execution_status(monitor)

    _verify_model_quality_job_description(
        sagemaker_session=sagemaker_session,
        monitor=monitor,
        endpoint_name=endpoint_name,
        ground_truth_input=ground_truth_input,
    )

    monitor.delete_monitoring_schedule()


@pytest.mark.slow_test
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MODEL_MONITORING_REGIONS,
    reason="ModelMonitoring is not yet supported in this region.",
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_run_model_quality_monitor_baseline(
    sagemaker_session,
    endpoint_name,
    data_path,
    ground_truth_input,
    upload_actual_data,
):
    monitor = ModelQualityMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        sagemaker_session=sagemaker_session,
        env=TEST_ENV,
        tags=TEST_TAGS,
    )

    baselining_job_name = utils.unique_name_from_base("model-quality-baselining-job")
    print("Creating baselining job: {}".format(baselining_job_name))
    monitor.suggest_baseline(
        baseline_dataset=data_path,
        dataset_format=DatasetFormat.csv(),
        problem_type=PROBLEM_TYPE,
        job_name=baselining_job_name,
        ground_truth_attribute=HEADER_OF_LABEL,
        inference_attribute=HEADER_OF_PREDICTED_LABEL,
    )

    monitoring_schedule_name = utils.unique_name_from_base("model-quality-suggest-baseline")
    s3_uri_monitoring_output = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        endpoint_name,
        monitoring_schedule_name,
        "monitor_output",
    )
    monitor.create_monitoring_schedule(
        endpoint_input=EndpointInput(
            endpoint_name=endpoint_name,
            destination=ENDPOINT_INPUT_LOCAL_PATH,
            start_time_offset=START_TIME_OFFSET,
            end_time_offset=END_TIME_OFFSET,
            inference_attribute=INFERENCE_ATTRIBUTE,
        ),
        ground_truth_input=ground_truth_input,
        problem_type=PROBLEM_TYPE,
        output_s3_uri=s3_uri_monitoring_output,
        monitor_schedule_name=monitoring_schedule_name,
        schedule_cron_expression=CRON,
    )
    _verify_execution_status(monitor)

    _verify_model_quality_job_description(
        sagemaker_session=sagemaker_session,
        monitor=monitor,
        endpoint_name=endpoint_name,
        ground_truth_input=ground_truth_input,
    )

    monitor.delete_monitoring_schedule()


def _verify_monitoring_schedule(monitor, schedule_status, schedule_cron_expression=CRON):
    desc = monitor.describe_schedule()
    assert desc["MonitoringScheduleName"] == monitor.monitoring_schedule_name
    assert (
        desc["MonitoringScheduleConfig"]["MonitoringJobDefinitionName"]
        == monitor.job_definition_name
    )
    assert desc["MonitoringScheduleStatus"] == schedule_status
    assert (
        desc["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
        == schedule_cron_expression
    )


def _verify_model_quality_job_description(
    sagemaker_session,
    monitor,
    endpoint_name,
    ground_truth_input,
    max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
):
    job_desc = sagemaker_session.sagemaker_client.describe_model_quality_job_definition(
        JobDefinitionName=monitor.job_definition_name
    )
    _verify_job_description(
        sagemaker_session=sagemaker_session,
        monitoring_schedule_name=monitor.monitoring_schedule_name,
        job_desc=job_desc,
        job_input_key="ModelQualityJobInput",
        monitor_output_key="ModelQualityJobOutputConfig",
        endpoint_name=endpoint_name,
        max_runtime_in_seconds=max_runtime_in_seconds,
    )
    assert START_TIME_OFFSET == job_desc["ModelQualityJobInput"]["EndpointInput"]["StartTimeOffset"]
    assert END_TIME_OFFSET == job_desc["ModelQualityJobInput"]["EndpointInput"]["EndTimeOffset"]
    assert ground_truth_input == job_desc["ModelQualityJobInput"]["GroundTruthS3Input"]["S3Uri"]


def _verify_explainability_job_description(
    sagemaker_session, monitor, endpoint_name, max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS
):
    job_desc = sagemaker_session.sagemaker_client.describe_model_explainability_job_definition(
        JobDefinitionName=monitor.job_definition_name
    )
    _verify_job_description(
        sagemaker_session=sagemaker_session,
        monitoring_schedule_name=monitor.monitoring_schedule_name,
        job_desc=job_desc,
        job_input_key="ModelExplainabilityJobInput",
        monitor_output_key="ModelExplainabilityJobOutputConfig",
        endpoint_name=endpoint_name,
        max_runtime_in_seconds=max_runtime_in_seconds,
    )


def _verify_job_description(
    sagemaker_session,
    monitoring_schedule_name,
    job_desc,
    job_input_key,
    monitor_output_key,
    endpoint_name,
    max_runtime_in_seconds,
):
    assert max_runtime_in_seconds == job_desc["StoppingCondition"]["MaxRuntimeInSeconds"]

    assert endpoint_name == job_desc[job_input_key]["EndpointInput"]["EndpointName"]

    assert INSTANCE_TYPE == job_desc["JobResources"]["ClusterConfig"]["InstanceType"]
    assert INSTANCE_COUNT == job_desc["JobResources"]["ClusterConfig"]["InstanceCount"]
    assert VOLUME_SIZE_IN_GB == job_desc["JobResources"]["ClusterConfig"]["VolumeSizeInGB"]
    assert "VolumeKmsKeyId" not in job_desc["JobResources"]["ClusterConfig"]
    assert 1 == len(job_desc[monitor_output_key]["MonitoringOutputs"])
    assert (
        "/opt/ml/processing/output"
        == job_desc[monitor_output_key]["MonitoringOutputs"][0]["S3Output"]["LocalPath"]
    )
    assert (
        "Continuous"
        == job_desc[monitor_output_key]["MonitoringOutputs"][0]["S3Output"]["S3UploadMode"]
    )
    s3_uri_monitoring_output = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        endpoint_name,
        monitoring_schedule_name,
        "monitor_output",
    )
    assert (
        s3_uri_monitoring_output
        == job_desc[monitor_output_key]["MonitoringOutputs"][0]["S3Output"]["S3Uri"]
    )


def _verify_execution_status(monitor):
    _wait_for_completion(monitor)
    executions = monitor.list_executions()
    assert len(executions) > 0
    schedule_desc = monitor.describe_schedule()
    execution_summary = schedule_desc.get("LastMonitoringExecutionSummary")
    last_execution_status = execution_summary["MonitoringExecutionStatus"]
    assert last_execution_status in ["Completed", "CompletedWithViolations"]


def _upload_actual_data(sagemaker_session, endpoint_name, actuals_s3_uri_base):
    current_hour_date_time = datetime.utcnow()
    previous_hour_date_time = current_hour_date_time - timedelta(hours=1)
    captures = os.path.join(DATA_DIR, "monitor/capture.jsonl")
    actuals = os.path.join(DATA_DIR, "monitor/actuals.jsonl")

    def _upload(s3_uri_base, input_file_name, target_time, file_name):
        time_folder = target_time.strftime("%Y/%m/%d/%H")
        time_str = str(target_time.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        s3_uri = os.path.join(s3_uri_base, time_folder, file_name)

        up_to_date_lines = []
        with open(input_file_name, "r") as input_file:
            for line in input_file:
                json_l = json.loads(line)
                json_l["eventMetadata"]["inferenceTime"] = time_str
                up_to_date_lines.append(json.dumps(json_l))

        file_target = "\n".join(up_to_date_lines)

        return S3Uploader.upload_string_as_file_body(
            file_target,
            desired_s3_uri=s3_uri,
            sagemaker_session=sagemaker_session,
        )

    capture_s3_uri_base = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        sagemaker_session.default_bucket_prefix,
        "model-monitor",
        "data-capture",
        endpoint_name,
        "AllTraffic",
    )

    capture_path_now = _upload(
        capture_s3_uri_base, captures, current_hour_date_time, "capture_data.jsonl"
    )
    capture_path = _upload(
        capture_s3_uri_base, captures, previous_hour_date_time, "capture_data.jsonl"
    )
    print("Uploading captured data to {} and {}.".format(capture_path_now, capture_path))

    if actuals_s3_uri_base is not None:
        _upload(actuals_s3_uri_base, actuals, current_hour_date_time, "actuals_data.jsonl")
        _upload(actuals_s3_uri_base, actuals, previous_hour_date_time, "actuals_data.jsonl")
        print("Uploading actuals data to {}.".format(actuals_s3_uri_base))
    return actuals_s3_uri_base


def _wait_for_completion(monitor):
    """Waits for the schedule to have an execution in a terminal status.

    Args:
        monitor (sagemaker.model_monitor.ModelMonitor): The monitor to watch.

    """
    for _ in retries(
        max_retry_count=200,
        exception_message_prefix="Waiting for the latest execution to be in a terminal status.",
        seconds_to_sleep=60,
    ):
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
