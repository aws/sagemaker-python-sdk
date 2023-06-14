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

from datetime import datetime
from typing import Callable

import pytest
import json
from botocore.exceptions import ClientError
from mock import Mock, patch
from sagemaker.remote_function.spark_config import SparkConfig

from sagemaker import Session
from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.feature_store.feature_processor._constants import (
    FEATURE_PROCESSOR_TAG_KEY,
    FEATURE_PROCESSOR_TAG_VALUE,
    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT,
)
from sagemaker.feature_store.feature_processor.feature_scheduler import (
    schedule,
    to_pipeline,
    execute,
    delete_schedule,
    describe,
    list_schedules,
)
from sagemaker.remote_function.job import (
    _JobSettings,
    SPARK_APP_SCRIPT_PATH,
    RUNTIME_SCRIPTS_CHANNEL_NAME,
    REMOTE_FUNCTION_WORKSPACE,
    SAGEMAKER_WHL_CHANNEL_NAME,
    SPARK_CONF_WORKSPACE,
    ENTRYPOINT_SCRIPT_NAME,
)
from sagemaker.workflow.parameters import Parameter, ParameterTypeEnum
from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
    SageMakerJobExceptionTypeEnum,
)

REGION = "us-west-2"
IMAGE = "image_uri"
BUCKET = "my-s3-bucket"
S3_URI = f"s3://{BUCKET}/keyprefix"
DEFAULT_IMAGE = (
    "153931337802.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:3.2-cpu-py39-v1.1"
)
PIPELINE_ARN = "pipeline_arn"
SCHEDULE_ARN = "schedule_arn"
SCHEDULE_ROLE_ARN = "my_schedule_role_arn"
EXECUTION_ROLE_ARN = "my_execution_role_arn"
VALID_SCHEDULE_STATE = "ENABLED"
INVALID_SCHEDULE_STATE = "invalid"
TEST_REGION = "us-west-2"
SAGEMAKER_SDK_WHL_FILE = (
    "s3://sagemaker-pathways/beta/pysdk/sagemaker-2.132.1.dev0-py2.py3-none-any.whl"
)
NOW = datetime.now()


def mock_session():
    session = Mock()
    session.default_bucket.return_value = BUCKET
    session.expand_role.return_value = EXECUTION_ROLE_ARN
    session.boto_region_name = TEST_REGION
    session.sagemaker_config = None
    session._append_sagemaker_config_tags.return_value = []
    session.default_bucket_prefix = None
    return session


def mock_pipeline():
    pipeline = Mock()
    pipeline.describe.return_value = {"PipelineArn": PIPELINE_ARN}
    pipeline.upsert.return_value = None
    return pipeline


def mock_event_bridge_scheduler_helper():
    helper = Mock()
    helper.upsert_schedule.return_value = SCHEDULE_ARN
    helper.delete_schedule.return_value = None
    helper.describe_schedule.return_value = {
        "Arn": "some_schedule_arn",
        "ScheduleExpression": "some_schedule_expression",
        "StartDate": NOW,
        "State": VALID_SCHEDULE_STATE,
        "Target": {"Arn": "some_pipeline_arn", "RoleArn": "some_schedule_role_arn"},
    }
    return helper


@pytest.fixture
def job_function():
    return Mock(Callable)


@pytest.fixture
def config_uploader():
    uploader = Mock()
    uploader.return_value = "some_s3_uri"
    uploader.prepare_and_upload_runtime_scripts.return_value = "some_s3_uri"
    return uploader


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.Pipeline",
    return_value=mock_pipeline(),
)
@patch(
    "sagemaker.remote_function.job._JobSettings._get_default_spark_image",
    return_value="some_image_uri",
)
@patch("sagemaker.feature_store.feature_processor._config_uploader.TrainingInput")
@patch("sagemaker.feature_store.feature_processor.feature_scheduler.TrainingStep")
@patch("sagemaker.feature_store.feature_processor.feature_scheduler.Estimator")
@patch(
    "sagemaker.feature_store.feature_processor._config_uploader.ConfigUploader"
    "._prepare_and_upload_spark_dependent_files",
    return_value=("path_a", "path_b", "path_c", "path_d"),
)
@patch(
    "sagemaker.feature_store.feature_processor._config_uploader.ConfigUploader._prepare_and_upload_dependencies",
    return_value="some_s3_uri",
)
@patch(
    "sagemaker.feature_store.feature_processor._config_uploader.ConfigUploader._prepare_and_upload_runtime_scripts",
    return_value="some_s3_uri",
)
@patch("sagemaker.feature_store.feature_processor.feature_scheduler.RuntimeEnvironmentManager")
@patch(
    "sagemaker.feature_store.feature_processor._config_uploader.ConfigUploader._prepare_and_upload_callable"
)
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_to_pipeline(
    get_execution_role,
    session,
    mock_upload_callable,
    mock_runtime_manager,
    mock_script_upload,
    mock_dependency_upload,
    mock_spark_dependency_upload,
    mock_estimator,
    mock_training_step,
    mock_training_input,
    mock_spark_image,
    pipeline,
):
    session.sagemaker_config = None
    session.boto_region_name = TEST_REGION
    session.expand_role.return_value = EXECUTION_ROLE_ARN
    spark_config = SparkConfig(submit_files=["file_a", "file_b", "file_c"])
    job_settings = _JobSettings(
        spark_config=spark_config,
        s3_root_uri=S3_URI,
        role=EXECUTION_ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
        sagemaker_session=session,
    )
    jobs_container_entrypoint = [
        "/bin/bash",
        f"/opt/ml/input/data/{RUNTIME_SCRIPTS_CHANNEL_NAME}/{ENTRYPOINT_SCRIPT_NAME}",
    ]
    jobs_container_entrypoint.extend(["--jars", "path_a"])
    jobs_container_entrypoint.extend(["--py-files", "path_b"])
    jobs_container_entrypoint.extend(["--files", "path_c"])
    jobs_container_entrypoint.extend([SPARK_APP_SCRIPT_PATH])
    container_args = ["--s3_base_uri", f"{S3_URI}/pipeline_name"]
    container_args.extend(["--region", session.boto_region_name])

    mock_feature_processor_config = Mock(mode=FeatureProcessorMode.PYSPARK)
    mock_feature_processor_config.mode.return_value = FeatureProcessorMode.PYSPARK

    wrapped_func = Mock(
        Callable,
        feature_processor_config=mock_feature_processor_config,
        job_settings=job_settings,
        wrapped_func=job_function,
    )
    wrapped_func.feature_processor_config.return_value = mock_feature_processor_config
    wrapped_func.job_settings.return_value = job_settings
    wrapped_func.wrapped_func.return_value = job_function

    pipeline_arn = to_pipeline(
        pipeline_name="pipeline_name",
        step=wrapped_func,
        role=EXECUTION_ROLE_ARN,
        max_retries=1,
        sagemaker_session=session,
    )
    assert pipeline_arn == PIPELINE_ARN

    assert mock_upload_callable.called_once_with(job_function)
    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()
    container_args.extend(["--client_python_version", mock_python_version])

    mock_script_upload.assert_called_once_with(
        spark_config,
        f"{S3_URI}/pipeline_name",
        None,
        session,
    )

    mock_dependency_upload.assert_called_once_with(
        local_dependencies_path,
        True,
        f"{S3_URI}/pipeline_name",
        None,
        session,
    )

    mock_spark_dependency_upload.assert_called_once_with(
        spark_config,
        f"{S3_URI}/pipeline_name",
        None,
        session,
    )

    mock_estimator.assert_called_once_with(
        role=EXECUTION_ROLE_ARN,
        max_run=86400,
        max_retry_attempts=1,
        output_path=f"{S3_URI}/pipeline_name",
        output_kms_key=None,
        image_uri="some_image_uri",
        input_mode="File",
        environment={
            "AWS_DEFAULT_REGION": "us-west-2",
            "REMOTE_FUNCTION_SECRET_KEY": job_settings.environment_variables[
                "REMOTE_FUNCTION_SECRET_KEY"
            ],
            "scheduled_time": Parameter(
                name="scheduled_time", parameter_type=ParameterTypeEnum.STRING
            ),
        },
        volume_size=30,
        volume_kms_key=None,
        instance_count=1,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
        container_entry_point=jobs_container_entrypoint,
        container_arguments=container_args,
        tags=[],
    )

    mock_training_step.assert_called_once_with(
        name="-".join(["pipeline_name", "feature-processor"]),
        estimator=mock_estimator(),
        inputs={
            RUNTIME_SCRIPTS_CHANNEL_NAME: mock_training_input(
                s3_data=mock_script_upload.return_value, s3_data_type="S3Prefix"
            ),
            REMOTE_FUNCTION_WORKSPACE: mock_training_input(
                s3_data=f"{S3_URI}/pipeline_name/sm_rf_user_ws", s3_data_type="S3Prefix"
            ),
            SPARK_CONF_WORKSPACE: mock_training_input(s3_data="path_d", s3_data_type="S3Prefix"),
            SAGEMAKER_WHL_CHANNEL_NAME: mock_training_input(
                s3_data=SAGEMAKER_SDK_WHL_FILE, s3_data_type="S3Prefix"
            ),
        },
        retry_policies=[
            StepRetryPolicy(
                exception_types=[
                    StepExceptionTypeEnum.SERVICE_FAULT,
                    StepExceptionTypeEnum.THROTTLING,
                ],
                max_attempts=1,
            ),
            SageMakerJobStepRetryPolicy(
                exception_types=[
                    SageMakerJobExceptionTypeEnum.INTERNAL_ERROR,
                    SageMakerJobExceptionTypeEnum.CAPACITY_ERROR,
                    SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT,
                ],
                max_attempts=1,
            ),
        ],
    )

    pipeline.assert_called_once_with(
        name="pipeline_name",
        steps=[mock_training_step()],
        sagemaker_session=session,
        parameters=[Parameter(name="scheduled_time", parameter_type=ParameterTypeEnum.STRING)],
    )

    pipeline().upsert.assert_called_once_with(
        role_arn=EXECUTION_ROLE_ARN,
        tags=[dict(Key=FEATURE_PROCESSOR_TAG_KEY, Value=FEATURE_PROCESSOR_TAG_VALUE)],
    )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_to_pipeline_not_wrapped_by_feature_processor(get_execution_role, session):
    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=EXECUTION_ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
    )
    wrapped_func = Mock(
        Callable,
        job_settings=job_settings,
        wrapped_func=job_function,
    )
    wrapped_func.job_settings.return_value = job_settings
    wrapped_func.wrapped_func.return_value = job_function

    with pytest.raises(
        ValueError,
        match="Please wrap step parameter with feature_processor decorator in order to use to_pipeline API.",
    ):
        to_pipeline(
            pipeline_name="pipeline_name", step=wrapped_func, role=EXECUTION_ROLE_ARN, max_retries=1
        )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch(
    "sagemaker.remote_function.job._JobSettings._get_default_spark_image",
    return_value="some_image_uri",
)
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_to_pipeline_wrong_mode(get_execution_role, mock_spark_image, session):
    spark_config = SparkConfig(submit_files=["file_a", "file_b", "file_c"])
    job_settings = _JobSettings(
        spark_config=spark_config,
        s3_root_uri=S3_URI,
        role=EXECUTION_ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
    )
    jobs_container_entrypoint = [
        "/bin/bash",
        f"/opt/ml/input/data/{RUNTIME_SCRIPTS_CHANNEL_NAME}/{ENTRYPOINT_SCRIPT_NAME}",
    ]
    jobs_container_entrypoint.extend(["--jars", "path_a"])
    jobs_container_entrypoint.extend(["--py-files", "path_b"])
    jobs_container_entrypoint.extend(["--files", "path_c"])
    jobs_container_entrypoint.extend([SPARK_APP_SCRIPT_PATH])
    container_args = ["--s3_base_uri", f"{S3_URI}/pipeline_name"]
    container_args.extend(["--region", TEST_REGION])

    mock_feature_processor_config = Mock(mode=FeatureProcessorMode.PYTHON)
    mock_feature_processor_config.mode.return_value = FeatureProcessorMode.PYTHON

    wrapped_func = Mock(
        Callable,
        feature_processor_config=mock_feature_processor_config,
        job_settings=job_settings,
        wrapped_func=job_function,
    )
    wrapped_func.feature_processor_config.return_value = mock_feature_processor_config
    wrapped_func.job_settings.return_value = job_settings
    wrapped_func.wrapped_func.return_value = job_function

    with pytest.raises(
        ValueError,
        match="Mode FeatureProcessorMode.PYTHON is not supported by to_pipeline API.",
    ):
        to_pipeline(
            pipeline_name="pipeline_name", step=wrapped_func, role=EXECUTION_ROLE_ARN, max_retries=1
        )


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
def test_schedule(helper):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.describe_pipeline = Mock(return_value={"PipelineArn": "my:arn"})

    schedule_arn = schedule(
        schedule_expression="some_schedule",
        state=VALID_SCHEDULE_STATE,
        start_date=NOW,
        pipeline_name=PIPELINE_ARN,
        role_arn=SCHEDULE_ROLE_ARN,
        sagemaker_session=session,
    )

    assert schedule_arn == SCHEDULE_ARN


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
def test_describe_both_exist(helper):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.describe_pipeline.return_value = {
        "PipelineArn": "some_pipeline_arn",
        "RoleArn": "some_execution_role_arn",
        "PipelineDefinition": json.dumps(
            {"Steps": [{"Arguments": {"RetryStrategy": {"MaximumRetryAttempts": 5}}}]}
        ),
    }
    describe_schedule_response = describe(
        pipeline_name="some_pipeline_arn", sagemaker_session=session
    )
    assert describe_schedule_response == dict(
        pipeline_arn="some_pipeline_arn",
        pipeline_execution_role_arn="some_execution_role_arn",
        max_retries=5,
        schedule_arn="some_schedule_arn",
        schedule_expression="some_schedule_expression",
        schedule_state=VALID_SCHEDULE_STATE,
        schedule_start_date=NOW.strftime(EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT),
        schedule_role="some_schedule_role_arn",
    )


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper.describe_schedule",
    return_value=None,
)
def test_describe_only_pipeline_exist(helper):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.describe_pipeline.return_value = {
        "PipelineArn": "some_pipeline_arn",
        "RoleArn": "some_execution_role_arn",
        "PipelineDefinition": json.dumps({"Steps": [{"Arguments": {}}]}),
    }
    helper.describe_schedule().return_value = None
    describe_schedule_response = describe(
        pipeline_name="some_pipeline_arn", sagemaker_session=session
    )
    assert describe_schedule_response == dict(
        pipeline_arn="some_pipeline_arn",
        pipeline_execution_role_arn="some_execution_role_arn",
    )


def test_list_schedules():
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.list_contexts.return_value = {
        "ContextSummaries": [{"Source": {"SourceUri": "some_pipeline_arn"}}]
    }
    pipeline_arns = list_schedules(session)
    assert pipeline_arns == ["some_pipeline_arn"]


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
def test_delete_schedule_both_exist(helper):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.delete_pipeline = Mock()
    delete_schedule(pipeline_name=PIPELINE_ARN, sagemaker_session=session)
    helper().delete_schedule.assert_called_once_with(PIPELINE_ARN)


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
def test_delete_schedule_not_exist(helper):
    helper.delete_schedule.side_effect = ClientError(
        error_response={"Error": {"Code": "ResourceNotFoundException"}},
        operation_name="update_schedule",
    )
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.delete_pipeline = Mock()
    delete_schedule(pipeline_name=PIPELINE_ARN, sagemaker_session=session)
    helper().delete_schedule.assert_called_once_with(PIPELINE_ARN)


def test_execute():
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.start_pipeline_execution = Mock(
        return_value={"PipelineExecutionArn": "my:arn"}
    )
    execution_arn = execute(
        pipeline_name="some_pipeline", execution_time=NOW, sagemaker_session=session
    )
    assert execution_arn == "my:arn"


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_remote_decorator_fields_consistency(get_execution_role, session):
    expected_remote_decorator_attributes = {
        "sagemaker_session",
        "python_sdk_whl_s3_uri",
        "environment_variables",
        "image_uri",
        "dependencies",
        "pre_execution_commands",
        "pre_execution_script",
        "include_local_workdir",
        "instance_type",
        "instance_count",
        "volume_size",
        "max_runtime_in_seconds",
        "max_retry_attempts",
        "keep_alive_period_in_seconds",
        "spark_config",
        "job_conda_env",
        "job_name_prefix",
        "encrypt_inter_container_traffic",
        "enable_network_isolation",
        "role",
        "s3_root_uri",
        "s3_kms_key",
        "volume_kms_key",
        "vpc_config",
        "tags",
    }

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=EXECUTION_ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
    )
    actual_attributes = {attribute for attribute, _ in job_settings.__dict__.items()}

    assert expected_remote_decorator_attributes == actual_attributes
