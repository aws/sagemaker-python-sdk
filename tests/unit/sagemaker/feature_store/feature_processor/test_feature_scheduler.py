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
from mock import Mock, patch, call

from sagemaker.feature_store.feature_processor.feature_scheduler import (
    FeatureProcessorLineageHandler,
)
from sagemaker.feature_store.feature_processor import (
    FeatureProcessorPipelineEvents,
    FeatureProcessorPipelineExecutionStatus,
)
from sagemaker.lineage.context import Context
from sagemaker.remote_function.spark_config import SparkConfig

from sagemaker import Session
from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.feature_store.feature_processor._constants import (
    FEATURE_PROCESSOR_TAG_KEY,
    FEATURE_PROCESSOR_TAG_VALUE,
    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT,
    PIPELINE_NAME_MAXIMUM_LENGTH,
)
from sagemaker.feature_store.feature_processor.feature_scheduler import (
    schedule,
    to_pipeline,
    execute,
    delete_schedule,
    describe,
    list_pipelines,
    put_trigger,
    enable_trigger,
    disable_trigger,
    delete_trigger,
    _validate_fg_lineage_resources,
    _validate_pipeline_lineage_resources,
)
from sagemaker.remote_function.job import (
    _JobSettings,
    SPARK_APP_SCRIPT_PATH,
    RUNTIME_SCRIPTS_CHANNEL_NAME,
    REMOTE_FUNCTION_WORKSPACE,
    ENTRYPOINT_SCRIPT_NAME,
    SPARK_CONF_CHANNEL_NAME,
)
from sagemaker.workflow.parameters import Parameter, ParameterTypeEnum
from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
    SageMakerJobExceptionTypeEnum,
)
import test_data_helpers as tdh

REGION = "us-west-2"
IMAGE = "image_uri"
BUCKET = "my-s3-bucket"
DEFAULT_BUCKET_PREFIX = "default_bucket_prefix"
S3_URI = f"s3://{BUCKET}/keyprefix"
DEFAULT_IMAGE = (
    "153931337802.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:3.2-cpu-py39-v1.1"
)
PIPELINE_ARN = "pipeline_arn"
SCHEDULE_ARN = "schedule_arn"
SCHEDULE_ROLE_ARN = "my_schedule_role_arn"
EXECUTION_ROLE_ARN = "my_execution_role_arn"
EVENT_BRIDGE_RULE_ARN = "arn:aws:events:us-west-2:123456789012:rule/test-rule"
VALID_SCHEDULE_STATE = "ENABLED"
INVALID_SCHEDULE_STATE = "invalid"
TEST_REGION = "us-west-2"
PIPELINE_CONTEXT_NAME_TAG_KEY = "sm-fs-fe:feature-engineering-pipeline-context-name"
PIPELINE_VERSION_CONTEXT_NAME_TAG_KEY = "sm-fs-fe:feature-engineering-pipeline-version-context-name"
NOW = datetime.now()
SAGEMAKER_SESSION_MOCK = Mock(Session)
CONTEXT_MOCK_01 = Mock(Context)
CONTEXT_MOCK_02 = Mock(Context)
CONTEXT_MOCK_03 = Mock(Context)
FEATURE_GROUP = tdh.DESCRIBE_FEATURE_GROUP_RESPONSE.copy()
PIPELINE = tdh.PIPELINE.copy()
TAGS = [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]


def mock_session():
    session = Mock()
    session.default_bucket.return_value = BUCKET
    session.default_bucket_prefix = DEFAULT_BUCKET_PREFIX
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
    helper.upsert_schedule.return_value = dict(ScheduleArn=SCHEDULE_ARN)
    helper.delete_schedule.return_value = None
    helper.describe_schedule.return_value = {
        "Arn": "some_schedule_arn",
        "ScheduleExpression": "some_schedule_expression",
        "StartDate": NOW,
        "State": VALID_SCHEDULE_STATE,
        "Target": {"Arn": "some_pipeline_arn", "RoleArn": "some_schedule_role_arn"},
    }
    return helper


def mock_event_bridge_rule_helper():
    helper = Mock()
    helper.describe_rule.return_value = {
        "Arn": "some_rule_arn",
        "EventPattern": "some_event_pattern",
        "State": "ENABLED",
    }
    return helper


def mock_feature_processor_lineage():
    return Mock(FeatureProcessorLineageHandler)


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
    "sagemaker.feature_store.feature_processor.feature_scheduler._validate_fg_lineage_resources",
    return_value=None,
)
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
    "sagemaker.feature_store.feature_processor._config_uploader.ConfigUploader._prepare_and_upload_workspace",
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
@patch(
    "sagemaker.feature_store.feature_processor.lineage."
    "_feature_processor_lineage.FeatureProcessorLineageHandler.create_lineage"
)
@patch(
    "sagemaker.feature_store.feature_processor.lineage."
    "_feature_processor_lineage.FeatureProcessorLineageHandler.get_pipeline_lineage_names",
    return_value=dict(
        pipeline_context_name="pipeline-context-name",
        pipeline_version_context_name="pipeline-version-context-name",
    ),
)
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_to_pipeline(
    get_execution_role,
    session,
    mock_get_pipeline_lineage_names,
    mock_create_lineage,
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
    lineage_validator,
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

    mock_feature_processor_config = Mock(
        mode=FeatureProcessorMode.PYSPARK, inputs=[tdh.FEATURE_PROCESSOR_INPUTS], output="some_fg"
    )
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
        tags=[("tag_key_1", "tag_value_1"), ("tag_key_2", "tag_value_2")],
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
        None,
        None,
        f"{S3_URI}/pipeline_name",
        None,
        session,
        None,
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
            SPARK_CONF_CHANNEL_NAME: mock_training_input(s3_data="path_d", s3_data_type="S3Prefix"),
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

    pipeline().upsert.assert_has_calls(
        [
            call(
                role_arn=EXECUTION_ROLE_ARN,
                tags=[
                    dict(Key=FEATURE_PROCESSOR_TAG_KEY, Value=FEATURE_PROCESSOR_TAG_VALUE),
                    dict(Key="tag_key_1", Value="tag_value_1"),
                    dict(Key="tag_key_2", Value="tag_value_2"),
                ],
            ),
            call(
                role_arn=EXECUTION_ROLE_ARN,
                tags=[
                    {
                        "Key": PIPELINE_CONTEXT_NAME_TAG_KEY,
                        "Value": "pipeline-context-name",
                    },
                    {
                        "Key": PIPELINE_VERSION_CONTEXT_NAME_TAG_KEY,
                        "Value": "pipeline-version-context-name",
                    },
                ],
            ),
        ]
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
            pipeline_name="pipeline_name",
            step=wrapped_func,
            role=EXECUTION_ROLE_ARN,
            max_retries=1,
        )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_to_pipeline_not_wrapped_by_remote(get_execution_role, session):
    mock_feature_processor_config = Mock(mode=FeatureProcessorMode.PYTHON)
    wrapped_func = Mock(
        Callable,
        feature_processor_config=mock_feature_processor_config,
        job_settings=None,
        wrapped_func=job_function,
    )
    wrapped_func.wrapped_func.return_value = job_function

    with pytest.raises(
        ValueError,
        match="Please wrap step parameter with remote decorator in order to use to_pipeline API.",
    ):
        to_pipeline(
            pipeline_name="pipeline_name",
            step=wrapped_func,
            role=EXECUTION_ROLE_ARN,
            max_retries=1,
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
            pipeline_name="pipeline_name",
            step=wrapped_func,
            role=EXECUTION_ROLE_ARN,
            max_retries=1,
        )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch(
    "sagemaker.remote_function.job._JobSettings._get_default_spark_image",
    return_value="some_image_uri",
)
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_to_pipeline_pipeline_name_length_limit_exceeds(
    get_execution_role, mock_spark_image, session
):
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

    with pytest.raises(
        ValueError,
        match="Pipeline name used by feature processor should be less than 80 "
        "characters. Please choose another pipeline name.",
    ):
        to_pipeline(
            pipeline_name="".join(["a" for _ in range(PIPELINE_NAME_MAXIMUM_LENGTH + 1)]),
            step=wrapped_func,
            role=EXECUTION_ROLE_ARN,
            max_retries=1,
        )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch(
    "sagemaker.remote_function.job._JobSettings._get_default_spark_image",
    return_value="some_image_uri",
)
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_to_pipeline_used_reserved_tags(get_execution_role, mock_spark_image, session):
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

    mock_feature_processor_config = Mock(
        mode=FeatureProcessorMode.PYSPARK, inputs=[tdh.FEATURE_PROCESSOR_INPUTS], output="some_fg"
    )
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

    with pytest.raises(
        ValueError,
        match="sm-fs-fe:created-from is a reserved tag key for to_pipeline API. Please choose another tag.",
    ):
        to_pipeline(
            pipeline_name="pipeline_name",
            step=wrapped_func,
            role=EXECUTION_ROLE_ARN,
            max_retries=1,
            tags=[("sm-fs-fe:created-from", "random")],
            sagemaker_session=session,
        )


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler"
    "._get_tags_from_pipeline_to_propagate_to_lineage_resources",
    return_value=TAGS,
)
@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler._validate_pipeline_lineage_resources",
    return_value=None,
)
@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.FeatureProcessorLineageHandler",
    return_value=mock_feature_processor_lineage(),
)
def test_schedule(lineage, helper, validation, get_tags):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.describe_pipeline = Mock(
        return_value={"PipelineArn": "my:arn", "CreationTime": NOW}
    )

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
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeRuleHelper",
    return_value=mock_event_bridge_rule_helper(),
)
@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper",
    return_value=mock_event_bridge_scheduler_helper(),
)
def test_describe_both_exist(mock_scheduler_helper, mock_rule_helper):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.describe_pipeline.return_value = PIPELINE
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
        trigger="some_rule_arn",
        event_pattern="some_event_pattern",
        trigger_state="ENABLED",
    )


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeRuleHelper.describe_rule",
    return_value=None,
)
@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler.EventBridgeSchedulerHelper.describe_schedule",
    return_value=None,
)
def test_describe_only_pipeline_exist(helper, mock_describe_rule):
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


def test_list_pipelines():
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.list_contexts.return_value = {
        "ContextSummaries": [
            {
                "Source": {
                    "SourceUri": "arn:aws:sagemaker:us-west-2:12345789012:pipeline/some_pipeline_name"
                }
            }
        ]
    }
    list_response = list_pipelines(session)
    assert list_response == [dict(pipeline_name="some_pipeline_name")]


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


@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler._validate_pipeline_lineage_resources",
    return_value=None,
)
def test_execute(validation):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.describe_pipeline = Mock(
        return_value={"PipelineArn": "my:arn", "CreationTime": NOW}
    )
    session.sagemaker_client.start_pipeline_execution = Mock(
        return_value={"PipelineExecutionArn": "my:arn"}
    )
    execution_arn = execute(
        pipeline_name="some_pipeline", execution_time=NOW, sagemaker_session=session
    )
    assert execution_arn == "my:arn"


def test_validate_fg_lineage_resources_happy_case():
    with patch.object(
        SAGEMAKER_SESSION_MOCK, "describe_feature_group", return_value=FEATURE_GROUP
    ) as fg_describe_method:
        with patch.object(
            Context, "load", side_effect=[CONTEXT_MOCK_01, CONTEXT_MOCK_02, CONTEXT_MOCK_03]
        ) as context_load:
            type(CONTEXT_MOCK_01).context_arn = "context-arn"
            type(CONTEXT_MOCK_02).context_arn = "context-arn-fep"
            type(CONTEXT_MOCK_03).context_arn = "context-arn-fep-ver"
            _validate_fg_lineage_resources(
                feature_group_name="some_fg",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            )
    fg_describe_method.assert_called_once_with(feature_group_name="some_fg")
    context_load.assert_has_calls(
        [
            call(
                context_name=f'{"some_fg"}-{FEATURE_GROUP["CreationTime"].strftime("%s")}'
                f"-feature-group-pipeline",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                context_name=f'{"some_fg"}-{FEATURE_GROUP["CreationTime"].strftime("%s")}'
                f"-feature-group-pipeline-version",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == context_load.call_count


def test_validete_fg_lineage_resources_rnf():
    with patch.object(SAGEMAKER_SESSION_MOCK, "describe_feature_group", return_value=FEATURE_GROUP):
        with patch.object(
            Context,
            "load",
            side_effect=ClientError(
                error_response={"Error": {"Code": "ResourceNotFound"}},
                operation_name="describe_context",
            ),
        ):
            feature_group_name = "some_fg"
            feature_group_creation_time = FEATURE_GROUP["CreationTime"].strftime("%s")
            context_name = f"{feature_group_name}-{feature_group_creation_time}"
            with pytest.raises(
                ValueError,
                match=f"Lineage resource {context_name} has not yet been created for feature group"
                f" {feature_group_name} or has already been deleted. Please try again later.",
            ):
                _validate_fg_lineage_resources(
                    feature_group_name="some_fg",
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                )


def test_validate_pipeline_lineage_resources_happy_case():
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.return_value = Mock()
    pipeline_name = "some_pipeline"
    with patch.object(
        session.sagemaker_client, "describe_pipeline", return_value=PIPELINE
    ) as pipeline_describe_method:
        with patch.object(
            Context, "load", side_effect=[CONTEXT_MOCK_01, CONTEXT_MOCK_02]
        ) as context_load:
            type(CONTEXT_MOCK_01).context_arn = "context-arn"
            type(CONTEXT_MOCK_01).properties = {"LastUpdateTime": NOW.strftime("%s")}
            type(CONTEXT_MOCK_02).context_arn = "context-arn-fep"
            _validate_pipeline_lineage_resources(
                pipeline_name=pipeline_name,
                sagemaker_session=session,
            )
    pipeline_describe_method.assert_called_once_with(PipelineName=pipeline_name)
    pipeline_creation_time = PIPELINE["CreationTime"].strftime("%s")
    last_updated_time = NOW.strftime("%s")
    context_load.assert_has_calls(
        [
            call(
                context_name=f"sm-fs-fe-{pipeline_name}-{pipeline_creation_time}-fep",
                sagemaker_session=session,
            ),
            call(
                context_name=f"sm-fs-fe-{pipeline_name}-{last_updated_time}-fep-ver",
                sagemaker_session=session,
            ),
        ]
    )
    assert 2 == context_load.call_count


def test_validate_pipeline_lineage_resources_rnf():
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    session.sagemaker_client.return_value = Mock()
    pipeline_name = "some_pipeline"
    with patch.object(session.sagemaker_client, "describe_pipeline", return_value=PIPELINE):
        with patch.object(
            Context,
            "load",
            side_effect=ClientError(
                error_response={"Error": {"Code": "ResourceNotFound"}},
                operation_name="describe_context",
            ),
        ):
            with pytest.raises(
                ValueError,
                match="Pipeline lineage resources have not been created yet or have"
                " already been deleted. Please try again later.",
            ):
                _validate_pipeline_lineage_resources(
                    pipeline_name=pipeline_name,
                    sagemaker_session=session,
                )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=EXECUTION_ROLE_ARN)
def test_remote_decorator_fields_consistency(get_execution_role, session):
    expected_remote_decorator_attributes = {
        "sagemaker_session",
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
        "use_spot_instances",
        "max_wait_time_in_seconds",
        "custom_file_filter",
        "disable_output_compression",
        "use_torchrun",
        "use_mpirun",
        "nproc_per_node",
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


@patch(
    "sagemaker.feature_store.feature_processor.lineage."
    "_feature_processor_lineage.FeatureProcessorLineageHandler.create_trigger_lineage"
)
@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.describe_rule",
    return_value={"EventPattern": "test-pattern"},
)
@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.add_tags"
)
@patch(
    "sagemaker.feature_store.feature_processor.feature_scheduler."
    "_get_tags_from_pipeline_to_propagate_to_lineage_resources",
    return_value=TAGS,
)
@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.put_target"
)
@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.put_rule",
    return_value="arn:aws:events:us-west-2:123456789012:rule/test-rule",
)
def test_put_trigger(
    mock_put_rule,
    mock_put_target,
    mock_get_tags,
    mock_add_tags,
    mock_describe_rule,
    mock_create_trigger_lineage,
):
    session = Mock(
        Session,
        sagemaker_client=Mock(
            describe_pipeline=Mock(return_value={"PipelineArn": "test-pipeline-arn"})
        ),
        boto_session=Mock(),
    )
    source_pipeline_events = [
        FeatureProcessorPipelineEvents(
            pipeline_name="test-pipeline",
            pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.SUCCEEDED],
        )
    ]
    put_trigger(
        source_pipeline_events=source_pipeline_events,
        target_pipeline="test-target-pipeline",
        state="Enabled",
        event_pattern="test-pattern",
        role_arn=SCHEDULE_ROLE_ARN,
        sagemaker_session=session,
    )

    mock_put_rule.assert_called_once_with(
        source_pipeline_events=source_pipeline_events,
        target_pipeline="test-target-pipeline",
        state="Enabled",
        event_pattern="test-pattern",
    )
    mock_put_target.assert_called_once_with(
        rule_name="test-rule",
        target_pipeline="test-target-pipeline",
        target_pipeline_parameters=None,
        role_arn=SCHEDULE_ROLE_ARN,
    )
    mock_add_tags.assert_called_once_with(rule_arn=EVENT_BRIDGE_RULE_ARN, tags=TAGS)
    mock_create_trigger_lineage.assert_called_once_with(
        pipeline_name="test-target-pipeline",
        trigger_arn=EVENT_BRIDGE_RULE_ARN,
        state="Enabled",
        tags=TAGS,
        event_pattern="test-pattern",
    )


@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.enable_rule"
)
def test_enable_trigger(mock_enable_rule):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    enable_trigger(pipeline_name="test-pipeline", sagemaker_session=session)
    mock_enable_rule.assert_called_once_with(rule_name="test-pipeline")


@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.disable_rule"
)
def test_disable_trigger(mock_disable_rule):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    disable_trigger(pipeline_name="test-pipeline", sagemaker_session=session)
    mock_disable_rule.assert_called_once_with(rule_name="test-pipeline")


@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.list_targets_by_rule",
    return_value=[{"Targets": [{"Id": "target_pipeline"}]}],
)
@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.remove_targets"
)
@patch(
    "sagemaker.feature_store.feature_processor._event_bridge_rule_helper.EventBridgeRuleHelper.delete_rule"
)
def test_delete_trigger(mock_delete_rule, mock_remove_targets, mock_list_targets_by_rule):
    session = Mock(Session, sagemaker_client=Mock(), boto_session=Mock())
    delete_trigger(pipeline_name="test-pipeline", sagemaker_session=session)
    mock_delete_rule.assert_called_once_with("test-pipeline")
    mock_list_targets_by_rule.assert_called_once_with("test-pipeline")
    mock_remove_targets.assert_called_once_with(rule_name="test-pipeline", ids=["target_pipeline"])
