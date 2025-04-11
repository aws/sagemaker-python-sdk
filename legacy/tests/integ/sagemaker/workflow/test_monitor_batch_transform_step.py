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
import logging
import os

import pytest

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
)
from sagemaker.s3 import S3Uploader, S3Downloader
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
)
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from tests.integ import (
    DATA_DIR,
)
from sagemaker.model_monitor import DatasetFormat, Statistics, Constraints
from sagemaker.xgboost import XGBoostModel
from sagemaker.transformer import Transformer
from sagemaker import get_execution_role, utils
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.check_job_config import CheckJobConfig
from tests.integ.retry import retries
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
)

_INSTANCE_COUNT = 1
_INSTANCE_TYPE = "ml.c5.xlarge"
_HEADERS = ["Label", "F1", "F2", "F3", "F4"]
_CHECK_FAIL_ERROR_MSG_CLARIFY = "ClientError: Clarify check failed. See violation report"
_PROBLEM_TYPE = "Regression"
_HEADER_OF_LABEL = "Label"
_HEADER_OF_PREDICTED_LABEL = "Prediction"
_CHECK_FAIL_ERROR_MSG_QUALITY = "ClientError: Quality check failed. See violation report"


@pytest.fixture(scope="module")
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture(scope="module")
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-clarify")


@pytest.fixture
def check_job_config(role, pipeline_session):
    return CheckJobConfig(
        role=role,
        instance_count=_INSTANCE_COUNT,
        instance_type=_INSTANCE_TYPE,
        volume_size_in_gb=60,
        sagemaker_session=pipeline_session,
    )


@pytest.fixture
def supplied_baseline_statistics_uri_param():
    return ParameterString(name="SuppliedBaselineStatisticsUri", default_value="")


@pytest.fixture
def supplied_baseline_constraints_uri_param():
    return ParameterString(name="SuppliedBaselineConstraintsUri", default_value="")


@pytest.fixture
def dataset(pipeline_session):
    dataset_local_path = os.path.join(DATA_DIR, "pipeline/clarify_check_step/dataset.csv")
    dataset_s3_uri = "s3://{}/{}/{}/{}/{}".format(
        pipeline_session.default_bucket(),
        "clarify_check_step",
        "input",
        "dataset",
        utils.unique_name_from_base("dataset"),
    )
    return S3Uploader.upload(dataset_local_path, dataset_s3_uri, sagemaker_session=pipeline_session)


@pytest.fixture
def data_config(pipeline_session, dataset):
    output_path = "s3://{}/{}/{}/{}".format(
        pipeline_session.default_bucket(),
        "clarify_check_step",
        "analysis_result",
        utils.unique_name_from_base("result"),
    )
    analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
        pipeline_session.default_bucket(),
        "clarify_check_step",
        "analysis_cfg",
        utils.unique_name_from_base("analysis_cfg"),
    )
    return DataConfig(
        s3_data_input_path=dataset,
        s3_output_path=output_path,
        s3_analysis_config_output_path=analysis_cfg_output_path,
        label="Label",
        headers=_HEADERS,
        dataset_type="text/csv",
    )


@pytest.fixture
def bias_config():
    return BiasConfig(
        label_values_or_threshold=[1],
        facet_name="F1",
        facet_values_or_threshold=[0.5],
        group_name="F2",
    )


@pytest.fixture
def data_bias_check_config(data_config, bias_config):
    return DataBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
    )


@pytest.fixture
def data_quality_baseline_dataset():
    return os.path.join(DATA_DIR, "pipeline/quality_check_step/data_quality/baseline_dataset.csv")


@pytest.fixture
def data_quality_check_config(data_quality_baseline_dataset):
    return DataQualityCheckConfig(
        baseline_dataset=data_quality_baseline_dataset,
        dataset_format=DatasetFormat.csv(header=False),
    )


@pytest.fixture
def data_quality_supplied_baseline_statistics(sagemaker_session):
    return Statistics.from_file_path(
        statistics_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/data_quality/statistics.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri


@pytest.fixture
def model_quality_baseline_dataset():
    return os.path.join(DATA_DIR, "pipeline/quality_check_step/model_quality/baseline_dataset.csv")


@pytest.fixture
def model_quality_check_config(model_quality_baseline_dataset):
    return ModelQualityCheckConfig(
        baseline_dataset=model_quality_baseline_dataset,
        dataset_format=DatasetFormat.csv(),
        problem_type=_PROBLEM_TYPE,
        inference_attribute=_HEADER_OF_LABEL,
        ground_truth_attribute=_HEADER_OF_PREDICTED_LABEL,
    )


@pytest.fixture
def model_quality_supplied_baseline_statistics(sagemaker_session):
    return Statistics.from_file_path(
        statistics_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/model_quality/statistics.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri


@pytest.fixture
def step_model_create(
    pipeline_session,
    role,
):

    xgb_model_data_s3 = pipeline_session.upload_data(
        path=os.path.join(os.path.join(DATA_DIR, "xgboost_abalone"), "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    xgb_model = XGBoostModel(
        model_data=xgb_model_data_s3,
        framework_version="1.3-1",
        role=role,
        sagemaker_session=pipeline_session,
        entry_point=os.path.join(os.path.join(DATA_DIR, "xgboost_abalone"), "inference.py"),
        enable_network_isolation=True,
    )

    create_model_step_args = xgb_model.create(
        instance_type="ml.m5.large",
    )
    step_model_create = ModelStep(
        name="MyModel",
        step_args=create_model_step_args,
    )
    return step_model_create


@pytest.fixture
def transform_args(
    pipeline_session,
    sagemaker_session,
    pipeline_name,
    step_model_create,
):

    transform_input = pipeline_session.upload_data(
        path=os.path.join(DATA_DIR, "xgboost_abalone", "abalone"),
        key_prefix="integ-test-data/xgboost_abalone/abalone",
    )

    transform_output = f"s3://{sagemaker_session.default_bucket()}/{pipeline_name}Transform"
    transformer = Transformer(
        model_name=step_model_create.properties.ModelName,
        strategy="SingleRecord",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=transform_output,
        sagemaker_session=pipeline_session,
    )
    transform_args = transformer.transform(
        data=transform_input,
        content_type="text/libsvm",
    )
    return transform_args


def test_monitor_batch_clarify_data_bias_pipeline_happycase(
    pipeline_session,
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    data_bias_check_config,
    supplied_baseline_constraints_uri_param,
    transform_args,
    step_model_create,
):

    data_bias_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/clarify_check_step/data_bias/good_cases/analysis.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri

    data_bias_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=data_bias_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        fail_on_violation=True,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_bias_check_step],
        parameters=[supplied_baseline_constraints_uri_param],
        sagemaker_session=pipeline_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        monitoring_analysis_cfg_json = S3Downloader.read_file(
            data_bias_check_config.monitoring_analysis_config_uri,
            pipeline_session,
        )
        monitoring_analysis_cfg = json.loads(monitoring_analysis_cfg_json)

        assert monitoring_analysis_cfg is not None and len(monitoring_analysis_cfg) > 0

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineConstraintsUri": data_bias_supplied_baseline_constraints
                },
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3

            for execution_step in execution_steps:
                failure_reason = execution_step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}. Retrying.."
                    )
                    continue

            for execution_step in execution_steps:
                assert execution_step["StepStatus"] == "Succeeded"
                if execution_step["StepName"] == "MonitorBatchTransformStep-transform":
                    transform_execution_end_time = execution_step["EndTime"]

                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    monitoring_execution_start_time = execution_step["StartTime"]
                    monitor_batch_data_bias_metadata = execution_step["Metadata"]["ClarifyCheck"]
                    assert not monitor_batch_data_bias_metadata["SkipCheck"]
                    assert not monitor_batch_data_bias_metadata["RegisterNewBaseline"]
                    assert not monitor_batch_data_bias_metadata.get("ViolationReport", "")
                    assert (
                        monitor_batch_data_bias_metadata["BaselineUsedForDriftCheckConstraints"]
                        == data_bias_supplied_baseline_constraints
                    )
                    assert (
                        monitor_batch_data_bias_metadata["BaselineUsedForDriftCheckConstraints"]
                        != monitor_batch_data_bias_metadata["CalculatedBaselineConstraints"]
                    )
            assert monitoring_execution_start_time > transform_execution_end_time
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_monitor_batch_clarify_data_bias_pipeline_bad_case(
    pipeline_session,
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    data_bias_check_config,
    supplied_baseline_constraints_uri_param,
    transform_args,
    step_model_create,
):

    data_bias_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/clarify_check_step/data_bias/bad_cases/analysis.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri

    data_bias_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=data_bias_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        fail_on_violation=True,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_bias_check_step],
        parameters=[supplied_baseline_constraints_uri_param],
        sagemaker_session=pipeline_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        monitoring_analysis_cfg_json = S3Downloader.read_file(
            data_bias_check_config.monitoring_analysis_config_uri,
            pipeline_session,
        )
        monitoring_analysis_cfg = json.loads(monitoring_analysis_cfg_json)

        assert monitoring_analysis_cfg is not None and len(monitoring_analysis_cfg) > 0

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineConstraintsUri": data_bias_supplied_baseline_constraints
                },
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3

            for execution_step in execution_steps:
                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    failure_reason = execution_step.get("FailureReason", "")
                    break
            if _CHECK_FAIL_ERROR_MSG_CLARIFY not in failure_reason:
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue
            assert execution_step["StepStatus"] == "Failed"
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_batch_transform_data_quality_step_pipeline_happycase(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    data_quality_check_config,
    data_quality_supplied_baseline_statistics,
    transform_args,
    step_model_create,
    pipeline_session,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/data_quality/good_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=data_quality_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        fail_on_violation=True,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=pipeline_session,
    )
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineStatisticsUri": data_quality_supplied_baseline_statistics,
                    "SuppliedBaselineConstraintsUri": data_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3
            for execution_step in execution_steps:
                failure_reason = execution_step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}. Retrying.."
                    )
                    continue

            for execution_step in execution_steps:
                assert execution_step["StepStatus"] == "Succeeded"
                if execution_step["StepName"] == "MonitorBatchTransformStep-transform":
                    transform_execution_end_time = execution_step["EndTime"]
                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    monitoring_execution_start_time = execution_step["StartTime"]
                    data_qual_metadata = execution_step["Metadata"]["QualityCheck"]
                    assert not data_qual_metadata["SkipCheck"]
                    assert not data_qual_metadata["RegisterNewBaseline"]
                    assert not data_qual_metadata.get("ViolationReport", "")
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                        == data_quality_supplied_baseline_constraints
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                        == data_quality_supplied_baseline_statistics
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                        != data_qual_metadata["CalculatedBaselineConstraints"]
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                        != data_qual_metadata["CalculatedBaselineStatistics"]
                    )
            assert monitoring_execution_start_time > transform_execution_end_time
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_batch_transform_data_quality_step_pipeline_failure_case(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    data_quality_check_config,
    data_quality_supplied_baseline_statistics,
    transform_args,
    step_model_create,
    pipeline_session,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/data_quality/bad_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=data_quality_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        fail_on_violation=True,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=pipeline_session,
    )
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineStatisticsUri": data_quality_supplied_baseline_statistics,
                    "SuppliedBaselineConstraintsUri": data_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3
            for execution_step in execution_steps:
                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    failure_reason = execution_step.get("FailureReason", "")
                    break
            if _CHECK_FAIL_ERROR_MSG_QUALITY not in failure_reason:
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue
            assert execution_step["StepStatus"] == "Failed"
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_batch_transform_model_quality_step_pipeline_happycase(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    model_quality_check_config,
    model_quality_supplied_baseline_statistics,
    transform_args,
    step_model_create,
    pipeline_session,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/model_quality/good_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=model_quality_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        fail_on_violation=True,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=pipeline_session,
    )
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineStatisticsUri": model_quality_supplied_baseline_statistics,
                    "SuppliedBaselineConstraintsUri": data_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3
            for execution_step in execution_steps:
                failure_reason = execution_step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}. Retrying.."
                    )
                    continue

            for execution_step in execution_steps:
                assert execution_step["StepStatus"] == "Succeeded"
                if execution_step["StepName"] == "MonitorBatchTransformStep-transform":
                    transform_execution_end_time = execution_step["EndTime"]
                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    monitoring_execution_start_time = execution_step["StartTime"]
                    data_qual_metadata = execution_step["Metadata"]["QualityCheck"]
                    assert not data_qual_metadata["SkipCheck"]
                    assert not data_qual_metadata["RegisterNewBaseline"]
                    assert not data_qual_metadata.get("ViolationReport", "")
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                        == data_quality_supplied_baseline_constraints
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                        == model_quality_supplied_baseline_statistics
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                        != data_qual_metadata["CalculatedBaselineConstraints"]
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                        != data_qual_metadata["CalculatedBaselineStatistics"]
                    )
            assert monitoring_execution_start_time > transform_execution_end_time
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_batch_transform_model_quality_step_pipeline_failure_case(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    model_quality_check_config,
    model_quality_supplied_baseline_statistics,
    transform_args,
    step_model_create,
    pipeline_session,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/model_quality/bad_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=model_quality_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        fail_on_violation=True,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=pipeline_session,
    )
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineStatisticsUri": model_quality_supplied_baseline_statistics,
                    "SuppliedBaselineConstraintsUri": data_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3
            for execution_step in execution_steps:
                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    failure_reason = execution_step.get("FailureReason", "")
                    break
            if _CHECK_FAIL_ERROR_MSG_QUALITY not in failure_reason:
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue
            assert execution_step["StepStatus"] == "Failed"
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_batch_transform_data_quality_step_pipeline_before_transformation(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    data_quality_check_config,
    data_quality_supplied_baseline_statistics,
    transform_args,
    step_model_create,
    pipeline_session,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/data_quality/good_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=data_quality_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        monitor_before_transform=True,
        fail_on_violation=True,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=pipeline_session,
    )
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineStatisticsUri": data_quality_supplied_baseline_statistics,
                    "SuppliedBaselineConstraintsUri": data_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3
            for execution_step in execution_steps:
                failure_reason = execution_step.get("FailureReason", "")
                if failure_reason != "":
                    logging.error(
                        f"Pipeline execution failed with error: {failure_reason}. Retrying.."
                    )
                    continue

            for execution_step in execution_steps:
                assert execution_step["StepStatus"] == "Succeeded"
                if execution_step["StepName"] == "MonitorBatchTransformStep-transform":
                    transform_execution_start_time = execution_step["StartTime"]

                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    monitoring_execution_end_time = execution_step["EndTime"]
                    data_qual_metadata = execution_step["Metadata"]["QualityCheck"]
                    assert not data_qual_metadata["SkipCheck"]
                    assert not data_qual_metadata["RegisterNewBaseline"]
                    assert not data_qual_metadata.get("ViolationReport", "")
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                        == data_quality_supplied_baseline_constraints
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                        == data_quality_supplied_baseline_statistics
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                        != data_qual_metadata["CalculatedBaselineConstraints"]
                    )
                    assert (
                        data_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                        != data_qual_metadata["CalculatedBaselineStatistics"]
                    )
            assert monitoring_execution_end_time < transform_execution_start_time
            break

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_batch_transform_model_quality_step_pipeline_failure_no_violation_case(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    model_quality_check_config,
    model_quality_supplied_baseline_statistics,
    transform_args,
    step_model_create,
    pipeline_session,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/model_quality/bad_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = MonitorBatchTransformStep(
        name="MonitorBatchTransformStep",
        transform_step_args=transform_args,
        monitor_configuration=model_quality_check_config,
        check_job_configuration=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
        fail_on_violation=False,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_model_create, data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=pipeline_session,
    )
    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]

        for _ in retries(
            max_retry_count=5,
            exception_message_prefix="Waiting for a successful execution of pipeline",
            seconds_to_sleep=10,
        ):
            execution = pipeline.start(
                parameters={
                    "SuppliedBaselineStatisticsUri": model_quality_supplied_baseline_statistics,
                    "SuppliedBaselineConstraintsUri": data_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            wait_pipeline_execution(execution=execution)
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 3
            for execution_step in execution_steps:
                if execution_step["StepName"] == "MonitorBatchTransformStep-monitoring":
                    failure_reason = execution_step.get("FailureReason", "")
                    break
            assert execution_step["StepStatus"] == "Succeeded"
            assert failure_reason == ""
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
