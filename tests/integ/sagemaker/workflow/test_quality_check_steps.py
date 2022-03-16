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

import logging
import os

import pytest
from botocore.exceptions import WaiterError

from sagemaker.workflow.parameters import ParameterString
from tests.integ import DATA_DIR

from sagemaker import get_execution_role, utils
from sagemaker.model_monitor import DatasetFormat, Statistics, Constraints
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.quality_check_step import (
    QualityCheckStep,
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from tests.integ.retry import retries

_INSTANCE_COUNT = 1
_INSTANCE_TYPE = "ml.c5.xlarge"

_PROBLEM_TYPE = "Regression"
_HEADER_OF_LABEL = "Label"
_HEADER_OF_PREDICTED_LABEL = "Prediction"
_HEADERS_OF_FEATURES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
_CHECK_FAIL_ERROR_MSG = "ClientError: Quality check failed. See violation report"


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-quality")


@pytest.fixture
def check_job_config(role, sagemaker_session):
    return CheckJobConfig(
        role=role,
        instance_count=_INSTANCE_COUNT,
        instance_type=_INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def supplied_baseline_statistics_uri_param():
    return ParameterString(name="SuppliedBaselineStatisticsUri", default_value="")


@pytest.fixture
def supplied_baseline_constraints_uri_param():
    return ParameterString(name="SuppliedBaselineConstraintsUri", default_value="")


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


def test_one_step_data_quality_pipeline_happycase(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    data_quality_check_config,
    data_quality_supplied_baseline_statistics,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/data_quality/good_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=False,
        register_new_baseline=False,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=sagemaker_session,
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

            try:
                execution.wait(delay=30, max_attempts=60)
            except WaiterError:
                pass
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if failure_reason != "":
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue
            assert execution_steps[0]["StepName"] == "DataQualityCheckStep"
            assert execution_steps[0]["StepStatus"] == "Succeeded"
            data_qual_metadata = execution_steps[0]["Metadata"]["QualityCheck"]
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
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_data_quality_pipeline_constraint_violation(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    data_quality_check_config,
    data_quality_supplied_baseline_statistics,
):
    data_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/data_quality/bad_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=False,
        register_new_baseline=False,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[data_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=sagemaker_session,
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

            try:
                execution.wait(delay=30, max_attempts=60)
            except WaiterError:
                pass
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if _CHECK_FAIL_ERROR_MSG not in failure_reason:
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue
            assert execution_steps[0]["StepName"] == "DataQualityCheckStep"
            assert execution_steps[0]["StepStatus"] == "Failed"
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_model_quality_pipeline_happycase(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    model_quality_check_config,
    model_quality_supplied_baseline_statistics,
):
    model_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/model_quality/good_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        register_new_baseline=False,
        skip_check=False,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[model_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=sagemaker_session,
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
                    "SuppliedBaselineConstraintsUri": model_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            try:
                execution.wait(delay=30, max_attempts=60)
            except WaiterError:
                pass
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if failure_reason != "":
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue
            assert execution_steps[0]["StepName"] == "ModelQualityCheckStep"
            assert execution_steps[0]["StepStatus"] == "Succeeded"
            model_qual_metadata = execution_steps[0]["Metadata"]["QualityCheck"]
            assert not model_qual_metadata["SkipCheck"]
            assert not model_qual_metadata["RegisterNewBaseline"]
            assert not model_qual_metadata.get("ViolationReport", "")
            assert (
                model_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                == model_quality_supplied_baseline_constraints
            )
            assert (
                model_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                == model_quality_supplied_baseline_statistics
            )
            assert (
                model_qual_metadata["BaselineUsedForDriftCheckConstraints"]
                != model_qual_metadata["CalculatedBaselineConstraints"]
            )
            assert (
                model_qual_metadata["BaselineUsedForDriftCheckStatistics"]
                != model_qual_metadata["CalculatedBaselineStatistics"]
            )
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_model_quality_pipeline_constraint_violation(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    supplied_baseline_statistics_uri_param,
    supplied_baseline_constraints_uri_param,
    model_quality_check_config,
    model_quality_supplied_baseline_statistics,
):
    model_quality_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/quality_check_step/model_quality/bad_cases/constraints.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        register_new_baseline=False,
        skip_check=False,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_uri_param,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[model_quality_check_step],
        parameters=[
            supplied_baseline_statistics_uri_param,
            supplied_baseline_constraints_uri_param,
        ],
        sagemaker_session=sagemaker_session,
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
                    "SuppliedBaselineConstraintsUri": model_quality_supplied_baseline_constraints,
                }
            )
            response = execution.describe()

            assert response["PipelineArn"] == create_arn

            try:
                execution.wait(delay=30, max_attempts=60)
            except WaiterError:
                pass
            execution_steps = execution.list_steps()

            assert len(execution_steps) == 1
            failure_reason = execution_steps[0].get("FailureReason", "")
            if _CHECK_FAIL_ERROR_MSG not in failure_reason:
                logging.error(f"Pipeline execution failed with error: {failure_reason}. Retrying..")
                continue
            assert execution_steps[0]["StepName"] == "ModelQualityCheckStep"
            assert execution_steps[0]["StepStatus"] == "Failed"
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
