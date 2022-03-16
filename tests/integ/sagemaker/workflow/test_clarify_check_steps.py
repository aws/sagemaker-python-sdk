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
from botocore.exceptions import WaiterError

from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
)
from sagemaker.s3 import S3Uploader, S3Downloader

from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
)
from sagemaker.workflow.parameters import ParameterString
from tests.integ import DATA_DIR

from sagemaker import get_execution_role, utils
from sagemaker.model_monitor import Constraints
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.check_job_config import CheckJobConfig
from tests.integ.retry import retries

_INSTANCE_COUNT = 1
_INSTANCE_TYPE = "ml.c5.xlarge"
_HEADERS = ["Label", "F1", "F2", "F3", "F4"]
_CHECK_FAIL_ERROR_MSG = "ClientError: Clarify check failed. See violation report"


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-clarify")


@pytest.fixture
def check_job_config(role, sagemaker_session):
    return CheckJobConfig(
        role=role,
        instance_count=_INSTANCE_COUNT,
        instance_type=_INSTANCE_TYPE,
        volume_size_in_gb=60,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def supplied_baseline_constraints_uri_param():
    return ParameterString(name="SuppliedBaselineConstraintsUri", default_value="")


@pytest.fixture
def dataset(sagemaker_session):
    dataset_local_path = os.path.join(DATA_DIR, "pipeline/clarify_check_step/dataset.csv")
    dataset_s3_uri = "s3://{}/{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "clarify_check_step",
        "input",
        "dataset",
        utils.unique_name_from_base("dataset"),
    )
    return S3Uploader.upload(
        dataset_local_path, dataset_s3_uri, sagemaker_session=sagemaker_session
    )


@pytest.fixture
def data_config(sagemaker_session, dataset):
    output_path = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "clarify_check_step",
        "analysis_result",
        utils.unique_name_from_base("result"),
    )
    analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
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


def test_one_step_data_bias_pipeline_happycase(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    data_bias_check_config,
    supplied_baseline_constraints_uri_param,
):
    data_bias_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/clarify_check_step/data_bias/good_cases/analysis.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=False,
        register_new_baseline=False,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[data_bias_check_step],
        parameters=[supplied_baseline_constraints_uri_param],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        monitoring_analysis_cfg_json = S3Downloader.read_file(
            data_bias_check_config.monitoring_analysis_config_uri,
            sagemaker_session,
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
            assert execution_steps[0]["StepName"] == "DataBiasCheckStep"
            assert execution_steps[0]["StepStatus"] == "Succeeded"
            data_bias_metadata = execution_steps[0]["Metadata"]["ClarifyCheck"]
            assert not data_bias_metadata["SkipCheck"]
            assert not data_bias_metadata["RegisterNewBaseline"]
            assert not data_bias_metadata.get("ViolationReport", "")
            assert (
                data_bias_metadata["BaselineUsedForDriftCheckConstraints"]
                == data_bias_supplied_baseline_constraints
            )
            assert (
                data_bias_metadata["BaselineUsedForDriftCheckConstraints"]
                != data_bias_metadata["CalculatedBaselineConstraints"]
            )
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_one_step_data_bias_pipeline_constraint_violation(
    sagemaker_session,
    role,
    pipeline_name,
    check_job_config,
    data_bias_check_config,
    supplied_baseline_constraints_uri_param,
):
    data_bias_supplied_baseline_constraints = Constraints.from_file_path(
        constraints_file_path=os.path.join(
            DATA_DIR, "pipeline/clarify_check_step/data_bias/bad_cases/analysis.json"
        ),
        sagemaker_session=sagemaker_session,
    ).file_s3_uri
    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=False,
        register_new_baseline=False,
        supplied_baseline_constraints=supplied_baseline_constraints_uri_param,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[data_bias_check_step],
        parameters=[supplied_baseline_constraints_uri_param],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        monitoring_analysis_cfg_json = S3Downloader.read_file(
            data_bias_check_config.monitoring_analysis_config_uri,
            sagemaker_session,
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
            assert execution_steps[0]["StepName"] == "DataBiasCheckStep"
            assert execution_steps[0]["StepStatus"] == "Failed"
            break
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
