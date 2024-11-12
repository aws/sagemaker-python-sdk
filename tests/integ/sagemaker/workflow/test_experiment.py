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

import os
import time

import pytest

from tests.integ.sagemaker.workflow.helpers import wait_pipeline_execution
from sagemaker.processing import ProcessingInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.dataset_definition.inputs import DatasetDefinition, AthenaDatasetDefinition
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterInteger,
)
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from tests.integ import DATA_DIR


@pytest.fixture
def pipeline_name():
    return f"my-pipeline-{int(time.time() * 10**7)}"


@pytest.fixture
def athena_dataset_definition(sagemaker_session_for_pipeline):
    return DatasetDefinition(
        local_path="/opt/ml/processing/input/add",
        data_distribution_type="FullyReplicated",
        input_mode="File",
        athena_dataset_definition=AthenaDatasetDefinition(
            catalog="AwsDataCatalog",
            database="default",
            work_group="workgroup",
            query_string='SELECT * FROM "default"."s3_test_table_$STAGE_$REGIONUNDERSCORED";',
            output_s3_uri=f"s3://{sagemaker_session_for_pipeline.default_bucket()}/add",
            output_format="JSON",
            output_compression="GZIP",
        ),
    )


def test_pipeline_execution_with_default_experiment_config(
    sagemaker_session_for_pipeline,
    smclient,
    role,
    sklearn_latest_version,
    cpu_instance_type,
    pipeline_name,
    athena_dataset_definition,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")
    inputs = [
        ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/"),
        ProcessingInput(dataset_definition=athena_dataset_definition),
    ]

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role=role,
        instance_type=cpu_instance_type,
        instance_count=instance_count,
        command=["python3"],
        sagemaker_session=sagemaker_session_for_pipeline,
        base_job_name="test-sklearn",
    )

    step_sklearn = ProcessingStep(
        name="sklearn-process",
        processor=sklearn_processor,
        inputs=inputs,
        code=script_path,
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_sklearn],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        pipeline.create(role)
        execution = pipeline.start(parameters={})

        wait_pipeline_execution(execution=execution, max_attempts=3)
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "sklearn-process"

        execution_id = execution.arn.split("/")[-1]

        # trial components
        trial_components = smclient.list_trial_components(TrialName=execution_id)
        assert len(trial_components["TrialComponentSummaries"]) == 1

        # trial details
        trial = smclient.describe_trial(TrialName=execution_id)
        assert pipeline_name == trial["ExperimentName"]
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_pipeline_execution_with_custom_experiment_config(
    sagemaker_session_for_pipeline,
    smclient,
    role,
    sklearn_latest_version,
    cpu_instance_type,
    pipeline_name,
    athena_dataset_definition,
):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")
    inputs = [
        ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/"),
        ProcessingInput(dataset_definition=athena_dataset_definition),
    ]

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role=role,
        instance_type=cpu_instance_type,
        instance_count=instance_count,
        command=["python3"],
        sagemaker_session=sagemaker_session_for_pipeline,
        base_job_name="test-sklearn",
    )

    step_sklearn = ProcessingStep(
        name="sklearn-process",
        processor=sklearn_processor,
        inputs=inputs,
        code=script_path,
    )

    experiment_name = f"my-experiment-{int(time.time() * 10**7)}"

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        pipeline_experiment_config=PipelineExperimentConfig(
            experiment_name=experiment_name,
            trial_name=Join(on="-", values=["my-trial", ExecutionVariables.PIPELINE_EXECUTION_ID]),
        ),
        steps=[step_sklearn],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        pipeline.create(role)
        execution = pipeline.start(parameters={})

        wait_pipeline_execution(execution=execution, max_attempts=3)
        execution_steps = execution.list_steps()
        assert len(execution_steps) == 1
        assert execution_steps[0]["StepName"] == "sklearn-process"

        execution_id = execution.arn.split("/")[-1]

        # trial components
        trial_components = smclient.list_trial_components(TrialName=f"my-trial-{execution_id}")
        assert len(trial_components["TrialComponentSummaries"]) == 1

        # trial details
        trial = smclient.describe_trial(TrialName=f"my-trial-{execution_id}")
        assert experiment_name == trial["ExperimentName"]
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
