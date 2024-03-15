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

import pytest

from sagemaker.processing import ProcessingInput
from tests.integ import DATA_DIR
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.step_outputs import get_step

from sagemaker.workflow.selective_execution_config import SelectiveExecutionConfig

from tests.integ.sagemaker.workflow.helpers import create_and_execute_pipeline
from sagemaker import utils, get_execution_role
from sagemaker.workflow.function_step import step
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

INSTANCE_TYPE = "ml.m5.large"


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("Selective-Pipeline")


def test_selective_execution_among_pure_function_steps(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_without_error
):
    # Test Selective Pipeline Execution on function step1 -> [select: function step2]
    os.environ["AWS_DEFAULT_REGION"] = region_name

    step_settings = dict(
        role=role,
        instance_type=INSTANCE_TYPE,
        image_uri=dummy_container_without_error,
        keep_alive_period_in_seconds=60,
    )

    @step(**step_settings)
    def generator() -> tuple:
        return 3, 4

    @step(**step_settings)
    def sum(a, b):
        """adds two numbers"""
        return a + b

    step_output_a = generator()
    step_output_b = sum(step_output_a[0], step_output_a[1])

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_output_b],
        sagemaker_session=sagemaker_session,
    )

    try:
        execution, _ = create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=2,
            last_step_name_prefix="sum",
            execution_parameters=dict(),
            step_status="Succeeded",
            step_result_type=int,
            step_result_value=7,
        )

        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=2,
            last_step_name_prefix="sum",
            execution_parameters=dict(),
            step_status="Succeeded",
            step_result_type=int,
            step_result_value=7,
            selective_execution_config=SelectiveExecutionConfig(
                source_pipeline_execution_arn=execution.arn,
                selected_steps=[get_step(step_output_b).name],
            ),
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_selective_execution_of_regular_step_referenced_by_function_step(
    sagemaker_session,
    role,
    pipeline_name,
    region_name,
    dummy_container_without_error,
    sklearn_latest_version,
):
    # Test Selective Pipeline Execution on regular step -> [select: function step]
    os.environ["AWS_DEFAULT_REGION"] = region_name

    script_path = os.path.join(DATA_DIR, "dummy_script.py")

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role=role,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        command=["python3"],
        sagemaker_session=sagemaker_session,
        base_job_name="test-sklearn",
    )

    step_sklearn = ProcessingStep(
        name="sklearn-process",
        processor=sklearn_processor,
        code=script_path,
    )

    @step(
        role=role,
        instance_type=INSTANCE_TYPE,
        image_uri=dummy_container_without_error,
        keep_alive_period_in_seconds=60,
    )
    def func_2(arg):
        return arg

    final_output = func_2(step_sklearn.properties.ProcessingJobStatus)

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[final_output],
        sagemaker_session=sagemaker_session,
    )

    try:
        execution, _ = create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=2,
            last_step_name_prefix="func",
            execution_parameters=dict(),
            step_status="Succeeded",
            step_result_type=str,
            step_result_value="Completed",
            wait_duration=600,
        )

        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=2,
            last_step_name_prefix="func",
            execution_parameters=dict(),
            step_status="Succeeded",
            step_result_type=str,
            step_result_value="Completed",
            wait_duration=600,
            selective_execution_config=SelectiveExecutionConfig(
                source_pipeline_execution_arn=execution.arn,
                selected_steps=[get_step(final_output).name],
            ),
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_selective_execution_of_function_step_referenced_by_regular_step(
    pipeline_session,
    role,
    pipeline_name,
    region_name,
    dummy_container_without_error,
    sklearn_latest_version,
):
    # Test Selective Pipeline Execution on function step -> [select: regular step]
    os.environ["AWS_DEFAULT_REGION"] = region_name
    processing_job_instance_counts = 2

    @step(
        name="step1",
        role=role,
        image_uri=dummy_container_without_error,
        instance_type=INSTANCE_TYPE,
        keep_alive_period_in_seconds=60,
    )
    def func(var: int):
        return 1, var

    step_output = func(processing_job_instance_counts)

    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")
    inputs = [
        ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/"),
    ]

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role=role,
        instance_type=INSTANCE_TYPE,
        instance_count=step_output[1],
        command=["python3"],
        sagemaker_session=pipeline_session,
        base_job_name="test-sklearn",
    )

    step_args = sklearn_processor.run(
        inputs=inputs,
        code=script_path,
    )
    process_step = ProcessingStep(
        name="MyProcessStep",
        step_args=step_args,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[process_step],
        sagemaker_session=pipeline_session,
    )

    try:
        execution, _ = create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=2,
            last_step_name_prefix=process_step.name,
            execution_parameters=dict(),
            step_status="Succeeded",
            wait_duration=1000,  # seconds
        )

        _, execution_steps2 = create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=2,
            last_step_name_prefix=process_step.name,
            execution_parameters=dict(),
            step_status="Succeeded",
            wait_duration=1000,  # seconds
            selective_execution_config=SelectiveExecutionConfig(
                source_pipeline_execution_arn=execution.arn,
                selected_steps=[process_step.name],
            ),
        )

        execution_proc_job = pipeline_session.describe_processing_job(
            execution_steps2[0]["Metadata"]["ProcessingJob"]["Arn"].split("/")[-1]
        )
        assert (
            execution_proc_job["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
            == processing_job_instance_counts
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
