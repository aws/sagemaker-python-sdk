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

import numpy
import pytest
import os
import random

from sagemaker import get_execution_role, utils
from sagemaker.config import load_sagemaker_config
from sagemaker.processing import ProcessingInput
from sagemaker.remote_function.errors import RemoteFunctionError
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.remote_function.core.serialization import CloudpickleSerializer
from sagemaker.s3 import S3Uploader
from sagemaker.s3_utils import s3_path_join
from sagemaker.workflow.conditions import ConditionLessThan
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.function_step import step
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
    ParameterBoolean,
)
from sagemaker.workflow.step_outputs import get_step
from sagemaker.workflow.steps import ProcessingStep

from tests.integ.sagemaker.workflow.helpers import (
    create_and_execute_pipeline,
    wait_pipeline_execution,
)
from tests.integ import DATA_DIR

INSTANCE_TYPE = "ml.m5.large"


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("Decorated-Step-Pipeline")


def test_compile_pipeline_with_function_steps(sagemaker_session, role, pipeline_name, region_name):
    @step(
        name="generate",
        role=role,
        instance_type=INSTANCE_TYPE,
    )
    def generate():
        """adds two numbers"""
        return random.randint(0, 100)

    @step(
        name="print",
        role=role,
        instance_type=INSTANCE_TYPE,
    )
    def print_result(result):
        print(result)

    generated = generate()
    conditional_print = ConditionStep(
        name="condition-step",
        # TODO: replace with the generated result
        conditions=[ConditionEquals(left=1, right=1)],
        if_steps=[print_result(generated)],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        sagemaker_session=sagemaker_session,
        steps=[generated, conditional_print],
    )

    try:
        pipeline.create(role_arn=role)

        # verify the artifacts are uploaded to the location specified by sagemaker_session
        assert (
            len(
                sagemaker_session.list_s3_files(
                    sagemaker_session.default_bucket(),
                    f"{sagemaker_session.default_bucket_prefix}/{pipeline_name}/generate",
                )
            )
            > 0
        )

        assert (
            len(
                sagemaker_session.list_s3_files(
                    sagemaker_session.default_bucket(),
                    f"{sagemaker_session.default_bucket_prefix}/{pipeline_name}/print",
                )
            )
            > 0
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_step_decorator_no_dependencies(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_without_error
):
    os.environ["AWS_DEFAULT_REGION"] = region_name

    @step(
        role=role,
        instance_type=INSTANCE_TYPE,
        image_uri=dummy_container_without_error,
        keep_alive_period_in_seconds=60,
    )
    def sum(a, b):
        """adds two numbers"""
        return a + b

    step_output_a = sum(2, 3)
    step_output_b = sum(5, 6)

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_output_a, step_output_b],
        sagemaker_session=sagemaker_session,
    )

    try:
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
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_step_decorator_with_execution_dependencies(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_without_error
):
    os.environ["AWS_DEFAULT_REGION"] = region_name

    @step(
        role=role,
        instance_type=INSTANCE_TYPE,
        image_uri=dummy_container_without_error,
        keep_alive_period_in_seconds=60,
    )
    def sum(a, b):
        """adds two numbers"""
        return a + b

    step_output_a = sum(2, 3)
    step_output_b = sum(5, 6)
    get_step(step_output_b).add_depends_on([step_output_a])

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_output_b],
        sagemaker_session=sagemaker_session,
    )

    try:
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
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_step_decorator_with_data_dependencies(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_without_error
):
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
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_step_decorator_with_pipeline_parameters(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_without_error
):
    os.environ["AWS_DEFAULT_REGION"] = region_name
    instance_type = ParameterString(name="TrainingInstanceCount", default_value=INSTANCE_TYPE)

    @step(
        role=role,
        instance_type=instance_type,
        image_uri=dummy_container_without_error,
        keep_alive_period_in_seconds=60,
    )
    def sum(a, b):
        """adds two numbers"""
        return a + b

    step_a = sum(2, 3)

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_type],
        steps=[step_a],
        sagemaker_session=sagemaker_session,
    )

    try:
        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=1,
            last_step_name_prefix="sum",
            execution_parameters=dict(TrainingInstanceCount="ml.m5.xlarge"),
            step_status="Succeeded",
            step_result_type=int,
            step_result_value=5,
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_passing_different_pipeline_variables_to_function(
    sagemaker_session,
    role,
    pipeline_name,
    region_name,
    dummy_container_without_error,
    sklearn_latest_version,
):
    os.environ["AWS_DEFAULT_REGION"] = region_name

    param_a = ParameterInteger(name="param_a", default_value=2)
    param_b = ParameterBoolean(name="param_b", default_value=True)
    param_c = ParameterFloat(name="param_c", default_value=2.0)
    param_d = ParameterString(name="param_d", default_value="string")

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
        keep_alive_period_in_seconds=600,
    )
    def func_1():
        return 1, 2, {"key": 3}

    @step(
        role=role,
        instance_type=INSTANCE_TYPE,
        image_uri=dummy_container_without_error,
        keep_alive_period_in_seconds=60,
    )
    def func_2(*args):
        return args

    first_output = func_1()
    final_output = func_2(
        param_a,
        param_b,
        param_c,
        param_d,
        step_sklearn.properties.ProcessingJobStatus,
        first_output[2]["key"],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[param_a, param_b, param_c, param_d],
        steps=[final_output],
        sagemaker_session=sagemaker_session,
    )

    try:
        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=3,
            last_step_name_prefix="func",
            execution_parameters=dict(param_a=3),
            step_status="Succeeded",
            step_result_type=tuple,
            step_result_value=(3, True, 2.0, "string", "Completed", 3),
            wait_duration=600,
        )

    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_step_decorator_with_pre_execution_script(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_without_error
):
    os.environ["AWS_DEFAULT_REGION"] = region_name
    pre_execution_script_path = os.path.join(DATA_DIR, "workflow", "pre_exec_commands")

    @step(
        role=role,
        instance_type=INSTANCE_TYPE,
        image_uri=dummy_container_without_error,
        pre_execution_script=pre_execution_script_path,
        keep_alive_period_in_seconds=60,
    )
    def validate_file_exists(files_exists, files_does_not_exist):
        for file_name in files_exists:
            if not os.path.exists(file_name):
                raise ValueError(f"file {file_name} should exist")

        for file_name in files_does_not_exist:
            if os.path.exists(file_name):
                raise ValueError(f"file {file_name} should not exist")

    step_a = validate_file_exists(["test_file_1", "test_file_3"], ["test_file_2"])

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_a],
        sagemaker_session=sagemaker_session,
    )

    try:
        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=1,
            last_step_name_prefix="validate_file_exists",
            execution_parameters=dict(),
            step_status="Succeeded",
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_step_decorator_with_include_local_workdir(
    sagemaker_session, role, pipeline_name, region_name, monkeypatch, dummy_container_without_error
):
    os.environ["AWS_DEFAULT_REGION"] = region_name
    source_dir_path = os.path.join(os.path.dirname(__file__))
    original_sagemaker_config = sagemaker_session.sagemaker_config
    with monkeypatch.context() as m:
        m.chdir(source_dir_path)
        sagemaker_config = load_sagemaker_config(
            [os.path.join(DATA_DIR, "workflow", "config.yaml")]
        )
        sagemaker_session.sagemaker_config = sagemaker_config
        dependencies_path = os.path.join(DATA_DIR, "workflow", "requirements.txt")

        @step(
            role=role,
            instance_type=INSTANCE_TYPE,
            dependencies=dependencies_path,
            keep_alive_period_in_seconds=300,
            image_uri=dummy_container_without_error,
        )
        def train(x):
            from workdir_helpers import local_module
            from workdir_helpers.nested_helper import local_module2

            output = local_module.square(x) + local_module2.cube(x)
            print(output)
            return output

        step_result = train(2)

        pipeline = Pipeline(
            name=pipeline_name,
            steps=[step_result],
            sagemaker_session=sagemaker_session,
        )

        try:
            create_and_execute_pipeline(
                pipeline=pipeline,
                pipeline_name=pipeline_name,
                region_name=region_name,
                role=role,
                no_of_steps=1,
                last_step_name_prefix="train",
                execution_parameters=dict(),
                step_status="Succeeded",
                step_result_type=int,
                step_result_value=12,
            )
        finally:
            try:
                pipeline.delete()
            except Exception:
                pass
    sagemaker_session.sagemaker_config = original_sagemaker_config


def test_decorator_with_conda_env(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_with_conda, conda_env_yml
):
    os.environ["AWS_DEFAULT_REGION"] = region_name

    @step(
        role=role,
        image_uri=dummy_container_with_conda,
        dependencies=conda_env_yml,
        instance_type=INSTANCE_TYPE,
        job_conda_env="integ_test_env",
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    step_a = cuberoot(8)

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_a],
        sagemaker_session=sagemaker_session,
    )

    try:
        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=1,
            last_step_name_prefix="cuberoot",
            execution_parameters=dict(),
            step_status="Succeeded",
            step_result_type=numpy.float64,
            step_result_value=2.0,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_decorator_step_failed(
    sagemaker_session,
    role,
    pipeline_name,
    region_name,
    dummy_container_without_error,
):
    os.environ["AWS_DEFAULT_REGION"] = region_name

    @step(
        role=role,
        image_uri=dummy_container_without_error,
        instance_type=INSTANCE_TYPE,
        keep_alive_period_in_seconds=60,
    )
    def divide(x, y):
        return x / y

    step_a = divide(10, 0)

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_a],
        sagemaker_session=sagemaker_session,
    )

    try:
        execution, execution_steps = create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=1,
            last_step_name_prefix="divide",
            execution_parameters=dict(),
            step_status="Failed",
        )

        step_name = execution_steps[0]["StepName"]
        with pytest.raises(RemoteFunctionError) as e:
            execution.result(step_name)
            assert f"step {step_name} is not in Completed status." in str(e)
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_decorator_step_with_json_get(
    sagemaker_session,
    role,
    pipeline_name,
    region_name,
    dummy_container_without_error,
):
    os.environ["AWS_DEFAULT_REGION"] = region_name

    step_settings = dict(
        role=role,
        image_uri=dummy_container_without_error,
        instance_type=INSTANCE_TYPE,
        keep_alive_period_in_seconds=60,
    )

    @step(name="step1", **step_settings)
    def func1() -> tuple:
        return 0, 1

    @step(name="step2", **step_settings)
    def func2():
        return 2

    @step(name="step3", **step_settings)
    def func3():
        return 3

    step_output1 = func1()
    step_output2 = func2()
    step_output3 = func3()

    cond_lt = ConditionLessThan(left=step_output1[1], right=step_output2)

    fail_step = FailStep(
        name="MyFailStep",
        error_message="Failed due to hitting in else branch",
    )

    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_lt],
        if_steps=[],
        else_steps=[fail_step],
        depends_on=[step_output3],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[cond_step],
        sagemaker_session=sagemaker_session,
    )

    try:
        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=4,  # The FailStep in else branch is not executed
            last_step_name_prefix="MyConditionStep",
            execution_parameters=dict(),
            step_status="Succeeded",
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_decorator_step_data_referenced_by_other_steps(
    pipeline_session,
    role,
    pipeline_name,
    region_name,
    dummy_container_without_error,
    sklearn_latest_version,
):
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
        _, execution_steps = create_and_execute_pipeline(
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

        execution_proc_job = pipeline_session.describe_processing_job(
            execution_steps[0]["Metadata"]["ProcessingJob"]["Arn"].split("/")[-1]
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


def test_decorator_step_checksum_mismatch(
    sagemaker_session, dummy_container_without_error, pipeline_name, role
):
    step_name = "original_func_step"

    @step(
        name=step_name,
        role=role,
        image_uri=dummy_container_without_error,
        instance_type=INSTANCE_TYPE,
        keep_alive_period_in_seconds=60,
    )
    def original_func(x):
        return x * x

    def updated_func(x):
        return x + 25

    pickled_updated_func = CloudpickleSerializer.serialize(updated_func)

    step_a = original_func(10)

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_a],
        sagemaker_session=sagemaker_session,
    )

    try:
        pipeline.create(role)

        pipeline_definition = json.loads(pipeline.describe()["PipelineDefinition"])
        step_container_args = pipeline_definition["Steps"][0]["Arguments"][
            "AlgorithmSpecification"
        ]["ContainerArguments"]
        s3_base_uri = step_container_args[step_container_args.index("--s3_base_uri") + 1]
        build_time = step_container_args[step_container_args.index("--func_step_s3_dir") + 1]

        # some other user updates the pickled function code
        S3Uploader.upload_bytes(
            pickled_updated_func,
            s3_path_join(s3_base_uri, step_name, build_time, "function", "payload.pkl"),
            kms_key=None,
            sagemaker_session=sagemaker_session,
        )
        execution = pipeline.start()
        wait_pipeline_execution(execution=execution, delay=20, max_attempts=20)
        execution_steps = execution.list_steps()

        assert execution_steps[0]["StepStatus"] == "Failed"
        assert (
            "Integrity check for the serialized function or data failed"
            in execution_steps[0]["FailureReason"]
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_with_user_and_workdir_set_in_the_image(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_with_user_and_workdir
):
    os.environ["AWS_DEFAULT_REGION"] = region_name
    dependencies_path = os.path.join(DATA_DIR, "workflow", "requirements.txt")

    @step(
        role=role,
        image_uri=dummy_container_with_user_and_workdir,
        dependencies=dependencies_path,
        instance_type=INSTANCE_TYPE,
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    step_a = cuberoot(8)

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_a],
        sagemaker_session=sagemaker_session,
    )

    try:
        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=1,
            last_step_name_prefix="cuberoot",
            execution_parameters=dict(),
            step_status="Succeeded",
            step_result_type=numpy.float64,
            step_result_value=2.0,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_with_user_and_workdir_set_in_the_image_client_error_case(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_with_user_and_workdir
):
    # This test aims to ensure client error in step decorated function
    # can be successfully surfaced and the job can be failed.
    os.environ["AWS_DEFAULT_REGION"] = region_name
    client_error_message = "Testing client error in job."

    @step(
        role=role,
        image_uri=dummy_container_with_user_and_workdir,
        instance_type=INSTANCE_TYPE,
    )
    def my_func():
        raise RuntimeError(client_error_message)

    step_a = my_func()

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_a],
        sagemaker_session=sagemaker_session,
    )

    try:
        _, execution_steps = create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=1,
            last_step_name_prefix=get_step(step_a).name,
            execution_parameters=dict(),
            step_status="Failed",
        )
        assert (
            f"ClientError: AlgorithmError: RuntimeError('{client_error_message}')"
            in execution_steps[0]["FailureReason"]
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_step_level_serialization(
    sagemaker_session, role, pipeline_name, region_name, dummy_container_without_error
):
    os.environ["AWS_DEFAULT_REGION"] = region_name

    _EXPECTED_STEP_A_OUTPUT = "This pipeline is a function."
    _EXPECTED_STEP_B_OUTPUT = "This generates a function arg."

    step_config = dict(
        role=role,
        image_uri=dummy_container_without_error,
        instance_type=INSTANCE_TYPE,
    )

    # This pipeline function may clash with the pipeline object
    # defined below.
    # However, if the function and args serialization happen in
    # step level, this clash won't happen.
    def pipeline():
        return _EXPECTED_STEP_A_OUTPUT

    @step(**step_config)
    def generator():
        return _EXPECTED_STEP_B_OUTPUT

    @step(**step_config)
    def func_with_collision(var: str):
        return f"{pipeline()} {var}"

    step_output_a = generator()
    step_output_b = func_with_collision(step_output_a)

    pipeline = Pipeline(  # noqa: F811
        name=pipeline_name,
        steps=[step_output_b],
        sagemaker_session=sagemaker_session,
    )

    try:
        create_and_execute_pipeline(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            region_name=region_name,
            role=role,
            no_of_steps=2,
            last_step_name_prefix=get_step(step_output_b).name,
            execution_parameters=dict(),
            step_status="Succeeded",
            step_result_type=str,
            step_result_value=f"{_EXPECTED_STEP_A_OUTPUT} {_EXPECTED_STEP_B_OUTPUT}",
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
