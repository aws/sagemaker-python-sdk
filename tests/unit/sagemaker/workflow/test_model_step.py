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

from mock import patch

import pytest

from sagemaker import Model, PipelineModel, Session, Processor
from sagemaker.chainer import ChainerModel
from sagemaker.estimator import Estimator
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.model import SCRIPT_PARAM_NAME, DIR_PARAM_NAME
from sagemaker.mxnet import MXNetModel
from sagemaker.parameter import IntegerParameter
from sagemaker.pytorch import PyTorchModel
from sagemaker.sklearn import SKLearnModel
from sagemaker.sparkml import SparkMLModel
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow._utils import REPACK_SCRIPT_LAUNCHER
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import (
    ModelStep,
    _REGISTER_MODEL_NAME_BASE,
    _CREATE_MODEL_NAME_BASE,
    _REPACK_MODEL_NAME_BASE,
)
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
)
from sagemaker.xgboost import XGBoostModel
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from tests.unit import DATA_DIR
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered
from tests.unit.sagemaker.workflow.conftest import BUCKET, ROLE

_IMAGE_URI = "fakeimage"
_INSTANCE_TYPE = "ml.m4.xlarge"

_SAGEMAKER_PROGRAM = SCRIPT_PARAM_NAME.upper()
_SAGEMAKER_SUBMIT_DIRECTORY = DIR_PARAM_NAME.upper()
_SCRIPT_NAME = "dummy_script.py"
_DIR_NAME = "/opt/ml/model/code"
_XGBOOST_PATH = os.path.join(DATA_DIR, "xgboost_abalone")
_TENSORFLOW_PATH = os.path.join(DATA_DIR, "tfs/tfs-test-entrypoint-and-dependencies")
_REPACK_OUTPUT_KEY_PREFIX = "code-output"
_MODEL_CODE_LOCATION = f"s3://{BUCKET}/{_REPACK_OUTPUT_KEY_PREFIX}"
_MODEL_CODE_LOCATION_TRAILING_SLASH = _MODEL_CODE_LOCATION + "/"


@pytest.fixture
def model_data_param():
    return ParameterString(name="ModelData", default_value="s3://my-bucket/file")


@pytest.fixture
def model(pipeline_session, model_data_param):
    return Model(
        name="MyModel",
        image_uri=_IMAGE_URI,
        model_data=model_data_param,
        sagemaker_session=pipeline_session,
        entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
        source_dir=f"{DATA_DIR}",
        role=ROLE,
    )


def test_register_model_with_runtime_repack(pipeline_session, model_data_param, model):
    custom_step = CustomStep("TestStep")
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
    )
    model_step = ModelStep(
        name="MyModelStep",
        step_args=step_args,
        retry_policies=dict(
            register_model_retry_policies=[
                StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
            ],
            repack_model_retry_policies=[
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.CAPACITY_ERROR], max_attempts=3
                )
            ],
        ),
        depends_on=["TestStep"],
        description="my model step description",
    )
    custom_step2 = CustomStep("TestStep2", depends_on=[model_step])
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_data_param],
        steps=[custom_step, model_step, custom_step2],
        sagemaker_session=pipeline_session,
    )
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 4
    expected_repack_step_name = f"MyModelStep-{_REPACK_MODEL_NAME_BASE}-MyModel"
    # Filter out the dummy custom step
    step_dsl_list = list(filter(lambda s: not s["Name"].startswith("TestStep"), step_dsl_list))
    for step in step_dsl_list[0:2]:
        if step["Type"] == "Training":
            assert step["Name"] == expected_repack_step_name
            assert len(step["DependsOn"]) == 1
            assert step["DependsOn"][0] == "TestStep"
            arguments = step["Arguments"]
            assert len(arguments["InputDataConfig"]) == 1
            assert arguments["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
                "Get": "Parameters.ModelData"
            }
            assert arguments["HyperParameters"]["inference_script"] == '"dummy_script.py"'
            assert arguments["HyperParameters"]["model_archive"] == {"Get": "Parameters.ModelData"}
            assert (
                arguments["HyperParameters"]["sagemaker_program"] == f'"{REPACK_SCRIPT_LAUNCHER}"'
            )
            assert "s3://" in arguments["HyperParameters"]["sagemaker_submit_directory"]
            assert arguments["HyperParameters"]["dependencies"] == "null"
            assert step["RetryPolicies"] == [
                {
                    "BackoffRate": 2.0,
                    "IntervalSeconds": 1,
                    "MaxAttempts": 3,
                    "ExceptionType": ["SageMaker.CAPACITY_ERROR"],
                }
            ]
            assert "repack a model with customer scripts" in step["Description"]
        elif step["Type"] == "RegisterModel":
            assert step["Name"] == f"MyModelStep-{_REGISTER_MODEL_NAME_BASE}"
            assert not step.get("DependsOn", None)
            arguments = step["Arguments"]
            assert arguments["ModelPackageGroupName"] == "MyModelPackageGroup"
            assert arguments["ModelApprovalStatus"] == "PendingManualApproval"
            assert len(arguments["InferenceSpecification"]["Containers"]) == 1
            container = arguments["InferenceSpecification"]["Containers"][0]
            assert container["Image"] == _IMAGE_URI
            assert container["ModelDataUrl"] == {
                "Get": f"Steps.{expected_repack_step_name}.ModelArtifacts.S3ModelArtifacts"
            }
            assert container["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
            assert container["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY] == _DIR_NAME
            assert step["RetryPolicies"] == [
                {
                    "BackoffRate": 2.0,
                    "IntervalSeconds": 1,
                    "MaxAttempts": 3,
                    "ExceptionType": ["Step.THROTTLING"],
                }
            ]
            assert "my model step description" in step["Description"]
        else:
            raise Exception("A step exists in the collection of an invalid type.")

    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "TestStep": ["MyModelStep-RepackModel-MyModel"],
            "MyModelStep-RepackModel-MyModel": ["MyModelStep-RegisterModel"],
            "MyModelStep-RegisterModel": ["TestStep2"],
            "TestStep2": [],
        }
    )


def test_create_model_with_runtime_repack(pipeline_session, model_data_param, model):
    step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    model_steps = ModelStep(
        name="MyModelStep",
        step_args=step_args,
        description="my model step description",
        retry_policies=[
            StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
        ],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_data_param],
        steps=[model_steps],
        sagemaker_session=pipeline_session,
    )
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 2
    expected_repack_step_name = f"MyModelStep-{_REPACK_MODEL_NAME_BASE}-MyModel"
    for step in step_dsl_list:
        if step["Type"] == "Training":
            assert step["Name"] == expected_repack_step_name
            arguments = step["Arguments"]
            assert len(arguments["InputDataConfig"]) == 1
            assert arguments["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
                "Get": "Parameters.ModelData"
            }
            assert arguments["HyperParameters"]["inference_script"] == '"dummy_script.py"'
            assert arguments["HyperParameters"]["model_archive"] == {"Get": "Parameters.ModelData"}
            assert (
                arguments["HyperParameters"]["sagemaker_program"] == f'"{REPACK_SCRIPT_LAUNCHER}"'
            )
            assert "s3://" in arguments["HyperParameters"]["sagemaker_submit_directory"]
            assert arguments["HyperParameters"]["dependencies"] == "null"
            assert "repack a model with customer scripts" in step["Description"]
            assert step["RetryPolicies"] == [
                {
                    "BackoffRate": 2.0,
                    "IntervalSeconds": 1,
                    "MaxAttempts": 3,
                    "ExceptionType": ["Step.THROTTLING"],
                }
            ]
        elif step["Type"] == "Model":
            assert step["Name"] == f"MyModelStep-{_CREATE_MODEL_NAME_BASE}"
            arguments = step["Arguments"]
            container = arguments["PrimaryContainer"]
            assert container["Image"] == _IMAGE_URI
            assert container["ModelDataUrl"] == {
                "Get": f"Steps.{expected_repack_step_name}.ModelArtifacts.S3ModelArtifacts"
            }
            assert container["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
            assert container["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY] == _DIR_NAME
            assert "my model step description" in step["Description"]
            assert step["RetryPolicies"] == [
                {
                    "BackoffRate": 2.0,
                    "IntervalSeconds": 1,
                    "MaxAttempts": 3,
                    "ExceptionType": ["Step.THROTTLING"],
                }
            ]
        else:
            raise Exception("A step exists in the collection of an invalid type.")
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyModelStep-CreateModel": [],
            "MyModelStep-RepackModel-MyModel": ["MyModelStep-CreateModel"],
        }
    )


def test_create_pipeline_model_with_runtime_repack(pipeline_session, model_data_param, model):
    # The model no need to runtime repack, as entry_point and source_dir are missing
    sparkml_model = SparkMLModel(
        name="MySparkMLModel",
        model_data=model_data_param,
        role=ROLE,
        sagemaker_session=pipeline_session,
        env={"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"},
    )
    # The model need to runtime repack
    ppl_model = PipelineModel(
        models=[sparkml_model, model], role=ROLE, sagemaker_session=pipeline_session
    )
    step_args = ppl_model.create(
        instance_type="c4.4xlarge",
    )
    model_steps = ModelStep(
        name="MyModelStep",
        step_args=step_args,
        retry_policies=dict(
            create_model_retry_policies=[
                StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
            ],
            repack_model_retry_policies=[
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.CAPACITY_ERROR], max_attempts=3
                )
            ],
        ),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_data_param],
        steps=[model_steps],
        sagemaker_session=pipeline_session,
    )
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 2
    expected_repack_step_name = f"MyModelStep-{_REPACK_MODEL_NAME_BASE}-MyModel"
    for step in step_dsl_list:
        if step["Type"] == "Training":
            assert step["Name"] == expected_repack_step_name
            arguments = step["Arguments"]
            assert len(arguments["InputDataConfig"]) == 1
            assert arguments["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
                "Get": "Parameters.ModelData"
            }
            assert arguments["HyperParameters"]["inference_script"] == '"dummy_script.py"'
            assert arguments["HyperParameters"]["model_archive"] == {"Get": "Parameters.ModelData"}
            assert (
                arguments["HyperParameters"]["sagemaker_program"] == f'"{REPACK_SCRIPT_LAUNCHER}"'
            )
            assert "s3://" in arguments["HyperParameters"]["sagemaker_submit_directory"]
            assert arguments["HyperParameters"]["dependencies"] == "null"
            assert step["RetryPolicies"] == [
                {
                    "BackoffRate": 2.0,
                    "IntervalSeconds": 1,
                    "MaxAttempts": 3,
                    "ExceptionType": ["SageMaker.CAPACITY_ERROR"],
                }
            ]
        elif step["Type"] == "Model":
            assert step["Name"] == f"MyModelStep-{_CREATE_MODEL_NAME_BASE}"
            arguments = step["Arguments"]
            containers = arguments["Containers"]
            assert len(containers) == 2
            assert containers[0]["Image"]
            assert containers[0]["ModelDataUrl"] == {"Get": "Parameters.ModelData"}
            assert containers[1]["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
            assert containers[1]["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY] == _DIR_NAME
            assert containers[1]["Image"] == _IMAGE_URI
            assert containers[1]["ModelDataUrl"] == {
                "Get": f"Steps.{expected_repack_step_name}.ModelArtifacts.S3ModelArtifacts"
            }
            assert containers[1]["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
            assert containers[1]["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY] == _DIR_NAME
            assert step["RetryPolicies"] == [
                {
                    "BackoffRate": 2.0,
                    "IntervalSeconds": 1,
                    "MaxAttempts": 3,
                    "ExceptionType": ["Step.THROTTLING"],
                }
            ]
        else:
            raise Exception("A step exists in the collection of an invalid type.")
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyModelStep-CreateModel": [],
            "MyModelStep-RepackModel-MyModel": ["MyModelStep-CreateModel"],
        }
    )


def test_register_pipeline_model_with_runtime_repack(pipeline_session, model_data_param):
    # The model no need to runtime repack, since source_dir is missing
    sparkml_model = SparkMLModel(
        model_data=model_data_param,
        role=ROLE,
        sagemaker_session=pipeline_session,
        env={"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"},
        entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
    )
    # The model need to runtime repack
    model = Model(
        image_uri=_IMAGE_URI,
        model_data=model_data_param,
        sagemaker_session=pipeline_session,
        entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
        source_dir=f"{DATA_DIR}",
        role=ROLE,
        env={"k": "v"},
    )
    model = PipelineModel(
        models=[sparkml_model, model], role=ROLE, sagemaker_session=pipeline_session
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
    )
    model_steps = ModelStep(
        name="MyModelStep",
        step_args=step_args,
    )
    custom_step = CustomStep("TestStep", input_data=model_steps.properties.ModelApprovalStatus)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_data_param],
        steps=[model_steps, custom_step],
        sagemaker_session=pipeline_session,
    )
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 3
    # Filter out the dummy custom step
    step_dsl_list = list(filter(lambda s: not s["Name"].startswith("TestStep"), step_dsl_list))
    expected_repack_step_name = f"MyModelStep-{_REPACK_MODEL_NAME_BASE}-1"
    for step in step_dsl_list:
        if step["Type"] == "Training":
            assert step["Name"] == expected_repack_step_name
            arguments = step["Arguments"]
            assert len(arguments["InputDataConfig"]) == 1
            assert arguments["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
                "Get": "Parameters.ModelData"
            }
            assert arguments["HyperParameters"]["inference_script"] == '"dummy_script.py"'
            assert arguments["HyperParameters"]["model_archive"] == {"Get": "Parameters.ModelData"}
            assert (
                arguments["HyperParameters"]["sagemaker_program"] == f'"{REPACK_SCRIPT_LAUNCHER}"'
            )
            assert "s3://" in arguments["HyperParameters"]["sagemaker_submit_directory"]
            assert arguments["HyperParameters"]["dependencies"] == "null"
        elif step["Type"] == "RegisterModel":
            assert step["Name"] == f"MyModelStep-{_REGISTER_MODEL_NAME_BASE}"
            arguments = step["Arguments"]
            assert arguments["ModelPackageGroupName"] == "MyModelPackageGroup"
            assert arguments["ModelApprovalStatus"] == "PendingManualApproval"
            containers = arguments["InferenceSpecification"]["Containers"]
            assert len(containers) == 2
            assert containers[0]["Image"]
            assert containers[0]["ModelDataUrl"] == {"Get": "Parameters.ModelData"}
            assert containers[0]["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
            assert "s3://" in containers[0]["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY]
            assert containers[1]["Image"] == _IMAGE_URI
            assert containers[1]["ModelDataUrl"] == {
                "Get": f"Steps.{expected_repack_step_name}.ModelArtifacts.S3ModelArtifacts"
            }
            assert containers[1]["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
            assert containers[1]["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY] == _DIR_NAME
            assert containers[1]["Environment"]["k"] == "v"
        else:
            raise Exception("A step exists in the collection of an invalid type.")
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyModelStep-RegisterModel": ["TestStep"],
            "MyModelStep-RepackModel-1": ["MyModelStep-RegisterModel"],
            "TestStep": [],
        }
    )


def test_register_model_without_repack(pipeline_session):
    # The model does not need repack as source_dir is missing
    # However, as entry_point is given, it uploads a separate sourcedir.tar.gaz
    # and specify it plus entry point with these env vars:
    # SAGEMAKER_SUBMIT_DIRECTORY, SAGEMAKER_PROGRAM
    model_data = ParameterString(name="ModelData", default_value="file://file")
    model_name = "MyModel"
    model = Model(
        name=model_name,
        image_uri=_IMAGE_URI,
        model_data=model_data,
        entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
        sagemaker_session=pipeline_session,
        role=ROLE,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
    )
    model_steps = ModelStep(
        name="MyModelStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_data],
        steps=[model_steps],
        sagemaker_session=pipeline_session,
    )
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 1
    assert step_dsl_list[0]["Name"] == f"MyModelStep-{_REGISTER_MODEL_NAME_BASE}"
    arguments = step_dsl_list[0]["Arguments"]
    assert arguments["ModelApprovalStatus"] == "PendingManualApproval"
    containers = arguments["InferenceSpecification"]["Containers"]
    assert len(containers) == 1
    assert containers[0]["Image"] == _IMAGE_URI
    assert containers[0]["ModelDataUrl"] == {"Get": "Parameters.ModelData"}
    assert containers[0]["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
    assert (
        containers[0]["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY]
        == f"s3://{BUCKET}/{model_name}/sourcedir.tar.gz"
    )
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered({"MyModelStep-RegisterModel": []})


@patch("sagemaker.utils.repack_model")
def test_create_model_with_compile_time_repack(mock_repack, pipeline_session):
    custom_step = CustomStep("TestStep")
    model_name = "MyModel"
    model = Model(
        name=model_name,
        image_uri=_IMAGE_URI,
        model_data=f"s3://{BUCKET}/model.tar.gz",
        sagemaker_session=pipeline_session,
        entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
        source_dir=f"{DATA_DIR}",
        role=ROLE,
    )
    step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    model_steps = ModelStep(name="MyModelStep", step_args=step_args, depends_on=["TestStep"])
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[model_steps, custom_step],
        sagemaker_session=pipeline_session,
    )
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 2
    assert step_dsl_list[0]["Name"] == "MyModelStep-CreateModel"
    arguments = step_dsl_list[0]["Arguments"]
    assert arguments["PrimaryContainer"]["Image"] == _IMAGE_URI
    assert (
        arguments["PrimaryContainer"]["ModelDataUrl"] == f"s3://{BUCKET}/{model_name}/model.tar.gz"
    )
    assert arguments["PrimaryContainer"]["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
    assert arguments["PrimaryContainer"]["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY] == _DIR_NAME
    assert len(step_dsl_list[0]["DependsOn"]) == 1
    assert step_dsl_list[0]["DependsOn"][0] == "TestStep"
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"MyModelStep-CreateModel": [], "TestStep": ["MyModelStep-CreateModel"]}
    )


def test_conditional_model_create_and_regis(
    pipeline_session,
    model_data_param,
    model,
):
    good_enough_input = ParameterInteger(name="GoodEnoughInput", default_value=1)
    # register model with runtime repack
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
    )
    step_model_regis = ModelStep(
        name="MyModelStepRegis",
        step_args=step_args,
    )
    # create model without runtime repack
    model.entry_point = None
    model.source_dir = None
    step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    step_model_create = ModelStep(
        name="MyModelStepCreate",
        step_args=step_args,
    )
    step_cond = ConditionStep(
        name="cond-good-enough",
        conditions=[
            ConditionGreaterThanOrEqualTo(left=good_enough_input, right=1),
        ],
        if_steps=[step_model_regis],
        else_steps=[step_model_create],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[good_enough_input, model_data_param],
        steps=[step_cond],
        sagemaker_session=pipeline_session,
    )
    cond_step_dsl = json.loads(pipeline.definition())["Steps"][0]
    step_dsl_list = cond_step_dsl["Arguments"]["IfSteps"] + cond_step_dsl["Arguments"]["ElseSteps"]
    assert len(step_dsl_list) == 3
    expected_repack_step_name = f"MyModelStepRegis-{_REPACK_MODEL_NAME_BASE}-MyModel"
    for step in step_dsl_list:
        if step["Type"] == "Training":
            assert step["Name"] == expected_repack_step_name
            arguments = step["Arguments"]
            assert len(arguments["InputDataConfig"]) == 1
            assert arguments["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == {
                "Get": "Parameters.ModelData"
            }
            assert arguments["HyperParameters"]["inference_script"] == '"dummy_script.py"'
            assert arguments["HyperParameters"]["model_archive"] == {"Get": "Parameters.ModelData"}
            assert (
                arguments["HyperParameters"]["sagemaker_program"] == f'"{REPACK_SCRIPT_LAUNCHER}"'
            )
            assert "s3://" in arguments["HyperParameters"]["sagemaker_submit_directory"]
            assert arguments["HyperParameters"]["dependencies"] == "null"
        elif step["Type"] == "RegisterModel":
            assert step["Name"] == f"MyModelStepRegis-{_REGISTER_MODEL_NAME_BASE}"
            arguments = step["Arguments"]
            assert arguments["ModelApprovalStatus"] == "PendingManualApproval"
            assert len(arguments["InferenceSpecification"]["Containers"]) == 1
            container = arguments["InferenceSpecification"]["Containers"][0]
            assert container["Image"] == _IMAGE_URI
            assert container["ModelDataUrl"] == {
                "Get": f"Steps.{expected_repack_step_name}.ModelArtifacts.S3ModelArtifacts"
            }
            assert container["Environment"][_SAGEMAKER_PROGRAM] == _SCRIPT_NAME
            assert container["Environment"][_SAGEMAKER_SUBMIT_DIRECTORY] == _DIR_NAME
        elif step["Type"] == "Model":
            assert step["Name"] == f"MyModelStepCreate-{_CREATE_MODEL_NAME_BASE}"
            arguments = step["Arguments"]
            container = arguments["PrimaryContainer"]
            assert container["Image"] == _IMAGE_URI
            assert container["ModelDataUrl"] == {"Get": "Parameters.ModelData"}
            assert not container.get("Environment", {})
        else:
            raise Exception("A step exists in the collection of an invalid type.")
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyModelStepCreate-CreateModel": [],
            "MyModelStepRegis-RegisterModel": [],
            "MyModelStepRegis-RepackModel-MyModel": ["MyModelStepRegis-RegisterModel"],
            "cond-good-enough": [
                "MyModelStepCreate-CreateModel",
                "MyModelStepRegis-RepackModel-MyModel",
            ],
        }
    )


@pytest.mark.parametrize(
    "test_input",
    [
        (
            SKLearnModel(
                name="MySKModel",
                model_data="dummy_model_data",
                image_uri=_IMAGE_URI,
                entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
                role=ROLE,
                enable_network_isolation=True,
                code_location=_MODEL_CODE_LOCATION_TRAILING_SLASH,
            ),
            2,
        ),
        (
            XGBoostModel(
                name="MYXGBoostModel",
                model_data="dummy_model_data",
                framework_version="1.11.0",
                image_uri=_IMAGE_URI,
                entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
                role=ROLE,
                enable_network_isolation=False,
            ),
            1,
        ),
        (
            PyTorchModel(
                name="MyPyTorchModel",
                model_data="dummy_model_data",
                image_uri=_IMAGE_URI,
                entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
                role=ROLE,
                framework_version="1.5.0",
                code_location=_MODEL_CODE_LOCATION_TRAILING_SLASH,
            ),
            2,
        ),
        (
            MXNetModel(
                name="MyMXNetModel",
                model_data="dummy_model_data",
                image_uri=_IMAGE_URI,
                entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
                role=ROLE,
                framework_version="1.2.0",
            ),
            1,
        ),
        (
            HuggingFaceModel(
                name="MyHuggingFaceModel",
                model_data="dummy_model_data",
                image_uri=_IMAGE_URI,
                entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
                role=ROLE,
            ),
            2,
        ),
        (
            TensorFlowModel(
                name="MyTensorFlowModel",
                model_data="dummy_model_data",
                image_uri=_IMAGE_URI,
                entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
                role=ROLE,
                code_location=_MODEL_CODE_LOCATION_TRAILING_SLASH,
            ),
            2,
        ),
        (
            ChainerModel(
                name="MyChainerModel",
                model_data="dummy_model_data",
                image_uri=_IMAGE_URI,
                entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
                role=ROLE,
            ),
            1,
        ),
    ],
)
def test_create_model_among_different_model_types(test_input, pipeline_session, model_data_param):
    def assert_test_result(steps: list):
        # If expected_step_num is 2, it means a runtime repack step is appended
        # If expected_step_num is 1, it means no runtime repack is needed
        assert len(steps) == expected_step_num
        if expected_step_num == 2:
            assert steps[0]["Type"] == "Training"
            if model.key_prefix is not None and model.key_prefix.startswith(
                _REPACK_OUTPUT_KEY_PREFIX
            ):
                assert steps[0]["Arguments"]["OutputDataConfig"]["S3OutputPath"] == (
                    f"{_MODEL_CODE_LOCATION}/{model.name}"
                )
            else:
                assert steps[0]["Arguments"]["OutputDataConfig"]["S3OutputPath"] == (
                    f"s3://{BUCKET}/{model.name}"
                )

    model, expected_step_num = test_input
    model.sagemaker_session = pipeline_session
    model.model_data = model_data_param
    create_model_step_args = model.create(
        instance_type="c4.4xlarge",
    )
    create_model_steps = ModelStep(
        name="MyModelStep",
        step_args=create_model_step_args,
    )
    assert_test_result(create_model_steps.request_dicts())

    register_model_step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
    )
    register_model_steps = ModelStep(
        name="MyModelStep",
        step_args=register_model_step_args,
    )
    assert_test_result(register_model_steps.request_dicts())


@pytest.mark.parametrize(
    "test_input",
    [
        # Assign entry_point and enable_network_isolation to the XGBoostModel
        # which will trigger a model runtime repacking and update
        # the container env with SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code
        (
            XGBoostModel(
                model_data="dummy_model_step",
                framework_version="1.3-1",
                role=ROLE,
                entry_point=os.path.join(_XGBOOST_PATH, "inference.py"),
                enable_network_isolation=True,
            ),
            {
                "expected_step_num": 2,
                f"{_SAGEMAKER_PROGRAM}": "inference.py",
                f"{_SAGEMAKER_SUBMIT_DIRECTORY}": f"{_DIR_NAME}",
            },
        ),
        # Assign entry_point only to the XGBoostModel
        # which will NOT trigger a model runtime repacking but update the container env with
        # SAGEMAKER_SUBMIT_DIRECTORY is a s3 uri pointing to the entry_point script
        (
            XGBoostModel(
                model_data="dummy_model_step",
                framework_version="1.3-1",
                role=ROLE,
                entry_point=os.path.join(_XGBOOST_PATH, "inference.py"),
            ),
            {
                "expected_step_num": 1,
                f"{_SAGEMAKER_PROGRAM}": "inference.py",
                f"{_SAGEMAKER_SUBMIT_DIRECTORY}": "s3://",
            },
        ),
        # Not assigning entry_point to the XGBoostModel
        # which will NOT trigger a model runtime repacking. Container env
        # SAGEMAKER_SUBMIT_DIRECTORY, SAGEMAKER_PROGRAM are empty
        (
            XGBoostModel(
                model_data="dummy_model_step",
                framework_version="1.3-1",
                role=ROLE,
                entry_point=None,
            ),
            {
                "expected_step_num": 1,
                f"{_SAGEMAKER_PROGRAM}": "",
                f"{_SAGEMAKER_SUBMIT_DIRECTORY}": "",
            },
        ),
        # Assign entry_point to the TensorFlowModel
        # which will trigger a model runtime repacking
        # TensorFlowModel does not configure the container Environment
        (
            TensorFlowModel(
                model_data="dummy_model_step",
                role=ROLE,
                image_uri=_IMAGE_URI,
                entry_point=os.path.join(_TENSORFLOW_PATH, "inference.py"),
            ),
            {
                "expected_step_num": 2,
                f"{_SAGEMAKER_PROGRAM}": None,
                f"{_SAGEMAKER_SUBMIT_DIRECTORY}": None,
            },
        ),
        # Not assigning entry_point to the TensorFlowModel
        # which will NOT trigger a model runtime repacking
        # TensorFlowModel does not configure the container Environment
        (
            TensorFlowModel(
                model_data="dummy_model_step",
                role=ROLE,
                image_uri=_IMAGE_URI,
            ),
            {
                "expected_step_num": 1,
                f"{_SAGEMAKER_PROGRAM}": None,
                f"{_SAGEMAKER_SUBMIT_DIRECTORY}": None,
            },
        ),
    ],
)
@patch("sagemaker.utils.repack_model")
def test_request_compare_of_register_model_under_different_sessions(
    mock_repack, test_input, pipeline_session, sagemaker_session, model_data_param
):
    model_package_group_name = "TestModelPackageGroup"
    model, expect = test_input
    expected_step_num = expect["expected_step_num"]

    # Get create model package request under PipelineSession
    model.model_data = model_data_param
    model.sagemaker_session = pipeline_session
    step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
    )
    step_model = ModelStep(
        name="MyModelStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[model_data_param],
        steps=[step_model],
        sagemaker_session=pipeline_session,
    )
    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == expected_step_num
    # the step arg is used as request to create model package in backend
    regis_step_arg = step_dsl_list[expected_step_num - 1]["Arguments"]
    _verify_register_model_container_definition(regis_step_arg, expect, dict)

    # Get create model package request under Session
    model.model_data = f"s3://{BUCKET}"
    model.sagemaker_session = sagemaker_session
    with patch.object(
        Session, "_intercept_create_request", return_value=dict(ModelPackageArn="arn:aws")
    ) as mock_method:
        model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status="Approved",
        )
        register_model_request = mock_method.call_args[0][0]
        _verify_register_model_container_definition(register_model_request, expect, str)
        register_model_request.pop("CertifyForMarketplace", None)
        assert regis_step_arg == register_model_request


def _verify_register_model_container_definition(
    request: dict, expect: dict, expected_model_data_type: type
):
    expected_submit_dir = expect[_SAGEMAKER_SUBMIT_DIRECTORY]
    expected_program = expect[_SAGEMAKER_PROGRAM]
    containers = request["InferenceSpecification"]["Containers"]
    assert len(containers) == 1
    isinstance(containers[0].pop("ModelDataUrl"), expected_model_data_type)
    container_env = containers[0]["Environment"]
    assert container_env.pop(_SAGEMAKER_PROGRAM, None) == expected_program
    submit_dir = container_env.pop(_SAGEMAKER_SUBMIT_DIRECTORY, None)
    if submit_dir and not submit_dir.startswith("s3://"):
        # exclude the s3 path assertion as it contains timestamp
        assert submit_dir == expected_submit_dir


def test_model_step_with_lambda_property_reference(pipeline_session):
    lambda_step = LambdaStep(
        name="MyLambda",
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda"
        ),
        outputs=[
            LambdaOutput(output_name="model_image", output_type=LambdaOutputTypeEnum.String),
            LambdaOutput(output_name="model_artifact", output_type=LambdaOutputTypeEnum.String),
        ],
    )

    model = PyTorchModel(
        name="MyModel",
        framework_version="1.8.0",
        py_version="py3",
        image_uri=lambda_step.properties.Outputs["model_image"],
        model_data=lambda_step.properties.Outputs["model_artifact"],
        sagemaker_session=pipeline_session,
        entry_point=f"{DATA_DIR}/{_SCRIPT_NAME}",
        role=ROLE,
    )

    step_create_model = ModelStep(name="mymodelstep", step_args=model.create())

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[lambda_step, step_create_model],
        sagemaker_session=pipeline_session,
    )
    steps = json.loads(pipeline.definition())["Steps"]
    repack_step = steps[1]
    assert repack_step["Arguments"]["InputDataConfig"][0]["DataSource"]["S3DataSource"][
        "S3Uri"
    ] == {"Get": "Steps.MyLambda.OutputParameters['model_artifact']"}
    register_step = steps[2]
    assert register_step["Arguments"]["PrimaryContainer"]["Image"] == {
        "Get": "Steps.MyLambda.OutputParameters['model_image']"
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyLambda": ["mymodelstep-CreateModel", "mymodelstep-RepackModel-MyModel"],
            "mymodelstep-CreateModel": [],
            "mymodelstep-RepackModel-MyModel": ["mymodelstep-CreateModel"],
        }
    )


@pytest.mark.parametrize(
    "inputs",
    [
        (
            Processor(
                image_uri=_IMAGE_URI,
                role=ROLE,
                instance_count=1,
                instance_type=_INSTANCE_TYPE,
            ),
            dict(target_fun="run", func_args={}),
        ),
        (
            Transformer(
                model_name="model_name",
                instance_type="ml.m5.xlarge",
                instance_count=1,
                output_path="s3://Transform",
            ),
            dict(
                target_fun="transform",
                func_args=dict(data="s3://data", job_name="test"),
            ),
        ),
        (
            HyperparameterTuner(
                estimator=Estimator(
                    role=ROLE,
                    instance_count=1,
                    instance_type=_INSTANCE_TYPE,
                    image_uri=_IMAGE_URI,
                ),
                objective_metric_name="test:acc",
                hyperparameter_ranges={"batch-size": IntegerParameter(64, 128)},
            ),
            dict(target_fun="fit", func_args={}),
        ),
        (
            Estimator(
                role=ROLE,
                instance_count=1,
                instance_type=_INSTANCE_TYPE,
                image_uri=_IMAGE_URI,
            ),
            dict(target_fun="fit", func_args={}),
        ),
    ],
)
def test_insert_wrong_step_args_into_model_step(inputs, pipeline_session):
    downstream_obj, target_func_cfg = inputs
    if isinstance(downstream_obj, HyperparameterTuner):
        downstream_obj.estimator.sagemaker_session = pipeline_session
    else:
        downstream_obj.sagemaker_session = pipeline_session
    func_name = target_func_cfg["target_fun"]
    func_args = target_func_cfg["func_args"]
    step_args = getattr(downstream_obj, func_name)(**func_args)

    with pytest.raises(ValueError) as error:
        ModelStep(
            name="MyModelStep",
            step_args=step_args,
        )

    assert "must be obtained from model.create() or model.register()" in str(error.value)


def test_pass_in_wrong_type_of_retry_policies(pipeline_session, model):
    sm_job_retry_policies = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.CAPACITY_ERROR], max_attempts=3
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
    )
    with pytest.raises(ValueError) as error:
        ModelStep(
            name="MyModelStep",
            step_args=step_args,
            retry_policies=dict(
                register_model_retry_policies=[sm_job_retry_policies],
                repack_model_retry_policies=[sm_job_retry_policies],
            ),
        )
    assert "SageMakerJobStepRetryPolicy is not allowed for a create/registe" in str(error.value)

    step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    with pytest.raises(ValueError) as error:
        ModelStep(
            name="MyModelStep",
            step_args=step_args,
            retry_policies=dict(
                create_model_retry_policies=[sm_job_retry_policies],
                repack_model_retry_policies=[sm_job_retry_policies],
            ),
        )
    assert "SageMakerJobStepRetryPolicy is not allowed for a create/registe" in str(error.value)


def test_register_model_step_with_model_package_name(pipeline_session):
    model = Model(
        name="MyModel",
        image_uri="my-image",
        model_data="s3://",
        sagemaker_session=pipeline_session,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_name="model-pkg-name-will-be-popped-out",
    )
    regis_model_step = ModelStep(
        name="MyModelStep",
        step_args=step_args,
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[regis_model_step],
        sagemaker_session=pipeline_session,
    )
    steps = json.loads(pipeline.definition())["Steps"]
    assert len(steps) == 1
    assert "ModelPackageName" not in steps[0]["Arguments"]
