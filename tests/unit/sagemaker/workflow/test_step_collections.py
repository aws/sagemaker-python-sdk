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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import os
import tempfile
import shutil
import pytest

from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.workflow._utils import REPACK_SCRIPT_LAUNCHER
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.model_step import (
    ModelStep,
    _CREATE_MODEL_NAME_BASE,
    _REPACK_MODEL_NAME_BASE,
)
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.utilities import list_to_request
from tests.unit import DATA_DIR

from sagemaker import PipelineModel
from sagemaker.estimator import Estimator
from sagemaker.model import Model, FrameworkModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import CreateModelInput, TransformInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.workflow.step_collections import (
    EstimatorTransformer,
    StepCollection,
    RegisterModel,
)
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum
from tests.unit.sagemaker.workflow.helpers import ordered, CustomStep
from tests.unit.sagemaker.workflow.conftest import IMAGE_URI, ROLE, BUCKET, REGION

MODEL_NAME = "gisele"
MODEL_REPACKING_IMAGE_URI = (
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-1-cpu-py3"
)


@pytest.fixture
def estimator(sagemaker_session):
    return Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="ml.c4.4xlarge",
        sagemaker_session=sagemaker_session,
        subnets=["abc", "def"],
        security_group_ids=["123", "456"],
    )


@pytest.fixture
def source_dir(request):
    wf = os.path.join(DATA_DIR, "workflow")
    tmp = tempfile.mkdtemp()
    shutil.copy2(os.path.join(wf, "inference.py"), os.path.join(tmp, "inference.py"))
    shutil.copy2(os.path.join(wf, "foo"), os.path.join(tmp, "foo"))

    def fin():
        shutil.rmtree(tmp)

    request.addfinalizer(fin)

    return tmp


@pytest.fixture
def model(sagemaker_session, source_dir):
    return FrameworkModel(
        image_uri=IMAGE_URI,
        model_data=f"s3://{BUCKET}/model.tar.gz",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        entry_point=f"{source_dir}/inference.py",
        name="modelName",
        vpc_config={"Subnets": ["abc", "def"], "SecurityGroupIds": ["123", "456"]},
    )


@pytest.fixture
def pipeline_model(sagemaker_session, model):
    return PipelineModel(
        models=[model],
        role=ROLE,
        sagemaker_session=sagemaker_session,
        vpc_config={"Subnets": ["abc", "def"], "SecurityGroupIds": ["123", "456"]},
    )


@pytest.fixture
def estimator_tf(sagemaker_session):
    return TensorFlow(
        entry_point="/some/script.py",
        framework_version="1.15.2",
        py_version="py3",
        role=ROLE,
        instance_type="ml.c4.2xlarge",
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def model_metrics():
    return ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{BUCKET}/metrics.csv",
            content_type="text/csv",
        )
    )


@pytest.fixture
def drift_check_baselines():
    return DriftCheckBaselines(
        model_constraints=MetricsSource(
            s3_uri=f"s3://{BUCKET}/constraints_metrics.csv",
            content_type="text/csv",
        )
    )


def test_step_collection():
    step_collection = StepCollection(
        name="MyStepCollection", steps=[CustomStep("MyStep1"), CustomStep("MyStep2")]
    )
    assert step_collection.request_dicts() == [
        {"Name": "MyStep1", "Type": "Training", "Arguments": dict()},
        {"Name": "MyStep2", "Type": "Training", "Arguments": dict()},
    ]


def test_step_collection_with_list_to_request():
    step_collection = StepCollection(
        name="MyStepCollection", steps=[CustomStep("MyStep1"), CustomStep("MyStep2")]
    )
    custom_step = CustomStep("MyStep3")
    assert list_to_request([step_collection, custom_step]) == [
        {"Name": "MyStep1", "Type": "Training", "Arguments": dict()},
        {"Name": "MyStep2", "Type": "Training", "Arguments": dict()},
        {"Name": "MyStep3", "Type": "Training", "Arguments": dict()},
    ]


def test_step_collection_properties(pipeline_session, sagemaker_session, source_dir):
    # ModelStep
    model = Model(
        name="MyModel",
        image_uri=IMAGE_URI,
        model_data=ParameterString(name="ModelData", default_value="s3://my-bucket/file"),
        sagemaker_session=pipeline_session,
        entry_point=f"{source_dir}/inference.py",
        source_dir=f"{source_dir}",
        role=ROLE,
    )
    step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    model_step_name = "MyModelStep"
    model_step = ModelStep(
        name=model_step_name,
        step_args=step_args,
    )
    steps = model_step.steps
    assert len(steps) == 2
    assert isinstance(steps[1], CreateModelStep)
    assert model_step.properties.ModelName.expr == {
        "Get": f"Steps.{model_step_name}-{_CREATE_MODEL_NAME_BASE}.ModelName"
    }

    # RegisterModel
    model.sagemaker_session = sagemaker_session
    model.entry_point = None
    model.source_dir = None
    register_model_step_name = "RegisterModelStep"
    register_model = RegisterModel(
        name=register_model_step_name,
        model=model,
        model_data="s3://",
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
    )
    steps = register_model.steps
    assert len(steps) == 1
    assert register_model.properties.ModelPackageName.expr == {
        "Get": f"Steps.{register_model_step_name}-RegisterModel.ModelPackageName"
    }

    assert register_model.properties.ModelPackageName._referenced_steps == [
        register_model.steps[-1]
    ]

    # Custom StepCollection
    step_collection = StepCollection(name="MyStepCollection")
    steps = step_collection.steps
    assert len(steps) == 0
    assert not step_collection.properties


def test_step_collection_is_depended_on(pipeline_session, sagemaker_session, source_dir):
    custom_step1 = CustomStep(name="MyStep1")
    model_name = "MyModel"
    model = Model(
        name=model_name,
        image_uri=IMAGE_URI,
        model_data=ParameterString(name="ModelData", default_value="s3://my-bucket/file"),
        sagemaker_session=pipeline_session,
        entry_point=f"{source_dir}/inference.py",
        source_dir=f"{source_dir}",
        role=ROLE,
    )
    step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    model_step_name = "MyModelStep"
    model_step = ModelStep(
        name=model_step_name,
        step_args=step_args,
    )

    # StepCollection object is depended on by another StepCollection object
    model.sagemaker_session = sagemaker_session
    register_model_name = "RegisterModelStep"
    register_model = RegisterModel(
        name=register_model_name,
        model=model,
        model_data="s3://",
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
        depends_on=["MyStep1", model_step],
    )

    # StepCollection objects are depended on by a step
    custom_step2 = CustomStep(
        name="MyStep2", depends_on=["MyStep1", model_step, register_model_name]
    )
    custom_step3 = CustomStep(
        name="MyStep3", depends_on=[custom_step1, model_step_name, register_model]
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[custom_step1, model_step, custom_step2, custom_step3, register_model],
    )
    step_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_list) == 7
    for step in step_list:
        if step["Name"] not in ["MyStep2", "MyStep3", f"{model_name}-RepackModel"]:
            assert "DependsOn" not in step
        elif step["Name"] == f"{model_name}-RepackModel":
            assert set(step["DependsOn"]) == {
                "MyStep1",
                f"{model_step_name}-{_REPACK_MODEL_NAME_BASE}-{model_name}",
                f"{model_step_name}-{_CREATE_MODEL_NAME_BASE}",
            }
        else:
            assert set(step["DependsOn"]) == {
                "MyStep1",
                f"{model_step_name}-{_REPACK_MODEL_NAME_BASE}-{model_name}",
                f"{model_step_name}-{_CREATE_MODEL_NAME_BASE}",
                f"{model_name}-RepackModel",
                f"{register_model_name}-RegisterModel",
            }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyStep1": ["MyStep2", "MyStep3", "MyModel-RepackModel"],
            "MyStep2": [],
            "MyStep3": [],
            "MyModelStep-RepackModel-MyModel": [
                "MyModelStep-CreateModel",
                "MyModel-RepackModel",
                "MyStep2",
                "MyStep3",
            ],
            "MyModelStep-CreateModel": ["MyStep2", "MyStep3", "MyModel-RepackModel"],
            "MyModel-RepackModel": ["MyStep2", "MyStep3"],
            "RegisterModelStep-RegisterModel": ["MyStep2", "MyStep3"],
        }
    )


def test_step_collection_in_condition_branch_is_depended_on(
    pipeline_session,
    sagemaker_session,
    source_dir,
):
    custom_step1 = CustomStep(name="MyStep1")

    # Define a step collection which will be inserted into the ConditionStep
    model_name = "MyModel"
    model = Model(
        name=model_name,
        image_uri=IMAGE_URI,
        model_data=ParameterString(name="ModelData", default_value="s3://my-bucket/file"),
        sagemaker_session=pipeline_session,
        entry_point=f"{source_dir}/inference.py",
        source_dir=f"{source_dir}",
        role=ROLE,
    )
    step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    model_step_name = "MyModelStep"
    model_step = ModelStep(
        name=model_step_name,
        step_args=step_args,
    )

    # Define another step collection which will be inserted into the ConditionStep
    # This StepCollection object depends on a StepCollection object in the ConditionStep
    # And a normal step outside ConditionStep
    model.sagemaker_session = sagemaker_session
    register_model_name = "RegisterModelStep"
    register_model = RegisterModel(
        name=register_model_name,
        model=model,
        model_data="s3://",
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
        depends_on=["MyStep1", model_step],
    )

    # StepCollection objects are depended on by a normal step in the ConditionStep
    custom_step2 = CustomStep(
        name="MyStep2", depends_on=["MyStep1", model_step, register_model_name]
    )
    # StepCollection objects are depended on by a normal step outside the ConditionStep
    custom_step3 = CustomStep(
        name="MyStep3", depends_on=[custom_step1, model_step_name, register_model]
    )

    cond_step = ConditionStep(
        name="CondStep",
        conditions=[ConditionEquals(left=2, right=1)],
        if_steps=[],
        else_steps=[model_step, register_model, custom_step2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step, custom_step1, custom_step3],
    )
    step_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_list) == 3
    for step in step_list:
        if step["Name"] == "MyStep1":
            assert "DependsOn" not in step
        elif step["Name"] == "CondStep":
            assert not step["Arguments"]["IfSteps"]
            for sub_step in step["Arguments"]["ElseSteps"]:
                if sub_step["Name"] == f"{model_name}-RepackModel":
                    assert set(sub_step["DependsOn"]) == {
                        "MyStep1",
                        f"{model_step_name}-{_REPACK_MODEL_NAME_BASE}-{model_name}",
                        f"{model_step_name}-{_CREATE_MODEL_NAME_BASE}",
                    }
                if sub_step["Name"] == "MyStep2":
                    assert set(sub_step["DependsOn"]) == {
                        "MyStep1",
                        f"{model_step_name}-{_REPACK_MODEL_NAME_BASE}-{model_name}",
                        f"{model_step_name}-{_CREATE_MODEL_NAME_BASE}",
                        f"{model_name}-RepackModel",
                        f"{register_model_name}-RegisterModel",
                    }
        else:
            assert set(step["DependsOn"]) == {
                "MyStep1",
                f"{model_step_name}-{_REPACK_MODEL_NAME_BASE}-{model_name}",
                f"{model_step_name}-{_CREATE_MODEL_NAME_BASE}",
                f"{model_name}-RepackModel",
                f"{register_model_name}-RegisterModel",
            }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "CondStep": [
                "MyModel-RepackModel",
                "MyModelStep-CreateModel",
                "MyModelStep-RepackModel-MyModel",
                "MyStep2",
                "RegisterModelStep-RegisterModel",
            ],
            "MyStep1": ["MyStep2", "MyStep3", "MyModel-RepackModel"],
            "MyStep2": [],
            "MyStep3": [],
            "MyModelStep-RepackModel-MyModel": [
                "MyModel-RepackModel",
                "MyModelStep-CreateModel",
                "MyStep2",
                "MyStep3",
            ],
            "MyModelStep-CreateModel": ["MyStep2", "MyStep3", "MyModel-RepackModel"],
            "MyModel-RepackModel": ["MyStep2", "MyStep3"],
            "RegisterModelStep-RegisterModel": ["MyStep2", "MyStep3"],
        }
    )


def test_condition_step_depends_on_step_collection():
    step1 = CustomStep(name="MyStep1")
    step2 = CustomStep(name="MyStep2", input_data=step1.properties)
    step_collection = StepCollection(name="MyStepCollection", steps=[step1, step2])
    cond_step = ConditionStep(
        name="MyConditionStep",
        depends_on=[step_collection],
        conditions=[ConditionEquals(left=2, right=1)],
        if_steps=[],
        else_steps=[],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step_collection, cond_step],
    )
    step_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_list) == 3
    for step in step_list:
        if step["Name"] != "MyConditionStep":
            continue
        assert step == {
            "Name": "MyConditionStep",
            "Type": "Condition",
            "DependsOn": ["MyStep1", "MyStep2"],
            "Arguments": {
                "Conditions": [
                    {
                        "Type": "Equals",
                        "LeftValue": 2,
                        "RightValue": 1,
                    },
                ],
                "IfSteps": [],
                "ElseSteps": [],
            },
        }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        [
            ("MyConditionStep", []),
            ("MyStep1", ["MyConditionStep", "MyStep2"]),
            ("MyStep2", ["MyConditionStep"]),
        ]
    )


def test_register_model(estimator, model_metrics, drift_check_baselines):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    register_model = RegisterModel(
        name="RegisterModelStep",
        estimator=estimator,
        model_data=model_data,
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        image_uri="012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
        model_package_group_name="mpg",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        approval_status="Approved",
        description="description",
        display_name="RegisterModelStep",
        depends_on=["TestStep"],
        tags=[{"Key": "myKey", "Value": "myValue"}],
        sample_payload_url="s3://test-bucket/model",
        task="IMAGE_CLASSIFICATION",
        framework="TENSORFLOW",
        framework_version="2.9",
        nearest_model_name="resnet50",
        data_input_configuration='{"input_1":[1,224,224,3]}',
    )
    assert ordered(register_model.request_dicts()) == ordered(
        [
            {
                "Name": "RegisterModelStep-RegisterModel",
                "Type": "RegisterModel",
                "DependsOn": ["TestStep"],
                "DisplayName": "RegisterModelStep",
                "Description": "description",
                "Arguments": {
                    "InferenceSpecification": {
                        "Containers": [
                            {
                                "Image": "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
                                "ModelDataUrl": f"s3://{BUCKET}/model.tar.gz",
                            }
                        ],
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "SkipModelValidation": "None",
                    "ModelMetrics": {
                        "Bias": {},
                        "Explainability": {},
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "DriftCheckBaselines": {
                        "ModelQuality": {
                            "Constraints": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/constraints_metrics.csv",
                            }
                        }
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                    "SamplePayloadUrl": "s3://test-bucket/model",
                    "Task": "IMAGE_CLASSIFICATION",
                },
            },
        ]
    )


def test_register_model_tf(estimator_tf, model_metrics, drift_check_baselines):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    register_model = RegisterModel(
        name="RegisterModelStep",
        estimator=estimator_tf,
        model_data=model_data,
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        approval_status="Approved",
        description="description",
        sample_payload_url="s3://test-bucket/model",
        task="IMAGE_CLASSIFICATION",
        framework="TENSORFLOW",
        framework_version="2.9",
        nearest_model_name="resnet50",
    )
    assert ordered(register_model.request_dicts()) == ordered(
        [
            {
                "Name": "RegisterModelStep-RegisterModel",
                "Type": "RegisterModel",
                "Description": "description",
                "Arguments": {
                    "InferenceSpecification": {
                        "Containers": [
                            {
                                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:1.15.2-cpu",
                                "ModelDataUrl": f"s3://{BUCKET}/model.tar.gz",
                            }
                        ],
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "SkipModelValidation": "None",
                    "ModelMetrics": {
                        "Bias": {},
                        "Explainability": {},
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "DriftCheckBaselines": {
                        "ModelQuality": {
                            "Constraints": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/constraints_metrics.csv",
                            }
                        }
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                    "SamplePayloadUrl": "s3://test-bucket/model",
                    "Task": "IMAGE_CLASSIFICATION",
                },
            },
        ]
    )


def test_register_model_sip(estimator, model_metrics, drift_check_baselines):
    model_list = [
        Model(image_uri="fakeimage1", model_data="Url1", env=[{"k1": "v1"}, {"k2": "v2"}]),
        Model(image_uri="fakeimage2", model_data="Url2", env=[{"k3": "v3"}, {"k4": "v4"}]),
    ]

    pipeline_model = PipelineModel(model_list, ROLE)

    register_model = RegisterModel(
        name="RegisterModelStep",
        estimator=estimator,
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        approval_status="Approved",
        description="description",
        model=pipeline_model,
        depends_on=["TestStep"],
        sample_payload_url="s3://test-bucket/model",
        task="IMAGE_CLASSIFICATION",
        framework="TENSORFLOW",
        framework_version="2.9",
        nearest_model_name="resnet50",
    )
    assert ordered(register_model.request_dicts()) == ordered(
        [
            {
                "Name": "RegisterModelStep-RegisterModel",
                "Type": "RegisterModel",
                "Description": "description",
                "DependsOn": ["TestStep"],
                "Arguments": {
                    "InferenceSpecification": {
                        "Containers": [
                            {
                                "Image": "fakeimage1",
                                "ModelDataUrl": "Url1",
                                "Environment": [{"k1": "v1"}, {"k2": "v2"}],
                                "Framework": "TENSORFLOW",
                                "FrameworkVersion": "2.9",
                                "NearestModelName": "resnet50",
                            },
                            {
                                "Image": "fakeimage2",
                                "ModelDataUrl": "Url2",
                                "Environment": [{"k3": "v3"}, {"k4": "v4"}],
                                "Framework": "TENSORFLOW",
                                "FrameworkVersion": "2.9",
                                "NearestModelName": "resnet50",
                            },
                        ],
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "SkipModelValidation": "None",
                    "ModelMetrics": {
                        "Bias": {},
                        "Explainability": {},
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "DriftCheckBaselines": {
                        "ModelQuality": {
                            "Constraints": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/constraints_metrics.csv",
                            }
                        }
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                    "SamplePayloadUrl": "s3://test-bucket/model",
                    "Task": "IMAGE_CLASSIFICATION",
                },
            },
        ]
    )


def test_register_model_with_model_repack_with_estimator(
    estimator,
    model_metrics,
    drift_check_baselines,
    source_dir,
):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    dummy_requirements = f"{DATA_DIR}/dummy_requirements.txt"
    register_model = RegisterModel(
        name="RegisterModelStep",
        estimator=estimator,
        model_data=model_data,
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        approval_status="Approved",
        description="description",
        entry_point=f"{source_dir}/inference.py",
        dependencies=[dummy_requirements],
        depends_on=["TestStep"],
        tags=[{"Key": "myKey", "Value": "myValue"}],
        sample_payload_url="s3://test-bucket/model",
        task="IMAGE_CLASSIFICATION",
        framework="TENSORFLOW",
        framework_version="2.9",
        nearest_model_name="resnet50",
    )

    request_dicts = register_model.request_dicts()
    assert len(request_dicts) == 2

    for request_dict in request_dicts:
        if request_dict["Type"] == "Training":
            assert request_dict["Name"] == "RegisterModelStep-RepackModel"
            assert len(request_dict["DependsOn"]) == 1
            assert request_dict["DependsOn"][0] == "TestStep"
            arguments = request_dict["Arguments"]
            assert BUCKET in arguments["HyperParameters"]["sagemaker_submit_directory"]
            arguments["HyperParameters"].pop("sagemaker_submit_directory")
            assert ordered(arguments) == ordered(
                {
                    "AlgorithmSpecification": {
                        "TrainingImage": MODEL_REPACKING_IMAGE_URI,
                        "TrainingInputMode": "File",
                    },
                    "DebugHookConfig": {
                        "CollectionConfigurations": [],
                        "S3OutputPath": f"s3://{BUCKET}/",
                    },
                    "ProfilerConfig": {"DisableProfiler": True},
                    "HyperParameters": {
                        "inference_script": '"inference.py"',
                        "dependencies": f'"{dummy_requirements}"',
                        "model_archive": '"s3://my-bucket/model.tar.gz"',
                        "sagemaker_program": f'"{REPACK_SCRIPT_LAUNCHER}"',
                        "sagemaker_container_log_level": "20",
                        "sagemaker_region": f'"{REGION}"',
                        "source_dir": "null",
                    },
                    "InputDataConfig": [
                        {
                            "ChannelName": "training",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataDistributionType": "FullyReplicated",
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": f"s3://{BUCKET}/model.tar.gz",
                                }
                            },
                        }
                    ],
                    "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/"},
                    "ResourceConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.large",
                        "VolumeSizeInGB": 30,
                    },
                    "RoleArn": ROLE,
                    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                    "VpcConfig": [
                        ("SecurityGroupIds", ["123", "456"]),
                        ("Subnets", ["abc", "def"]),
                    ],
                }
            )
        elif request_dict["Type"] == "RegisterModel":
            assert request_dict["Name"] == "RegisterModelStep-RegisterModel"
            assert "DependsOn" not in request_dict
            arguments = request_dict["Arguments"]
            assert len(arguments["InferenceSpecification"]["Containers"]) == 1
            assert (
                arguments["InferenceSpecification"]["Containers"][0]["Image"]
                == estimator.training_image_uri()
            )
            assert isinstance(
                arguments["InferenceSpecification"]["Containers"][0]["ModelDataUrl"], Properties
            )
            del arguments["InferenceSpecification"]["Containers"]
            assert ordered(arguments) == ordered(
                {
                    "InferenceSpecification": {
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "SkipModelValidation": "None",
                    "ModelMetrics": {
                        "Bias": {},
                        "Explainability": {},
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "DriftCheckBaselines": {
                        "ModelQuality": {
                            "Constraints": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/constraints_metrics.csv",
                            }
                        }
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                    "SamplePayloadUrl": "s3://test-bucket/model",
                    "Task": "IMAGE_CLASSIFICATION",
                }
            )
        else:
            raise Exception("A step exists in the collection of an invalid type.")


def test_register_model_with_model_repack_with_model(model, model_metrics, drift_check_baselines):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    register_model = RegisterModel(
        name="RegisterModelStep",
        model=model,
        model_data=model_data,
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        approval_status="Approved",
        description="description",
        depends_on=["TestStep"],
        tags=[{"Key": "myKey", "Value": "myValue"}],
    )

    request_dicts = register_model.request_dicts()
    assert len(request_dicts) == 2

    for request_dict in request_dicts:
        if request_dict["Type"] == "Training":
            assert request_dict["Name"] == "modelName-RepackModel"
            assert len(request_dict["DependsOn"]) == 1
            assert request_dict["DependsOn"][0] == "TestStep"
            arguments = request_dict["Arguments"]
            assert BUCKET in arguments["HyperParameters"]["sagemaker_submit_directory"]
            arguments["HyperParameters"].pop("sagemaker_submit_directory")
            assert ordered(arguments) == ordered(
                {
                    "AlgorithmSpecification": {
                        "TrainingImage": MODEL_REPACKING_IMAGE_URI,
                        "TrainingInputMode": "File",
                    },
                    "DebugHookConfig": {
                        "CollectionConfigurations": [],
                        "S3OutputPath": f"s3://{BUCKET}/",
                    },
                    "ProfilerConfig": {"DisableProfiler": True},
                    "HyperParameters": {
                        "inference_script": '"inference.py"',
                        "model_archive": '"s3://my-bucket/model.tar.gz"',
                        "sagemaker_program": f'"{REPACK_SCRIPT_LAUNCHER}"',
                        "sagemaker_container_log_level": "20",
                        "sagemaker_region": f'"{REGION}"',
                        "dependencies": "null",
                        "source_dir": "null",
                    },
                    "InputDataConfig": [
                        {
                            "ChannelName": "training",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataDistributionType": "FullyReplicated",
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": f"s3://{BUCKET}/model.tar.gz",
                                }
                            },
                        }
                    ],
                    "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/"},
                    "ResourceConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.large",
                        "VolumeSizeInGB": 30,
                    },
                    "RoleArn": ROLE,
                    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                    "VpcConfig": [
                        ("SecurityGroupIds", ["123", "456"]),
                        ("Subnets", ["abc", "def"]),
                    ],
                }
            )
        elif request_dict["Type"] == "RegisterModel":
            assert request_dict["Name"] == "RegisterModelStep-RegisterModel"
            assert "DependsOn" not in request_dict
            arguments = request_dict["Arguments"]
            assert len(arguments["InferenceSpecification"]["Containers"]) == 1
            assert arguments["InferenceSpecification"]["Containers"][0]["Image"] == model.image_uri
            assert isinstance(
                arguments["InferenceSpecification"]["Containers"][0]["ModelDataUrl"], Properties
            )
            del arguments["InferenceSpecification"]["Containers"]
            assert ordered(arguments) == ordered(
                {
                    "InferenceSpecification": {
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "SkipModelValidation": "None",
                    "ModelMetrics": {
                        "Bias": {},
                        "Explainability": {},
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "DriftCheckBaselines": {
                        "ModelQuality": {
                            "Constraints": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/constraints_metrics.csv",
                            }
                        }
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                }
            )
        else:
            raise Exception("A step exists in the collection of an invalid type.")


def test_register_model_with_model_repack_with_pipeline_model(
    pipeline_model, model_metrics, drift_check_baselines
):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    service_fault_retry_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.SERVICE_FAULT], max_attempts=10
    )
    register_model = RegisterModel(
        name="RegisterModelStep",
        model=pipeline_model,
        model_data=model_data,
        content_types=["content_type"],
        response_types=["response_type"],
        inference_instances=["inference_instance"],
        transform_instances=["transform_instance"],
        model_package_group_name="mpg",
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
        approval_status="Approved",
        description="description",
        depends_on=["TestStep"],
        repack_model_step_retry_policies=[service_fault_retry_policy],
        register_model_step_retry_policies=[service_fault_retry_policy],
        tags=[{"Key": "myKey", "Value": "myValue"}],
    )

    request_dicts = register_model.request_dicts()
    assert len(request_dicts) == 2

    for request_dict in request_dicts:
        if request_dict["Type"] == "Training":
            assert request_dict["Name"] == "modelName-RepackModel"
            assert len(request_dict["DependsOn"]) == 1
            assert request_dict["DependsOn"][0] == "TestStep"
            arguments = request_dict["Arguments"]
            assert BUCKET in arguments["HyperParameters"]["sagemaker_submit_directory"]
            arguments["HyperParameters"].pop("sagemaker_submit_directory")
            assert ordered(arguments) == ordered(
                {
                    "AlgorithmSpecification": {
                        "TrainingImage": MODEL_REPACKING_IMAGE_URI,
                        "TrainingInputMode": "File",
                    },
                    "DebugHookConfig": {
                        "CollectionConfigurations": [],
                        "S3OutputPath": f"s3://{BUCKET}/",
                    },
                    "ProfilerConfig": {"DisableProfiler": True},
                    "HyperParameters": {
                        "dependencies": "null",
                        "inference_script": '"inference.py"',
                        "model_archive": '"s3://my-bucket/model.tar.gz"',
                        "sagemaker_program": f'"{REPACK_SCRIPT_LAUNCHER}"',
                        "sagemaker_container_log_level": "20",
                        "sagemaker_region": f'"{REGION}"',
                        "source_dir": "null",
                    },
                    "InputDataConfig": [
                        {
                            "ChannelName": "training",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataDistributionType": "FullyReplicated",
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": f"s3://{BUCKET}/model.tar.gz",
                                }
                            },
                        }
                    ],
                    "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/"},
                    "ResourceConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.large",
                        "VolumeSizeInGB": 30,
                    },
                    "RoleArn": ROLE,
                    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                    "VpcConfig": [
                        ("SecurityGroupIds", ["123", "456"]),
                        ("Subnets", ["abc", "def"]),
                    ],
                }
            )
        elif request_dict["Type"] == "RegisterModel":
            assert request_dict["Name"] == "RegisterModelStep-RegisterModel"
            assert "DependsOn" not in request_dict
            arguments = request_dict["Arguments"]
            assert len(arguments["InferenceSpecification"]["Containers"]) == 1
            assert (
                arguments["InferenceSpecification"]["Containers"][0]["Image"]
                == pipeline_model.models[0].image_uri
            )
            assert isinstance(
                arguments["InferenceSpecification"]["Containers"][0]["ModelDataUrl"], Properties
            )
            del arguments["InferenceSpecification"]["Containers"]
            assert ordered(arguments) == ordered(
                {
                    "InferenceSpecification": {
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "SkipModelValidation": "None",
                    "ModelMetrics": {
                        "Bias": {},
                        "Explainability": {},
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "DriftCheckBaselines": {
                        "ModelQuality": {
                            "Constraints": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/constraints_metrics.csv",
                            }
                        }
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                }
            )
        else:
            raise Exception("A step exists in the collection of an invalid type.")


def test_estimator_transformer(estimator):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    model_inputs = CreateModelInput(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    service_fault_retry_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.SERVICE_FAULT], max_attempts=10
    )
    transform_inputs = TransformInput(data=f"s3://{BUCKET}/transform_manifest")
    estimator_transformer = EstimatorTransformer(
        name="EstimatorTransformerStep",
        estimator=estimator,
        model_data=model_data,
        model_inputs=model_inputs,
        instance_count=1,
        instance_type="ml.c4.4xlarge",
        transform_inputs=transform_inputs,
        depends_on=["TestStep"],
        model_step_retry_policies=[service_fault_retry_policy],
        transform_step_retry_policies=[service_fault_retry_policy],
        repack_model_step_retry_policies=[service_fault_retry_policy],
    )
    request_dicts = estimator_transformer.request_dicts()
    assert len(request_dicts) == 2

    for request_dict in request_dicts:
        if request_dict["Type"] == "Model":
            assert request_dict == {
                "Name": "EstimatorTransformerStepCreateModelStep",
                "Type": "Model",
                "DependsOn": ["TestStep"],
                "RetryPolicies": [service_fault_retry_policy.to_request()],
                "Arguments": {
                    "ExecutionRoleArn": "DummyRole",
                    "PrimaryContainer": {
                        "Environment": {},
                        "Image": "fakeimage",
                        "ModelDataUrl": "s3://my-bucket/model.tar.gz",
                    },
                },
            }
        elif request_dict["Type"] == "Transform":
            assert request_dict["Name"] == "EstimatorTransformerStepTransformStep"
            assert request_dict["RetryPolicies"] == [service_fault_retry_policy.to_request()]
            arguments = request_dict["Arguments"]
            assert isinstance(arguments["ModelName"], Properties)
            arguments.pop("ModelName")
            assert "DependsOn" not in request_dict
            assert arguments == {
                "TransformInput": {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET}/transform_manifest",
                        }
                    }
                },
                "TransformOutput": {"S3OutputPath": None},
                "TransformResources": {"InstanceCount": 1, "InstanceType": "ml.c4.4xlarge"},
            }
        else:
            raise Exception("A step exists in the collection of an invalid type.")


def test_estimator_transformer_with_model_repack_with_estimator(estimator, source_dir):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    model_inputs = CreateModelInput(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    service_fault_retry_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.SERVICE_FAULT], max_attempts=10
    )
    transform_inputs = TransformInput(data=f"s3://{BUCKET}/transform_manifest")
    estimator_transformer = EstimatorTransformer(
        name="EstimatorTransformerStep",
        estimator=estimator,
        model_data=model_data,
        model_inputs=model_inputs,
        instance_count=1,
        instance_type="ml.c4.4xlarge",
        transform_inputs=transform_inputs,
        depends_on=["TestStep"],
        model_step_retry_policies=[service_fault_retry_policy],
        transform_step_retry_policies=[service_fault_retry_policy],
        repack_model_step_retry_policies=[service_fault_retry_policy],
        entry_point=f"{source_dir}/inference.py",
    )
    request_dicts = estimator_transformer.request_dicts()
    assert len(request_dicts) == 3

    for request_dict in request_dicts:
        if request_dict["Type"] == "Training":
            assert request_dict["Name"] == "EstimatorTransformerStepRepackModel"
            assert request_dict["DependsOn"] == ["TestStep"]
            assert request_dict["RetryPolicies"] == [service_fault_retry_policy.to_request()]
            arguments = request_dict["Arguments"]
            # pop out the dynamic generated fields
            arguments["HyperParameters"].pop("sagemaker_submit_directory")
            assert arguments == {
                "AlgorithmSpecification": {
                    "TrainingInputMode": "File",
                    "TrainingImage": MODEL_REPACKING_IMAGE_URI,
                },
                "ProfilerConfig": {"DisableProfiler": True},
                "OutputDataConfig": {"S3OutputPath": "s3://my-bucket/"},
                "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
                "ResourceConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                    "VolumeSizeInGB": 30,
                },
                "RoleArn": "DummyRole",
                "InputDataConfig": [
                    {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "s3://my-bucket/model.tar.gz",
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "ChannelName": "training",
                    }
                ],
                "HyperParameters": {
                    "inference_script": '"inference.py"',
                    "model_archive": '"s3://my-bucket/model.tar.gz"',
                    "dependencies": "null",
                    "source_dir": "null",
                    "sagemaker_program": f'"{REPACK_SCRIPT_LAUNCHER}"',
                    "sagemaker_container_log_level": "20",
                    "sagemaker_region": '"us-west-2"',
                },
                "VpcConfig": {"Subnets": ["abc", "def"], "SecurityGroupIds": ["123", "456"]},
                "DebugHookConfig": {
                    "S3OutputPath": "s3://my-bucket/",
                    "CollectionConfigurations": [],
                },
            }
        elif request_dict["Type"] == "Model":
            assert request_dict["Name"] == "EstimatorTransformerStepCreateModelStep"
            assert request_dict["RetryPolicies"] == [service_fault_retry_policy.to_request()]
            arguments = request_dict["Arguments"]
            assert isinstance(arguments["PrimaryContainer"]["ModelDataUrl"], Properties)
            arguments["PrimaryContainer"].pop("ModelDataUrl")
            assert "DependsOn" not in request_dict
            assert arguments == {
                "ExecutionRoleArn": "DummyRole",
                "PrimaryContainer": {
                    "Environment": {},
                    "Image": "fakeimage",
                },
            }
        elif request_dict["Type"] == "Transform":
            assert request_dict["Name"] == "EstimatorTransformerStepTransformStep"
            assert request_dict["RetryPolicies"] == [service_fault_retry_policy.to_request()]
            arguments = request_dict["Arguments"]
            assert isinstance(arguments["ModelName"], Properties)
            arguments.pop("ModelName")
            assert "DependsOn" not in request_dict
            assert arguments == {
                "TransformInput": {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET}/transform_manifest",
                        }
                    }
                },
                "TransformOutput": {"S3OutputPath": None},
                "TransformResources": {"InstanceCount": 1, "InstanceType": "ml.c4.4xlarge"},
            }
        else:
            raise Exception("A step exists in the collection of an invalid type.")
