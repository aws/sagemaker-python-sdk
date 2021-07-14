# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
import tempfile
import shutil
import pytest

from tests.unit import DATA_DIR

import sagemaker

from mock import (
    Mock,
    PropertyMock,
)

from sagemaker import PipelineModel
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import CreateModelInput, TransformInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import (
    Step,
    StepTypeEnum,
)
from sagemaker.workflow.step_collections import (
    EstimatorTransformer,
    StepCollection,
    RegisterModel,
)
from tests.unit.sagemaker.workflow.helpers import ordered

REGION = "us-west-2"
BUCKET = "my-bucket"
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"
MODEL_NAME = "gisele"
MODEL_REPACKING_IMAGE_URI = (
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
)


class CustomStep(Step):
    def __init__(self, name):
        super(CustomStep, self).__init__(name, StepTypeEnum.TRAINING)
        self._properties = Properties(path=f"Steps.{name}")

    @property
    def arguments(self):
        return dict()

    @property
    def properties(self):
        return self._properties


@pytest.fixture
def boto_session():
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name=REGION)
    session_mock.resource.return_value = resource_mock

    return session_mock


@pytest.fixture
def client():
    """Mock client.

    Considerations when appropriate:

        * utilize botocore.stub.Stubber
        * separate runtime client from client
    """
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture
def sagemaker_session(boto_session, client):
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=client,
        sagemaker_runtime_client=client,
        default_bucket=BUCKET,
    )


@pytest.fixture
def estimator(sagemaker_session):
    return Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="ml.c4.4xlarge",
        sagemaker_session=sagemaker_session,
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
def source_dir(request):
    wf = os.path.join(DATA_DIR, "workflow")
    tmp = tempfile.mkdtemp()
    shutil.copy2(os.path.join(wf, "inference.py"), os.path.join(tmp, "inference.py"))
    shutil.copy2(os.path.join(wf, "foo"), os.path.join(tmp, "foo"))

    def fin():
        shutil.rmtree(tmp)

    request.addfinalizer(fin)

    return tmp


def test_step_collection():
    step_collection = StepCollection(steps=[CustomStep("MyStep1"), CustomStep("MyStep2")])
    assert step_collection.request_dicts() == [
        {"Name": "MyStep1", "Type": "Training", "Arguments": dict()},
        {"Name": "MyStep2", "Type": "Training", "Arguments": dict()},
    ]


def test_register_model(estimator, model_metrics):
    model_data = f"s3://{BUCKET}/model.tar.gz"
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
        approval_status="Approved",
        description="description",
        depends_on=["TestStep"],
        tags=[{"Key": "myKey", "Value": "myValue"}],
    )
    assert ordered(register_model.request_dicts()) == ordered(
        [
            {
                "Name": "RegisterModelStep",
                "Type": "RegisterModel",
                "DependsOn": ["TestStep"],
                "Arguments": {
                    "InferenceSpecification": {
                        "Containers": [
                            {"Image": "fakeimage", "ModelDataUrl": f"s3://{BUCKET}/model.tar.gz"}
                        ],
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "ModelMetrics": {
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                    "Tags": [{"Key": "myKey", "Value": "myValue"}],
                },
            },
        ]
    )


def test_register_model_tf(estimator_tf, model_metrics):
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
        approval_status="Approved",
        description="description",
    )
    assert ordered(register_model.request_dicts()) == ordered(
        [
            {
                "Name": "RegisterModelStep",
                "Type": "RegisterModel",
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
                    "ModelMetrics": {
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                },
            },
        ]
    )


def test_register_model_sip(estimator, model_metrics):
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
        approval_status="Approved",
        description="description",
        model=pipeline_model,
        depends_on=["TestStep"],
    )
    assert ordered(register_model.request_dicts()) == ordered(
        [
            {
                "Name": "RegisterModelStep",
                "Type": "RegisterModel",
                "DependsOn": ["TestStep"],
                "Arguments": {
                    "InferenceSpecification": {
                        "Containers": [
                            {
                                "Image": "fakeimage1",
                                "ModelDataUrl": "Url1",
                                "Environment": [{"k1": "v1"}, {"k2": "v2"}],
                            },
                            {
                                "Image": "fakeimage2",
                                "ModelDataUrl": "Url2",
                                "Environment": [{"k3": "v3"}, {"k4": "v4"}],
                            },
                        ],
                        "SupportedContentTypes": ["content_type"],
                        "SupportedRealtimeInferenceInstanceTypes": ["inference_instance"],
                        "SupportedResponseMIMETypes": ["response_type"],
                        "SupportedTransformInstanceTypes": ["transform_instance"],
                    },
                    "ModelApprovalStatus": "Approved",
                    "ModelMetrics": {
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
                },
            },
        ]
    )


def test_register_model_with_model_repack(estimator, model_metrics):
    model_data = f"s3://{BUCKET}/model.tar.gz"
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
        approval_status="Approved",
        description="description",
        entry_point=f"{DATA_DIR}/dummy_script.py",
        depends_on=["TestStep"],
    )

    request_dicts = register_model.request_dicts()
    assert len(request_dicts) == 2

    for request_dict in request_dicts:
        if request_dict["Type"] == "Training":
            assert request_dict["Name"] == "RegisterModelStepRepackModel"
            assert len(request_dict["DependsOn"]) == 1
            assert request_dict["DependsOn"][0] == "TestStep"
            arguments = request_dict["Arguments"]
            repacker_job_name = arguments["HyperParameters"]["sagemaker_job_name"]
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
                    "HyperParameters": {
                        "inference_script": '"dummy_script.py"',
                        "model_archive": '"model.tar.gz"',
                        "sagemaker_submit_directory": '"s3://{}/{}/source/sourcedir.tar.gz"'.format(
                            BUCKET, repacker_job_name.replace('"', "")
                        ),
                        "sagemaker_program": '"_repack_model.py"',
                        "sagemaker_container_log_level": "20",
                        "sagemaker_job_name": repacker_job_name,
                        "sagemaker_region": f'"{REGION}"',
                    },
                    "InputDataConfig": [
                        {
                            "ChannelName": "training",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataDistributionType": "FullyReplicated",
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": f"s3://{BUCKET}",
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
                }
            )
        elif request_dict["Type"] == "RegisterModel":
            assert request_dict["Name"] == "RegisterModelStep"
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
                    "ModelMetrics": {
                        "ModelQuality": {
                            "Statistics": {
                                "ContentType": "text/csv",
                                "S3Uri": f"s3://{BUCKET}/metrics.csv",
                            },
                        },
                    },
                    "ModelPackageDescription": "description",
                    "ModelPackageGroupName": "mpg",
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
    )
    request_dicts = estimator_transformer.request_dicts()
    assert len(request_dicts) == 2
    for request_dict in request_dicts:
        if request_dict["Type"] == "Model":
            assert request_dict == {
                "Name": "EstimatorTransformerStepCreateModelStep",
                "Type": "Model",
                "DependsOn": ["TestStep"],
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
