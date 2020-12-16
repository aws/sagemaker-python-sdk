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

import pytest
import sagemaker

from mock import (
    Mock,
    PropertyMock,
)

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, TransformInput, CreateModelInput
from sagemaker.model import Model
from sagemaker.processing import (
    Processor,
    ProcessingInput,
)
from sagemaker.transformer import Transformer
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import (
    ProcessingStep,
    Step,
    StepTypeEnum,
    TrainingStep,
    TransformStep,
    CreateModelStep,
)

REGION = "us-west-2"
BUCKET = "my-bucket"
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"
MODEL_NAME = "gisele"


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


def test_custom_step():
    step = CustomStep("MyStep")
    assert step.to_request() == {"Name": "MyStep", "Type": "Training", "Arguments": dict()}


def test_training_step(sagemaker_session):
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=sagemaker_session,
    )
    inputs = TrainingInput(f"s3://{BUCKET}/train_manifest")
    step = TrainingStep(
        name="MyTrainingStep",
        estimator=estimator,
        inputs=inputs,
    )
    assert step.to_request() == {
        "Name": "MyTrainingStep",
        "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingImage": IMAGE_URI, "TrainingInputMode": "File"},
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "FullyReplicated",
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET}/train_manifest",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/"},
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "c4.4xlarge",
                "VolumeSizeInGB": 30,
            },
            "RoleArn": ROLE,
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        },
    }
    assert step.properties.TrainingJobName.expr == {"Get": "Steps.MyTrainingStep.TrainingJobName"}


def test_processing_step(sagemaker_session):
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="ml.m4.4xlarge",
        sagemaker_session=sagemaker_session,
    )
    inputs = [
        ProcessingInput(
            source=f"s3://{BUCKET}/processing_manifest",
            destination="processing_manifest",
        )
    ]
    step = ProcessingStep(
        name="MyProcessingStep",
        processor=processor,
        inputs=inputs,
        outputs=[],
    )
    assert step.to_request() == {
        "Name": "MyProcessingStep",
        "Type": "Processing",
        "Arguments": {
            "AppSpecification": {"ImageUri": "fakeimage"},
            "ProcessingInputs": [
                {
                    "InputName": "input-1",
                    "AppManaged": False,
                    "S3Input": {
                        "LocalPath": "processing_manifest",
                        "S3CompressionType": "None",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3Uri": "s3://my-bucket/processing_manifest",
                    },
                }
            ],
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m4.4xlarge",
                    "VolumeSizeInGB": 30,
                }
            },
            "RoleArn": "DummyRole",
        },
    }
    assert step.properties.ProcessingJobName.expr == {
        "Get": "Steps.MyProcessingStep.ProcessingJobName"
    }


def test_create_model_step(sagemaker_session):
    model = Model(
        image_uri=IMAGE_URI,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )
    inputs = CreateModelInput(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    step = CreateModelStep(
        name="MyCreateModelStep",
        model=model,
        inputs=inputs,
    )

    assert step.to_request() == {
        "Name": "MyCreateModelStep",
        "Type": "Model",
        "Arguments": {
            "ExecutionRoleArn": "DummyRole",
            "PrimaryContainer": {"Environment": {}, "Image": "fakeimage"},
        },
    }
    assert step.properties.ModelName.expr == {"Get": "Steps.MyCreateModelStep.ModelName"}


def test_transform_step(sagemaker_session):
    transformer = Transformer(
        model_name=MODEL_NAME,
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=sagemaker_session,
    )
    inputs = TransformInput(data=f"s3://{BUCKET}/transform_manifest")
    step = TransformStep(
        name="MyTransformStep",
        transformer=transformer,
        inputs=inputs,
    )
    assert step.to_request() == {
        "Name": "MyTransformStep",
        "Type": "Transform",
        "Arguments": {
            "ModelName": "gisele",
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://my-bucket/transform_manifest",
                    }
                }
            },
            "TransformOutput": {"S3OutputPath": None},
            "TransformResources": {
                "InstanceCount": 1,
                "InstanceType": "c4.4xlarge",
            },
        },
    }
    assert step.properties.TransformJobName.expr == {
        "Get": "Steps.MyTransformStep.TransformJobName"
    }


def test_properties_describe_training_job_response():
    prop = Properties("Steps.MyStep", "DescribeTrainingJobResponse")
    some_prop_names = ["TrainingJobName", "TrainingJobArn", "HyperParameters", "OutputDataConfig"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.CreationTime.expr == {"Get": "Steps.MyStep.CreationTime"}
    assert prop.OutputDataConfig.S3OutputPath.expr == {
        "Get": "Steps.MyStep.OutputDataConfig.S3OutputPath"
    }


def test_properties_describe_processing_job_response():
    prop = Properties("Steps.MyStep", "DescribeProcessingJobResponse")
    some_prop_names = ["ProcessingInputs", "ProcessingOutputConfig", "ProcessingEndTime"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.ProcessingJobName.expr == {"Get": "Steps.MyStep.ProcessingJobName"}
    assert prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri.expr == {
        "Get": "Steps.MyStep.ProcessingOutputConfig.Outputs['MyOutputName'].S3Output.S3Uri"
    }
