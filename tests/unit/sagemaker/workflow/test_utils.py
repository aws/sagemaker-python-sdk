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

import os
import shutil
import tempfile

import pytest
import sagemaker

from mock import (
    Mock,
    PropertyMock,
)

from sagemaker.estimator import Estimator
from sagemaker.workflow._utils import _RepackModelStep
from tests.unit import DATA_DIR

REGION = "us-west-2"
BUCKET = "my-bucket"
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"


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
        instance_type="c4.4xlarge",
        sagemaker_session=sagemaker_session,
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


def test_repack_model_step(estimator):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    entry_point = f"{DATA_DIR}/dummy_script.py"
    step = _RepackModelStep(
        name="MyRepackModelStep",
        sagemaker_session=estimator.sagemaker_session,
        role=estimator.role,
        model_data=model_data,
        entry_point=entry_point,
        depends_on=["TestStep"],
    )
    request_dict = step.to_request()

    hyperparameters = request_dict["Arguments"]["HyperParameters"]
    assert hyperparameters["inference_script"] == '"dummy_script.py"'
    assert hyperparameters["model_archive"] == '"model.tar.gz"'
    assert hyperparameters["sagemaker_program"] == '"_repack_model.py"'

    del request_dict["Arguments"]["HyperParameters"]
    del request_dict["Arguments"]["AlgorithmSpecification"]["TrainingImage"]
    assert request_dict == {
        "Name": "MyRepackModelStep",
        "Type": "Training",
        "DependsOn": ["TestStep"],
        "Arguments": {
            "AlgorithmSpecification": {"TrainingInputMode": "File"},
            "DebugHookConfig": {"CollectionConfigurations": [], "S3OutputPath": "s3://my-bucket/"},
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
        },
    }
    assert step.properties.TrainingJobName.expr == {
        "Get": "Steps.MyRepackModelStep.TrainingJobName"
    }


def test_repack_model_step_with_source_dir(estimator, source_dir):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    entry_point = "inference.py"
    step = _RepackModelStep(
        name="MyRepackModelStep",
        sagemaker_session=estimator.sagemaker_session,
        role=estimator.role,
        model_data=model_data,
        entry_point=entry_point,
        source_dir=source_dir,
    )
    request_dict = step.to_request()
    assert os.path.isfile(f"{source_dir}/_repack_model.py")

    hyperparameters = request_dict["Arguments"]["HyperParameters"]
    assert hyperparameters["inference_script"] == '"inference.py"'
    assert hyperparameters["model_archive"] == '"model.tar.gz"'
    assert hyperparameters["sagemaker_program"] == '"_repack_model.py"'

    del request_dict["Arguments"]["HyperParameters"]
    del request_dict["Arguments"]["AlgorithmSpecification"]["TrainingImage"]
    assert request_dict == {
        "Name": "MyRepackModelStep",
        "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingInputMode": "File"},
            "DebugHookConfig": {"CollectionConfigurations": [], "S3OutputPath": "s3://my-bucket/"},
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
        },
    }
    assert step.properties.TrainingJobName.expr == {
        "Get": "Steps.MyRepackModelStep.TrainingJobName"
    }
