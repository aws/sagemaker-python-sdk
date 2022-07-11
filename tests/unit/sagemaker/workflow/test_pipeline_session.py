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
"""This module contains code to test ``sagemaker.workflow.pipeline_session.PipelineSession``"""
from __future__ import absolute_import

import pytest
from mock import Mock, PropertyMock

from sagemaker import Model
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession

from botocore.config import Config

from tests.unit import DATA_DIR

_REGION = "us-west-2"
_ROLE = "DummyRole"
_BUCKET = "my-bucket"


def test_pipeline_session_init(sagemaker_client_config, boto_session):
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )

    sess = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
    )
    assert sess.sagemaker_client is not None
    assert sess.default_bucket() is not None
    assert sess.context is None


@pytest.fixture
def client_mock():
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
def boto_session_mock(client_mock):
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=_ROLE)

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name=_REGION)
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client_mock

    return session_mock


@pytest.fixture
def pipeline_session_mock(boto_session_mock, client_mock):
    return PipelineSession(
        boto_session=boto_session_mock,
        sagemaker_client=client_mock,
        default_bucket=_BUCKET,
    )


def test_pipeline_session_context_for_model_step(pipeline_session_mock):
    model = Model(
        name="MyModel",
        image_uri="fakeimage",
        model_data=ParameterString(name="ModelData", default_value="s3://my-bucket/file"),
        sagemaker_session=pipeline_session_mock,
        entry_point=f"{DATA_DIR}/dummy_script.py",
        source_dir=f"{DATA_DIR}",
        role=_ROLE,
    )
    # CreateModelStep requires runtime repack
    create_step_args = model.create(
        instance_type="c4.4xlarge",
        accelerator_type="ml.eia1.medium",
    )
    # The context should be cleaned up before return
    assert pipeline_session_mock.context is None
    assert create_step_args.create_model_request
    assert not create_step_args.create_model_package_request
    assert len(create_step_args.need_runtime_repack) == 1

    # _RegisterModelStep does not require runtime repack
    model.entry_point = None
    model.source_dir = None
    register_step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
        task="IMAGE_CLASSIFICATION",
        sample_payload_url="s3://test-bucket/model",
        framework="TENSORFLOW",
        framework_version="2.9",
        nearest_model_name="resnet50",
        data_input_configuration='{"input_1":[1,224,224,3]}',
    )
    # The context should be cleaned up before return
    assert not pipeline_session_mock.context
    assert not register_step_args.create_model_request
    assert register_step_args.create_model_package_request
    assert len(register_step_args.need_runtime_repack) == 0


def test_pipeline_session_context_for_model_step_without_instance_types(
    pipeline_session_mock,
):
    model = Model(
        name="MyModel",
        image_uri="fakeimage",
        model_data=ParameterString(name="ModelData", default_value="s3://my-bucket/file"),
        sagemaker_session=pipeline_session_mock,
        entry_point=f"{DATA_DIR}/dummy_script.py",
        source_dir=f"{DATA_DIR}",
        role=_ROLE,
    )
    register_step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        model_package_group_name="MyModelPackageGroup",
        task="IMAGE_CLASSIFICATION",
        sample_payload_url="s3://test-bucket/model",
        framework="TENSORFLOW",
        framework_version="2.9",
        nearest_model_name="resnet50",
        data_input_configuration='{"input_1":[1,224,224,3]}',
    )

    expected_output = {
        "ModelPackageGroupName": "MyModelPackageGroup",
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": "fakeimage",
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "dummy_script.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                        "SAGEMAKER_REGION": "us-west-2",
                    },
                    "ModelDataUrl": ParameterString(
                        name="ModelData",
                        default_value="s3://my-bucket/file",
                    ),
                    "Framework": "TENSORFLOW",
                    "FrameworkVersion": "2.9",
                    "NearestModelName": "resnet50",
                    "ModelInput": {
                        "DataInputConfig": '{"input_1":[1,224,224,3]}',
                    },
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedRealtimeInferenceInstanceTypes": None,
            "SupportedTransformInstanceTypes": None,
        },
        "CertifyForMarketplace": False,
        "ModelApprovalStatus": "PendingManualApproval",
        "SamplePayloadUrl": "s3://test-bucket/model",
        "Task": "IMAGE_CLASSIFICATION",
    }

    assert register_step_args.create_model_package_request == expected_output
