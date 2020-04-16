# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from mock import Mock, patch

from sagemaker.model import Model

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"

REGION = "us-west-2"

NEO_REGION_ACCOUNT = "301217895009"
DESCRIBE_COMPILATION_JOB_RESPONSE = {
    "CompilationJobStatus": "Completed",
    "ModelArtifacts": {"S3ModelArtifacts": "s3://output-path/model.tar.gz"},
}

EC2_REGION_LIST = [
    "us-east-2",
    "us-east-1",
    "us-west-1",
    "us-west-2",
    "ap-east-1",
    "ap-south-1",
    "ap-northeast-3",
    "ap-northeast-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "ca-central-1",
    "cn-north-1",
    "cn-northwest-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "us-gov-east-1",
    "us-gov-west-1",
]

NEO_REGION_LIST = [
    "us-west-1",
    "us-west-2",
    "us-east-1",
    "us-east-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-north-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-east-1",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "sa-east-1",
    "ca-central-1",
    "me-south-1",
    "cn-north-1",
    "cn-northwest-1",
    "us-gov-west-1",
]


@pytest.fixture
def sagemaker_session():
    return Mock(boto_region_name=REGION)


def _create_model(sagemaker_session=None):
    return Model(MODEL_DATA, MODEL_IMAGE, sagemaker_session=sagemaker_session)


def test_compile_model_for_inferentia(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="ml_inf",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        framework_version="1.15.0",
        job_name="compile-model",
    )
    assert (
        "{}.dkr.ecr.{}.amazonaws.com/sagemaker-neo-tensorflow:1.15.0-inf-py3".format(
            NEO_REGION_ACCOUNT, REGION
        )
        == model.image
    )
    assert model._is_compiled_model is True


def test_compile_model_for_edge_device(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="deeplens",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
    )
    assert model._is_compiled_model is False


def test_compile_model_for_edge_device_tflite(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="deeplens",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tflite",
        job_name="tflite-compile-model",
    )
    assert model._is_compiled_model is False


def test_compile_model_for_cloud(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
    )
    assert model._is_compiled_model is True


def test_compile_model_for_cloud_tflite(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tflite",
        job_name="tflite-compile-model",
    )
    assert model._is_compiled_model is True


@patch("sagemaker.session.Session")
def test_compile_creates_session(session):
    session.return_value.boto_region_name = REGION

    model = _create_model()
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
    )

    assert session.return_value == model.sagemaker_session


def test_check_neo_region(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    for region_name in EC2_REGION_LIST:
        if region_name in NEO_REGION_LIST:
            assert model.check_neo_region(region_name) is True
        else:
            assert model.check_neo_region(region_name) is False
