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

import pytest
from mock import Mock, patch

from sagemaker.model import Model

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"

IMAGE_URI = "inference-container-uri"

REGION = "us-west-2"

NEO_REGION_ACCOUNT = "301217895009"
DESCRIBE_COMPILATION_JOB_RESPONSE = {
    "CompilationJobStatus": "Completed",
    "ModelArtifacts": {"S3ModelArtifacts": "s3://output-path/model.tar.gz"},
    "InferenceImage": IMAGE_URI,
}


@pytest.fixture
def sagemaker_session():
    session = Mock(
        boto_region_name=REGION,
        default_bucket_prefix=None,
    )
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


def _create_model(sagemaker_session=None):
    return Model(MODEL_IMAGE, MODEL_DATA, role="role", sagemaker_session=sagemaker_session)


def test_compile_model_for_inferentia(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="ml_inf1",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        framework_version="1.15.0",
        job_name="compile-model",
    )
    assert DESCRIBE_COMPILATION_JOB_RESPONSE["InferenceImage"] == model.image_uri
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


@pytest.mark.xfail(reason="tflite images are not available yet.")
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


def test_compile_model_linux_arm64_nvidia(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family=None,
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
        target_platform_os="LINUX",
        target_platform_arch="ARM64",
        target_platform_accelerator="NVIDIA",
        compiler_options={"gpu-code": "sm_72", "trt-ver": "6.0.1", "cuda-ver": "10.1"},
    )
    assert model._is_compiled_model is False


def test_compile_model_android_armv7(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family=None,
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
        target_platform_os="ANDROID",
        target_platform_arch="ARM_EABI",
        compiler_options={"ANDROID_PLATFORM": 25, "mattr": ["+neon"]},
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


@pytest.mark.xfail(reason="tflite images are not available yet.")
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
    session.return_value.sagemaker_config = {}

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


def test_compile_validates_framework():
    model = _create_model()

    with pytest.raises(ValueError) as e:
        model.compile(
            target_instance_family="ml_c4",
            input_shape={"data": [1, 3, 1024, 1024]},
            output_path="s3://output",
            role="role",
        )

    assert "You must specify framework" in str(e)

    with pytest.raises(ValueError) as e:
        model.compile(
            target_instance_family="ml_c4",
            input_shape={"data": [1, 3, 1024, 1024]},
            output_path="s3://output",
            role="role",
            framework="not-a-real-framework",
        )

    assert "You must provide valid framework" in str(e)


def test_compile_validates_job_name():
    model = _create_model()

    with pytest.raises(ValueError) as e:
        model.compile(
            target_instance_family="ml_c4",
            input_shape={"data": [1, 3, 1024, 1024]},
            output_path="s3://output",
            role="role",
            framework="tensorflow",
        )

    assert "You must provide a compilation job name" in str(e)


def test_compile_validates_model_data():
    model = Model(MODEL_IMAGE)

    with pytest.raises(ValueError) as e:
        model.compile(
            target_instance_family="ml_c4",
            input_shape={"data": [1, 3, 1024, 1024]},
            output_path="s3://output",
            role="role",
            framework="tensorflow",
            job_name="compile-model",
        )

    assert "You must provide an S3 path to the compressed model artifacts." in str(e)


def test_deploy_honors_provided_model_name(sagemaker_session):
    model = _create_model(sagemaker_session)
    model._is_compiled_model = True

    model_name = "foo"
    model.name = model_name

    model.deploy(1, "ml.c4.xlarge")
    assert model_name == model.name


def test_deploy_add_compiled_model_suffix_to_generated_resource_names(sagemaker_session):
    model = _create_model(sagemaker_session)
    model._is_compiled_model = True

    model.deploy(1, "ml.c4.xlarge")
    assert model.name.startswith("mi-ml-c4")
    assert model.endpoint_name.startswith("mi-ml-c4")


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
def test_deploy_add_compiled_model_suffix_to_endpoint_name_from_model_name(sagemaker_session):
    model = _create_model(sagemaker_session)
    model._is_compiled_model = True

    model_name = "foo"
    model.name = model_name

    model.deploy(1, "ml.c4.xlarge")
    assert model.endpoint_name.startswith("{}-ml-c4".format(model_name))


def test_compile_with_framework_version_15(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )

    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="pytorch",
        framework_version="1.5",
        job_name="compile-model",
    )

    assert IMAGE_URI == model.image_uri


def test_compile_with_framework_version_16(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )

    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="pytorch",
        framework_version="1.6",
        job_name="compile-model",
    )

    assert IMAGE_URI == model.image_uri


@patch("sagemaker.session.Session")
def test_compile_with_pytorch_neo_in_ml_inf(session):
    session.return_value.boto_region_name = REGION
    session.return_value.sagemaker_config = {}

    model = _create_model()
    model.compile(
        target_instance_family="ml_inf1",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="pytorch",
        framework_version="1.6",
        job_name="compile-model",
    )

    assert (
        "{}.dkr.ecr.{}.amazonaws.com/sagemaker-inference-pytorch:1.6-cpu-py3".format(
            NEO_REGION_ACCOUNT, REGION
        )
        != model.image_uri
    )


@patch("sagemaker.session.Session")
def test_compile_with_tensorflow_neo_in_ml_inf(session):
    session.return_value.boto_region_name = REGION
    session.return_value.sagemaker_config = {}

    model = _create_model()
    model.compile(
        target_instance_family="ml_inf1",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        framework_version="1.15",
        job_name="compile-model",
    )

    assert (
        "{}.dkr.ecr.{}.amazonaws.com/sagemaker-inference-tensorflow:1.15-cpu-py3".format(
            NEO_REGION_ACCOUNT, REGION
        )
        != model.image_uri
    )


def test_compile_validates_framework_version(sagemaker_session):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value={
            "CompilationJobStatus": "Completed",
            "ModelArtifacts": {"S3ModelArtifacts": "s3://output-path/model.tar.gz"},
            "InferenceImage": None,
        }
    )
    model = _create_model(sagemaker_session)
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="pytorch",
        framework_version="1.6.1",
        job_name="compile-model",
    )

    assert model.image_uri is None
