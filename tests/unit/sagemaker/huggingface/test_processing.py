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
from mock import Mock, patch, MagicMock

from sagemaker.huggingface.processing import HuggingFaceProcessor
from sagemaker.fw_utils import UploadedCode
from sagemaker.session_settings import SessionSettings

from .huggingface_utils import get_full_gpu_image_uri, GPU_INSTANCE_TYPE, REGION

BUCKET_NAME = "mybucket"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
ECR_HOSTNAME = "ecr.us-west-2.amazonaws.com"
CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"
MOCKED_S3_URI = "s3://mocked_s3_uri_from_upload_data"


@pytest.fixture(autouse=True)
def mock_create_tar_file():
    with patch("sagemaker.utils.create_tar_file", MagicMock()) as create_tar_file:
        yield create_tar_file


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)

    session_mock.upload_data = Mock(name="upload_data", return_value=MOCKED_S3_URI)
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = ROLE
    # For tests which doesn't verify config file injection, operate with empty config
    session_mock.sagemaker_config = {}
    return session_mock


@pytest.fixture()
def uploaded_code(
    s3_prefix="s3://mocked_s3_uri_from_upload_data/my_job_name/source/sourcedir.tar.gz",
    script_name="processing_code.py",
):
    return UploadedCode(s3_prefix=s3_prefix, script_name=script_name)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_huggingface_processor_with_required_parameters(
    exists_mock,
    isfile_mock,
    botocore_resolver,
    sagemaker_session,
    huggingface_training_version,
    huggingface_pytorch_training_version,
    huggingface_pytorch_training_py_version,
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = HuggingFaceProcessor(
        role=ROLE,
        instance_type=GPU_INSTANCE_TYPE,
        transformers_version=huggingface_training_version,
        pytorch_version=huggingface_pytorch_training_version,
        py_version=huggingface_pytorch_training_py_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args_modular_code(processor._current_job_name)
    expected_args["app_specification"]["ImageUri"] = get_full_gpu_image_uri(
        huggingface_training_version,
        f"pytorch{huggingface_pytorch_training_version}",
    )

    sagemaker_session.process.assert_called_with(**expected_args)


def _get_expected_args_modular_code(job_name, code_s3_uri=f"s3://{BUCKET_NAME}"):
    return {
        "inputs": [
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_uri}/{job_name}/source/sourcedir.tar.gz",
                    "LocalPath": "/opt/ml/processing/input/code/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "entrypoint",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_uri}/{job_name}/source/runproc.sh",
                    "LocalPath": "/opt/ml/processing/input/entrypoint",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {"Outputs": []},
        "experiment_config": None,
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": GPU_INSTANCE_TYPE,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": CUSTOM_IMAGE_URI,
            "ContainerEntrypoint": [
                "/bin/bash",
                "/opt/ml/processing/input/entrypoint/runproc.sh",
            ],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
        "experiment_config": None,
    }
