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
from unittest.mock import Mock, patch
from sagemaker.jumpstart.curated_hub.accessors.file_generator import (
    generate_file_infos_from_model_specs,
    generate_file_infos_from_s3_location,
)
from sagemaker.jumpstart.curated_hub.types import FileInfo, S3ObjectLocation

from sagemaker.jumpstart.types import JumpStartModelSpecs
from tests.unit.sagemaker.jumpstart.constants import BASE_SPEC
from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec


@pytest.fixture()
def s3_client():
    mock_s3_client = Mock()
    mock_s3_client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "my-key-one", "Size": 123456789, "LastModified": "08-14-1997 00:00:00"}
        ]
    }
    mock_s3_client.head_object.return_value = {
        "ContentLength": 123456789,
        "LastModified": "08-14-1997 00:00:00",
    }
    return mock_s3_client


def test_s3_path_file_generator_happy_path(s3_client):
    s3_client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "my-key-one", "Size": 123456789, "LastModified": "08-14-1997 00:00:00"},
            {"Key": "my-key-one", "Size": 10101010, "LastModified": "08-14-1997 00:00:00"},
        ]
    }

    mock_hub_bucket = S3ObjectLocation(bucket="mock-bucket", key="mock-key")
    response = generate_file_infos_from_s3_location(mock_hub_bucket, s3_client)

    s3_client.list_objects_v2.assert_called_once()
    assert response == [
        FileInfo("mock-bucket", "my-key-one", 123456789, "08-14-1997 00:00:00"),
        FileInfo("mock-bucket", "my-key-one", 10101010, "08-14-1997 00:00:00"),
    ]


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_model_specs_file_generator_happy_path(patched_get_model_specs, s3_client):
    patched_get_model_specs.side_effect = get_spec_from_base_spec

    specs = JumpStartModelSpecs(BASE_SPEC)
    studio_specs = {"defaultDataKey": "model_id123"}
    response = generate_file_infos_from_model_specs(specs, studio_specs, "us-west-2", s3_client)

    s3_client.head_object.assert_called()
    patched_get_model_specs.assert_called()

    assert response == [
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
            123456789,
            "08-14-1997 00:00:00",
        ),
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
            123456789,
            "08-14-1997 00:00:00",
        ),
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
            123456789,
            "08-14-1997 00:00:00",
        ),
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
            123456789,
            "08-14-1997 00:00:00",
        ),
        FileInfo("jumpstart-cache-prod-us-west-2", "model_id123", 123456789, "08-14-1997 00:00:00"),
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "pytorch-notebooks/pytorch-ic-mobilenet-v2-inference.ipynb",
            123456789,
            "08-14-1997 00:00:00",
        ),
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "pytorch-metadata/pytorch-ic-mobilenet-v2.md",
            123456789,
            "08-14-1997 00:00:00",
        ),
    ]


def test_s3_path_file_generator_with_no_objects(s3_client):
    s3_client.list_objects_v2.return_value = {"Contents": []}

    mock_hub_bucket = S3ObjectLocation(bucket="mock-bucket", key="mock-key")
    response = generate_file_infos_from_s3_location(mock_hub_bucket, s3_client)

    s3_client.list_objects_v2.assert_called_once()
    assert response == []

    s3_client.list_objects_v2.reset_mock()

    s3_client.list_objects_v2.return_value = {}

    mock_hub_bucket = S3ObjectLocation(bucket="mock-bucket", key="mock-key")
    response = generate_file_infos_from_s3_location(mock_hub_bucket, s3_client)

    s3_client.list_objects_v2.assert_called_once()
    assert response == []

@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_specs_file_generator_training_unsupported(patched_get_model_specs, s3_client):
    specs = Mock()
    specs.model_id = "mock_model_123"
    specs.training_supported = False
    specs.gated_bucket = False
    specs.hosting_prepacked_artifact_key = "/my/inference/tarball.tgz"
    specs.hosting_script_key = "/my/inference/script.py"

    response = generate_file_infos_from_model_specs(specs, {}, "us-west-2", s3_client)

    assert response == [
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "/my/inference/tarball.tgz",
            123456789,
            "08-14-1997 00:00:00",
        ),
        FileInfo(
            "jumpstart-cache-prod-us-west-2",
            "/my/inference/script.py",
            123456789,
            "08-14-1997 00:00:00",
        ),
    ]

@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_specs_file_generator_gated_model(patched_get_model_specs, s3_client):
    specs = Mock()
    specs.model_id = "mock_model_123"
    specs.gated_bucket = True
    specs.training_supported = True
    specs.hosting_prepacked_artifact_key = "/my/inference/tarball.tgz"
    specs.hosting_script_key = "/my/inference/script.py"
    specs.training_prepacked_artifact_key = "/my/training/tarball.tgz"
    specs.training_script_key = "/my/training/script.py"

    response = generate_file_infos_from_model_specs(specs, {}, "us-west-2", s3_client)

    assert response == []