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
from sagemaker.jumpstart.curated_hub.accessors.filegenerator import FileGenerator, ModelSpecsFileGenerator, S3PathFileGenerator
from sagemaker.jumpstart.curated_hub.accessors.fileinfo import FileInfo

from sagemaker.jumpstart.curated_hub.accessors.objectlocation import S3ObjectLocation
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
    generator = S3PathFileGenerator("us-west-2", s3_client)
    response = generator.format(mock_hub_bucket)

    s3_client.list_objects_v2.assert_called_once()
    assert response == [
        FileInfo("my-key-one", 123456789, "08-14-1997 00:00:00"),
        FileInfo("my-key-one", 10101010, "08-14-1997 00:00:00"),
    ]


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_model_specs_file_generator_happy_path(patched_get_model_specs, s3_client):
    patched_get_model_specs.side_effect = get_spec_from_base_spec

    specs = JumpStartModelSpecs(BASE_SPEC)
    studio_specs = {"defaultDataKey": "model_id123"}
    generator = ModelSpecsFileGenerator("us-west-2", s3_client, studio_specs)
    response = generator.format(specs)

    s3_client.head_object.assert_called()
    patched_get_model_specs.assert_called()
    # TODO: Figure out why object attrs aren't being compared
    assert response == [
        FileInfo("my-key-one", 123456789, "08-14-1997 00:00:00"),
        FileInfo("my-key-two", 123456789, "08-14-1997 00:00:00"),
        FileInfo("my-key-three", 123456789, "08-14-1997 00:00:00"),
        FileInfo("my-key-four", 123456789, "08-14-1997 00:00:00"),
        FileInfo("my-key-five", 123456789, "08-14-1997 00:00:00"),
        FileInfo("my-key-six", 123456789, "08-14-1997 00:00:00"),
        FileInfo("my-key-seven", 123456789, "08-14-1997 00:00:00"),
    ]


def test_s3_path_file_generator_with_no_objects(s3_client):
    s3_client.list_objects_v2.return_value = {"Contents": []}

    mock_hub_bucket = S3ObjectLocation(bucket="mock-bucket", key="mock-key")
    generator = S3PathFileGenerator("us-west-2", s3_client)
    response = generator.format(mock_hub_bucket)

    s3_client.list_objects_v2.assert_called_once()
    assert response == []

    s3_client.list_objects_v2.reset_mock()

    s3_client.list_objects_v2.return_value = {}

    mock_hub_bucket = S3ObjectLocation(bucket="mock-bucket", key="mock-key")
    generator = S3PathFileGenerator("us-west-2", s3_client)
    response = generator.format(mock_hub_bucket)

    s3_client.list_objects_v2.assert_called_once()
    assert response == []
