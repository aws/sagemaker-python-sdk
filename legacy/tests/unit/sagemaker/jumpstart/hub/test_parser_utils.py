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
from sagemaker.jumpstart.hub.parser_utils import camel_to_snake
from sagemaker.jumpstart.hub.parsers import make_model_specs_from_describe_hub_content_response
from sagemaker.jumpstart.hub.interfaces import HubModelDocument
from tests.unit.sagemaker.jumpstart.constants import HUB_MODEL_DOCUMENT_DICTS
from unittest.mock import MagicMock
from sagemaker.jumpstart.types import HubContentType


REGION = "us-east-1"
ACCOUNT_ID = "123456789123"
HUB_NAME = "mock-hub-name"


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("camelCase", "camel_case"),
        ("PascalCase", "pascal_case"),
        ("already_snake", "already_snake"),
        ("", ""),
        ("A", "a"),
        ("PascalCase123", "pascal_case123"),
        ("123StartWithNumber", "123_start_with_number"),
    ],
)
def test_parse_camelCase(input_string, expected):
    assert expected == camel_to_snake(input_string)


def test_make_model_specs_from_describe_hub_content_response():
    mock_describe_response = MagicMock()
    region = "us-west-2"
    mock_describe_response.hub_content_type = HubContentType.MODEL
    mock_describe_response.get_hub_region.return_value = region
    mock_describe_response.hub_content_version = "1.0.0"
    json_obj = HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"]
    mock_describe_response.hub_content_document = HubModelDocument(json_obj=json_obj, region=region)

    make_model_specs_from_describe_hub_content_response(mock_describe_response)
