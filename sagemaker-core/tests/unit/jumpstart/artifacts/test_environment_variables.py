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
"""Unit tests for JumpStart environment variable retrieval."""
from __future__ import absolute_import

import datetime
import json
import os
from unittest.mock import patch

import pytest

from sagemaker.core.jumpstart.artifacts.environment_variables import (
    _retrieve_default_environment_variables,
)
from sagemaker.core.jumpstart.hub.interfaces import DescribeHubContentResponse
from sagemaker.core.jumpstart.hub.parsers import (
    make_model_specs_from_describe_hub_content_response,
)

HUB_ARN = "arn:aws:sagemaker:us-east-1:123456789012:hub/my-private-hub"


@pytest.fixture
def private_hub_model_specs():
    """Model specs parsed from a private hub document, the way the JumpStart cache builds them."""
    path = os.path.join(os.path.dirname(__file__), "..", "hub_content_document.json")
    with open(path, "r") as f:
        hub_content_document = json.load(f)
    response = DescribeHubContentResponse(
        {
            "CreationTime": datetime.datetime(2024, 1, 1),
            "DocumentSchemaVersion": "2.0.0",
            "HubArn": HUB_ARN,
            "HubContentArn": (
                "arn:aws:sagemaker:us-east-1:123456789012:hub-content/"
                "my-private-hub/Model/meta-textgeneration-llama-2-13b-f/1.0.0"
            ),
            "HubContentName": "meta-textgeneration-llama-2-13b-f",
            "HubContentType": "Model",
            "HubContentVersion": "1.0.0",
            "HubContentStatus": "Available",
            "HubName": "my-private-hub",
            "HubContentDocument": json.dumps(hub_content_document),
        }
    )
    return make_model_specs_from_describe_hub_content_response(response)


@patch(
    "sagemaker.core.jumpstart.artifacts.environment_variables.verify_model_region_and_return_specs"
)
def test_private_hub_instance_specific_environment_variable_overrides_default(
    mock_verify_model_region_and_return_specs, private_hub_model_specs
):
    """An instance specific value must replace the model default rather than being added
    under a mangled second key, which is what ended up in the container environment before."""
    mock_verify_model_region_and_return_specs.return_value = private_hub_model_specs

    environment_variables = _retrieve_default_environment_variables(
        model_id="meta-textgeneration-llama-2-13b-f",
        model_version="1.0.0",
        hub_arn=HUB_ARN,
        region="us-east-1",
        instance_type="ml.g5.48xlarge",
        sagemaker_session=None,
    )

    gpu_keys = [key for key in environment_variables if key.replace("_", "").lower() == "smnumgpus"]
    assert gpu_keys == ["SM_NUM_GPUS"]
    assert environment_variables["SM_NUM_GPUS"] == "8"
    assert mock_verify_model_region_and_return_specs.call_args.kwargs["hub_arn"] == HUB_ARN
