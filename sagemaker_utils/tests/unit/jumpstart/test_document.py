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
"""Test for JumpStart Document."""
from __future__ import absolute_import

import json
import os
from botocore.exceptions import ClientError

import pytest
from unittest.mock import patch

from sagemaker_core.resources import HubContent
from sagemaker.utils.jumpstart.document import get_hub_content_document
from sagemaker.utils.jumpstart.configs import JumpStartConfig
from sagemaker.utils.jumpstart.model import HubContentDocument

DEFAULT_ROLE = "arn:aws:iam::123456789012:role/role-name"
DEFAULT_REGION = "us-west-2"


@pytest.fixture(scope="function")
def jumpstart_session():
    with patch("sagemaker_core.helper.session_helper.Session") as mock_session:
        session_instance = mock_session.return_value
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.boto_region_name = DEFAULT_REGION
        yield session_instance


@pytest.fixture(scope="function")
def valid_hub_content():
    """Fixture to create a valid HubContentDocument."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir, "hub_content_document.json"), "r") as f:
        hub_content_document = json.load(f)
        return HubContent(
            hub_name="SageMakerPublicHub",
            hub_content_name="meta-textgeneration-llama-2-13b-f",
            hub_content_version="1.0.0",
            hub_content_type="Model",
            hub_content_document=json.dumps(hub_content_document),
        )


def test_get_hub_content_document_happy(valid_hub_content, jumpstart_session):
    """Test HubContentDocument initialization for all documents."""

    jumpstart_config = JumpStartConfig(model_id="meta-textgeneration-llama-2-13b-f")

    with patch("sagemaker.utils.jumpstart.document.HubContent.get") as mock_get:
        mock_get.return_value = valid_hub_content
        hub_content_document = get_hub_content_document(
            jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
        )
        assert isinstance(hub_content_document, HubContentDocument)


def test_get_hub_content_document_failure(jumpstart_session):
    """Test HubContentDocument initialization for all documents."""

    jumpstart_config = JumpStartConfig(model_id="non-existent-model-id")

    with patch("sagemaker.utils.jumpstart.document.HubContent.get") as mock_get:
        mock_get.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFound"}},
            operation_name="DescribeHubContent",
        )
        with pytest.raises(ClientError) as excinfo:
            get_hub_content_document(
                jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
            )
