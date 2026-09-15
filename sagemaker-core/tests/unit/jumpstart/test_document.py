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

from sagemaker.core.resources import HubContent
from sagemaker.core.jumpstart.document import get_hub_content_and_document
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.jumpstart.models import HubContentDocument

DEFAULT_ROLE = "arn:aws:iam::123456789012:role/role-name"
DEFAULT_REGION = "us-west-2"


@pytest.fixture(scope="function")
def jumpstart_session():
    with patch("sagemaker.core.helper.session_helper.Session") as mock_session:
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

    with patch("sagemaker.core.jumpstart.document.HubContent.get") as mock_get:
        mock_get.return_value = valid_hub_content
        hub_content, hub_content_document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
        )
        assert isinstance(hub_content_document, HubContentDocument)
        # assert isinstance(hub_content, HubContent)


def test_get_hub_content_document_failure(jumpstart_session):
    """Test HubContentDocument initialization for all documents."""

    jumpstart_config = JumpStartConfig(model_id="non-existent-model-id")

    with patch("sagemaker.core.jumpstart.document.HubContent.get") as mock_get:
        mock_get.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFound"}},
            operation_name="DescribeHubContent",
        )
        with pytest.raises(ClientError):
            get_hub_content_and_document(
                jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
            )


# ---------------------------------------------------------------------------
# Tests for private-hub content-type probing + hub_content_name alias support.
#
# A private hub can contain either a ModelReference (a pointer to a public
# model) or a privately-owned Model. get_hub_content_and_document() must not
# guess from the hub name; it probes ModelReference first, then falls back to
# Model. The public hub only holds Models. It also honors hub_content_name when
# the content is filed under an alias differing from model_id.
#
# Note: distinct model_id / hub_name values are used per test to avoid the
# module-level lru_cache on get_hub_content_and_document returning a stale
# result across tests.
# ---------------------------------------------------------------------------


def _hub_content(hub_name, name, content_type, doc):
    return HubContent(
        hub_name=hub_name,
        hub_content_name=name,
        hub_content_version="1.0.0",
        hub_content_type=content_type,
        hub_content_document=json.dumps(doc),
    )


def _not_found():
    return ClientError(
        error_response={"Error": {"Code": "ResourceNotFound"}},
        operation_name="DescribeHubContent",
    )


def _load_doc():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir, "hub_content_document.json"), "r") as f:
        return json.load(f)


def test_public_hub_uses_model_type_only(jumpstart_session):
    """Public hub: resolve as Model, and never probe ModelReference."""
    doc = _load_doc()
    jumpstart_config = JumpStartConfig(model_id="probe-public-model")

    with patch("sagemaker.core.jumpstart.document.HubContent.get") as mock_get:
        mock_get.return_value = _hub_content(
            "SageMakerPublicHub", "probe-public-model", "Model", doc
        )
        hub_content, _ = get_hub_content_and_document(
            jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
        )

    assert hub_content.hub_content_type == "Model"
    # Public hub must be looked up exactly once, as Model.
    assert mock_get.call_count == 1
    assert mock_get.call_args.kwargs["hub_content_type"] == "Model"


def test_private_hub_resolves_model_reference_first(jumpstart_session):
    """Private hub holding a ModelReference: first probe (ModelReference) hits."""
    doc = _load_doc()
    jumpstart_config = JumpStartConfig(model_id="probe-ref-model", hub_name="my-private-hub-ref")

    with patch("sagemaker.core.jumpstart.document.HubContent.get") as mock_get:
        mock_get.return_value = _hub_content(
            "my-private-hub-ref", "probe-ref-model", "ModelReference", doc
        )
        hub_content, _ = get_hub_content_and_document(
            jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
        )

    assert hub_content.hub_content_type == "ModelReference"
    # ModelReference is tried first and succeeds -> single call.
    assert mock_get.call_count == 1
    assert mock_get.call_args.kwargs["hub_content_type"] == "ModelReference"


def test_private_hub_falls_back_to_model(jumpstart_session):
    """Private hub holding a privately-owned Model: ModelReference misses, then
    the Model fallback resolves it (the core of the fix)."""
    doc = _load_doc()
    jumpstart_config = JumpStartConfig(
        model_id="probe-private-model", hub_name="my-private-hub-model"
    )

    with patch("sagemaker.core.jumpstart.document.HubContent.get") as mock_get:
        mock_get.side_effect = [
            _not_found(),  # ModelReference lookup misses
            _hub_content(  # Model fallback resolves
                "my-private-hub-model", "probe-private-model", "Model", doc
            ),
        ]
        hub_content, _ = get_hub_content_and_document(
            jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
        )

    assert hub_content.hub_content_type == "Model"
    # Two probes: ModelReference (miss) then Model (hit).
    assert mock_get.call_count == 2
    assert [c.kwargs["hub_content_type"] for c in mock_get.call_args_list] == [
        "ModelReference",
        "Model",
    ]


def test_private_hub_honors_hub_content_name_alias(jumpstart_session):
    """When hub_content_name is set (alias differs from model_id), the lookup
    must use the alias, not the model_id."""
    doc = _load_doc()
    jumpstart_config = JumpStartConfig(
        model_id="probe-alias-public-id",
        hub_name="my-private-hub-alias",
        hub_content_name="the-alias-name",
    )

    with patch("sagemaker.core.jumpstart.document.HubContent.get") as mock_get:
        mock_get.return_value = _hub_content(
            "my-private-hub-alias", "the-alias-name", "ModelReference", doc
        )
        get_hub_content_and_document(
            jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
        )

    # Lookup used the alias, not the model_id.
    assert mock_get.call_args.kwargs["hub_content_name"] == "the-alias-name"


def test_private_hub_not_found_as_either_type_raises(jumpstart_session):
    """Private hub where neither ModelReference nor Model exists: raise."""
    jumpstart_config = JumpStartConfig(
        model_id="probe-missing-model", hub_name="my-private-hub-missing"
    )

    with patch("sagemaker.core.jumpstart.document.HubContent.get") as mock_get:
        mock_get.side_effect = [_not_found(), _not_found()]
        with pytest.raises(ClientError):
            get_hub_content_and_document(
                jumpstart_config=jumpstart_config, sagemaker_session=jumpstart_session
            )
        # Both content types were attempted before giving up.
        assert mock_get.call_count == 2
