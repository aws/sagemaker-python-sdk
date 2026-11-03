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
"""Test for JumpStart HubContentDocument Model."""

from __future__ import absolute_import

import json
import os
from sagemaker.core.jumpstart.models import (
    CapabilityEnum,
    HostingComponentsModel,
    HubContentDocument,
)

TEST_HUB_CONTENT_DOCUMENT = "hub_content_document.json"

MINIMAL_DOCUMENT = {
    "ModelTypes": ["OPEN_WEIGHTS"],
    "Url": "https://example.com",
    "TrainingSupported": True,
    "IncrementalTrainingSupported": False,
}


def test_init_hub_content_document():
    """Test HubContentDocument initialization."""

    with open(os.path.join(os.path.dirname(__file__), TEST_HUB_CONTENT_DOCUMENT), "r") as f:
        hub_content_document = json.load(f)
    hub_content_document_instance = HubContentDocument(**hub_content_document)
    assert isinstance(hub_content_document_instance, HubContentDocument)


def test_capability_enum_has_customization():
    """CUSTOMIZATION is a valid, documented capability value."""

    assert CapabilityEnum.CUSTOMIZATION.value == "CUSTOMIZATION"


def test_hub_content_document_parses_customization_capability():
    """A document whose Capabilities include CUSTOMIZATION parses successfully."""

    document = HubContentDocument(**MINIMAL_DOCUMENT, Capabilities=["TRAINING", "CUSTOMIZATION"])
    assert CapabilityEnum.CUSTOMIZATION in document.Capabilities


def test_hub_content_document_parses_all_known_capabilities():
    """Every documented CapabilityEnum value parses without error."""

    known = [capability.value for capability in CapabilityEnum]
    document = HubContentDocument(**MINIMAL_DOCUMENT, Capabilities=known)
    assert all(isinstance(capability, CapabilityEnum) for capability in document.Capabilities)
    assert len(document.Capabilities) == len(known)


def test_hub_content_document_tolerates_unknown_capability(caplog):
    """An unknown capability is kept as a string with a warning, not rejected."""

    with caplog.at_level("WARNING", logger="sagemaker.core.jumpstart.models"):
        document = HubContentDocument(
            **MINIMAL_DOCUMENT, Capabilities=["TRAINING", "SOME_FUTURE_CAPABILITY"]
        )

    assert "SOME_FUTURE_CAPABILITY" in document.Capabilities
    assert isinstance(document.Capabilities[-1], str)
    assert CapabilityEnum.TRAINING in document.Capabilities
    assert "SOME_FUTURE_CAPABILITY" in caplog.text


def test_hosting_components_model_tolerates_unknown_capability():
    """The base-class Capabilities field also tolerates unknown values."""

    component = HostingComponentsModel(Capabilities=["CUSTOMIZATION", "SOME_FUTURE_CAPABILITY"])
    assert CapabilityEnum.CUSTOMIZATION in component.Capabilities
    assert "SOME_FUTURE_CAPABILITY" in component.Capabilities
