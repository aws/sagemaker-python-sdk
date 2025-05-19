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
from sagemaker.utils.jumpstart.models import HubContentDocument


TEST_HUB_CONTENT_DOCUMENT = "hub_content_document.json"


def test_init_hub_content_document():
    """Test HubContentDocument initialization."""

    with open(os.path.join(os.path.dirname(__file__), TEST_HUB_CONTENT_DOCUMENT), "r") as f:
        hub_content_document = json.load(f)
    hub_content_document_instance = HubContentDocument(**hub_content_document)
    assert isinstance(hub_content_document_instance, HubContentDocument)
