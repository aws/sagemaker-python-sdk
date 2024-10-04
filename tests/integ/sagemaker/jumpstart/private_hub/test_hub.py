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
from sagemaker.jumpstart.hub.hub import Hub

from tests.integ.sagemaker.jumpstart.utils import (
    get_sm_session,
)
from tests.integ.sagemaker.jumpstart.utils import (
    get_test_suite_id,
)
from tests.integ.sagemaker.jumpstart.constants import (
    HUB_NAME_PREFIX,
)


@pytest.fixture
def hub_instance():
    HUB_NAME = f"{HUB_NAME_PREFIX}-{get_test_suite_id()}"
    hub = Hub(HUB_NAME, sagemaker_session=get_sm_session())
    yield hub


def test_private_hub(setup, hub_instance):
    # Createhub
    create_hub_response = hub_instance.create(
        description="This is a Test Private Hub.",
        display_name="PySDK integration tests Hub",
        search_keywords=["jumpstart-sdk-integ-test"],
    )

    # Create Hub Verifications
    assert create_hub_response is not None

    # Describe Hub
    hub_description = hub_instance.describe()
    assert hub_description is not None

    # Delete Hub
    delete_hub_response = hub_instance.delete()
    assert delete_hub_response is not None
