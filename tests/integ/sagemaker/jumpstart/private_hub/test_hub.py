import pytest
import os
from unittest.mock import MagicMock, patch
from sagemaker.jumpstart.hub.hub import Hub
from sagemaker.jumpstart.constants import JUMPSTART_LOGGER

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
    HUB_NAME=f"{HUB_NAME_PREFIX}-{get_test_suite_id()}"
    hub = Hub(HUB_NAME, sagemaker_session=get_sm_session())
    yield hub

def test_private_hub(setup, hub_instance):
    #Createhub
    create_hub_response = hub_instance.create(
      description="This is a Test Private Hub.",
      display_name="malavhs Test hub",
      search_keywords=["jumpstart-sdk-integ-test"],
    )

    #Create Hub Verifications
    assert create_hub_response is not None

    #Describe Hub
    hub_description = hub_instance.describe()
    assert hub_description is not None

    #Delete Hub
    delete_hub_response = hub_instance.delete()
    assert delete_hub_response is not None 