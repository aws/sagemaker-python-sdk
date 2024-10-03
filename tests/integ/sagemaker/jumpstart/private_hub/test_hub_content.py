import pytest
import os
from sagemaker.jumpstart.hub.hub import Hub

from sagemaker.jumpstart.hub.interfaces import DescribeHubContentResponse
from tests.integ.sagemaker.jumpstart.utils import (
    get_sm_session,
)
from tests.integ.sagemaker.jumpstart.utils import (
    get_public_hub_model_arn
)
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
)
import tests


def test_hub_model_reference(setup):
    model_id = "meta-textgenerationneuron-llama-3-2-1b-instruct"

    hub_instance = Hub(hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME], sagemaker_session=get_sm_session())

    #Create Model Reference
    create_model_response = hub_instance.create_model_reference(
        model_arn = get_public_hub_model_arn(hub_instance, model_id)
    )
    assert create_model_response is not None  

    #Describe Model
    describe_model_response = hub_instance.describe_model(
        model_name = model_id
    )
    assert describe_model_response is not None
    assert type(describe_model_response) == DescribeHubContentResponse

    #Delete Model Reference
    delete_model_response = hub_instance.delete_model_reference(model_name=model_id)
    assert delete_model_response is not None  
