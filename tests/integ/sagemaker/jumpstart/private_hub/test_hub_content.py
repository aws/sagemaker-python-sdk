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
import os
from sagemaker.jumpstart.hub.hub import Hub

from sagemaker.jumpstart.hub.interfaces import DescribeHubContentResponse
from tests.integ.sagemaker.jumpstart.utils import (
    get_sm_session,
)
from tests.integ.sagemaker.jumpstart.utils import get_public_hub_model_arn
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
)


def test_hub_model_reference(setup):
    model_id = "meta-textgenerationneuron-llama-3-2-1b-instruct"

    hub_instance = Hub(
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME], sagemaker_session=get_sm_session()
    )

    create_model_response = hub_instance.create_model_reference(
        model_arn=get_public_hub_model_arn(hub_instance, model_id)
    )
    assert create_model_response is not None

    describe_model_response = hub_instance.describe_model(model_name=model_id)
    assert describe_model_response is not None
    assert isinstance(describe_model_response, DescribeHubContentResponse)
    assert describe_model_response.hub_content_name == model_id
    assert describe_model_response.hub_content_type == "ModelReference"

    delete_model_response = hub_instance.delete_model_reference(model_name=model_id)
    assert delete_model_response is not None
