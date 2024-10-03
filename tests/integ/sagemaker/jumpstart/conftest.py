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
import boto3
import pytest
from botocore.config import Config
from sagemaker.jumpstart.hub.hub import Hub
from sagemaker.session import Session
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
    HUB_NAME_PREFIX,
    JUMPSTART_TAG,
    SM_JUMPSTART_PUBLIC_HUB_NAME,
)

from sagemaker.jumpstart.types import (
    HubContentType,
)

from sagemaker.jumpstart.constants import JUMPSTART_LOGGER

from tests.integ.sagemaker.jumpstart.utils import (
    get_test_artifact_bucket,
    get_test_suite_id,
    get_sm_session,
)

from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME


def _setup():
    print("Setting up...")
    test_suit_id = get_test_suite_id()
    test_hub_name = f"{HUB_NAME_PREFIX}{test_suit_id}"
    test_hub_description = "PySDK Integ Test Private Hub"
    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID: test_suit_id})
    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME: test_hub_name})
    hub = Hub(
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME], sagemaker_session=get_sm_session()
    )
    hub.create(description=test_hub_description)
    describe_hub_response = hub.describe()
    JUMPSTART_LOGGER.info(f"Describe Hub {describe_hub_response}")


def _teardown():
    print("Tearing down...")

    test_cache_bucket = get_test_artifact_bucket()

    test_suite_id = os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]
    test_hub_name = os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME]

    boto3_session = boto3.Session(region_name=JUMPSTART_DEFAULT_REGION_NAME)

    sagemaker_client = boto3_session.client(
        "sagemaker",
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
    )

    sagemaker_session = Session(boto_session=boto3_session, sagemaker_client=sagemaker_client)

    search_endpoints_result = sagemaker_client.search(
        Resource="Endpoint",
        SearchExpression={
            "Filters": [
                {"Name": f"Tags.{JUMPSTART_TAG}", "Operator": "Equals", "Value": test_suite_id}
            ]
        },
    )

    endpoint_names = [
        endpoint_info["Endpoint"]["EndpointName"]
        for endpoint_info in search_endpoints_result["Results"]
    ]
    endpoint_config_names = [
        endpoint_info["Endpoint"]["EndpointConfigName"]
        for endpoint_info in search_endpoints_result["Results"]
    ]
    model_names = list(
        filter(
            lambda elt: elt is not None,
            [
                sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)[
                    "ProductionVariants"
                ][0].get("ModelName")
                for endpoint_config_name in endpoint_config_names
            ],
        )
    )

    inference_component_names = []
    for endpoint_name in endpoint_names:
        for (
            inference_component_name
        ) in sagemaker_session.list_and_paginate_inference_component_names_associated_with_endpoint(
            endpoint_name=endpoint_name
        ):
            inference_component_names.append(inference_component_name)

    # delete inference components for test-suite-tagged endpoints
    for inference_component_name in inference_component_names:
        sagemaker_session.delete_inference_component(
            inference_component_name=inference_component_name, wait=True
        )

    # delete test-suite-tagged endpoints
    for endpoint_name in endpoint_names:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    # delete endpoint configs for test-suite-tagged endpoints
    for endpoint_config_name in endpoint_config_names:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

    # delete models for test-suite-tagged endpoints
    for model_name in model_names:
        sagemaker_client.delete_model(ModelName=model_name)

    # delete test artifact/cache s3 folder
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(test_cache_bucket)
    bucket.objects.filter(Prefix=test_suite_id + "/").delete()

    # delete private hubs
    _delete_hubs(sagemaker_session)


def _delete_hubs(sagemaker_session):
    # list Hubs created by PySDK integration tests
    list_hub_response = sagemaker_session.list_hubs(name_contains=HUB_NAME_PREFIX)

    for hub in list_hub_response["HubSummaries"]:
        if hub["HubName"] != SM_JUMPSTART_PUBLIC_HUB_NAME:
            # delete all hub contents first
            _delete_hub_contents(sagemaker_session, hub["HubName"])
            JUMPSTART_LOGGER.info(f"Deleting {hub['HubName']}")
            sagemaker_session.delete_hub(hub["HubName"])


def _delete_hub_contents(sagemaker_session, test_hub_name):
    # list hub_contents for the given hub
    list_hub_content_response = sagemaker_session.list_hub_contents(
        hub_name=test_hub_name, hub_content_type=HubContentType.MODEL_REFERENCE.value
    )
    JUMPSTART_LOGGER.info(f"Listing HubContents {list_hub_content_response}")

    # delete hub_contents for the given hub
    for models in list_hub_content_response["HubContentSummaries"]:
        sagemaker_session.delete_hub_content_reference(
            hub_name=test_hub_name,
            hub_content_type=HubContentType.MODEL_REFERENCE.value,
            hub_content_name=models["HubContentName"],
        )


@pytest.fixture(scope="session", autouse=True)
def setup(request):
    _setup()

    request.addfinalizer(_teardown)
