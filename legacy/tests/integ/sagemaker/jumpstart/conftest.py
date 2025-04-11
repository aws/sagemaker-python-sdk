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
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.hub.hub import Hub
from sagemaker.session import Session
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
    HUB_NAME_PREFIX,
    JUMPSTART_TAG,
)

from sagemaker.jumpstart.types import (
    HubContentType,
)


from tests.integ.sagemaker.jumpstart.utils import (
    get_test_artifact_bucket,
    get_test_suite_id,
    get_sm_session,
    with_exponential_backoff,
)


def _setup():
    print("Setting up...")
    test_suite_id = get_test_suite_id()
    test_hub_name = f"{HUB_NAME_PREFIX}{test_suite_id}"
    test_hub_description = "PySDK Integ Test Private Hub"

    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID: test_suite_id})
    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME: test_hub_name})

    # Create a private hub to use for the test session
    hub = Hub(
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME], sagemaker_session=get_sm_session()
    )
    hub.create(description=test_hub_description)


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
    _delete_hubs(sagemaker_session, test_hub_name)


def _delete_hubs(sagemaker_session, hub_name):
    # list and delete all hub contents first
    list_hub_content_response = sagemaker_session.list_hub_contents(
        hub_name=hub_name, hub_content_type=HubContentType.MODEL_REFERENCE.value
    )
    for model in list_hub_content_response["HubContentSummaries"]:
        _delete_hub_contents(sagemaker_session, hub_name, model)

    sagemaker_session.delete_hub(hub_name)


@with_exponential_backoff()
def _delete_hub_contents(sagemaker_session, hub_name, model):
    sagemaker_session.delete_hub_content_reference(
        hub_name=hub_name,
        hub_content_type=HubContentType.MODEL_REFERENCE.value,
        hub_content_name=model["HubContentName"],
    )


@pytest.fixture(scope="session", autouse=True)
def setup(request):
    _setup()

    request.addfinalizer(_teardown)
