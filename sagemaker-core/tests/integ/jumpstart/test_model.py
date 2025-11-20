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

from botocore.config import Config
import boto3
import json
import os

import pytest

from sagemaker.core.jumpstart.document import HubContentDocument


@pytest.fixture(scope="module")
def sm_client():
    """Fixture to create a SageMaker client."""
    config = Config(retries=dict(max_attempts=10, mode="adaptive"))
    return boto3.client("sagemaker", region_name="us-west-2", config=config)


def test_all_hub_content_documents(sm_client):
    """Test HubContentDocument initialization for all documents."""

    next_token = None
    while True:
        if next_token:
            response = sm_client.list_hub_contents(
                HubName="SageMakerPublicHub",
                HubContentType="Model",
                NextToken=next_token,
            )
        else:
            response = sm_client.list_hub_contents(
                HubName="SageMakerPublicHub",
                HubContentType="Model",
            )

        for summary in response["HubContentSummaries"]:
            content = sm_client.describe_hub_content(
                HubName="SageMakerPublicHub",
                HubContentType="Model",
                HubContentName=summary["HubContentName"],
            )
            content_document = json.loads(content["HubContentDocument"])
            print(content["HubContentName"])
            
            # Skip models with RecipeCollection field (not yet supported)
            if "RecipeCollection" in content_document:
                continue
                
            hub_content_document = HubContentDocument(**content_document)
            assert isinstance(hub_content_document, HubContentDocument)

        next_token = response.get("NextToken")
        if not next_token:
            break


def test_specific_hub_content_document(sm_client):
    """Test HubContentDocument initialization for a specific document.

    This is a placeholder in case any new models are added and break the previous test.
    """
    # Add the model IDs you want to test
    model_to_test = []

    for model_id in model_to_test:
        content = sm_client.describe_hub_content(
            HubName="SageMakerPublicHub",
            HubContentType="Model",
            HubContentName=model_id,
        )
        content_document = json.loads(content["HubContentDocument"])

        # write document to file in local directory
        local_dir = os.path.dirname(__file__)
        with open(os.path.join(local_dir, f"{model_id}.json"), "w") as f:
            json.dump(content_document, f, indent=4)
        hub_content_document = HubContentDocument(**content_document)
        assert isinstance(hub_content_document, HubContentDocument)

        # delete the file after test
        os.remove(os.path.join(local_dir, f"{model_id}.json"))
