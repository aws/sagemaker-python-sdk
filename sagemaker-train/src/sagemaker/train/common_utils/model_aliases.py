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
"""Model name alias mapping: user-friendly IDs → SageMaker Hub content names.

This file maps Bedrock-style model identifiers (what users see in the Bedrock
console) to the corresponding SageMaker Hub content names (what the Hub API
expects for DescribeHubContent calls).

Users can pass either format to SFTTrainer, DPOTrainer, RLVRTrainer, etc.
The SDK normalizes to the Hub content name before making API calls.

MAINTENANCE:
    When a new model is onboarded to SageMaker Hub, add an entry here mapping
    its user-facing Bedrock-style ID to its Hub content name. The Hub content
    name is the value of HubContentName in the DescribeHubContent response.

    To find the Hub content name for a model:
        import boto3
        client = boto3.client('sagemaker', region_name='us-west-2')
        response = client.list_hub_contents(
            HubName='SageMakerPublicHub',
            HubContentType='Model',
        )
        for summary in response['HubContentSummaries']:
            print(summary['HubContentName'])

    If a user passes a string that is NOT in this map, it is assumed to already
    be a valid Hub content name and is passed through unchanged.
"""

# Bedrock-style model ID → SageMaker Hub content name
MODEL_NAME_ALIASES = {
    # Nova v2 models
    "amazon.nova-2-lite-v1": "nova-textgeneration-lite-v2",
    # Nova v1 models
    "amazon.nova-lite-v1": "nova-textgeneration-lite",
    "amazon.nova-pro-v1": "nova-textgeneration-pro",
    "amazon.nova-micro-v1": "nova-textgeneration-micro",
}

# SageMaker Hub content name → Bedrock cross-region inference model ID (region-agnostic).
# The geographic prefix (e.g., "us.") is prepended at runtime based on session region.
# Derived from MODEL_NAME_ALIASES — each Bedrock model ID is the alias key + ":0".
# Only includes models that support Bedrock cross-region inference for LLMAJ.
NOVA_BEDROCK_MODEL_IDS = {
    hub_name: f"{bedrock_id}:0"
    for bedrock_id, hub_name in MODEL_NAME_ALIASES.items()
    if "nova" in hub_name.lower()
}


def normalize_model_name(model_name: str) -> str:
    """Normalize a model name to its SageMaker Hub content name.

    If the model name matches a known Bedrock-style ID (e.g. "amazon.nova-lite-v2"),
    returns the corresponding Hub content name (e.g. "nova-textgeneration-lite-v2").

    If the model name is not in the alias map, it is returned unchanged — assumed
    to already be a valid Hub content name.

    Args:
        model_name: User-provided model identifier string.

    Returns:
        Normalized model name suitable for Hub API calls.
    """
    return MODEL_NAME_ALIASES.get(model_name, model_name)
