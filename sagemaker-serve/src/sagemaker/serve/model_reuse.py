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
"""Model source tag-based resource reuse utilities."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MODEL_SOURCE_TAG_KEY = "sagemaker.amazonaws.com/model-source"

_TAG_VALUE_MAX_LENGTH = 256
_TAG_TRUNCATE_PREFIX_LENGTH = 224
_TAG_HASH_SUFFIX_LENGTH = 31

_ACTIVE_STATUSES = {"Active", "InService"}
_CREATING_STATUSES = {"Creating"}
_FAILED_STATUSES = {"Failed"}


def normalize_tag_value(value: str) -> str:
    """Normalize a tag value to fit within the 256-character AWS tag limit.

    If the value is <= 256 chars, returns as-is.
    Otherwise, truncates to 224 chars + "-" + 31 hex chars of SHA-256.
    """
    if len(value) <= _TAG_VALUE_MAX_LENGTH:
        return value
    hash_suffix = hashlib.sha256(value.encode()).hexdigest()[:_TAG_HASH_SUFFIX_LENGTH]
    return f"{value[:_TAG_TRUNCATE_PREFIX_LENGTH]}-{hash_suffix}"


def find_existing_bedrock_model(
    bedrock_client,
    source_id: str,
    poll_interval: int = 30,
    max_wait: int = 900,
) -> Optional[str]:
    """Find an existing Bedrock custom model tagged with a matching source id.

    Enumerates custom models and matches on the
    ``sagemaker.amazonaws.com/model-source`` tag, then validates the model
    status before returning it for reuse.

    Args:
        bedrock_client: A boto3 Bedrock client.
        source_id: Raw source identifier (will be normalized).
        poll_interval: Seconds between status polls for "Creating" resources.
        max_wait: Maximum wait time for "Creating" resources.

    Returns:
        Model ARN if an active/ready model is found, None otherwise.

    Raises:
        TimeoutError: If a creating model doesn't become ready within max_wait.
    """
    tag_value = normalize_tag_value(source_id)
    try:
        resource_arn = _find_bedrock_model_arn_by_tag(bedrock_client, tag_value)
    except Exception as e:
        logger.warning("Could not list Bedrock custom models: %s. Proceeding without.", e)
        return None

    if not resource_arn:
        return None

    return _resolve_ready_arn(
        bedrock_client, resource_arn, check_bedrock_model_status, poll_interval, max_wait
    )


def find_active_bedrock_deployment_for_model(bedrock_client, model_arn: str) -> Optional[str]:
    """Find an existing active custom model deployment for a Bedrock model.

    Args:
        bedrock_client: A boto3 Bedrock client.
        model_arn: ARN of the custom model whose deployment to reuse.

    Returns:
        The ARN of an existing Active deployment on the model, or None.
    """
    try:
        next_token = None
        while True:
            kwargs = {"nextToken": next_token} if next_token else {}
            response = bedrock_client.list_custom_model_deployments(**kwargs)
            for summary in response.get("modelDeploymentSummaries", []):
                if summary.get("modelArn") != model_arn:
                    continue
                if summary.get("status") in _ACTIVE_STATUSES:
                    return summary.get("customModelDeploymentArn")
            next_token = response.get("nextToken")
            if not next_token:
                return None
    except Exception as e:
        logger.warning(
            "Could not list Bedrock custom model deployments: %s. Proceeding without.", e
        )
        return None


def find_existing_sagemaker_endpoint(
    sagemaker_client,
    source_id: str,
    poll_interval: int = 30,
    max_wait: int = 900,
) -> Optional[str]:
    """Find an existing SageMaker endpoint tagged with a matching source id.

    Enumerates endpoints and matches on the
    ``sagemaker.amazonaws.com/model-source`` tag, then validates the endpoint
    status before returning it for reuse.

    Args:
        sagemaker_client: A boto3 SageMaker client.
        source_id: Raw source identifier (will be normalized).
        poll_interval: Seconds between status polls for "Creating" resources.
        max_wait: Maximum wait time for "Creating" resources.

    Returns:
        Endpoint ARN if an in-service/ready endpoint is found, None otherwise.

    Raises:
        TimeoutError: If a creating endpoint doesn't become ready within max_wait.
    """
    tag_value = normalize_tag_value(source_id)
    try:
        resource_arn = _find_sagemaker_endpoint_arn_by_tag(sagemaker_client, tag_value)
    except Exception as e:
        logger.warning("Could not list SageMaker endpoints: %s. Proceeding without.", e)
        return None

    if not resource_arn:
        return None

    return _resolve_ready_arn(
        sagemaker_client, resource_arn, check_sagemaker_endpoint_status, poll_interval, max_wait
    )


def _find_bedrock_model_arn_by_tag(bedrock_client, tag_value: str) -> Optional[str]:
    """Return the ARN of the first Bedrock custom model carrying the source tag."""
    next_token = None
    while True:
        kwargs = {"nextToken": next_token} if next_token else {}
        response = bedrock_client.list_custom_models(**kwargs)
        for summary in response.get("modelSummaries", []):
            arn = summary.get("modelArn")
            if arn and _bedrock_resource_has_tag(bedrock_client, arn, tag_value):
                return arn
        next_token = response.get("nextToken")
        if not next_token:
            return None


def _bedrock_resource_has_tag(bedrock_client, resource_arn: str, tag_value: str) -> bool:
    """Return True if the Bedrock resource carries the source tag with tag_value."""
    tags = bedrock_client.list_tags_for_resource(resourceARN=resource_arn).get("tags", [])
    return any(
        tag.get("key") == MODEL_SOURCE_TAG_KEY and tag.get("value") == tag_value
        for tag in tags
    )


def _find_sagemaker_endpoint_arn_by_tag(sagemaker_client, tag_value: str) -> Optional[str]:
    """Return the ARN of the first SageMaker endpoint carrying the source tag."""
    next_token = None
    while True:
        kwargs = {"NextToken": next_token} if next_token else {}
        response = sagemaker_client.list_endpoints(**kwargs)
        for endpoint in response.get("Endpoints", []):
            arn = endpoint.get("EndpointArn")
            if arn and _sagemaker_resource_has_tag(sagemaker_client, arn, tag_value):
                return arn
        next_token = response.get("NextToken")
        if not next_token:
            return None


def _sagemaker_resource_has_tag(sagemaker_client, resource_arn: str, tag_value: str) -> bool:
    """Return True if the SageMaker resource carries the source tag with tag_value."""
    tags = sagemaker_client.list_tags(ResourceArn=resource_arn).get("Tags", [])
    return any(
        tag.get("Key") == MODEL_SOURCE_TAG_KEY and tag.get("Value") == tag_value
        for tag in tags
    )


def _resolve_ready_arn(
    client,
    resource_arn: str,
    status_checker: Callable,
    poll_interval: int,
    max_wait: int,
) -> Optional[str]:
    """Validate a resource's status and return its ARN only when ready.

    Returns the ARN for active resources, polls creating resources until ready,
    and returns None for failed or unexpected statuses.
    """
    try:
        status = status_checker(client, resource_arn)
    except Exception as e:
        logger.warning("Could not check resource status: %s. Proceeding without.", e)
        return None

    if status in _ACTIVE_STATUSES:
        return resource_arn

    if status in _FAILED_STATUSES:
        logger.warning("Found resource %s in Failed status. Proceeding to create new.", resource_arn)
        return None

    if status in _CREATING_STATUSES:
        return _poll_until_ready(client, resource_arn, status_checker, poll_interval, max_wait)

    logger.warning("Resource %s has unexpected status '%s'. Proceeding to create new.", resource_arn, status)
    return None


def _poll_until_ready(
    client,
    resource_arn: str,
    status_checker: Callable,
    poll_interval: int,
    max_wait: int,
) -> Optional[str]:
    """Poll a resource in Creating status until it becomes ready or times out."""
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        try:
            status = status_checker(client, resource_arn)
        except Exception as e:
            logger.warning("Could not check resource status during poll: %s. Proceeding without.", e)
            return None

        if status in _ACTIVE_STATUSES:
            return resource_arn

        if status in _FAILED_STATUSES:
            logger.warning(
                "Resource %s transitioned to Failed during poll. Proceeding to create new.",
                resource_arn,
            )
            return None

        if status not in _CREATING_STATUSES:
            logger.warning(
                "Resource %s has unexpected status '%s' during poll. Proceeding to create new.",
                resource_arn,
                status,
            )
            return None

    raise TimeoutError(
        f"Resource {resource_arn} did not become ready within {max_wait} seconds."
    )


def build_source_tag(source_id: str) -> dict:
    """Build a tag dict for the model source."""
    return {"key": MODEL_SOURCE_TAG_KEY, "value": normalize_tag_value(source_id)}


def check_bedrock_model_status(bedrock_client, model_arn: str) -> str:
    """Return the status of a Bedrock custom model."""
    try:
        response = bedrock_client.get_custom_model(modelIdentifier=model_arn)
        return response["modelStatus"]
    except Exception as e:
        logger.warning("Could not get Bedrock model status: %s. Proceeding without.", e)
        raise


def check_sagemaker_endpoint_status(sagemaker_client, endpoint_arn: str) -> str:
    """Return the status of a SageMaker endpoint."""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=_arn_to_name(endpoint_arn))
        return response["EndpointStatus"]
    except Exception as e:
        logger.warning("Could not get endpoint status: %s. Proceeding without.", e)
        raise


def _arn_to_name(arn: str) -> str:
    """Extract the resource name from an ARN (last segment after '/')."""
    return arn.rsplit("/", 1)[-1]
