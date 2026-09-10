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

from botocore.exceptions import ClientError

import boto3

logger = logging.getLogger(__name__)

MODEL_SOURCE_TAG_KEY = "sagemaker.amazonaws.com/model-source"

_TAG_VALUE_MAX_LENGTH = 256
_TAG_TRUNCATE_PREFIX_LENGTH = 224
_TAG_HASH_SUFFIX_LENGTH = 31

_ACTIVE_STATUSES = {"Active", "InService"}
_CREATING_STATUSES = {"Creating"}
_FAILED_STATUSES = {"Failed"}
_ACCESS_DENIED_CODE = "AccessDeniedException"


def normalize_tag_value(value: str) -> str:
    """Normalize a tag value to fit within the 256-character AWS tag limit.

    If the value is <= 256 chars, returns as-is.
    Otherwise, truncates to 224 chars + "-" + 31 hex chars of SHA-256.
    """
    if len(value) <= _TAG_VALUE_MAX_LENGTH:
        return value
    hash_suffix = hashlib.sha256(value.encode()).hexdigest()[:_TAG_HASH_SUFFIX_LENGTH]
    return f"{value[:_TAG_TRUNCATE_PREFIX_LENGTH]}-{hash_suffix}"


def _reraise_if_access_denied(error: ClientError, permission: str) -> None:
    """Surface a denied reuse-discovery call as an actionable PermissionError.

    Called only from ``reuse_resources=True`` discovery paths. A denied
    discovery call otherwise looks the same as "nothing to reuse", so reuse
    falls through to creating the resource. But the resource created on a prior
    run is still there under the same deterministic name, so the create fails
    with a confusing ``ValidationException: ... already exists`` that never
    mentions the real cause. Re-raising here names the missing IAM permission
    instead.

    Args:
        error: The ClientError raised by a discovery call.
        permission: The IAM action reuse discovery requires (named in the message).

    Raises:
        PermissionError: If ``error`` is an ``AccessDeniedException``.
    """
    if error.response.get("Error", {}).get("Code") == _ACCESS_DENIED_CODE:
        raise PermissionError(
            f"reuse_resources=True requires the '{permission}' permission to discover "
            f"reusable resources. Add it to the execution role's policy, or set "
            f"reuse_resources=False to always create new resources."
        ) from error


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
    except ClientError as e:
        _reraise_if_access_denied(e, "bedrock:ListTagsForResource")
        logger.warning("Could not list Bedrock custom models: %s. Proceeding without.", e)
        return None
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
    except ClientError as e:
        _reraise_if_access_denied(e, "bedrock:ListCustomModelDeployments")
        logger.warning(
            "Could not list Bedrock custom model deployments: %s. Proceeding without.", e
        )
        return None
    except Exception as e:
        logger.warning(
            "Could not list Bedrock custom model deployments: %s. Proceeding without.", e
        )
        return None


def find_existing_imported_model(
    bedrock_client,
    source_id: str,
) -> Optional[str]:
    """Find an existing completed Bedrock imported model matching a source id.

    Enumerates imported models (via ``list_imported_models``) and matches on the
    ``sagemaker.amazonaws.com/model-source`` tag.

    Args:
        bedrock_client: A boto3 Bedrock client.
        source_id: Raw source identifier (will be normalized).

    Returns:
        The imported-model ARN (``.../imported-model/...``) if a match is found,
        None otherwise.
    """
    tag_value = normalize_tag_value(source_id)

    try:
        resource_arn = _find_imported_model_arn_by_tag(bedrock_client, tag_value)
    except ClientError as e:
        _reraise_if_access_denied(e, "bedrock:ListTagsForResource")
        logger.warning("Could not list Bedrock imported models: %s. Proceeding without.", e)
        return None
    except Exception as e:
        logger.warning("Could not list Bedrock imported models: %s. Proceeding without.", e)
        return None

    return resource_arn


def find_existing_model_import_job(
    bedrock_client,
    source_id: str,
) -> Optional[str]:
    """Find an in-progress Bedrock model import job matching a source id.

    Enumerates in-progress import jobs (via ``list_model_import_jobs``) and
    matches on the ``sagemaker.amazonaws.com/model-source`` tag. Use this when
    ``find_existing_imported_model`` returns None to detect an import that is
    already running for the same source.

    Args:
        bedrock_client: A boto3 Bedrock client.
        source_id: Raw source identifier (will be normalized).

    Returns:
        The import-job ARN (``.../model-import-job/...``) if a matching
        in-progress job is found, None otherwise.
    """
    tag_value = normalize_tag_value(source_id)

    try:
        job_arn = _find_in_progress_import_job_by_tag(bedrock_client, tag_value)
    except ClientError as e:
        _reraise_if_access_denied(e, "bedrock:ListTagsForResource")
        logger.warning("Could not list Bedrock import jobs: %s. Proceeding without.", e)
        return None
    except Exception as e:
        logger.warning("Could not list Bedrock import jobs: %s. Proceeding without.", e)
        return None

    if job_arn:
        logger.info("Found in-progress import job %s with matching model-source tag.", job_arn)

    return job_arn


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
    except ClientError as e:
        _reraise_if_access_denied(e, "sagemaker:ListTags")
        logger.warning("Could not list SageMaker endpoints: %s. Proceeding without.", e)
        return None
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


def _find_imported_model_arn_by_tag(bedrock_client, tag_value: str) -> Optional[str]:
    """Return the ARN of the first Bedrock imported model carrying the source tag."""
    next_token = None
    while True:
        kwargs = {"nextToken": next_token} if next_token else {}
        response = bedrock_client.list_imported_models(**kwargs)
        for summary in response.get("modelSummaries", []):
            arn = summary.get("modelArn")
            if arn and _bedrock_resource_has_tag(bedrock_client, arn, tag_value):
                return arn
        next_token = response.get("nextToken")
        if not next_token:
            return None


# The Bedrock ListModelImportJobs API only accepts the enum values
# {Completed, InProgress, Failed} for statusEquals.
_IMPORT_JOB_IN_PROGRESS_STATUSES = {"InProgress"}

def _find_in_progress_import_job_by_tag(bedrock_client, tag_value: str) -> Optional[str]:
    """Return the job ARN of an in-progress import job carrying the source tag.

    Searches jobs in the InProgress state.
    """
    for status_filter in _IMPORT_JOB_IN_PROGRESS_STATUSES:
        next_token = None
        while True:
            kwargs = {"statusEquals": status_filter}
            if next_token:
                kwargs["nextToken"] = next_token
            response = bedrock_client.list_model_import_jobs(**kwargs)
            for summary in response.get("modelImportJobSummaries", []):
                job_arn = summary.get("jobArn")
                if job_arn and _bedrock_resource_has_tag(bedrock_client, job_arn, tag_value):
                    return job_arn
            next_token = response.get("nextToken")
            if not next_token:
                break
    return None


def _bedrock_resource_has_tag(bedrock_client, resource_arn: str, tag_value: str) -> bool:
    """Return True if the Bedrock resource carries the source tag with tag_value."""
    tags = bedrock_client.list_tags_for_resource(resourceARN=resource_arn).get("tags", [])
    return any(
        tag.get("key") == MODEL_SOURCE_TAG_KEY and tag.get("value") == tag_value
        for tag in tags
    )


def _find_resource_arn_by_tagging_api(
    sagemaker_client, tag_value: str, resource_type: str
) -> Optional[str]:
    """Use Resource Groups Tagging API to find a resource by tag (fast path).

    Makes a single server-side filtered query instead of iterating through all
    resources and calling list_tags on each one.

    Args:
        sagemaker_client: A boto3 SageMaker client (used for region detection
            and as fallback for creating the tagging client).
        tag_value: The normalized tag value to search for.
        resource_type: The resource type filter (e.g. "sagemaker:model",
            "sagemaker:endpoint").

    Returns:
        The resource ARN if found, empty string "" if no match (signals fast path
        completed successfully with no results), or None if the tagging API is
        unavailable (signals caller should fall back to the slow scan).
    """
    try:
        region = sagemaker_client.meta.region_name
        if not region:
            return None

        tagging_client = boto3.client("resourcegroupstaggingapi", region_name=region)

        pagination_token = ""
        while True:
            kwargs = {
                "TagFilters": [
                    {"Key": MODEL_SOURCE_TAG_KEY, "Values": [tag_value]}
                ],
                "ResourceTypeFilters": [resource_type],
            }
            if pagination_token:
                kwargs["PaginationToken"] = pagination_token

            response = tagging_client.get_resources(**kwargs)
            for mapping in response.get("ResourceTagMappingList", []):
                arn = mapping.get("ResourceARN")
                if arn:
                    return arn

            pagination_token = response.get("PaginationToken", "")
            if not pagination_token:
                return ""  # Fast path succeeded, no matching resource found

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == _ACCESS_DENIED_CODE:
            logger.debug(
                "Resource Groups Tagging API access denied (tag:GetResources). "
                "Falling back to paginated list+list_tags scan."
            )
            return None  # Signal caller to use fallback
        logger.debug("Resource Groups Tagging API call failed: %s. Using fallback.", e)
        return None
    except Exception as e:
        logger.debug("Resource Groups Tagging API unavailable: %s. Using fallback.", e)
        return None


def find_sagemaker_model_arn_by_tag(sagemaker_client, tag_value: str) -> Optional[str]:
    """Return the ARN of the first SageMaker Model carrying the source tag.

    Uses the Resource Groups Tagging API for efficient server-side filtering
    when available, falling back to paginated list+list_tags if denied.

    Args:
        sagemaker_client: A boto3 SageMaker client.
        tag_value: The normalized tag value to match.

    Returns:
        Model ARN if found, None otherwise.
    """
    # Try the fast path first: Resource Groups Tagging API
    arn = _find_resource_arn_by_tagging_api(
        sagemaker_client, tag_value, resource_type="sagemaker:model"
    )
    if arn is not None:
        return arn if arn != "" else None

    # Fallback: paginate through all models and check tags individually
    next_token = None
    while True:
        kwargs = {"SortBy": "CreationTime", "SortOrder": "Descending"}
        if next_token:
            kwargs["NextToken"] = next_token
        response = sagemaker_client.list_models(**kwargs)
        for model_summary in response.get("Models", []):
            model_arn = model_summary.get("ModelArn")
            if model_arn and _sagemaker_resource_has_tag(sagemaker_client, model_arn, tag_value):
                return model_arn
        next_token = response.get("NextToken")
        if not next_token:
            return None


def _find_sagemaker_endpoint_arn_by_tag(sagemaker_client, tag_value: str) -> Optional[str]:
    """Return the ARN of the first SageMaker endpoint carrying the source tag.

    Uses the Resource Groups Tagging API for efficient server-side filtering
    when available, falling back to paginated list+list_tags if denied.
    """
    # Try the fast path first: Resource Groups Tagging API
    arn = _find_resource_arn_by_tagging_api(
        sagemaker_client, tag_value, resource_type="sagemaker:endpoint"
    )
    if arn is not None:
        return arn if arn != "" else None

    # Fallback: paginate through all endpoints and check tags individually
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
        logger.info(
            "Existing resource %s is still Creating; polling every %ds up to %ds "
            "before it can be reused",
            resource_arn,
            poll_interval,
            max_wait,
        )
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

        logger.info(
            "Polling resource %s: status='%s' (%ds/%ds elapsed)",
            resource_arn,
            status,
            elapsed,
            max_wait,
        )

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
