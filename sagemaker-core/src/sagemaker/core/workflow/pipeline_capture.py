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
"""Shared capture paths for producers running under a ``PipelineSession``."""
from __future__ import absolute_import

from typing import Any, Dict, List, Optional

from sagemaker.core.apiutils._boto_functions import to_pascal_case
from sagemaker.core.shapes import Tag
from sagemaker.core.utils.utils import serialize


def capture_create_job_request(
    pipeline_session,
    *,
    job_name: str,
    role_arn: str,
    job_category: str,
    schema_version: str,
    job_config: Dict[str, Any],
    tags: Optional[List[Any]] = None,
    customer_details: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture a ``CreateJob`` request for ``JobStep`` composition instead of submitting it.

    ``JobName`` stays in the request: ``JobStep``'s custom-job-prefix handling
    requires the key to be present. ``job_config`` is the raw config dict --
    ``JobStep`` scopes ``OutputDataConfig.S3OutputPath`` per execution and encodes
    the document at definition time.

    Args:
        pipeline_session (PipelineSession): The capturing session.
        job_name (str): Client-minted name; replaced by the service at execution.
        role_arn (str): The execution role for the job.
        job_category (str): ``CreateJob`` job category.
        schema_version (str): ``JobConfigDocument`` schema version.
        job_config (Dict[str, Any]): The raw job configuration dict.
        tags (Optional[List[Any]]): Tags in the caller's wire form; ``serialize``
            emits ``Tag`` shapes as ``{"Key", "Value"}``.
        customer_details (Optional[Dict[str, Any]]): ``CustomerDetails`` envelope
            member for producers whose direct path sends it (data preparation).
            Session-derived, so it is stable across executions.
    """
    request: Dict[str, Any] = {
        "JobName": job_name,
        "RoleArn": role_arn,
        "JobCategory": job_category,
        "JobConfigSchemaVersion": schema_version,
        "JobConfigDocument": job_config,
    }
    if customer_details is not None:
        request["CustomerDetails"] = customer_details
    if tags is not None:
        request["Tags"] = tags
    pipeline_session._intercept_create_request(serialize(request), None, "create_job")


def capture_training_request(pipeline_session, create_args: Dict[str, Any]) -> None:
    """Capture a ``CreateTrainingJob`` request for ``TrainingStep`` composition.

    ``create_args`` are ``TrainingJob.create`` keyword arguments with data channels
    already resolved, so datasets must be concrete at authoring time; only the job
    name is deferred to execution. Client-resolution members (``session``,
    ``region``) are dropped, and dict-form tags are coerced through the ``Tag``
    model so ``serialize`` emits the wire form ``TrainingJob.create`` produces.

    Args:
        pipeline_session (PipelineSession): The capturing session.
        create_args (Dict[str, Any]): ``TrainingJob.create`` keyword arguments.
    """
    pipeline_args = {k: v for k, v in create_args.items() if k not in ("session", "region")}
    pipeline_args.pop("training_job_name", None)
    request = {to_pascal_case(k): v for k, v in pipeline_args.items()}
    if request.get("Tags"):
        # Dict-form tags arrive in both shapes: lowercase keys from the SDK's own
        # tag builders, PascalCase from callers passing wire-form dicts through.
        request["Tags"] = [
            Tag(key=tag.get("key", tag.get("Key")), value=tag.get("value", tag.get("Value")))
            if isinstance(tag, dict)
            else tag
            for tag in request["Tags"]
        ]
    pipeline_session._intercept_create_request(serialize(request), None, "train")
