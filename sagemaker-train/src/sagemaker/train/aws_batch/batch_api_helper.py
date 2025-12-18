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
"""The module provides helper function for Batch Submit/Describe/Terminal job APIs."""
from __future__ import absolute_import

import json
from typing import List, Dict, Optional
from sagemaker.train.aws_batch.constants import (
    SAGEMAKER_TRAINING,
    DEFAULT_TIMEOUT,
    DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
)
from sagemaker.train.aws_batch.boto_client import get_batch_boto_client


def _submit_service_job(
    training_payload: Dict,
    job_name: str,
    job_queue: str,
    retry_config: Optional[Dict] = None,
    scheduling_priority: Optional[int] = None,
    timeout: Optional[Dict] = None,
    share_identifier: Optional[str] = None,
    tags: Optional[Dict] = None,
) -> Dict:
    """Batch submit_service_job API helper function.

    Args:
        training_payload: a dict containing a dict of arguments for Training job.
        job_name: Batch job name.
        job_queue: Batch job queue ARN.
        retry_config: Batch job retry configuration.
        scheduling_priority: An integer representing scheduling priority.
        timeout: Set with value of timeout if specified, else default to 1 day.
        share_identifier: value of shareIdentifier if specified.
        tags: A dict of string to string representing Batch tags.

    Returns:
        A dict containing jobArn, jobName and jobId.
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    client = get_batch_boto_client()
    training_payload_tags = training_payload.pop("Tags", None)
    payload = {
        "jobName": job_name,
        "jobQueue": job_queue,
        "retryStrategy": DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        "serviceJobType": SAGEMAKER_TRAINING,
        "serviceRequestPayload": json.dumps(training_payload),
        "timeoutConfig": timeout,
    }
    if retry_config:
        payload["retryStrategy"] = retry_config
    if scheduling_priority:
        payload["schedulingPriority"] = scheduling_priority
    if share_identifier:
        payload["shareIdentifier"] = share_identifier
    if tags or training_payload_tags:
        payload["tags"] = __merge_tags(tags, training_payload_tags)
    return client.submit_service_job(**payload)


def _describe_service_job(job_id: str) -> Dict:
    """Batch describe_service_job API helper function.

    Args:
        job_id: Job ID used.

    Returns: a dict. See the sample below
    {
        'attempts': [
            {
                'serviceResourceId': {
                    'name': 'string',
                    'value': 'string'
                },
                'startedAt': 123,
                'stoppedAt': 123,
                'statusReason': 'string'
            },
        ],
        'createdAt': 123,
        'isTerminated': True|False,
        'jobArn': 'string',
        'jobId': 'string',
        'jobName': 'string',
        'jobQueue': 'string',
        'retryStrategy': {
            'attempts': 123
        },
        'schedulingPriority': 123,
        'serviceRequestPayload': 'string',
        'serviceJobType': 'EKS'|'ECS'|'ECS_FARGATE'|'SAGEMAKER_TRAINING',
        'shareIdentifier': 'string',
        'startedAt': 123,
        'status': 'SUBMITTED'|'PENDING'|'RUNNABLE'|'STARTING'|'RUNNING'|'SUCCEEDED'|'FAILED',
        'statusReason': 'string',
        'stoppedAt': 123,
        'tags': {
            'string': 'string'
        },
        'timeout': {
            'attemptDurationSeconds': 123
        }
    }
    """
    client = get_batch_boto_client()
    return client.describe_service_job(jobId=job_id)


def _terminate_service_job(job_id: str, reason: Optional[str] = "default terminate reason") -> Dict:
    """Batch terminate_service_job API helper function.

    Args:
        job_id: Job ID
        reason: A string representing terminate reason.

    Returns: an empty dict
    """
    client = get_batch_boto_client()
    return client.terminate_service_job(jobId=job_id, reason=reason)


def _list_service_job(
    job_queue: str,
    job_status: Optional[str] = None,
    filters: Optional[List] = None,
    next_token: Optional[str] = None,
) -> Dict:
    """Batch list_service_job API helper function.

    Args:
        job_queue: Batch job queue ARN.
        job_status: Batch job status.
        filters: A list of Dict. Each contains a filter.
        next_token: Used to retrieve data in next page.

    Returns: A generator containing list results.

    """
    client = get_batch_boto_client()
    payload = {"jobQueue": job_queue}
    if filters:
        payload["filters"] = filters
    if next_token:
        payload["nextToken"] = next_token
    if job_status:
        payload["jobStatus"] = job_status
    part_of_jobs = client.list_service_jobs(**payload)
    next_token = part_of_jobs.get("nextToken")
    yield part_of_jobs
    if next_token:
        yield from _list_service_job(job_queue, job_status, filters, next_token)


def __merge_tags(batch_tags: Optional[Dict], training_tags: Optional[List]) -> Optional[Dict]:
    """Merges Batch and training payload tags.

    Returns a copy of Batch tags merged with training payload tags.  Training payload tags take
    precedence in the case of key conflicts.

    :param batch_tags: A dict of string to string representing Batch tags.
    :param training_tags: A list of `{"Key": "string", "Value": "string"}` objects representing
    training payload tags.
    :return: A dict of string to string representing batch tags merged with training tags.
    batch_tags is returned unaltered if training_tags is None or empty.
    """
    if not training_tags:
        return batch_tags

    training_tags_to_merge = {tag["Key"]: tag["Value"] for tag in training_tags}
    batch_tags_copy = batch_tags.copy() if batch_tags else {}
    batch_tags_copy.update(training_tags_to_merge)

    return batch_tags_copy
