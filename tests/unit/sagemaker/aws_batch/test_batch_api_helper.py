from __future__ import absolute_import
from sagemaker.aws_batch.batch_api_helper import (
    submit_service_job,
    terminate_service_job,
    describe_service_job,
    list_service_job,
    __merge_tags,
)

import json
import pytest
from mock.mock import patch

from sagemaker.aws_batch.constants import (
    DEFAULT_TIMEOUT,
    DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
    SAGEMAKER_TRAINING,
)
from .mock_client import MockClient
from .constants import (
    JOB_NAME,
    JOB_QUEUE,
    SCHEDULING_PRIORITY,
    JOB_ID,
    REASON,
    SHARE_IDENTIFIER,
    BATCH_TAGS,
    TRAINING_TAGS,
    TRAINING_TAGS_DUPLICATING_BATCH_TAGS,
    TRAINING_TAGS_CONVERTED_TO_BATCH_TAGS,
    MERGED_TAGS,
    MERGED_TAGS_TRAINING_OVERRIDE,
    JOB_STATUS_RUNNING,
    NEXT_TOKEN,
)


@patch("sagemaker.aws_batch.batch_api_helper.get_batch_boto_client")
def test_submit_service_job(patched_get_batch_boto_client):
    patched_get_batch_boto_client.return_value = MockClient()
    training_payload = {}
    resp = submit_service_job(
        training_payload,
        JOB_NAME,
        JOB_QUEUE,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        DEFAULT_TIMEOUT,
        SHARE_IDENTIFIER,
        BATCH_TAGS,
    )
    assert resp["jobName"] == JOB_NAME
    assert "jobArn" in resp
    assert "jobId" in resp


@patch("sagemaker.aws_batch.batch_api_helper.get_batch_boto_client")
@patch("sagemaker.aws_batch.batch_api_helper.__merge_tags")
@pytest.mark.parametrize(
    "batch_tags,training_tags",
    [
        (BATCH_TAGS, TRAINING_TAGS),
        (None, TRAINING_TAGS),
        ({}, TRAINING_TAGS),
        (BATCH_TAGS, None),
        (BATCH_TAGS, []),
    ],
)
def test_submit_service_job_called_with_merged_tags(
    patched_merge_tags, patched_get_batch_boto_client, batch_tags, training_tags
):
    mock_client = MockClient()
    patched_get_batch_boto_client.return_value = mock_client
    patched_merge_tags.return_value = MERGED_TAGS

    with patch.object(
        mock_client, "submit_service_job", wraps=mock_client.submit_service_job
    ) as wrapped_submit_service_job:
        training_payload = {"Tags": training_tags}
        resp = submit_service_job(
            training_payload,
            JOB_NAME,
            JOB_QUEUE,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            DEFAULT_TIMEOUT,
            SHARE_IDENTIFIER,
            batch_tags,
        )
        assert resp["jobName"] == JOB_NAME
        assert "jobArn" in resp
        assert "jobId" in resp
        patched_merge_tags.assert_called_once_with(batch_tags, training_tags)
        wrapped_submit_service_job.assert_called_once_with(
            jobName=JOB_NAME,
            jobQueue=JOB_QUEUE,
            retryStrategy=DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            serviceJobType=SAGEMAKER_TRAINING,
            serviceRequestPayload=json.dumps(training_payload),
            timeoutConfig=DEFAULT_TIMEOUT,
            schedulingPriority=SCHEDULING_PRIORITY,
            shareIdentifier=SHARE_IDENTIFIER,
            tags={**MERGED_TAGS},
        )


@patch("sagemaker.aws_batch.batch_api_helper.get_batch_boto_client")
@patch("sagemaker.aws_batch.batch_api_helper.__merge_tags")
def test_submit_service_job_not_called_with_tags(patched_merge_tags, patched_get_batch_boto_client):
    mock_client = MockClient()
    patched_get_batch_boto_client.return_value = mock_client
    patched_merge_tags.return_value = MERGED_TAGS

    with patch.object(
        mock_client, "submit_service_job", wraps=mock_client.submit_service_job
    ) as wrapped_submit_service_job:
        training_payload = {}
        resp = submit_service_job(
            training_payload,
            JOB_NAME,
            JOB_QUEUE,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            DEFAULT_TIMEOUT,
            SHARE_IDENTIFIER,
        )
        assert resp["jobName"] == JOB_NAME
        assert "jobArn" in resp
        assert "jobId" in resp
        patched_merge_tags.assert_not_called()
        wrapped_submit_service_job.assert_called_once_with(
            jobName=JOB_NAME,
            jobQueue=JOB_QUEUE,
            retryStrategy=DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            serviceJobType=SAGEMAKER_TRAINING,
            serviceRequestPayload=json.dumps(training_payload),
            timeoutConfig=DEFAULT_TIMEOUT,
            schedulingPriority=SCHEDULING_PRIORITY,
            shareIdentifier=SHARE_IDENTIFIER,
        )


@patch("sagemaker.aws_batch.batch_api_helper.get_batch_boto_client")
def test_describe_service_job(patched_get_batch_boto_client):
    patched_get_batch_boto_client.return_value = MockClient()
    resp = describe_service_job(job_id=JOB_ID)
    assert resp["jobId"] == JOB_ID


@patch("sagemaker.aws_batch.batch_api_helper.get_batch_boto_client")
def test_terminate_service_job(patched_get_batch_boto_client):
    patched_get_batch_boto_client.return_value = MockClient()
    resp = terminate_service_job(job_id=JOB_ID, reason=REASON)
    assert len(resp) == 0


@patch("sagemaker.aws_batch.batch_api_helper.get_batch_boto_client")
def test_list_service_job_has_next_token(patched_get_batch_boto_client):
    patched_get_batch_boto_client.return_value = MockClient()
    gen = list_service_job(job_queue=None, job_status=JOB_STATUS_RUNNING, next_token=NEXT_TOKEN)
    resp = next(gen)
    assert resp["nextToken"] == NEXT_TOKEN


@patch("sagemaker.aws_batch.batch_api_helper.get_batch_boto_client")
def test_list_service_job_no_next_token(patched_get_batch_boto_client):
    patched_get_batch_boto_client.return_value = MockClient()
    gen = list_service_job(job_queue=None, job_status=JOB_STATUS_RUNNING, next_token=None)
    resp = next(gen)
    assert resp["nextToken"] is None


@pytest.mark.parametrize(
    "batch_tags,training_tags,expected",
    [
        (BATCH_TAGS, TRAINING_TAGS, MERGED_TAGS),
        (BATCH_TAGS, TRAINING_TAGS_DUPLICATING_BATCH_TAGS, MERGED_TAGS_TRAINING_OVERRIDE),
        (BATCH_TAGS, None, BATCH_TAGS),
        (BATCH_TAGS, [], BATCH_TAGS),
        (None, TRAINING_TAGS, TRAINING_TAGS_CONVERTED_TO_BATCH_TAGS),
        ({}, TRAINING_TAGS, TRAINING_TAGS_CONVERTED_TO_BATCH_TAGS),
    ],
)
def test___merge_tags(batch_tags, training_tags, expected):
    result = __merge_tags(batch_tags=batch_tags, training_tags=training_tags)
    assert result == expected
