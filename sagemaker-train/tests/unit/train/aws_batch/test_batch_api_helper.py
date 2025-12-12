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
"""Unit tests for batch_api_helper module"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from sagemaker.train.aws_batch.batch_api_helper import (
    submit_service_job,
    describe_service_job,
    terminate_service_job,
    list_service_job,
)
from .conftest import (
    JOB_NAME,
    JOB_QUEUE,
    JOB_ID,
    REASON,
    BATCH_TAGS,
    TRAINING_TAGS,
    TRAINING_TAGS_CONVERTED,
    MERGED_TAGS,
    DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
    TIMEOUT_CONFIG,
    SCHEDULING_PRIORITY,
    SHARE_IDENTIFIER,
    SUBMIT_SERVICE_JOB_RESP,
    DESCRIBE_SERVICE_JOB_RESP_RUNNING,
    LIST_SERVICE_JOB_RESP_EMPTY,
    LIST_SERVICE_JOB_RESP_WITH_JOBS,
    LIST_SERVICE_JOB_RESP_WITH_NEXT_TOKEN,
    TRAINING_JOB_PAYLOAD,
    NEXT_TOKEN,
    JOB_STATUS_RUNNING,
)


class TestSubmitServiceJob:
    """Tests for submit_service_job function"""

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_submit_service_job_basic(self, mock_get_client):
        """Test basic submit_service_job call"""
        mock_client = Mock()
        mock_client.submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP
        mock_get_client.return_value = mock_client

        result = submit_service_job(
            TRAINING_JOB_PAYLOAD,
            JOB_NAME,
            JOB_QUEUE,
        )

        assert result["jobArn"] == SUBMIT_SERVICE_JOB_RESP["jobArn"]
        assert result["jobName"] == SUBMIT_SERVICE_JOB_RESP["jobName"]
        assert result["jobId"] == SUBMIT_SERVICE_JOB_RESP["jobId"]
        mock_client.submit_service_job.assert_called_once()

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_submit_service_job_with_all_params(self, mock_get_client):
        """Test submit_service_job with all optional parameters"""
        mock_client = Mock()
        mock_client.submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP
        mock_get_client.return_value = mock_client

        result = submit_service_job(
            TRAINING_JOB_PAYLOAD,
            JOB_NAME,
            JOB_QUEUE,
            retry_config=DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            scheduling_priority=SCHEDULING_PRIORITY,
            timeout=TIMEOUT_CONFIG,
            share_identifier=SHARE_IDENTIFIER,
            tags=BATCH_TAGS,
        )

        assert result["jobArn"] == SUBMIT_SERVICE_JOB_RESP["jobArn"]
        call_kwargs = mock_client.submit_service_job.call_args[1]
        assert call_kwargs["jobName"] == JOB_NAME
        assert call_kwargs["jobQueue"] == JOB_QUEUE
        assert call_kwargs["schedulingPriority"] == SCHEDULING_PRIORITY
        assert call_kwargs["shareIdentifier"] == SHARE_IDENTIFIER
        assert call_kwargs["timeoutConfig"] == TIMEOUT_CONFIG

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_submit_service_job_with_tags(self, mock_get_client):
        """Test submit_service_job merges batch and training tags"""
        mock_client = Mock()
        mock_client.submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP
        mock_get_client.return_value = mock_client

        payload = TRAINING_JOB_PAYLOAD.copy()
        payload["Tags"] = TRAINING_TAGS

        result = submit_service_job(
            payload,
            JOB_NAME,
            JOB_QUEUE,
            tags=BATCH_TAGS,
        )

        assert result["jobArn"] == SUBMIT_SERVICE_JOB_RESP["jobArn"]
        call_kwargs = mock_client.submit_service_job.call_args[1]
        assert "tags" in call_kwargs
        # Verify tags were merged
        merged = call_kwargs["tags"]
        assert merged["batch-key"] == "batch-value"
        assert merged["training-key"] == "training-value"

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_submit_service_job_payload_serialized(self, mock_get_client):
        """Test that training payload is JSON serialized"""
        mock_client = Mock()
        mock_client.submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP
        mock_get_client.return_value = mock_client

        submit_service_job(
            TRAINING_JOB_PAYLOAD,
            JOB_NAME,
            JOB_QUEUE,
        )

        call_kwargs = mock_client.submit_service_job.call_args[1]
        payload_str = call_kwargs["serviceRequestPayload"]
        # Verify it's a JSON string
        parsed = json.loads(payload_str)
        assert parsed["TrainingJobName"] == TRAINING_JOB_PAYLOAD["TrainingJobName"]


class TestDescribeServiceJob:
    """Tests for describe_service_job function"""

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_describe_service_job(self, mock_get_client):
        """Test describe_service_job returns job details"""
        mock_client = Mock()
        mock_client.describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_RUNNING
        mock_get_client.return_value = mock_client

        result = describe_service_job(JOB_ID)

        assert result["jobId"] == JOB_ID
        assert result["status"] == "RUNNING"
        mock_client.describe_service_job.assert_called_once_with(jobId=JOB_ID)


class TestTerminateServiceJob:
    """Tests for terminate_service_job function"""

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_terminate_service_job(self, mock_get_client):
        """Test terminate_service_job calls terminate API"""
        mock_client = Mock()
        mock_client.terminate_service_job.return_value = {}
        mock_get_client.return_value = mock_client

        result = terminate_service_job(JOB_ID, REASON)

        assert result == {}
        mock_client.terminate_service_job.assert_called_once_with(
            jobId=JOB_ID, reason=REASON
        )

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_terminate_service_job_default_reason(self, mock_get_client):
        """Test terminate_service_job with default reason"""
        mock_client = Mock()
        mock_client.terminate_service_job.return_value = {}
        mock_get_client.return_value = mock_client

        terminate_service_job(JOB_ID)

        call_kwargs = mock_client.terminate_service_job.call_args[1]
        assert call_kwargs["jobId"] == JOB_ID
        assert "reason" in call_kwargs


class TestListServiceJob:
    """Tests for list_service_job function"""

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_list_service_job_empty(self, mock_get_client):
        """Test list_service_job with no jobs"""
        mock_client = Mock()
        mock_client.list_service_jobs.return_value = LIST_SERVICE_JOB_RESP_EMPTY
        mock_get_client.return_value = mock_client

        gen = list_service_job(JOB_QUEUE)
        result = next(gen)

        assert result["jobSummaryList"] == []
        assert result["nextToken"] is None

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_list_service_job_with_jobs(self, mock_get_client):
        """Test list_service_job returns jobs"""
        mock_client = Mock()
        mock_client.list_service_jobs.return_value = LIST_SERVICE_JOB_RESP_WITH_JOBS
        mock_get_client.return_value = mock_client

        gen = list_service_job(JOB_QUEUE)
        result = next(gen)

        assert len(result["jobSummaryList"]) == 2
        assert result["jobSummaryList"][0]["jobName"] == JOB_NAME

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_list_service_job_with_pagination(self, mock_get_client):
        """Test list_service_job handles pagination"""
        mock_client = Mock()
        mock_client.list_service_jobs.side_effect = [
            LIST_SERVICE_JOB_RESP_WITH_NEXT_TOKEN,
            LIST_SERVICE_JOB_RESP_EMPTY,
        ]
        mock_get_client.return_value = mock_client

        gen = list_service_job(JOB_QUEUE)
        first_result = next(gen)
        assert first_result["nextToken"] == NEXT_TOKEN

        second_result = next(gen)
        assert second_result["jobSummaryList"] == []

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_list_service_job_with_filters(self, mock_get_client):
        """Test list_service_job with filters"""
        mock_client = Mock()
        mock_client.list_service_jobs.return_value = LIST_SERVICE_JOB_RESP_WITH_JOBS
        mock_get_client.return_value = mock_client

        filters = [{"name": "JOB_NAME", "values": [JOB_NAME]}]
        gen = list_service_job(JOB_QUEUE, filters=filters)
        result = next(gen)

        call_kwargs = mock_client.list_service_jobs.call_args[1]
        assert call_kwargs["filters"] == filters

    @patch("sagemaker.train.aws_batch.batch_api_helper.get_batch_boto_client")
    def test_list_service_job_with_status(self, mock_get_client):
        """Test list_service_job with job status filter"""
        mock_client = Mock()
        mock_client.list_service_jobs.return_value = LIST_SERVICE_JOB_RESP_WITH_JOBS
        mock_get_client.return_value = mock_client

        gen = list_service_job(JOB_QUEUE, job_status=JOB_STATUS_RUNNING)
        result = next(gen)

        call_kwargs = mock_client.list_service_jobs.call_args[1]
        assert call_kwargs["jobStatus"] == JOB_STATUS_RUNNING
