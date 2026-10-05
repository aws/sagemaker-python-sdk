from __future__ import absolute_import
from typing import Optional, List, Dict
from .constants import (
    JOB_ARN,
    JOB_ID,
    FIRST_LIST_SERVICE_JOB_RESP,
    EMPTY_LIST_SERVICE_JOB_RESP,
    JOB_STATUS_RUNNING,
    TIMEOUT_CONFIG,
)


class MockClient:
    def submit_service_job(
        self,
        jobName,
        jobQueue,
        serviceRequestPayload,
        serviceJobType,
        retryStrategy: Optional[Dict] = None,
        schedulingPriority: Optional[int] = None,
        shareIdentifier: Optional[str] = "",
        tags: Optional[Dict] = None,
        timeoutConfig: Optional[Dict] = TIMEOUT_CONFIG,
    ):
        return {"jobArn": JOB_ARN, "jobName": jobName, "jobId": JOB_ID}

    def describe_service_job(self, jobId):
        return {"jobId": jobId}

    def terminate_service_job(self, jobId, reason):
        return {}

    def list_service_jobs(
        self,
        jobQueue,
        jobStatus: Optional[str] = JOB_STATUS_RUNNING,
        nextToken: Optional[str] = "",
        filters: Optional[List] = [],
    ):
        if nextToken:
            return FIRST_LIST_SERVICE_JOB_RESP
        else:
            return EMPTY_LIST_SERVICE_JOB_RESP
