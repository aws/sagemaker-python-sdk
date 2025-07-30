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
"""Define QueuedJob class for AWS Batch service"""
from __future__ import absolute_import

import logging
import time
import asyncio
from typing import Optional, Dict
import nest_asyncio
from sagemaker.estimator import Estimator
from .batch_api_helper import terminate_service_job, describe_service_job
from .exception import NoTrainingJob, MissingRequiredArgument
from ..utils import get_training_job_name_from_training_job_arn
from .constants import JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, POLL_IN_SECONDS

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


class TrainingQueuedJob:
    """TrainingQueuedJob class for AWS Batch service.

    With this class, customers are able to attach the latest training job to an estimator.
    """

    def __init__(self, job_arn: str, job_name: str):
        self.job_arn = job_arn
        self.job_name = job_name
        self._no_training_job_status = {"SUBMITTED", "PENDING", "RUNNABLE"}

    def get_estimator(self) -> Estimator:
        """Attach the latest training job to an estimator and return.

        Returns: an Estimator instance.

        """
        describe_resp = self.describe()
        job_status = describe_resp.get("status", "")
        if self._training_job_created(job_status):
            if "latestAttempt" not in describe_resp:
                raise MissingRequiredArgument("No LatestAttempt in describe call")
            new_training_job_name = _get_new_training_job_name_from_latest_attempt(
                describe_resp["latestAttempt"]
            )
            output_estimator = _construct_estimator_from_training_job_name(new_training_job_name)
            _remove_system_tags_in_place_in_estimator_object(output_estimator)
            return output_estimator

        _output_attempt_history(describe_resp)
        raise NoTrainingJob("No Training job created. Job is still waiting in queue")

    def terminate(self, reason: Optional[str] = "Default terminate reason") -> None:
        """Terminate Batch job.

        Args:
            reason: Reason for terminating a job.

        Returns: None

        """
        terminate_service_job(self.job_arn, reason)

    def describe(self) -> Dict:
        """Describe Batch job.

        Returns: A dict which includes job parameters, job status, attempts and so on.

        """
        return describe_service_job(self.job_arn)

    def _training_job_created(self, status: str) -> bool:
        """Return True if a Training job has been created

        Args:
            status: Job status returned from Batch API.

        Returns: a boolean indicating whether a Training job has been created.

        """
        return status not in self._no_training_job_status

    def result(self, timeout: int = None) -> Dict:
        """Fetch the terminal result of the Batch job.

        Args:
            timeout: The time to wait for the Batch job to complete. Defaults to ``None``.

        Returns: The results of the Batch job, represented as a Dict.

        """
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.fetch_job_results(timeout))
        resp = loop.run_until_complete(task)
        return resp

    async def fetch_job_results(self, timeout: int = None) -> Dict:
        """Async method that waits for the Batch job to complete or until timeout.

        Args:
            timeout: The time to wait for the Batch job to complete. Defaults to ``None``.

        Returns: The results of the Batch job, represented as a Dict, or an Error.

        """
        self.wait(timeout)

        describe_resp = self.describe()
        if describe_resp.get("status", "") == JOB_STATUS_COMPLETED:
            return describe_resp
        if describe_resp.get("status", "") == JOB_STATUS_FAILED:
            raise RuntimeError(describe_resp["statusReason"])
        raise TimeoutError("Reached timeout before the Batch job reached a terminal status")

    def wait(self, timeout: int = None) -> Dict:
        """Wait for the Batch job to finish.

        This method blocks on the job completing for up to the timeout value (if specified).
        If timeout is ``None``, this method will block until the job is completed.

        Args:
            timeout (int): Timeout in seconds to wait until the job is completed. ``None`` by
            default.

        Returns: The last describe_service_job response for the Batch job.
        """
        request_end_time = time.time() + timeout if timeout else None
        describe_resp = self.describe()
        job_status = describe_resp.get("status", "")
        job_completed = job_status in (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED)

        while not job_completed:
            if timeout and time.time() > request_end_time:
                logging.info(
                    "Timeout exceeded: %d seconds elapsed. Returning current results", timeout
                )
                break
            if job_status in (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED):
                break

            time.sleep(POLL_IN_SECONDS)
            describe_resp = self.describe()
            job_status = describe_resp.get("status", "")
            job_completed = job_status in (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED)

        return describe_resp


def _construct_estimator_from_training_job_name(training_job_name: str) -> Estimator:
    """Build Estimator instance from payload.

    Args:
        training_job_name: Training job name.

    Returns: an Estimator instance.

    """
    return Estimator.attach(training_job_name)


def _output_attempt_history(describe_resp: Dict) -> None:
    """Print attempt history if no Training job created.

    Args:
        describe_resp: Describe response from Batch API.

    Returns: None

    """
    has_seen_status_reason = False
    for i, attempt_dict in enumerate(describe_resp.get("attempts", [])):
        if "statusReason" in attempt_dict:
            logging.info("Attempt %d - %s", i + 1, attempt_dict["statusReason"])
            has_seen_status_reason = True
    if not has_seen_status_reason:
        logging.info("No attempts found or no statusReason found.")


def _get_new_training_job_name_from_latest_attempt(latest_attempt: Dict) -> str:
    """Extract new Training job name from latest attempt in Batch Describe response.

    Args:
        latest_attempt: a Dict containing Training job arn.

    Returns: new Training job name or None if not found.

    """
    training_job_arn = latest_attempt.get("serviceResourceId", {}).get("value", None)
    return get_training_job_name_from_training_job_arn(training_job_arn)


def _remove_system_tags_in_place_in_estimator_object(estimator: Estimator) -> None:
    """Remove system tags in place.

    Args:
        estimator: input Estimator object.

    Returns: None. Remove system tags in place.

    """
    new_tags = []
    for tag_dict in estimator.tags:
        if not tag_dict.get("Key", "").startswith("aws:"):
            new_tags.append(tag_dict)
    estimator.tags = new_tags
