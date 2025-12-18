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
"""Define Queue class for AWS Batch service"""
from __future__ import absolute_import

from typing import Dict, Optional, List
import logging
from sagemaker.train.model_trainer import ModelTrainer, Mode
from .training_queued_job import TrainingQueuedJob
from .batch_api_helper import _submit_service_job, _list_service_job
from .exception import MissingRequiredArgument
from .constants import DEFAULT_TIMEOUT, JOB_STATUS_RUNNING


class TrainingQueue:
    """TrainingQueue class for AWS Batch service

    With this class, customers are able to create a new queue and submit jobs to AWS Batch Service.
    """

    def __init__(self, queue_name: str):
        self.queue_name = queue_name

    def submit(
        self,
        training_job: ModelTrainer,
        inputs,
        job_name: Optional[str] = None,
        retry_config: Optional[Dict] = None,
        priority: Optional[int] = None,
        share_identifier: Optional[str] = None,
        timeout: Optional[Dict] = None,
        tags: Optional[Dict] = None,
        experiment_config: Optional[Dict] = None,
    ) -> TrainingQueuedJob:
        """Submit a queued job and return a QueuedJob object.

        Args:
            training_job: Training job ModelTrainer object.
            inputs: Training job inputs.
            job_name: Batch job name.
            retry_config: Retry configuration for Batch job.
            priority: Scheduling priority for Batch job.
            share_identifier: Share identifier for Batch job.
            timeout: Timeout configuration for Batch job.
            tags: Tags apply to Batch job. These tags are for Batch job only.
            experiment_config: Experiment management configuration.
                Optionally, the dict can contain four keys:
                'ExperimentName', 'TrialName', 'TrialComponentDisplayName' and 'RunName'.

        Returns: a TrainingQueuedJob object with Batch job ARN and job name.

        """
        if not isinstance(training_job, ModelTrainer):
            raise TypeError(
                "training_job must be an instance of ModelTrainer, "
                f"but got {type(training_job)}"
            )

        if training_job.training_mode != Mode.SAGEMAKER_TRAINING_JOB:
            raise ValueError(
                "TrainingQueue requires using a ModelTrainer with Mode.SAGEMAKER_TRAINING_JOB"
            )
        if experiment_config is not None:
            logging.warning(
                "ExperimentConfig is not supported for ModelTrainer. "
                "It will be ignored when submitting the job."
            )
        training_payload = training_job._create_training_job_args(
            input_data_config=inputs, boto3=True
        )

        if timeout is None:
            timeout = DEFAULT_TIMEOUT
        if job_name is None:
            job_name = training_payload["TrainingJobName"]

        resp = _submit_service_job(
            training_payload,
            job_name,
            self.queue_name,
            retry_config,
            priority,
            timeout,
            share_identifier,
            tags,
        )
        if "jobArn" not in resp or "jobName" not in resp:
            raise MissingRequiredArgument(
                "jobArn or jobName is missing in response from Batch submit_service_job API"
            )
        return TrainingQueuedJob(resp["jobArn"], resp["jobName"])

    def map(
        self,
        training_job: ModelTrainer,
        inputs,
        job_names: Optional[List[str]] = None,
        retry_config: Optional[Dict] = None,
        priority: Optional[int] = None,
        share_identifier: Optional[str] = None,
        timeout: Optional[Dict] = None,
        tags: Optional[Dict] = None,
        experiment_config: Optional[Dict] = None,
    ) -> List[TrainingQueuedJob]:
        """Submit queued jobs to the provided estimator and return a list of TrainingQueuedJob objects.

        Args:
            training_job: Training job ModelTrainer object.
            inputs: List of Training job inputs.
            job_names: List of Batch job names.
            retry_config: Retry config for the Batch jobs.
            priority: Scheduling priority for the Batch jobs.
            share_identifier: Share identifier for the Batch jobs.
            timeout: Timeout configuration for the Batch jobs.
            tags: Tags apply to Batch job. These tags are for Batch job only.
            experiment_config: Experiment management configuration.
                Optionally, the dict can contain four keys:
                'ExperimentName', 'TrialName', 'TrialComponentDisplayName' and 'RunName'.

        Returns: a list of TrainingQueuedJob objects with each Batch job ARN and job name.

        """
        if experiment_config is None:
            experiment_config = {}

        if job_names is not None:
            if len(job_names) != len(inputs):
                raise ValueError(
                    "When specified, the number of job names must match the number of inputs"
                )
        else:
            job_names = [None] * len(inputs)

        queued_batch_job_list = []
        for index, value in enumerate(inputs):
            queued_batch_job = self.submit(
                training_job,
                value,
                job_names[index],
                retry_config,
                priority,
                share_identifier,
                timeout,
                tags,
                experiment_config,
            )
            queued_batch_job_list.append(queued_batch_job)

        return queued_batch_job_list

    def list_jobs(
        self, job_name: Optional[str] = None, status: Optional[str] = JOB_STATUS_RUNNING
    ) -> List[TrainingQueuedJob]:
        """List Batch jobs according to job_name or status.

        Args:
            job_name: Batch job name.
            status: Batch job status.

        Returns: A list of QueuedJob.

        """
        filters = None
        if job_name:
            filters = [{"name": "JOB_NAME", "values": [job_name]}]
            status = None  # job_status is ignored when job_name is specified.
        jobs_to_return = []
        next_token = None
        for job_result_dict in _list_service_job(self.queue_name, status, filters, next_token):
            for job_result in job_result_dict.get("jobSummaryList", []):
                if "jobArn" in job_result and "jobName" in job_result:
                    jobs_to_return.append(
                        TrainingQueuedJob(job_result["jobArn"], job_result["jobName"])
                    )
                else:
                    logging.warning("Missing JobArn or JobName in Batch ListJobs API")
                    continue
        return jobs_to_return

    def get_job(self, job_name):
        """Get a Batch job according to job_name.

        Args:
        job_name: Batch job name.

        Returns: The QueuedJob with name matching job_name.

        """
        jobs_to_return = self.list_jobs(job_name)
        if len(jobs_to_return) == 0:
            raise ValueError(f"Cannot find job: {job_name}")
        return jobs_to_return[0]
