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
import re
from typing import Optional, Dict
import nest_asyncio
from sagemaker.core.resources import TrainingJob
from sagemaker.core.shapes import Unassigned
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import (
    Compute,
    Networking,
    StoppingCondition,
    SourceCode,
    TrainingImageConfig,
)
from .batch_api_helper import _terminate_service_job, _describe_service_job
from .exception import NoTrainingJob, MissingRequiredArgument
from ..utils import _get_training_job_name_from_training_job_arn
from .constants import JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, POLL_IN_SECONDS

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


class TrainingQueuedJob:
    """TrainingQueuedJob class for AWS Batch service.

    With this class, customers are able to attach the latest training job to a ModelTrainer.
    """

    def __init__(self, job_arn: str, job_name: str):
        self.job_arn = job_arn
        self.job_name = job_name
        self._no_training_job_status = {"SUBMITTED", "PENDING", "RUNNABLE"}

    def get_model_trainer(self) -> ModelTrainer:
        """Attach the latest training job to a ModelTrainer and return.

        Returns: a ModelTrainer instance.

        """
        describe_resp = self.describe()
        job_status = describe_resp.get("status", "")
        if self._training_job_created(job_status):
            if "latestAttempt" not in describe_resp:
                raise MissingRequiredArgument("No LatestAttempt in describe call")
            new_training_job_name = _get_new_training_job_name_from_latest_attempt(
                describe_resp["latestAttempt"]
            )
            output_model_trainer = _construct_model_trainer_from_training_job_name(
                new_training_job_name
            )
            _remove_system_tags_in_place_in_model_trainer_object(output_model_trainer)
            return output_model_trainer

        _output_attempt_history(describe_resp)
        raise NoTrainingJob("No Training job created. Job is still waiting in queue")

    def terminate(self, reason: Optional[str] = "Default terminate reason") -> None:
        """Terminate Batch job.

        Args:
            reason: Reason for terminating a job.

        Returns: None

        """
        _terminate_service_job(self.job_arn, reason)

    def describe(self) -> Dict:
        """Describe Batch job.

        Returns: A dict which includes job parameters, job status, attempts and so on.

        """
        return _describe_service_job(self.job_arn)

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


def _construct_model_trainer_from_training_job_name(training_job_name: str) -> ModelTrainer:
    """Build ModelTrainer instance from training job name.

    Args:
        training_job_name: Training job name.

    Returns: a ModelTrainer instance with _latest_training_job set.

    """
    # Step 1: Get the TrainingJob resource
    training_job = TrainingJob.get(training_job_name=training_job_name)

    # Step 2: Extract parameters from training_job to reconstruct ModelTrainer
    init_params = {}

    # Required/common parameters
    init_params["role"] = training_job.role_arn
    init_params["base_job_name"] = _extract_base_job_name(training_job_name)

    # Training image or algorithm
    if training_job.algorithm_specification and not isinstance(training_job.algorithm_specification, Unassigned):
        if (training_job.algorithm_specification.training_image and 
            not isinstance(training_job.algorithm_specification.training_image, Unassigned)):
            init_params["training_image"] = training_job.algorithm_specification.training_image
        if (training_job.algorithm_specification.algorithm_name and 
            not isinstance(training_job.algorithm_specification.algorithm_name, Unassigned)):
            init_params["algorithm_name"] = training_job.algorithm_specification.algorithm_name
        if (training_job.algorithm_specification.training_input_mode and 
            not isinstance(training_job.algorithm_specification.training_input_mode, Unassigned)):
            init_params["training_input_mode"] = training_job.algorithm_specification.training_input_mode

    # Compute config
    if training_job.resource_config and not isinstance(training_job.resource_config, Unassigned):
        compute_params = {}
        
        if (training_job.resource_config.instance_type and 
            not isinstance(training_job.resource_config.instance_type, Unassigned)):
            compute_params["instance_type"] = training_job.resource_config.instance_type
        if (training_job.resource_config.instance_count and 
            not isinstance(training_job.resource_config.instance_count, Unassigned)):
            compute_params["instance_count"] = training_job.resource_config.instance_count
        if (training_job.resource_config.volume_size_in_gb and 
            not isinstance(training_job.resource_config.volume_size_in_gb, Unassigned)):
            compute_params["volume_size_in_gb"] = training_job.resource_config.volume_size_in_gb
        
        # Add managed spot training if enabled (available directly on TrainingJob)
        if training_job.enable_managed_spot_training and not isinstance(training_job.enable_managed_spot_training, Unassigned):
            compute_params["enable_managed_spot_training"] = training_job.enable_managed_spot_training
            
        if compute_params:  # Only create Compute if we have valid params
            init_params["compute"] = Compute(**compute_params)

    # Output config - pass the raw training job output config directly
    if training_job.output_data_config and not isinstance(training_job.output_data_config, Unassigned):
        init_params["output_data_config"] = training_job.output_data_config

    # Stopping condition
    if training_job.stopping_condition and not isinstance(training_job.stopping_condition, Unassigned):
        if (training_job.stopping_condition.max_runtime_in_seconds and 
            not isinstance(training_job.stopping_condition.max_runtime_in_seconds, Unassigned)):
            init_params["stopping_condition"] = StoppingCondition(
                max_runtime_in_seconds=training_job.stopping_condition.max_runtime_in_seconds,
            )

    # Networking
    if training_job.vpc_config and not isinstance(training_job.vpc_config, Unassigned):
        networking_params = {}
        
        if (training_job.vpc_config.subnets and 
            not isinstance(training_job.vpc_config.subnets, Unassigned)):
            networking_params["subnets"] = training_job.vpc_config.subnets
        if (training_job.vpc_config.security_group_ids and 
            not isinstance(training_job.vpc_config.security_group_ids, Unassigned)):
            networking_params["security_group_ids"] = training_job.vpc_config.security_group_ids
        
        # Add network isolation if present (available directly on TrainingJob)
        if training_job.enable_network_isolation and not isinstance(training_job.enable_network_isolation, Unassigned):
            networking_params["enable_network_isolation"] = training_job.enable_network_isolation
            
        # Add inter-container traffic encryption if present (available directly on TrainingJob)
        if training_job.enable_inter_container_traffic_encryption and not isinstance(training_job.enable_inter_container_traffic_encryption, Unassigned):
            networking_params["enable_inter_container_traffic_encryption"] = training_job.enable_inter_container_traffic_encryption
            
        if networking_params:  # Only create Networking if we have valid params
            init_params["networking"] = Networking(**networking_params)

    # Hyperparameters
    if training_job.hyper_parameters and not isinstance(training_job.hyper_parameters, Unassigned):
        init_params["hyperparameters"] = training_job.hyper_parameters

    # Environment
    if training_job.environment and not isinstance(training_job.environment, Unassigned):
        init_params["environment"] = training_job.environment

    # Checkpoint config
    if training_job.checkpoint_config and not isinstance(training_job.checkpoint_config, Unassigned):
        init_params["checkpoint_config"] = training_job.checkpoint_config

    # Step 3: Create ModelTrainer
    model_trainer = ModelTrainer(**init_params)

    # Step 4: Set _latest_training_job
    model_trainer._latest_training_job = training_job

    return model_trainer


def _extract_base_job_name(training_job_name: str) -> str:
    """Extract base job name from full training job name.

    Args:
        training_job_name: Full training job name.

    Returns: Base job name.

    """
    # Use the same regex pattern as PySDK V2's base_from_name() function
    # Matches timestamps like: YYYY-MM-DD-HH-MM-SS-SSS or YYMMDD-HHMM
    match = re.match(r"^(.+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}|\d{6}-\d{4})", training_job_name)
    return match.group(1) if match else training_job_name


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
    return _get_training_job_name_from_training_job_arn(training_job_arn)


def _remove_system_tags_in_place_in_model_trainer_object(model_trainer: ModelTrainer) -> None:
    """Remove system tags in place.

    Args:
        model_trainer: input ModelTrainer object.

    Returns: None. Remove system tags in place.

    """
    if model_trainer.tags:
        filtered_tags = []
        for tag in model_trainer.tags:
            # Handle both V2 dict format {"Key": "...", "Value": "..."} and V3 object format with .key attribute
            if isinstance(tag, dict):
                # V2 format
                if not tag.get("Key", "").startswith("aws:"):
                    filtered_tags.append(tag)
            else:
                # V3 format - assume it has .key attribute
                if hasattr(tag, 'key') and not tag.key.startswith("aws:"):
                    filtered_tags.append(tag)
                elif hasattr(tag, 'Key') and not tag.Key.startswith("aws:"):
                    # Fallback for other formats
                    filtered_tags.append(tag)
                else:
                    # If we can't determine the key, keep the tag to be safe
                    filtered_tags.append(tag)
        
        model_trainer.tags = filtered_tags
