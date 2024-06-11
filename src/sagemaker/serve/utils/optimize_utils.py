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
"""Holds the util functions used for the optimize function"""
from __future__ import absolute_import

import time
import logging

from sagemaker import Session

# TODO: determine how long optimization jobs take
OPTIMIZE_POLLER_MAX_TIMEOUT_SECS = 300
OPTIMIZE_POLLER_INTERVAL_SECS = 30

logger = logging.getLogger(__name__)


def _is_compatible_with_compilation(instance_type: str) -> bool:
    """Checks whether an instance is compatible with compilation.

    Args:
        instance_type (str): The instance type used for the compilation job.

    Returns:
        bool: Whether the given instance type is compatible with compilation.
    """
    return instance_type.startswith("ml.inf") or instance_type.startswith("ml.trn")


def _poll_optimization_job(job_name: str, sagemaker_session: Session) -> bool:
    """Polls optimization job status until success.

    Args:
        job_name (str): The name of the optimization job.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed.

    Returns:
        bool: Whether the optimization job was successful.
    """
    logger.info("Polling status of optimization job %s", job_name)
    start_time = time.time()
    while time.time() - start_time < OPTIMIZE_POLLER_MAX_TIMEOUT_SECS:
        result = sagemaker_session.sagemaker_client.describe_optimization_job(job_name)
        # TODO: use correct condition to determine whether optimization job is complete
        if result is not None:
            return result
        time.sleep(OPTIMIZE_POLLER_INTERVAL_SECS)
