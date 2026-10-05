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
"""The file defines constants used for Batch API helper functions."""

from __future__ import absolute_import

SAGEMAKER_TRAINING = "SAGEMAKER_TRAINING"
DEFAULT_ATTEMPT_DURATION_IN_SECONDS = 86400  # 1 day in seconds.
DEFAULT_TIMEOUT = {"attemptDurationSeconds": DEFAULT_ATTEMPT_DURATION_IN_SECONDS}
POLL_IN_SECONDS = 5
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_COMPLETED = "SUCCEEDED"
JOB_STATUS_FAILED = "FAILED"
DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG = {
    "attempts": 1,
    "evaluateOnExit": [
        {
            "action": "RETRY",
            "onStatusReason": "Received status from SageMaker:InternalServerError: "
            "We encountered an internal error. Please try again.",
        },
        {"action": "EXIT", "onStatusReason": "*"},
    ],
}
