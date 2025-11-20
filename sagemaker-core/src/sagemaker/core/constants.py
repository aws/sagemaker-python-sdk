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
"""Constants used across the SageMaker Core package.

This module contains constant values that are shared across different
components of the SageMaker SDK.
"""
from __future__ import absolute_import

# Script mode environment variable names
# These constants define the environment variable names used in SageMaker
# containers for script mode execution

SCRIPT_PARAM_NAME = "sagemaker_program"
"""Environment variable name for the entry point script name."""

DIR_PARAM_NAME = "sagemaker_submit_directory"
"""Environment variable name for the directory containing the entry point script.

This constant specifies the S3 location or local path where the training/inference
code and dependencies are located.
"""

CONTAINER_LOG_LEVEL_PARAM_NAME = "sagemaker_container_log_level"
"""Environment variable name for the container log level."""

JOB_NAME_PARAM_NAME = "sagemaker_job_name"
"""Environment variable name for the SageMaker job name."""

MODEL_SERVER_WORKERS_PARAM_NAME = "sagemaker_model_server_workers"
"""Environment variable name for the number of model server workers.

This constant specifies how many worker processes the model server should use
for handling inference requests. More workers can improve throughput for
CPU-bound models.
"""

SAGEMAKER_REGION_PARAM_NAME = "sagemaker_region"
"""Environment variable name for the AWS region."""

SAGEMAKER_OUTPUT_LOCATION = "sagemaker_s3_output"
"""Environment variable name for the S3 output location."""

# Neo compilation frameworks
NEO_ALLOWED_FRAMEWORKS = set(
    ["mxnet", "tensorflow", "keras", "pytorch", "onnx", "xgboost", "tflite"]
)
"""Set of frameworks allowed for Neo compilation.

Neo is SageMaker's model compilation service that optimizes models for
specific hardware targets.
"""

__all__ = [
    "SCRIPT_PARAM_NAME",
    "DIR_PARAM_NAME",
    "CONTAINER_LOG_LEVEL_PARAM_NAME",
    "JOB_NAME_PARAM_NAME",
    "MODEL_SERVER_WORKERS_PARAM_NAME",
    "SAGEMAKER_REGION_PARAM_NAME",
    "SAGEMAKER_OUTPUT_LOCATION",
    "NEO_ALLOWED_FRAMEWORKS",
]
