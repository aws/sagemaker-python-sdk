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
"""Defines enum values."""

from __future__ import absolute_import

import logging
from enum import Enum


LOGGER = logging.getLogger("sagemaker")


class EndpointType(Enum):
    """Types of endpoint"""

    MODEL_BASED = "ModelBased"  # Amazon SageMaker Model Based Endpoint
    INFERENCE_COMPONENT_BASED = (
        "InferenceComponentBased"  # Amazon SageMaker Inference Component Based Endpoint
    )


class RoutingStrategy(Enum):
    """Strategy for routing https traffics."""

    RANDOM = "RANDOM"
    """The endpoint routes each request to a randomly chosen instance.
    """
    LEAST_OUTSTANDING_REQUESTS = "LEAST_OUTSTANDING_REQUESTS"
    """The endpoint routes requests to the specific instances that have
    more capacity to process them.
    """


class Tag(str, Enum):
    """Enum class for tag keys to apply to models."""

    OPTIMIZATION_JOB_NAME = "sagemaker-sdk:optimization-job-name"
    SPECULATIVE_DRAFT_MODEL_PROVIDER = "sagemaker-sdk:speculative-draft-model-provider"
    FINE_TUNING_MODEL_PATH = "sagemaker-sdk:fine-tuning-model-path"
    FINE_TUNING_JOB_NAME = "sagemaker-sdk:fine-tuning-job-name"
