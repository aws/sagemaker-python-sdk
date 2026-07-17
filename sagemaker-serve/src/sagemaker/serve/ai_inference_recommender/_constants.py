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
"""Constants for the AI inference recommender module."""
from __future__ import absolute_import

from enum import Enum

MAX_INSTANCE_TYPES = 3

FEATURE_GATING_RUNBOOK_URL = (
    "https://docs.aws.amazon.com/sagemaker/latest/dg/"
    "generative-ai-inference-recommendations.html"
)


class PerformanceTarget(str, Enum):
    """Optimization goal for a recommendation job."""

    THROUGHPUT = "throughput"
    TTFT_MS = "ttft-ms"
    COST = "cost"


class InferenceFramework(str, Enum):
    """Inference framework to benchmark a recommendation against."""

    LMI = "LMI"
    VLLM = "VLLM"
