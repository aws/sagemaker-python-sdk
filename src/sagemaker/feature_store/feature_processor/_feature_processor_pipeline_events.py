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
"""Contains data classes for the Feature Processor Pipeline Events."""
from __future__ import absolute_import

from typing import List
import attr
from sagemaker.feature_store.feature_processor._enums import FeatureProcessorPipelineExecutionStatus


@attr.s(frozen=True)
class FeatureProcessorPipelineEvents:
    """Immutable data class containing the execution events for a FeatureProcessor pipeline.

    This class is used for creating event based triggers for feature processor pipelines.
    """

    pipeline_name: str = attr.ib()
    pipeline_execution_status: List[FeatureProcessorPipelineExecutionStatus] = attr.ib()
