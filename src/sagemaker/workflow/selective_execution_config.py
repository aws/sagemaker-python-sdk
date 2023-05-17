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
"""Pipeline Parallelism Configuration"""
from __future__ import absolute_import
from typing import List
from sagemaker.workflow.entities import RequestType


class SelectiveExecutionConfig:
    """Selective execution config config for SageMaker pipeline."""

    def __init__(self, source_pipeline_execution_arn: str, selected_steps: List[str]):
        """Create a SelectiveExecutionConfig

        Args:
            source_pipeline_execution_arn (str): ARN of the previously executed pipeline execution.
                The given arn pipeline execution status can be either Failed or Success.
            selected_steps (List[str]): List of step names that pipeline users want to run
                in new subworkflow-execution. The steps must be connected.
        """
        self.source_pipeline_execution_arn = source_pipeline_execution_arn
        self.selected_steps = selected_steps

    def _build_selected_steps_from_list(self) -> RequestType:
        """Get the request structure for list of selected steps"""
        selected_step_list = []
        for selected_step in self.selected_steps:
            selected_step_list.append(dict(StepName=selected_step))
        return selected_step_list

    def to_request(self) -> RequestType:
        """Convert SelectiveExecutionConfig object to request dict."""
        request = {}

        if self.source_pipeline_execution_arn is not None:
            request["SourcePipelineExecutionArn"] = self.source_pipeline_execution_arn

        if self.selected_steps is not None:
            request["SelectedSteps"] = self._build_selected_steps_from_list()

        return request
