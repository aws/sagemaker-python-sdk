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
"""Unit tests for workflow selective_execution_config."""
from __future__ import absolute_import

from sagemaker.mlops.workflow.selective_execution_config import SelectiveExecutionConfig


def test_selective_execution_config_init():
    config = SelectiveExecutionConfig(
        selected_steps=["step1", "step2"],
        source_pipeline_execution_arn="arn:aws:sagemaker:us-west-2:123456789012:pipeline/test/execution/exec-123"
    )
    assert config.selected_steps == ["step1", "step2"]
    assert "exec-123" in config.source_pipeline_execution_arn


def test_selective_execution_config_to_request():
    config = SelectiveExecutionConfig(
        selected_steps=["step1", "step2"],
        source_pipeline_execution_arn="arn:test"
    )
    request = config.to_request()
    assert request["SourcePipelineExecutionArn"] == "arn:test"
    assert len(request["SelectedSteps"]) == 2
    assert request["SelectedSteps"][0]["StepName"] == "step1"


def test_selective_execution_config_reference_latest():
    config = SelectiveExecutionConfig(
        selected_steps=["step1"],
        reference_latest_execution=True
    )
    assert config.reference_latest_execution is True
