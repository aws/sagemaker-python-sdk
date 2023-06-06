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
"""Unit tests for SelectiveExecutionConfig module"""

from __future__ import absolute_import

from sagemaker.workflow.selective_execution_config import SelectiveExecutionConfig


def test_SelectiveExecutionConfig():
    selective_execution_config = SelectiveExecutionConfig(
        source_pipeline_execution_arn="foo-arn", selected_steps=["step-1", "step-2", "step-3"]
    )
    assert selective_execution_config.to_request() == {
        "SelectedSteps": [{"StepName": "step-1"}, {"StepName": "step-2"}, {"StepName": "step-3"}],
        "SourcePipelineExecutionArn": "foo-arn",
    }
