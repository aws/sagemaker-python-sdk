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
"""Unit tests for workflow pipeline_experiment_config."""
from __future__ import absolute_import

from sagemaker.mlops.workflow.pipeline_experiment_config import (
    PipelineExperimentConfig, PipelineExperimentConfigProperties
)
from sagemaker.core.workflow.execution_variables import ExecutionVariables


def test_pipeline_experiment_config_init():
    config = PipelineExperimentConfig(
        experiment_name="test-experiment",
        trial_name="test-trial"
    )
    assert config.experiment_name == "test-experiment"
    assert config.trial_name == "test-trial"


def test_pipeline_experiment_config_with_execution_variables():
    config = PipelineExperimentConfig(
        experiment_name=ExecutionVariables.PIPELINE_NAME,
        trial_name=ExecutionVariables.PIPELINE_EXECUTION_ID
    )
    request = config.to_request()
    assert "ExperimentName" in request
    assert "TrialName" in request


def test_pipeline_experiment_config_properties():
    assert PipelineExperimentConfigProperties.EXPERIMENT_NAME.name == "ExperimentName"
    assert PipelineExperimentConfigProperties.TRIAL_NAME.name == "TrialName"
