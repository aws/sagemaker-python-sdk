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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker.workflow.pipeline_experiment_config import (
    PipelineExperimentConfig,
    PipelineExperimentConfigProperties,
)


def test_pipeline_experiment_config():
    config = PipelineExperimentConfig("experiment-name", "trial-name")
    assert config.to_request() == {"ExperimentName": "experiment-name", "TrialName": "trial-name"}


def test_pipeline_experiment_config_property():
    var = PipelineExperimentConfigProperties.EXPERIMENT_NAME
    assert var.expr == {"Get": "PipelineExperimentConfig.ExperimentName"}
