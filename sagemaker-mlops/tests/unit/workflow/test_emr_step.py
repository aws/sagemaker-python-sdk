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
"""Unit tests for workflow emr_step."""
from __future__ import absolute_import

import pytest

from sagemaker.mlops.workflow.emr_step import EMRStep, EMRStepConfig
from sagemaker.mlops.workflow.steps import StepTypeEnum
from sagemaker.core.workflow.properties import Properties


def test_emr_step_config_init():
    config = EMRStepConfig(jar="s3://bucket/my.jar", args=["arg1", "arg2"])
    assert config.jar == "s3://bucket/my.jar"
    assert config.args == ["arg1", "arg2"]


def test_emr_step_config_to_request():
    config = EMRStepConfig(jar="s3://bucket/my.jar", args=["arg1"])
    request = config.to_request()
    assert request["HadoopJarStep"]["Jar"] == "s3://bucket/my.jar"
    assert request["HadoopJarStep"]["Args"] == ["arg1"]


def test_emr_step_with_cluster_id():
    config = EMRStepConfig(jar="s3://bucket/my.jar")
    step = EMRStep(
        name="emr-step",
        display_name="EMR Step",
        description="Test EMR step",
        cluster_id="j-123456",
        step_config=config,
    )
    assert step.name == "emr-step"
    assert step.step_type == StepTypeEnum.EMR


def test_emr_step_with_cluster_config():
    config = EMRStepConfig(jar="s3://bucket/my.jar")
    cluster_config = {
        "Instances": {"InstanceGroups": [{"InstanceType": "m5.xlarge", "InstanceCount": 1}]}
    }
    step = EMRStep(
        name="emr-step",
        display_name="EMR Step",
        description="Test EMR step",
        cluster_id=None,
        step_config=config,
        cluster_config=cluster_config,
    )
    assert step.name == "emr-step"


def test_emr_step_without_cluster_id_or_config_raises_error():
    config = EMRStepConfig(jar="s3://bucket/my.jar")
    with pytest.raises(ValueError, match="must have either cluster_id or cluster_config"):
        EMRStep(
            name="emr-step",
            display_name="EMR Step",
            description="Test EMR step",
            cluster_id=None,
            step_config=config,
        )


def test_emr_step_with_both_cluster_id_and_config_raises_error():
    config = EMRStepConfig(jar="s3://bucket/my.jar")
    with pytest.raises(ValueError, match="can not have both cluster_id"):
        EMRStep(
            name="emr-step",
            display_name="EMR Step",
            description="Test EMR step",
            cluster_id="j-123456",
            step_config=config,
            cluster_config={"Instances": {}},
        )

def test_emr_step_with_output_args():
    config = EMRStepConfig(jar="s3://bucket/my.jar", args=["arg1"], output_args={"output": "s3://bucket/my/output/path"})
    step = EMRStep(
        name="emr-step",
        display_name="EMR Step",
        description="Test EMR step",
        cluster_id="j-123456",
        step_config=config,
    )
    assert "output" in step.emr_outputs
    assert isinstance(step.emr_outputs["output"], Properties)
