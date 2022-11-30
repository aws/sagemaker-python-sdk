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
from __future__ import absolute_import

import json

import pytest

from mock import Mock

from sagemaker.workflow.emr_step import EMRStep, EMRStepConfig
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.parameters import ParameterString
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name="us-west-2")
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name="us-west-2",
        config=None,
        local_mode=False,
    )
    return session_mock


def test_emr_step_with_one_step_config(sagemaker_session):
    emr_step_config = EMRStepConfig(
        jar="s3:/script-runner/script-runner.jar",
        args=["--arg_0", "arg_0_value"],
        main_class="com.my.main",
        properties=[{"Key": "Foo", "Value": "Foo_value"}, {"Key": "Bar", "Value": "Bar_value"}],
    )

    emr_step = EMRStep(
        name="MyEMRStep",
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id="MyClusterID",
        step_config=emr_step_config,
        depends_on=["TestStep"],
        cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
    )
    emr_step.add_depends_on(["SecondTestStep"])
    assert emr_step.to_request() == {
        "Name": "MyEMRStep",
        "Type": "EMR",
        "Arguments": {
            "ClusterId": "MyClusterID",
            "StepConfig": {
                "HadoopJarStep": {
                    "Args": ["--arg_0", "arg_0_value"],
                    "Jar": "s3:/script-runner/script-runner.jar",
                    "MainClass": "com.my.main",
                    "Properties": [
                        {"Key": "Foo", "Value": "Foo_value"},
                        {"Key": "Bar", "Value": "Bar_value"},
                    ],
                }
            },
        },
        "DependsOn": ["TestStep", "SecondTestStep"],
        "DisplayName": "MyEMRStep",
        "Description": "MyEMRStepDescription",
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
    }

    assert emr_step.properties.ClusterId == "MyClusterID"
    assert emr_step.properties.ActionOnFailure.expr == {"Get": "Steps.MyEMRStep.ActionOnFailure"}
    assert emr_step.properties.Config.Args.expr == {"Get": "Steps.MyEMRStep.Config.Args"}
    assert emr_step.properties.Config.Jar.expr == {"Get": "Steps.MyEMRStep.Config.Jar"}
    assert emr_step.properties.Config.MainClass.expr == {"Get": "Steps.MyEMRStep.Config.MainClass"}
    assert emr_step.properties.Id.expr == {"Get": "Steps.MyEMRStep.Id"}
    assert emr_step.properties.Name.expr == {"Get": "Steps.MyEMRStep.Name"}
    assert emr_step.properties.Status.State.expr == {"Get": "Steps.MyEMRStep.Status.State"}
    assert emr_step.properties.Status.FailureDetails.Reason.expr == {
        "Get": "Steps.MyEMRStep.Status.FailureDetails.Reason"
    }


def test_pipeline_interpolates_emr_outputs(sagemaker_session):
    custom_step = CustomStep("TestStep")
    parameter = ParameterString("MyStr")

    emr_step_config_1 = EMRStepConfig(
        jar="s3:/script-runner/script-runner_1.jar",
        args=["--arg_0", "arg_0_value"],
        main_class="com.my.main",
        properties=[{"Key": "Foo", "Value": "Foo_value"}, {"Key": "Bar", "Value": "Bar_value"}],
    )

    step_emr_1 = EMRStep(
        name="emr_step_1",
        cluster_id="MyClusterID",
        display_name="emr_step_1",
        description="MyEMRStepDescription",
        depends_on=[custom_step],
        step_config=emr_step_config_1,
    )

    emr_step_config_2 = EMRStepConfig(jar="s3:/script-runner/script-runner_2.jar")

    step_emr_2 = EMRStep(
        name="emr_step_2",
        cluster_id="MyClusterID",
        display_name="emr_step_2",
        description="MyEMRStepDescription",
        depends_on=[custom_step],
        step_config=emr_step_config_2,
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step_emr_1, step_emr_2, custom_step],
        sagemaker_session=sagemaker_session,
    )

    assert json.loads(pipeline.definition()) == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [{"Name": "MyStr", "Type": "String"}],
        "PipelineExperimentConfig": {
            "ExperimentName": {"Get": "Execution.PipelineName"},
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
        "Steps": [
            {
                "Name": "emr_step_1",
                "Type": "EMR",
                "Arguments": {
                    "ClusterId": "MyClusterID",
                    "StepConfig": {
                        "HadoopJarStep": {
                            "Args": ["--arg_0", "arg_0_value"],
                            "Jar": "s3:/script-runner/script-runner_1.jar",
                            "MainClass": "com.my.main",
                            "Properties": [
                                {"Key": "Foo", "Value": "Foo_value"},
                                {"Key": "Bar", "Value": "Bar_value"},
                            ],
                        }
                    },
                },
                "DependsOn": ["TestStep"],
                "Description": "MyEMRStepDescription",
                "DisplayName": "emr_step_1",
            },
            {
                "Name": "emr_step_2",
                "Type": "EMR",
                "Arguments": {
                    "ClusterId": "MyClusterID",
                    "StepConfig": {
                        "HadoopJarStep": {"Jar": "s3:/script-runner/script-runner_2.jar"}
                    },
                },
                "Description": "MyEMRStepDescription",
                "DisplayName": "emr_step_2",
                "DependsOn": ["TestStep"],
            },
            {
                "Name": "TestStep",
                "Type": "Training",
                "Arguments": {},
            },
        ],
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"emr_step_1": [], "emr_step_2": [], "TestStep": ["emr_step_1", "emr_step_2"]}
    )
