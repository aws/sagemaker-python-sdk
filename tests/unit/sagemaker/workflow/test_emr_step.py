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

from sagemaker.workflow.emr_step import EMRStep, EMRStepConfig
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.parameters import ParameterString
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


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


g_emr_step_config = EMRStepConfig(jar="s3:/script-runner/script-runner.jar")
g_emr_step_name = "MyEMRStep"
g_prefix = "EMRStep " + g_emr_step_name + " "
g_prefix_with_in = "In EMRStep " + g_emr_step_name + ", "
g_cluster_config = {
    "Instances": {
        "InstanceGroups": [
            {
                "Name": "Master Instance Group",
                "InstanceRole": "MASTER",
                "InstanceCount": 1,
                "InstanceType": "m1.small",
                "Market": "ON_DEMAND",
            }
        ],
        "InstanceCount": 1,
        "HadoopVersion": "MyHadoopVersion",
    },
    "AmiVersion": "3.8.0",
    "AdditionalInfo": "MyAdditionalInfo",
}


def test_emr_step_throws_exception_when_both_cluster_id_and_cluster_config_are_present():
    with pytest.raises(Exception) as exceptionInfo:
        EMRStep(
            name=g_emr_step_name,
            display_name="MyEMRStep",
            description="MyEMRStepDescription",
            step_config=g_emr_step_config,
            cluster_id="MyClusterID",
            cluster_config=g_cluster_config,
            depends_on=["TestStep"],
            cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
        )
    expected_error_msg = (
        g_prefix + "can not have both cluster_id or cluster_config. "
        "If user wants to use cluster_config, then they "
        "have to explicitly set cluster_id as None"
    )
    actual_error_msg = exceptionInfo.value.args[0]

    assert actual_error_msg == expected_error_msg


def test_emr_step_throws_exception_when_both_cluster_id_and_cluster_config_are_none():
    with pytest.raises(Exception) as exceptionInfo:
        EMRStep(
            name=g_emr_step_name,
            display_name="MyEMRStep",
            description="MyEMRStepDescription",
            cluster_id=None,
            step_config=g_emr_step_config,
            depends_on=["TestStep"],
            cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
        )
    expected_error_msg = g_prefix + "must have either cluster_id or cluster_config"
    actual_error_msg = exceptionInfo.value.args[0]

    assert actual_error_msg == expected_error_msg


def test_emr_step_with_valid_cluster_config():
    emr_step = EMRStep(
        name=g_emr_step_name,
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id=None,
        cluster_config=g_cluster_config,
        step_config=g_emr_step_config,
        cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
    )

    assert emr_step.to_request() == {
        "Name": "MyEMRStep",
        "Type": "EMR",
        "Arguments": {
            "StepConfig": {"HadoopJarStep": {"Jar": "s3:/script-runner/script-runner.jar"}},
            "ClusterConfig": {
                "AdditionalInfo": "MyAdditionalInfo",
                "AmiVersion": "3.8.0",
                "Instances": {
                    "HadoopVersion": "MyHadoopVersion",
                    "InstanceCount": 1,
                    "InstanceGroups": [
                        {
                            "InstanceCount": 1,
                            "InstanceRole": "MASTER",
                            "InstanceType": "m1.small",
                            "Market": "ON_DEMAND",
                            "Name": "Master Instance Group",
                        }
                    ],
                },
            },
        },
        "DisplayName": "MyEMRStep",
        "Description": "MyEMRStepDescription",
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
    }

    pipeline = Pipeline(name="MyPipeline", steps=[emr_step])

    assert json.loads(pipeline.definition()) == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [],
        "PipelineExperimentConfig": {
            "ExperimentName": {"Get": "Execution.PipelineName"},
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
        "Steps": [
            {
                "Name": "MyEMRStep",
                "Type": "EMR",
                "Arguments": {
                    "StepConfig": {"HadoopJarStep": {"Jar": "s3:/script-runner/script-runner.jar"}},
                    "ClusterConfig": {
                        "AdditionalInfo": "MyAdditionalInfo",
                        "AmiVersion": "3.8.0",
                        "Instances": {
                            "HadoopVersion": "MyHadoopVersion",
                            "InstanceCount": 1,
                            "InstanceGroups": [
                                {
                                    "InstanceCount": 1,
                                    "InstanceRole": "MASTER",
                                    "InstanceType": "m1.small",
                                    "Market": "ON_DEMAND",
                                    "Name": "Master Instance Group",
                                }
                            ],
                        },
                    },
                },
                "DisplayName": "MyEMRStep",
                "Description": "MyEMRStepDescription",
                "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
            }
        ],
    }


@pytest.mark.parametrize(
    "invalid_cluster_config, expected_error_msg",
    [
        (
            {
                "Name": "someName",
                "Instances": {
                    "InstanceGroups": [
                        {
                            "Name": "Master Instance Group",
                        }
                    ],
                },
            },
            g_prefix_with_in + "cluster_config should not contain any of "
            "Name, AutoTerminationPolicy and/or Steps",
        ),
        (
            {
                "AutoTerminationPolicy": {},
                "Instances": {
                    "InstanceGroups": [
                        {
                            "Name": "Master Instance Group",
                        }
                    ],
                },
            },
            g_prefix_with_in + "cluster_config should not contain any of "
            "Name, AutoTerminationPolicy and/or Steps",
        ),
        (
            {
                "Steps": [],
                "Instances": {
                    "InstanceGroups": [
                        {
                            "Name": "Master Instance Group",
                        }
                    ],
                },
            },
            g_prefix_with_in + "cluster_config should not contain any of "
            "Name, AutoTerminationPolicy and/or Steps",
        ),
        (
            {
                "AmiVersion": "3.8.0",
                "AdditionalInfo": "MyAdditionalInfo",
            },
            g_prefix_with_in + "cluster_config must contain Instances",
        ),
        (
            {
                "Instances": {},
            },
            g_prefix_with_in + "Instances should contain either "
            "InstanceGroups or InstanceFleets",
        ),
        (
            {
                "Instances": {
                    "InstanceGroups": [
                        {
                            "Name": "Master Instance Group",
                        }
                    ],
                    "InstanceFleets": [
                        {
                            "Name": "Master Instance Fleets",
                        }
                    ],
                },
            },
            g_prefix_with_in + "Instances should contain either "
            "InstanceGroups or InstanceFleets",
        ),
        (
            {
                "Instances": {
                    "InstanceGroups": [
                        {
                            "Name": "Master Instance Group",
                        }
                    ],
                    "KeepJobFlowAliveWhenNoSteps": True,
                },
            },
            g_prefix_with_in + "Instances should not contain "
            "KeepJobFlowAliveWhenNoSteps or "
            "TerminationProtected",
        ),
        (
            {
                "Instances": {
                    "InstanceGroups": [
                        {
                            "Name": "Master Instance Group",
                        }
                    ],
                    "TerminationProtected": True,
                },
            },
            g_prefix_with_in + "Instances should not contain "
            "KeepJobFlowAliveWhenNoSteps or "
            "TerminationProtected",
        ),
    ],
)
def test_emr_step_throws_exception_when_cluster_config_contains_restricted_entities(
    invalid_cluster_config, expected_error_msg
):
    with pytest.raises(Exception) as exceptionInfo:
        EMRStep(
            name=g_emr_step_name,
            display_name="MyEMRStep",
            description="MyEMRStepDescription",
            cluster_id=None,
            step_config=g_emr_step_config,
            cluster_config=invalid_cluster_config,
            depends_on=["TestStep"],
            cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
        )

    actual_error_msg = exceptionInfo.value.args[0]

    assert actual_error_msg == expected_error_msg
