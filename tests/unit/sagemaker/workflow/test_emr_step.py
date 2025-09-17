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

from sagemaker.workflow.emr_step import (
    EMRStep,
    EMRStepConfig,
    ERR_STR_WITH_NAME_AUTO_TERMINATION_OR_STEPS,
    ERR_STR_WITHOUT_INSTANCE,
    ERR_STR_WITH_KEEPJOBFLOW_OR_TERMINATIONPROTECTED,
    ERR_STR_BOTH_OR_NONE_INSTANCEGROUPS_OR_INSTANCEFLEETS,
    ERR_STR_WITH_BOTH_CLUSTER_ID_AND_CLUSTER_CFG,
    ERR_STR_WITHOUT_CLUSTER_ID_AND_CLUSTER_CFG,
    ERR_STR_WITH_EXEC_ROLE_ARN_AND_WITHOUT_CLUSTER_ID,
)
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


@pytest.mark.parametrize("execution_role_arn", [None, "arn:aws:iam:000000000000:role/runtime-role"])
def test_emr_step_with_one_step_config(sagemaker_session, execution_role_arn):
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
        execution_role_arn=execution_role_arn,
    )
    emr_step.add_depends_on(["SecondTestStep"])

    expected_request = {
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

    if execution_role_arn is not None:
        expected_request["Arguments"]["ExecutionRoleArn"] = execution_role_arn

    assert emr_step.to_request() == expected_request
    assert emr_step.properties.ClusterId == "MyClusterID"
    assert (
        emr_step.properties.ExecutionRoleArn == execution_role_arn
        if execution_role_arn is not None
        else True
    )
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
    assert emr_step.properties.Status.FailureDetails.Reason._referenced_steps == [emr_step]


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

    pipeline_def = json.loads(pipeline.definition())
    assert ordered(pipeline_def) == ordered(
        {
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
    )
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"emr_step_1": [], "emr_step_2": [], "TestStep": ["emr_step_1", "emr_step_2"]}
    )


g_emr_step_config = EMRStepConfig(jar="s3:/script-runner/script-runner.jar")
g_emr_step_name = "MyEMRStep"
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
    with pytest.raises(ValueError) as exceptionInfo:
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
    expected_error_msg = ERR_STR_WITH_BOTH_CLUSTER_ID_AND_CLUSTER_CFG.format(
        step_name=g_emr_step_name
    )
    actual_error_msg = exceptionInfo.value.args[0]

    assert actual_error_msg == expected_error_msg


def test_emr_step_throws_exception_when_both_cluster_id_and_cluster_config_are_none():
    with pytest.raises(ValueError) as exceptionInfo:
        EMRStep(
            name=g_emr_step_name,
            display_name="MyEMRStep",
            description="MyEMRStepDescription",
            cluster_id=None,
            step_config=g_emr_step_config,
            depends_on=["TestStep"],
            cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
        )
    expected_error_msg = ERR_STR_WITHOUT_CLUSTER_ID_AND_CLUSTER_CFG.format(
        step_name=g_emr_step_name
    )
    actual_error_msg = exceptionInfo.value.args[0]

    assert actual_error_msg == expected_error_msg


def test_emr_step_throws_exception_when_both_execution_role_arn_and_cluster_config_are_present():
    with pytest.raises(ValueError) as exceptionInfo:
        EMRStep(
            name=g_emr_step_name,
            display_name="MyEMRStep",
            description="MyEMRStepDescription",
            step_config=g_emr_step_config,
            cluster_id=None,
            cluster_config=g_cluster_config,
            depends_on=["TestStep"],
            cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
            execution_role_arn="arn:aws:iam:000000000000:role/some-role",
        )
    expected_error_msg = ERR_STR_WITH_EXEC_ROLE_ARN_AND_WITHOUT_CLUSTER_ID.format(
        step_name=g_emr_step_name
    )
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
            ERR_STR_WITH_NAME_AUTO_TERMINATION_OR_STEPS.format(step_name=g_emr_step_name),
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
            ERR_STR_WITH_NAME_AUTO_TERMINATION_OR_STEPS.format(step_name=g_emr_step_name),
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
            ERR_STR_WITH_NAME_AUTO_TERMINATION_OR_STEPS.format(step_name=g_emr_step_name),
        ),
        (
            {
                "AmiVersion": "3.8.0",
                "AdditionalInfo": "MyAdditionalInfo",
            },
            ERR_STR_WITHOUT_INSTANCE.format(step_name=g_emr_step_name),
        ),
        (
            {
                "Instances": {},
            },
            ERR_STR_BOTH_OR_NONE_INSTANCEGROUPS_OR_INSTANCEFLEETS.format(step_name=g_emr_step_name),
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
            ERR_STR_BOTH_OR_NONE_INSTANCEGROUPS_OR_INSTANCEFLEETS.format(step_name=g_emr_step_name),
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
            ERR_STR_WITH_KEEPJOBFLOW_OR_TERMINATIONPROTECTED.format(step_name=g_emr_step_name),
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
            ERR_STR_WITH_KEEPJOBFLOW_OR_TERMINATIONPROTECTED.format(step_name=g_emr_step_name),
        ),
    ],
)
def test_emr_step_throws_exception_when_cluster_config_contains_restricted_entities(
    invalid_cluster_config, expected_error_msg
):
    with pytest.raises(ValueError) as exceptionInfo:
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


def test_emr_step_with_retry_policies(sagemaker_session):
    """Test EMRStep with retry policies."""
    emr_step_config = EMRStepConfig(
        jar="s3:/script-runner/script-runner.jar",
        args=["--arg_0", "arg_0_value"],
        main_class="com.my.main",
        properties=[{"Key": "Foo", "Value": "Foo_value"}, {"Key": "Bar", "Value": "Bar_value"}],
    )

    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        ),
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.THROTTLING],
            interval_seconds=5,
            max_attempts=5,
            backoff_rate=1.5,
        ),
    ]

    emr_step = EMRStep(
        name="MyEMRStep",
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id="MyClusterID",
        step_config=emr_step_config,
        depends_on=["TestStep"],
        cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
        retry_policies=retry_policies,
    )

    expected_request = {
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
        "DependsOn": ["TestStep"],
        "DisplayName": "MyEMRStep",
        "Description": "MyEMRStepDescription",
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
        "RetryPolicies": [
            {
                "ExceptionType": ["Step.SERVICE_FAULT"],
                "IntervalSeconds": 1,
                "MaxAttempts": 3,
                "BackoffRate": 2.0,
            },
            {
                "ExceptionType": ["Step.THROTTLING"],
                "IntervalSeconds": 5,
                "MaxAttempts": 5,
                "BackoffRate": 1.5,
            },
        ],
    }

    assert emr_step.to_request() == expected_request


def test_emr_step_with_retry_policies_and_cluster_config():
    """Test EMRStep with both retry policies and cluster configuration."""
    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        )
    ]

    emr_step = EMRStep(
        name=g_emr_step_name,
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id=None,
        cluster_config=g_cluster_config,
        step_config=g_emr_step_config,
        cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
        retry_policies=retry_policies,
    )

    expected_request = {
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
        "RetryPolicies": [
            {
                "ExceptionType": ["Step.SERVICE_FAULT"],
                "IntervalSeconds": 1,
                "MaxAttempts": 3,
                "BackoffRate": 2.0,
            }
        ],
    }

    assert emr_step.to_request() == expected_request


def test_emr_step_with_retry_policy_expire_after():
    """Test EMRStep with retry policy using expire_after_mins."""
    emr_step_config = EMRStepConfig(
        jar="s3:/script-runner/script-runner.jar",
        args=["--arg_0", "arg_0_value"],
    )

    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            expire_after_mins=30,
            backoff_rate=2.0,
        )
    ]

    emr_step = EMRStep(
        name="MyEMRStep",
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id="MyClusterID",
        step_config=emr_step_config,
        retry_policies=retry_policies,
    )

    expected_request = {
        "Name": "MyEMRStep",
        "Type": "EMR",
        "Arguments": {
            "ClusterId": "MyClusterID",
            "StepConfig": {
                "HadoopJarStep": {
                    "Args": ["--arg_0", "arg_0_value"],
                    "Jar": "s3:/script-runner/script-runner.jar",
                }
            },
        },
        "DisplayName": "MyEMRStep",
        "Description": "MyEMRStepDescription",
        "RetryPolicies": [
            {
                "ExceptionType": ["Step.SERVICE_FAULT"],
                "IntervalSeconds": 1,
                "ExpireAfterMin": 30,
                "BackoffRate": 2.0,
            }
        ],
    }

    assert emr_step.to_request() == expected_request


def test_emr_step_with_all_exception_types():
    """Test EMRStep with all available exception types."""
    emr_step_config = EMRStepConfig(jar="s3:/script-runner/script-runner.jar")

    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT, StepExceptionTypeEnum.THROTTLING],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        )
    ]

    emr_step = EMRStep(
        name="MyEMRStep",
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id="MyClusterID",
        step_config=emr_step_config,
        retry_policies=retry_policies,
    )

    expected_request = {
        "Name": "MyEMRStep",
        "Type": "EMR",
        "Arguments": {
            "ClusterId": "MyClusterID",
            "StepConfig": {
                "HadoopJarStep": {
                    "Jar": "s3:/script-runner/script-runner.jar",
                }
            },
        },
        "DisplayName": "MyEMRStep",
        "Description": "MyEMRStepDescription",
        "RetryPolicies": [
            {
                "ExceptionType": ["Step.SERVICE_FAULT", "Step.THROTTLING"],
                "IntervalSeconds": 1,
                "MaxAttempts": 3,
                "BackoffRate": 2.0,
            }
        ],
    }

    assert emr_step.to_request() == expected_request


def test_pipeline_interpolates_emr_outputs_with_retry_policies(sagemaker_session):
    """Test pipeline definition with EMR steps that have retry policies."""
    custom_step = CustomStep("TestStep")
    parameter = ParameterString("MyStr")

    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        )
    ]

    step_emr = EMRStep(
        name="emr_step_1",
        cluster_id="MyClusterID",
        display_name="emr_step_1",
        description="MyEMRStepDescription",
        depends_on=[custom_step],
        step_config=EMRStepConfig(jar="s3:/script-runner/script-runner.jar"),
        retry_policies=retry_policies,
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step_emr, custom_step],
        sagemaker_session=sagemaker_session,
    )

    pipeline_def = json.loads(pipeline.definition())
    assert "RetryPolicies" in pipeline_def["Steps"][0]


def test_emr_step_with_retry_policies_and_execution_role():
    """Test EMRStep with both retry policies and execution role."""
    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        )
    ]

    emr_step = EMRStep(
        name="MyEMRStep",
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id="MyClusterID",
        step_config=g_emr_step_config,
        execution_role_arn="arn:aws:iam:000000000000:role/role",
        retry_policies=retry_policies,
    )

    request = emr_step.to_request()
    assert "RetryPolicies" in request
    assert "ExecutionRoleArn" in request["Arguments"]


def test_emr_step_properties_with_retry_policies():
    """Test EMRStep properties when retry policies are provided."""
    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        )
    ]

    emr_step = EMRStep(
        name="MyEMRStep",
        display_name="MyEMRStep",
        description="MyEMRStepDescription",
        cluster_id="MyClusterID",
        step_config=g_emr_step_config,
        retry_policies=retry_policies,
    )

    # Verify properties still work with retry policies
    assert emr_step.properties.ClusterId == "MyClusterID"
    assert emr_step.properties.Status.State.expr == {"Get": "Steps.MyEMRStep.Status.State"}
