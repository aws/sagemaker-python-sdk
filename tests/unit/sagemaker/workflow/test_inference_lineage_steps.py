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
"""Unit tests for the inference and lineage pipeline step types (v2).

Passthrough ``arguments: Dict[str, Any]`` API. Top-level argument keys
are validated client-side against the public AWS API input shape
(botocore service model); fields known to be rejected by SageMaker
Pipelines fail fast at construction. Values are not validated -- they
may be pipeline variables. Full schema validation remains server-side.
"""

from __future__ import absolute_import

import pytest

from sagemaker.workflow.endpoint_step import EndpointConfigStep, EndpointStep
from sagemaker.workflow.inference_component_step import InferenceComponentStep
from sagemaker.workflow.lineage_step import LineageStep
from sagemaker.workflow.steps import CacheConfig, StepTypeEnum

# ---------- EndpointConfigStep ----------


def test_endpoint_config_step_basic():
    step = EndpointConfigStep(
        name="Cfg",
        arguments={
            "EndpointConfigName": "MyCfg",
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "m",
                    "InstanceType": "ml.m5.large",
                    "InitialInstanceCount": 1,
                }
            ],
        },
    )
    assert step.step_type == StepTypeEnum.ENDPOINT_CONFIG
    assert step.arguments["EndpointConfigName"] == "MyCfg"


def test_endpoint_config_step_to_request_includes_cache_and_retry():
    step = EndpointConfigStep(
        name="Cfg",
        arguments={"EndpointConfigName": "MyCfg", "ProductionVariants": []},
        display_name="Create Config",
        description="desc",
        cache_config=CacheConfig(enable_caching=True, expire_after="P30D"),
    )
    req = step.to_request()
    assert req["Type"] == "EndpointConfig"
    assert req["DisplayName"] == "Create Config"
    assert req["Description"] == "desc"
    assert req["CacheConfig"] == {"Enabled": True, "ExpireAfter": "P30D"}


def test_endpoint_config_step_accepts_full_api_surface():
    step = EndpointConfigStep(
        name="Cfg",
        arguments={
            "EndpointConfigName": "MyCfg",
            "ProductionVariants": [],
            "KmsKeyId": "arn:aws:kms:...",
            "ExecutionRoleArn": "arn:aws:iam:...",
            "AsyncInferenceConfig": {"OutputConfig": {"S3OutputPath": "s3://x/"}},
            "VpcConfig": {"SecurityGroupIds": ["sg-0"], "Subnets": ["subnet-0"]},
            "EnableNetworkIsolation": False,
            "ShadowProductionVariants": [],
        },
    )
    args = step.arguments
    assert args["KmsKeyId"] == "arn:aws:kms:..."
    assert args["ExecutionRoleArn"] == "arn:aws:iam:..."


def test_endpoint_config_step_requires_arguments():
    with pytest.raises(ValueError):
        EndpointConfigStep(name="Cfg", arguments=None)


# ---------- EndpointStep ----------


def test_endpoint_step_basic():
    step = EndpointStep(
        name="Deploy",
        arguments={"EndpointName": "ep", "EndpointConfigName": "cfg"},
    )
    assert step.step_type == StepTypeEnum.ENDPOINT
    assert step.arguments == {"EndpointName": "ep", "EndpointConfigName": "cfg"}


def test_endpoint_step_cache_config():
    step = EndpointStep(
        name="Deploy",
        arguments={"EndpointName": "ep", "EndpointConfigName": "cfg"},
        cache_config=CacheConfig(enable_caching=True),
    )
    req = step.to_request()
    assert req["Type"] == "Endpoint"
    assert req["CacheConfig"] == {"Enabled": True}


def test_endpoint_step_rejects_retry_policies_kwarg():
    with pytest.raises(TypeError):
        EndpointStep(
            name="Deploy",
            arguments={"EndpointName": "ep", "EndpointConfigName": "cfg"},
            retry_policies=[],
        )


# ---------- InferenceComponentStep ----------


def test_inference_component_step_basic():
    step = InferenceComponentStep(
        name="IC",
        arguments={
            "InferenceComponentName": "ic",
            "EndpointName": "ep",
            "VariantName": "v",
            "Specification": {
                "ModelName": "m",
                "ComputeResourceRequirements": {
                    "MinMemoryRequiredInMb": 1024,
                    "NumberOfCpuCoresRequired": 2.0,
                },
            },
            "RuntimeConfig": {"CopyCount": 1},
        },
    )
    assert step.step_type == StepTypeEnum.INFERENCE_COMPONENT
    assert step.arguments["Specification"]["ModelName"] == "m"


def test_inference_component_step_rejects_retry_policies_kwarg():
    with pytest.raises(TypeError):
        InferenceComponentStep(name="IC", arguments={}, retry_policies=[])


# ---------- LineageStep ----------


def test_lineage_step_basic():
    step = LineageStep(
        name="Rec",
        arguments={
            "Actions": [{"ActionName": "a1", "ActionType": "ModelTraining", "Status": "Completed"}],
            "Artifacts": [
                {
                    "ArtifactName": "art1",
                    "ArtifactType": "Model",
                    "Source": {"SourceUri": "s3://x/y"},
                }
            ],
            "Associations": [
                {
                    "Source": {"Name": "a1", "Type": "Action"},
                    "Destination": {"Name": "art1", "Type": "Artifact"},
                    "AssociationType": "Produced",
                }
            ],
        },
    )
    assert step.step_type == StepTypeEnum.LINEAGE
    assert len(step.arguments["Actions"]) == 1
    assert len(step.arguments["Associations"]) == 1


def test_lineage_step_partial_arguments():
    step = LineageStep(
        name="Rec",
        arguments={"Actions": [{"ActionName": "a", "ActionType": "T", "Status": "Completed"}]},
    )
    assert "Actions" in step.arguments


def test_lineage_step_requires_at_least_one_recognized_key():
    with pytest.raises(ValueError):
        LineageStep(name="Rec", arguments={})
    with pytest.raises(ValueError):
        LineageStep(name="Rec", arguments={"Bogus": []})


def test_lineage_step_properties():
    step = LineageStep(name="Rec", arguments={"Actions": []})
    for field in ("ActionArns", "ArtifactArns", "ContextArns", "Associations"):
        assert hasattr(step.properties, field)


# ---------- Cross-cutting ----------


def test_step_type_enum_values():
    assert StepTypeEnum.ENDPOINT_CONFIG.value == "EndpointConfig"
    assert StepTypeEnum.ENDPOINT.value == "Endpoint"
    assert StepTypeEnum.INFERENCE_COMPONENT.value == "InferenceComponent"
    assert StepTypeEnum.LINEAGE.value == "Lineage"


def test_depends_on_accepts_string_list():
    step = EndpointStep(
        name="Deploy",
        arguments={"EndpointName": "ep", "EndpointConfigName": "cfg"},
        depends_on=["Prev"],
    )
    req = step.to_request()
    assert req["DependsOn"] == ["Prev"]


# ---------- Client-side argument validation ----------


def test_endpoint_config_step_rejects_unsupported_fields():
    """DataCaptureConfig and ExplainerConfig exist in the public API but
    are rejected by SageMaker Pipelines -- fail fast with a clear error."""
    for field in ("DataCaptureConfig", "ExplainerConfig"):
        with pytest.raises(ValueError, match=field):
            EndpointConfigStep(
                name="Cfg",
                arguments={
                    "EndpointConfigName": "cfg",
                    "ProductionVariants": [],
                    field: {},
                },
            )


def test_endpoint_step_rejects_unsupported_deployment_config():
    with pytest.raises(ValueError, match="DeploymentConfig"):
        EndpointStep(
            name="Deploy",
            arguments={
                "EndpointName": "ep",
                "EndpointConfigName": "cfg",
                "DeploymentConfig": {},
            },
        )


def test_unknown_argument_key_rejected():
    """Keys outside the operation's input shape fail fast at construction."""
    with pytest.raises(ValueError, match="Bogus"):
        EndpointConfigStep(
            name="Cfg",
            arguments={"EndpointConfigName": "cfg", "Bogus": 1},
        )
    with pytest.raises(ValueError, match="Bogus"):
        InferenceComponentStep(
            name="IC",
            arguments={"InferenceComponentName": "ic", "Bogus": 1},
        )


def test_empty_arguments_rejected():
    for cls, valid_key in (
        (EndpointConfigStep, "EndpointConfigName"),
        (EndpointStep, "EndpointName"),
        (InferenceComponentStep, "InferenceComponentName"),
    ):
        with pytest.raises(ValueError):
            cls(name="x", arguments={})
        # sanity: a single valid key constructs fine
        assert cls(name="x", arguments={valid_key: "v"}).arguments == {valid_key: "v"}


def test_pipeline_variable_values_pass_validation():
    """Only top-level keys are validated -- values may be pipeline
    variables (Get expressions) at any position."""
    step = EndpointStep(
        name="Deploy",
        arguments={
            "EndpointName": {"Get": "Parameters.EndpointName"},
            "EndpointConfigName": {"Get": "Steps.Cfg.EndpointConfigName"},
        },
    )
    assert step.arguments["EndpointName"] == {"Get": "Parameters.EndpointName"}


def test_post_construction_mutation_caught_at_serialization():
    """Injecting an unsupported field after construction is caught when
    the arguments property is read (i.e., at pipeline serialization)."""
    step = EndpointConfigStep(
        name="Cfg",
        arguments={"EndpointConfigName": "cfg", "ProductionVariants": []},
    )
    step._arguments["DataCaptureConfig"] = {}
    with pytest.raises(ValueError, match="DataCaptureConfig"):
        _ = step.arguments


def test_lineage_step_rejects_unknown_keys_alongside_recognized():
    with pytest.raises(ValueError, match="Bogus"):
        LineageStep(name="Rec", arguments={"Actions": [], "Bogus": []})
