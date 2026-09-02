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
"""Unit tests for the inference and lineage pipeline step types.

The inference steps (EndpointConfigStep, EndpointStep,
InferenceComponentStep) follow the ``step_args`` convention: the step
arguments are captured by calling the corresponding session method under
a ``PipelineSession``, which intercepts the request instead of calling
the service.
"""

from __future__ import absolute_import

from unittest.mock import Mock

import pytest

from sagemaker.core.workflow.pipeline_context import PipelineSession, _JobStepArguments
from sagemaker.mlops.workflow.endpoint_step import EndpointConfigStep, EndpointStep
from sagemaker.mlops.workflow.inference_component_step import InferenceComponentStep
from sagemaker.core.lineage.action import Action
from sagemaker.core.lineage.artifact import Artifact
from sagemaker.core.lineage.association import Association
from sagemaker.core.lineage.context import Context
from sagemaker.mlops.workflow.lineage_step import LineageStep
from sagemaker.mlops.workflow.retry import (
    StepExceptionTypeEnum,
    StepRetryPolicy,
)
from sagemaker.mlops.workflow.steps import CacheConfig, StepTypeEnum

ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"


@pytest.fixture
def pipeline_session():
    """A PipelineSession with a mocked client -- no AWS calls are made."""
    return PipelineSession(
        boto_session=Mock(region_name="us-west-2"),
        sagemaker_client=Mock(),
    )


@pytest.fixture
def endpoint_config_step_args(pipeline_session):
    return pipeline_session.endpoint_from_production_variants(
        name="my-config",
        production_variants=[
            {
                "ModelName": "my-model",
                "VariantName": "AllTraffic",
                "InstanceType": "ml.m5.large",
                "InitialInstanceCount": 1,
            }
        ],
        kms_key="arn:aws:kms:us-west-2:123456789012:key/abc",
    )


@pytest.fixture
def endpoint_step_args(pipeline_session):
    return pipeline_session.create_endpoint(endpoint_name="my-endpoint", config_name="my-config")


@pytest.fixture
def inference_component_step_args(pipeline_session):
    return pipeline_session.create_inference_component(
        inference_component_name="my-component",
        endpoint_name="my-endpoint",
        variant_name="AllTraffic",
        specification={"ModelName": "my-model"},
        runtime_config={"CopyCount": 2},
    )


# ---------- step_args capture via PipelineSession ----------


def test_capture_does_not_call_service(pipeline_session, endpoint_config_step_args):
    assert isinstance(endpoint_config_step_args, _JobStepArguments)
    assert not pipeline_session.sagemaker_client.create_endpoint_config.called
    assert not pipeline_session.sagemaker_client.create_endpoint.called


def test_captured_request_content(endpoint_config_step_args):
    args = endpoint_config_step_args.args
    assert args["EndpointConfigName"] == "my-config"
    assert args["KmsKeyId"] == "arn:aws:kms:us-west-2:123456789012:key/abc"
    assert args["ProductionVariants"][0]["ModelName"] == "my-model"


# ---------- EndpointConfigStep ----------


def test_endpoint_config_step_basic(endpoint_config_step_args):
    step = EndpointConfigStep(name="Cfg", step_args=endpoint_config_step_args)
    assert step.step_type == StepTypeEnum.ENDPOINT_CONFIG
    assert step.arguments["EndpointConfigName"] == "my-config"
    req = step.to_request()
    assert req["Type"] == "EndpointConfig"
    assert req["Name"] == "Cfg"


def test_endpoint_config_step_to_request_includes_cache_and_retry(
    endpoint_config_step_args,
):
    step = EndpointConfigStep(
        name="Cfg",
        step_args=endpoint_config_step_args,
        cache_config=CacheConfig(enable_caching=True, expire_after="P30D"),
        retry_policies=[
            StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
        ],
    )
    req = step.to_request()
    assert req["CacheConfig"]["Enabled"] is True
    assert req["RetryPolicies"][0]["MaxAttempts"] == 3


def test_endpoint_config_step_rejects_wrong_producer(endpoint_step_args):
    with pytest.raises(ValueError, match="endpoint_from_production_variants"):
        EndpointConfigStep(name="Cfg", step_args=endpoint_step_args)


def test_endpoint_config_step_rejects_raw_dict():
    with pytest.raises(TypeError):
        EndpointConfigStep(name="Cfg", step_args={"EndpointConfigName": "x"})


def test_endpoint_config_step_properties(endpoint_config_step_args):
    step = EndpointConfigStep(name="Cfg", step_args=endpoint_config_step_args)
    assert step.properties.EndpointConfigName.expr == {"Get": "Steps.Cfg.EndpointConfigName"}


# ---------- EndpointStep ----------


def test_endpoint_step_basic(endpoint_step_args):
    step = EndpointStep(name="Deploy", step_args=endpoint_step_args)
    assert step.step_type == StepTypeEnum.ENDPOINT
    assert step.arguments["EndpointName"] == "my-endpoint"
    assert step.arguments["EndpointConfigName"] == "my-config"
    assert step.to_request()["Type"] == "Endpoint"


def test_endpoint_step_cache_config(endpoint_step_args):
    step = EndpointStep(
        name="Deploy",
        step_args=endpoint_step_args,
        cache_config=CacheConfig(enable_caching=True, expire_after="P30D"),
    )
    assert step.to_request()["CacheConfig"]["Enabled"] is True


def test_endpoint_step_rejects_retry_policies_kwarg(endpoint_step_args):
    """EndpointStep is not retryable -- constructor must not accept retry_policies."""
    with pytest.raises(TypeError):
        EndpointStep(name="Deploy", step_args=endpoint_step_args, retry_policies=[])


def test_endpoint_step_rejects_wrong_producer(endpoint_config_step_args):
    with pytest.raises(ValueError, match="create_endpoint"):
        EndpointStep(name="Deploy", step_args=endpoint_config_step_args)


def test_endpoint_step_properties(endpoint_step_args):
    step = EndpointStep(name="Deploy", step_args=endpoint_step_args)
    assert step.properties.EndpointName.expr == {"Get": "Steps.Deploy.EndpointName"}


# ---------- InferenceComponentStep ----------


def test_inference_component_step_basic(inference_component_step_args):
    step = InferenceComponentStep(name="IC", step_args=inference_component_step_args)
    assert step.step_type == StepTypeEnum.INFERENCE_COMPONENT
    args = step.arguments
    assert args["InferenceComponentName"] == "my-component"
    assert args["EndpointName"] == "my-endpoint"
    assert args["VariantName"] == "AllTraffic"
    assert args["Specification"] == {"ModelName": "my-model"}
    assert args["RuntimeConfig"] == {"CopyCount": 2}


def test_inference_component_step_default_runtime_config(pipeline_session):
    step_args = pipeline_session.create_inference_component(
        inference_component_name="ic",
        endpoint_name="ep",
        variant_name="v",
        specification={"ModelName": "m"},
    )
    step = InferenceComponentStep(name="IC", step_args=step_args)
    assert step.arguments["RuntimeConfig"] == {"CopyCount": 1}


def test_inference_component_step_rejects_retry_policies_kwarg(
    inference_component_step_args,
):
    with pytest.raises(TypeError):
        InferenceComponentStep(
            name="IC", step_args=inference_component_step_args, retry_policies=[]
        )


def test_inference_component_step_rejects_wrong_producer(endpoint_step_args):
    with pytest.raises(ValueError, match="create_inference_component"):
        InferenceComponentStep(name="IC", step_args=endpoint_step_args)


def test_inference_component_step_properties(inference_component_step_args):
    step = InferenceComponentStep(name="IC", step_args=inference_component_step_args)
    assert step.properties.InferenceComponentName.expr == {"Get": "Steps.IC.InferenceComponentName"}


# ---------- plain Session behavior is unchanged ----------


def test_plain_session_still_calls_service():
    from sagemaker.core.helper.session_helper import Session

    session = Session(boto_session=Mock(region_name="us-west-2"), sagemaker_client=Mock())
    session.sagemaker_client.create_endpoint.return_value = {"EndpointArn": "arn:x"}
    name = session.create_endpoint(endpoint_name="ep", config_name="cfg", wait=False)
    assert name == "ep"
    assert session.sagemaker_client.create_endpoint.called


# ---------- LineageStep ----------


@pytest.fixture
def action_step_args(pipeline_session):
    return Action.create(
        action_name="act1",
        source_uri="s3://bucket/model.tar.gz",
        source_type="S3ETag",
        action_type="ModelTraining",
        status="Completed",
        sagemaker_session=pipeline_session,
    )


def test_lineage_step_action(pipeline_session, action_step_args):
    step = LineageStep(name="RecA", step_args=action_step_args)
    assert step.step_type == StepTypeEnum.LINEAGE
    args = step.arguments
    assert list(args.keys()) == ["Actions"]
    assert args["Actions"][0]["ActionName"] == "act1"
    assert args["Actions"][0]["Source"]["SourceUri"] == "s3://bucket/model.tar.gz"
    assert not pipeline_session.sagemaker_client.create_action.called


def test_lineage_step_artifact(pipeline_session):
    step_args = Artifact.create(
        artifact_name="art1",
        source_uri="s3://bucket/data",
        artifact_type="Model",
        sagemaker_session=pipeline_session,
    )
    step = LineageStep(name="RecB", step_args=step_args)
    assert list(step.arguments.keys()) == ["Artifacts"]
    assert step.arguments["Artifacts"][0]["ArtifactName"] == "art1"


def test_lineage_step_context(pipeline_session):
    step_args = Context.create(
        context_name="ctx1",
        source_uri="s3://bucket/ctx",
        context_type="Endpoint",
        sagemaker_session=pipeline_session,
    )
    step = LineageStep(name="RecC", step_args=step_args)
    assert list(step.arguments.keys()) == ["Contexts"]
    assert step.arguments["Contexts"][0]["ContextName"] == "ctx1"


def test_lineage_step_association_translates_arns(pipeline_session, action_step_args):
    action_step = LineageStep(name="RecA", step_args=action_step_args)
    step_args = Association.create(
        source_arn=action_step.properties.ActionArns["act1"],
        destination_arn="arn:aws:sagemaker:us-west-2:123456789012:artifact/abc",
        association_type="Produced",
        sagemaker_session=pipeline_session,
    )
    step = LineageStep(name="RecD", step_args=step_args, depends_on=[action_step])
    entity = step.arguments["Associations"][0]
    # AddAssociation's SourceArn/DestinationArn become entity references.
    assert entity["Source"]["Arn"].expr == {"Get": "Steps.RecA.ActionArns['act1']"}
    assert entity["Destination"] == {"Arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/abc"}
    assert entity["AssociationType"] == "Produced"
    assert not pipeline_session.sagemaker_client.add_association.called


def test_lineage_step_rejects_wrong_producer(endpoint_step_args):
    with pytest.raises(ValueError, match="Action.create"):
        LineageStep(name="Rec", step_args=endpoint_step_args)


def test_lineage_step_rejects_raw_dict():
    with pytest.raises(TypeError):
        LineageStep(name="Rec", step_args={"Actions": []})


def test_lineage_step_properties(action_step_args):
    step = LineageStep(name="Rec", step_args=action_step_args)
    for field in ("ActionArns", "ArtifactArns", "ContextArns", "Associations"):
        assert hasattr(step.properties, field)
    assert step.properties.ArtifactArns["x"].expr == {"Get": "Steps.Rec.ArtifactArns['x']"}


def test_lineage_create_on_plain_session_calls_service():
    from sagemaker.core.helper.session_helper import Session

    session = Session(boto_session=Mock(region_name="us-west-2"), sagemaker_client=Mock())
    session.sagemaker_client.create_action.return_value = {"ActionArn": "arn:x"}
    result = Action.create(
        action_name="a",
        source_uri="s3://b",
        source_type="S3ETag",
        action_type="T",
        status="Completed",
        sagemaker_session=session,
    )
    assert session.sagemaker_client.create_action.called
    assert not isinstance(result, _JobStepArguments)


# ---------- Cross-cutting ----------


def test_all_steps_importable_from_init():
    from sagemaker.mlops.workflow import (  # noqa: F401
        EndpointConfigStep,
        EndpointStep,
        InferenceComponentStep,
        LineageStep,
    )


def test_step_type_enum_values():
    assert StepTypeEnum.ENDPOINT_CONFIG.value == "EndpointConfig"
    assert StepTypeEnum.ENDPOINT.value == "Endpoint"
    assert StepTypeEnum.INFERENCE_COMPONENT.value == "InferenceComponent"
    assert StepTypeEnum.LINEAGE.value == "Lineage"


def test_depends_on_accepts_step_and_string(endpoint_config_step_args, endpoint_step_args):
    cfg_step = EndpointConfigStep(name="Cfg", step_args=endpoint_config_step_args)
    step = EndpointStep(name="Deploy", step_args=endpoint_step_args, depends_on=[cfg_step, "Other"])
    req = step.to_request()
    assert req["DependsOn"] == [cfg_step, "Other"]
