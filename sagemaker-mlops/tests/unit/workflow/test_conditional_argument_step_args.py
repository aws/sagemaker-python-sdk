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
"""Conditional argument expressions passed through step_args builders."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.network import NetworkConfig
from sagemaker.core.parameter import IntegerParameter
from sagemaker.core.processing import Processor
from sagemaker.core.transformer import Transformer
from sagemaker.core.workflow import ElseIf, If, Omit
from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.core.workflow.parameters import ParameterBoolean, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep, TransformStep, TuningStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train.configs import (
    Compute,
    Networking,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.tuner import HyperparameterTuner

IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/img:latest"
ROLE = "arn:aws:iam::000000000000:role/test-role"
ENABLED = ParameterBoolean("Enabled", default_value=False)
VALUE = ParameterString("Value", default_value="")
EXPECTED = {
    "Std:If": {
        "Condition": {
            "Type": "Equals",
            "LeftValue": {"Get": "Parameters.Enabled"},
            "RightValue": True,
        },
        "Then": {"Get": "Parameters.Value"},
        "Else": {"Std:Omit": True},
    }
}


def _optional_value():
    return If(condition=ConditionEquals(left=ENABLED, right=True), then=VALUE, else_=Omit())


@pytest.fixture
def session():
    session = PipelineSession.__new__(PipelineSession)
    session._context = None
    session.boto_session = MagicMock()
    session.boto_session.region_name = "us-west-2"
    session._region_name = "us-west-2"
    session.sagemaker_config = {}
    session.default_bucket = lambda: "bucket"
    session.default_bucket_prefix = None
    session.local_mode = False
    session.settings = MagicMock(include_jumpstart_tags=False)
    session.sagemaker_client = MagicMock()
    return session


@pytest.fixture(autouse=True)
def skip_role_validation():
    passthrough = dict(side_effect=lambda provided_role, **kwargs: provided_role)
    with (
        patch("sagemaker.train.defaults.resolve_and_validate_role", **passthrough),
        patch("sagemaker.serve.model_builder.resolve_and_validate_role", **passthrough),
    ):
        yield


def _step_arguments(step, session):
    pipeline = Pipeline(
        name="P", parameters=[ENABLED, VALUE], steps=[step], sagemaker_session=session
    )
    return json.loads(pipeline.definition())["Steps"][0]["Arguments"]


def _model_trainer(session, **overrides):
    kwargs = dict(
        training_image=IMAGE,
        role=ROLE,
        base_job_name="job",
        sagemaker_session=session,
        compute=Compute(instance_type="ml.m5.large", instance_count=1),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=60),
        output_data_config=OutputDataConfig(s3_output_path="s3://bucket/output"),
    )
    kwargs.update(overrides)
    return ModelTrainer(**kwargs)


def test_model_trainer_scalar_map_entry_and_list_element(session):
    trainer = _model_trainer(
        session,
        output_data_config=OutputDataConfig(
            s3_output_path="s3://bucket/output", kms_key_id=_optional_value()
        ),
        hyperparameters={"top_k": _optional_value()},
        environment={"KEY": _optional_value()},
        networking=Networking(subnets=[_optional_value()], security_group_ids=["sg-1"]),
    )

    arguments = _step_arguments(TrainingStep(name="Train", step_args=trainer.train()), session)

    assert arguments["OutputDataConfig"]["KmsKeyId"] == EXPECTED
    assert arguments["HyperParameters"]["top_k"] == EXPECTED
    assert arguments["Environment"]["KEY"] == EXPECTED
    assert arguments["VpcConfig"]["Subnets"] == [EXPECTED]


def test_processor_scalar_map_entry_and_list_element(session):
    processor = Processor(
        role=ROLE,
        image_uri=IMAGE,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
        env={"KEY": _optional_value()},
        network_config=NetworkConfig(subnets=[_optional_value()], security_group_ids=["sg-1"]),
    )

    arguments = _step_arguments(ProcessingStep(name="Process", step_args=processor.run()), session)

    assert arguments["Environment"]["KEY"] == EXPECTED
    assert arguments["NetworkConfig"]["VpcConfig"]["Subnets"] == [EXPECTED]


def test_transformer_scalar_and_map_entry(session):
    transformer = Transformer(
        model_name="model",
        instance_count=1,
        instance_type="ml.m5.large",
        output_path="s3://bucket/output",
        output_kms_key=_optional_value(),
        accept=_optional_value(),
        env={"KEY": _optional_value()},
        base_transform_job_name="transform",
        sagemaker_session=session,
    )
    step_args = transformer.transform(data="s3://bucket/input", content_type=_optional_value())

    arguments = _step_arguments(TransformStep(name="Transform", step_args=step_args), session)

    assert arguments["TransformOutput"]["KmsKeyId"] == EXPECTED
    assert arguments["TransformOutput"]["Accept"] == EXPECTED
    assert arguments["TransformInput"]["ContentType"] == EXPECTED
    assert arguments["Environment"]["KEY"] == EXPECTED


def test_tuner_keeps_if_outermost_on_static_hyperparameters(session):
    trainer = _model_trainer(session, hyperparameters={"top_k": _optional_value(), "lr": "0.1"})
    tuner = HyperparameterTuner(
        model_trainer=trainer,
        objective_metric_name="acc",
        hyperparameter_ranges={"depth": IntegerParameter(1, 2)},
        metric_definitions=[{"Name": "acc", "Regex": "acc=(.*)"}],
    )

    arguments = _step_arguments(TuningStep(name="Tune", step_args=tuner.tune()), session)

    static = arguments["TrainingJobDefinition"]["StaticHyperParameters"]
    assert static["top_k"] == EXPECTED
    assert static["lr"] == "0.1"


def test_model_builder_map_entry(session):
    builder = ModelBuilder(
        image_uri=IMAGE,
        s3_model_data_url="s3://bucket/model.tar.gz",
        role_arn=ROLE,
        env_vars={"KEY": _optional_value()},
        sagemaker_session=session,
    )
    step = ModelStep(name="Model", step_args=builder.build(model_name="model"))

    arguments = _step_arguments(step, session)

    assert arguments["PrimaryContainer"]["Environment"]["KEY"] == EXPECTED


def test_else_if_and_nested_if_through_builder(session):
    env = ParameterString("Env", default_value="none")
    kms = If(
        condition=ConditionEquals(left=env, right="prod"),
        then="prod-key",
        else_if=[ElseIf(condition=ConditionEquals(left=env, right="beta"), then="beta-key")],
        else_=If(condition=ConditionEquals(left=ENABLED, right=True), then=VALUE, else_=Omit()),
    )
    trainer = _model_trainer(
        session,
        output_data_config=OutputDataConfig(s3_output_path="s3://bucket/output", kms_key_id=kms),
    )
    pipeline = Pipeline(
        name="P",
        parameters=[env, ENABLED, VALUE],
        steps=[TrainingStep(name="Train", step_args=trainer.train())],
        sagemaker_session=session,
    )

    arguments = json.loads(pipeline.definition())["Steps"][0]["Arguments"]

    assert arguments["OutputDataConfig"]["KmsKeyId"] == {
        "Std:If": {
            "Condition": {
                "Type": "Equals",
                "LeftValue": {"Get": "Parameters.Env"},
                "RightValue": "prod",
            },
            "Then": "prod-key",
            "ElseIf": [
                {
                    "Condition": {
                        "Type": "Equals",
                        "LeftValue": {"Get": "Parameters.Env"},
                        "RightValue": "beta",
                    },
                    "Then": "beta-key",
                }
            ],
            "Else": EXPECTED,
        }
    }


@pytest.mark.parametrize("branch", ["then", "else_if", "else"])
def test_step_property_in_any_branch_creates_dependency(session, branch):
    train = TrainingStep(name="Train", step_args=_model_trainer(session).train())
    model_uri = train.properties.ModelArtifacts.S3ModelArtifacts
    values = {"then": "x", "else_if": "x", "else": "x", branch: model_uri}
    processor = Processor(
        role=ROLE,
        image_uri=IMAGE,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
        env={
            "MODEL": If(
                condition=ConditionEquals(left=ENABLED, right=True),
                then=values["then"],
                else_if=[
                    ElseIf(
                        condition=ConditionEquals(left=ENABLED, right=False),
                        then=values["else_if"],
                    )
                ],
                else_=values["else"],
            )
        },
    )
    process = ProcessingStep(name="Process", step_args=processor.run())
    pipeline = Pipeline(
        name="P",
        parameters=[ENABLED, VALUE],
        steps=[train, process],
        sagemaker_session=session,
    )

    assert PipelineGraph.from_pipeline(pipeline).adjacency_list["Train"] == ["Process"]


def test_bare_omit_is_rejected_when_definition_is_built(session):
    trainer = _model_trainer(
        session,
        output_data_config=OutputDataConfig(s3_output_path="s3://bucket/output", kms_key_id=Omit()),
    )

    with pytest.raises(ValueError, match="complete then or else_ branch"):
        _step_arguments(TrainingStep(name="Train", step_args=trainer.train()), session)
