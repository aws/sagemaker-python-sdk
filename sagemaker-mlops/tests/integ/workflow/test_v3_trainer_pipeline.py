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
"""Integration tests for V3 trainer PipelineSession support (SFT/DPO/RLAIF/RLVR).

Tests that V3 fine-tuning trainers work with PipelineSession and TrainingStep,
producing valid pipeline definitions that can be created and executed.

Ref: https://github.com/aws/sagemaker-python-sdk/issues/6163
"""
import json
import os
import pytest
import time
import uuid

import boto3

from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.common import TrainingType
from sagemaker.core.workflow.pipeline_context import PipelineSession, _StepArguments
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.workflow.steps import TrainingStep
from sagemaker.mlops.workflow.pipeline import Pipeline

MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"

# Data files kept small (20 rows each) and repeated in-memory to exceed the
# largest trainer batch size (128 for DPO/RLAIF/RLVR).
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "v3_trainer")


def _load_repeated(filename, repeat):
    with open(os.path.join(_DATA_DIR, filename)) as f:
        lines = [line.strip() for line in f if line.strip()]
    return "\n".join(lines * repeat)


# Repeat to get 140 rows (> max batch_size 128 used by DPO/RLAIF/RLVR trainers)
SFT_TRAINING_DATA = _load_repeated("sft_train.jsonl", 7)
PREFERENCE_TRAINING_DATA = _load_repeated("preference_train.jsonl", 7)


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def pipeline_session():
    return PipelineSession()


@pytest.fixture
def role():
    return get_execution_role()


@pytest.fixture
def model_package_group(sagemaker_session):
    """Create a temporary model package group and delete it after the test.

    Cleanup is thorough: member model packages are deleted first (otherwise
    delete_model_package_group fails on a non-empty group and would leak
    resources in the CI account over time). Cleanup errors are logged rather
    than silently swallowed so real leaks surface in CI output.
    """
    name = f"integ-test-v3-trainer-{uuid.uuid4().hex[:8]}"
    client = sagemaker_session.sagemaker_client
    client.create_model_package_group(
        ModelPackageGroupName=name,
        ModelPackageGroupDescription="Integ test for V3 trainer pipeline",
    )
    yield name

    # Delete any model packages created inside the group before deleting the group itself.
    try:
        paginator = client.get_paginator("list_model_packages")
        for page in paginator.paginate(ModelPackageGroupName=name):
            for pkg in page.get("ModelPackageSummaryList", []):
                pkg_arn = pkg["ModelPackageArn"]
                try:
                    client.delete_model_package(ModelPackageName=pkg_arn)
                except Exception as exc:  # pragma: no cover - best effort
                    print(f"Warning: failed to delete model package {pkg_arn}: {exc}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Warning: failed to list model packages in group {name}: {exc}")

    try:
        client.delete_model_package_group(ModelPackageGroupName=name)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Warning: failed to delete model package group {name}: {exc}")


@pytest.fixture
def sft_training_data_uri(sagemaker_session):
    """SFT training data: messages format (create or update, never delete)."""
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    data_key = "integ-test-v3-trainer/sft/train.jsonl"

    s3_client = boto3.client("s3", region_name=region)
    s3_client.put_object(
        Bucket=bucket, Key=data_key, Body=SFT_TRAINING_DATA.encode()
    )

    return f"s3://{bucket}/{data_key}"


@pytest.fixture
def preference_training_data_uri(sagemaker_session):
    """DPO/RLAIF/RLVR training data: chosen/rejected preference format."""
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    data_key = "integ-test-v3-trainer/preference/train.jsonl"

    s3_client = boto3.client("s3", region_name=region)
    s3_client.put_object(
        Bucket=bucket, Key=data_key, Body=PREFERENCE_TRAINING_DATA.encode()
    )

    return f"s3://{bucket}/{data_key}"


def _assert_valid_pipeline_definition(trainer, step_name, pipeline_session):
    """Shared assertion logic for all trainer definition tests.

    Verifies: train() -> TrainingStep -> step.arguments -> Pipeline.definition()
    produces valid PascalCase JSON with no leaked session/region objects.
    """
    # train() should return _StepArguments (no job launched)
    step_args = trainer.train()
    assert isinstance(step_args, _StepArguments)
    assert step_args.func is not None
    assert step_args.func_args[0] is trainer

    # TrainingStep should accept the step_args
    step = TrainingStep(name=step_name, step_args=step_args)

    # step.arguments should produce valid PascalCase dict
    arguments = step.arguments
    assert isinstance(arguments, dict)
    assert "session" not in arguments, "Leaked boto session object"
    assert "region" not in arguments, "Leaked region string"

    # Keys should be PascalCase
    non_none_keys = [k for k in arguments.keys() if arguments[k] is not None]
    assert any(k[0].isupper() for k in non_none_keys), (
        f"Expected PascalCase keys, got: {non_none_keys}"
    )

    # Tags should have PascalCase Key/Value
    tags = arguments.get("Tags", [])
    for tag in tags:
        assert "Key" in tag and "Value" in tag, f"Tag not PascalCase: {tag}"

    # Pipeline definition should be valid JSON
    pipeline_name = f"integ-test-{step_name.lower()}-{uuid.uuid4().hex[:8]}"
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step],
        sagemaker_session=pipeline_session,
    )
    definition = json.loads(pipeline.definition())
    assert len(definition["Steps"]) == 1
    assert definition["Steps"][0]["Type"] == "Training"
    assert "RoleArn" in definition["Steps"][0]["Arguments"]

    return definition


def _assert_pipeline_create_and_execute(
    trainer_cls,
    step_name,
    sagemaker_session,
    pipeline_session,
    role,
    model_package_group,
    training_data_uri,
    post_init_fn=None,
):
    """Shared execution test logic for all trainers.

    Creates a real pipeline, starts execution, and verifies the training
    step is dispatched (reaches Executing or terminal state). Cleans up after.

    Note: This incurs AWS costs (serverless fine-tuning).
    """
    pipeline_name = f"integ-test-{step_name.lower()}-{uuid.uuid4().hex[:8]}"

    try:
        # Build pipeline
        trainer = trainer_cls(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            training_dataset=training_data_uri,
            model_package_group=model_package_group,
            sagemaker_session=pipeline_session,
            accept_eula=True,
        )

        # Skip validation split -- integ test data is minimal
        trainer.hyperparameters.train_val_split_ratio = 1.0

        if post_init_fn:
            post_init_fn(trainer)

        step_args = trainer.train()
        step = TrainingStep(name=step_name, step_args=step_args)

        pipeline = Pipeline(
            name=pipeline_name,
            steps=[step],
            sagemaker_session=pipeline_session,
        )

        # Create and start
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()

        # Poll until pipeline execution succeeds (full E2E validation)
        timeout = 1800  # 30 minutes for serverless fine-tuning
        start_time = time.time()
        status = "Unknown"

        while time.time() - start_time < timeout:
            execution_desc = execution.describe()
            status = execution_desc["PipelineExecutionStatus"]

            if status == "Succeeded":
                break
            elif status in ("Failed", "Stopped"):
                steps = (
                    sagemaker_session.sagemaker_client
                    .list_pipeline_execution_steps(
                        PipelineExecutionArn=execution_desc[
                            "PipelineExecutionArn"
                        ]
                    )["PipelineExecutionSteps"]
                )
                failures = [
                    f"{s['StepName']}: {s.get('FailureReason', 'Unknown')}"
                    for s in steps if s.get("FailureReason")
                ]
                pytest.fail(
                    f"Pipeline execution {status}.\n"
                    + "\n".join(failures or ["No details available"])
                )

            time.sleep(60)
        else:
            pytest.fail(
                f"Pipeline timed out after {timeout}s. Status: {status}"
            )

    finally:
        # Cleanup pipeline only -- S3 data cleaned by training_data_uri fixture
        try:
            sagemaker_session.sagemaker_client.delete_pipeline(
                PipelineName=pipeline_name
            )
        except Exception:
            pass


class TestSFTTrainerPipelineIntegration:
    """Integration tests for SFTTrainer with PipelineSession."""

    def test_sft_trainer_pipeline_definition_is_valid(
        self, pipeline_session, model_package_group, sft_training_data_uri
    ):
        """SFTTrainer.train() produces a valid pipeline definition."""
        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            training_dataset=sft_training_data_uri,
            model_package_group=model_package_group,
            sagemaker_session=pipeline_session,
            accept_eula=True,
        )
        _assert_valid_pipeline_definition(trainer, "SFTFineTune", pipeline_session)

    def test_sft_trainer_pipeline_create_and_execute(
        self, sagemaker_session, pipeline_session, role, model_package_group,
        sft_training_data_uri,
    ):
        """SFTTrainer pipeline can be created and executed on SageMaker."""
        _assert_pipeline_create_and_execute(
            SFTTrainer,
            "SFTFineTune",
            sagemaker_session,
            pipeline_session,
            role,
            model_package_group,
            sft_training_data_uri,
        )


class TestDPOTrainerPipelineIntegration:
    """Integration tests for DPOTrainer with PipelineSession."""

    def test_dpo_trainer_pipeline_definition_is_valid(
        self, pipeline_session, model_package_group, preference_training_data_uri
    ):
        """DPOTrainer.train() produces a valid pipeline definition."""
        trainer = DPOTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            training_dataset=preference_training_data_uri,
            model_package_group=model_package_group,
            sagemaker_session=pipeline_session,
            accept_eula=True,
        )
        _assert_valid_pipeline_definition(trainer, "DPOFineTune", pipeline_session)

    def test_dpo_trainer_pipeline_create_and_execute(
        self, sagemaker_session, pipeline_session, role, model_package_group,
        preference_training_data_uri,
    ):
        """DPOTrainer pipeline can be created and executed on SageMaker."""
        _assert_pipeline_create_and_execute(
            DPOTrainer,
            "DPOFineTune",
            sagemaker_session,
            pipeline_session,
            role,
            model_package_group,
            preference_training_data_uri,
        )


class TestRLAIFTrainerPipelineIntegration:
    """Integration tests for RLAIFTrainer with PipelineSession."""

    def test_rlaif_trainer_pipeline_definition_is_valid(
        self, pipeline_session, model_package_group, sft_training_data_uri
    ):
        """RLAIFTrainer.train() produces a valid pipeline definition."""
        trainer = RLAIFTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            training_dataset=sft_training_data_uri,
            model_package_group=model_package_group,
            sagemaker_session=pipeline_session,
            accept_eula=True,
        )
        _assert_valid_pipeline_definition(
            trainer, "RLAIFFineTune", pipeline_session
        )


class TestRLVRTrainerPipelineIntegration:
    """Integration tests for RLVRTrainer with PipelineSession."""

    def test_rlvr_trainer_pipeline_definition_is_valid(
        self, pipeline_session, model_package_group, sft_training_data_uri
    ):
        """RLVRTrainer.train() produces a valid pipeline definition."""
        trainer = RLVRTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            training_dataset=sft_training_data_uri,
            model_package_group=model_package_group,
            sagemaker_session=pipeline_session,
            accept_eula=True,
        )
        # RLVR requires a reward signal
        trainer.hyperparameters.preset_reward_function = "prime_code"
        _assert_valid_pipeline_definition(
            trainer, "RLVRFineTune", pipeline_session
        )
