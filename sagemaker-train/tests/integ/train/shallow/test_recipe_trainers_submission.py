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
"""Shallow submission tests for the recipe trainers (SFT / DPO / RLVR / CPT).

These trainers do far more request-shaping than ``ModelTrainer``: they resolve a
foundation model, select and render a training recipe, derive a resource config
from it, and translate datasets into channels. All of that lands in the
``CreateTrainingJob`` payload, and the training backend validates it -- including
recipe *acceptance*, which is checked after the request validators and rejects
with ``"No valid recipes found for the given request"``.

That makes submit-then-stop unusually valuable for this family: a recipe
regression is invisible to unit tests (which mock the service) and today is only
caught by a full, expensive training run.

Serverless (recipe-selected compute) is the default path. Where a test pins
``TrainingJobCompute`` it is asserting the serverful path specifically, since the
two produce materially different payloads.
"""

from __future__ import absolute_import

import os

import pytest
from sagemaker.core import shapes
from sagemaker.core.training.configs import HyperPodCompute, TrainingJobCompute
from sagemaker.train.common import TrainingType
from sagemaker.train.cpt_trainer import CPTTrainer
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.sft_trainer import SFTTrainer

from .harness import (
    MAX_RUNTIME_IN_SECONDS,
    assert_rejected,
    assert_submitted,
    submitted,
    unique_name,
)

# Small, publicly available instruct model. Kept small deliberately: these tests
# never train, so model size only affects how long recipe/artifact resolution
# takes during submission.
MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"

# Reused from the existing dry-run suite so both suites exercise the same
# already-provisioned model package group rather than each needing their own.
MODEL_PACKAGE_GROUP = (
    "arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models"
)

# An accelerator type is required for the serverful recipe path: the recipes for
# these trainers will not resolve onto a CPU instance, so unlike the
# ModelTrainer suite we cannot use ml.m5.large here. The job is still stopped
# immediately, so this requests capacity only transiently.
SERVERFUL_INSTANCE_TYPE = "ml.g5.12xlarge"

# RLAIF requires a reward model and prompt; without them the request is refused
# before it reaches the validation this suite cares about. Values match the
# existing test_rlaif_trainer_integration.py so both suites exercise the same
# already-entitled reward model.
RLAIF_REWARD_MODEL_ID = "openai.gpt-oss-120b-1:0"
RLAIF_REWARD_PROMPT = "Builtin.Summarize"

# Per-trainer extra constructor arguments. Everything else is shared, which is
# what lets these four trainers be covered by one parametrized body instead of
# four near-identical files.
_TRAINER_EXTRA_KWARGS = {
    "RLAIFTrainer": {
        "reward_model_id": RLAIF_REWARD_MODEL_ID,
        "reward_prompt": RLAIF_REWARD_PROMPT,
    },
}

# Every recipe trainer takes the same core arguments, so the per-trainer test
# bodies stay a single call.
RECIPE_TRAINERS = [
    pytest.param(SFTTrainer, id="sft"),
    pytest.param(DPOTrainer, id="dpo"),
    pytest.param(RLVRTrainer, id="rlvr"),
    pytest.param(RLAIFTrainer, id="rlaif"),
]

# Subset that accepts an explicit TrainingJobCompute. RLAIFTrainer takes no
# ``compute`` argument at all (verified against the SDK), so it has no serverful
# path and is excluded rather than being expected to fail.
SERVERFUL_CAPABLE_TRAINERS = [t for t in RECIPE_TRAINERS if t.values[0] is not RLAIFTrainer]


def _stopping_condition():
    return shapes.StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS)


def _trainer(trainer_cls, sagemaker_session, dataset, name, **overrides):
    """Build a recipe trainer in its minimal accepted configuration.

    ``accept_eula=True`` is required for gated foundation models; without it the
    request is refused before it reaches the interesting validation.

    Trainer-specific required arguments come from ``_TRAINER_EXTRA_KWARGS`` so
    adding another trainer to ``RECIPE_TRAINERS`` stays a two-line change.
    """
    kwargs = dict(
        model=MODEL_ID,
        training_type=TrainingType.LORA,
        model_package_group=MODEL_PACKAGE_GROUP,
        training_dataset=dataset,
        accept_eula=True,
        sagemaker_session=sagemaker_session,
        base_job_name=name,
        stopping_condition=_stopping_condition(),
    )
    kwargs.update(_TRAINER_EXTRA_KWARGS.get(trainer_cls.__name__, {}))
    kwargs.update(overrides)
    return trainer_cls(**kwargs)


class TestServerlessSubmission:
    """The default path: compute is derived from the selected recipe.

    Recipe selection and resource-config generation happen server-side after the
    request validators, so acceptance here is the only cheap proof that the
    SDK's recipe payload is still valid.
    """

    @pytest.mark.parametrize("trainer_cls", RECIPE_TRAINERS)
    def test_minimal_request_is_accepted(self, trainer_cls, sagemaker_session, train_data_uri):
        name = unique_name(f"shallow-{trainer_cls.__name__.lower()}")
        trainer = _trainer(trainer_cls, sagemaker_session, train_data_uri, name)

        with submitted(trainer) as job:
            assert_submitted(job)

    @pytest.mark.parametrize("trainer_cls", RECIPE_TRAINERS)
    def test_with_validation_dataset(
        self, trainer_cls, sagemaker_session, train_data_uri, validation_data_uri
    ):
        """A validation dataset adds a second channel, which is resolved against
        S3 independently of the training channel."""
        name = unique_name(f"shallow-{trainer_cls.__name__.lower()}-val")
        trainer = _trainer(
            trainer_cls,
            sagemaker_session,
            train_data_uri,
            name,
            validation_dataset=validation_data_uri,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    @pytest.mark.parametrize("trainer_cls", RECIPE_TRAINERS)
    def test_datasets_passed_to_train_override_constructor(
        self, trainer_cls, sagemaker_session, train_data_uri
    ):
        """``train(training_dataset=...)`` overrides the constructor value.

        Worth asserting server-side: if the override were dropped, the payload
        would silently reference the wrong data and only a real run would reveal
        it.
        """
        name = unique_name(f"shallow-{trainer_cls.__name__.lower()}-override")
        trainer = _trainer(trainer_cls, sagemaker_session, None, name)

        with submitted(trainer, training_dataset=train_data_uri) as job:
            assert_submitted(job)

    # Only LORA is parametrized. Verified against AWS: for MODEL_ID there is no
    # serverless (SMTJ) recipe for full fine-tuning --
    #   ValueError: No recipes found with Smtj for technique: SFT,
    #   training_type:TrainingType.FULL
    # so a FULL case here would assert a recipe-catalogue limitation rather than
    # SDK behaviour. Kept parametrized so FULL can be re-added against a model
    # that supports it, rather than the distinction being silently dropped.
    @pytest.mark.parametrize("training_type", [TrainingType.LORA])
    def test_training_types(self, sagemaker_session, train_data_uri, training_type):
        """LoRA and full fine-tuning select different recipes, so each must be
        independently accepted."""
        suffix = str(getattr(training_type, "value", training_type)).lower()
        name = unique_name(f"shallow-sft-{suffix}")
        trainer = _trainer(
            SFTTrainer,
            sagemaker_session,
            train_data_uri,
            name,
            training_type=training_type,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    @pytest.mark.gpu_intensive
    def test_cpt_trainer_is_accepted(self, sagemaker_session, train_data_uri):
        """Continued pre-training, which submits only via HyperPod.

        Kept out of RECIPE_TRAINERS because its constructor genuinely differs:
        verified against the SDK, ``CPTTrainer`` accepts no ``training_type``
        (there is no LoRA/full distinction for continued pre-training) and its
        ``compute`` is ``HyperPodCompute``-only.

        Marked ``gpu_intensive`` and skipped unless a cluster is configured. CPT
        refuses to submit without one --

            ValueError: CPT requires HyperPod compute.
            Pass compute=HyperPodCompute(...) when creating the CPTTrainer.

        -- and HyperPod submits to a pre-provisioned cluster rather than through
        CreateTrainingJob, so there is nothing this suite can create on demand.
        Written in the shallow style anyway so it becomes gate-eligible by
        dropping one marker once a cluster exists in the PR account.
        """
        cluster_name = os.environ.get("SHALLOW_HYPERPOD_CLUSTER")
        if not cluster_name:
            pytest.skip("CPT requires HyperPod; set SHALLOW_HYPERPOD_CLUSTER to run")

        name = unique_name("shallow-cpt")
        trainer = CPTTrainer(
            model=MODEL_ID,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=train_data_uri,
            accept_eula=True,
            sagemaker_session=sagemaker_session,
            base_job_name=name,
            stopping_condition=_stopping_condition(),
            compute=HyperPodCompute(cluster_name=cluster_name),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestServerfulSubmission:
    """Explicit ``TrainingJobCompute`` produces a materially different payload
    from the recipe-derived serverless path, including a resource config the
    backend validates against the recipe.

    RLAIF is absent from this class on purpose: verified against the SDK,
    ``RLAIFTrainer.__init__`` has no ``compute`` parameter at all, so it has no
    serverful path to exercise. It is still covered by every serverless case in
    ``TestServerlessSubmission``.
    """

    @pytest.mark.parametrize("trainer_cls", SERVERFUL_CAPABLE_TRAINERS)
    def test_explicit_compute_is_accepted(self, trainer_cls, sagemaker_session, train_data_uri):
        name = unique_name(f"shallow-{trainer_cls.__name__.lower()}-serverful")
        trainer = _trainer(
            trainer_cls,
            sagemaker_session,
            train_data_uri,
            name,
            compute=TrainingJobCompute(instance_type=SERVERFUL_INSTANCE_TYPE, instance_count=1),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestOutputAndTracking:
    """Output location and MLflow tracking are validated server-side."""

    def test_explicit_s3_output_path(self, sagemaker_session, train_data_uri, output_path):
        name = unique_name("shallow-sft-output")
        trainer = _trainer(
            SFTTrainer,
            sagemaker_session,
            train_data_uri,
            name,
            s3_output_path=output_path,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_disable_output_compression(self, sagemaker_session, train_data_uri):
        """Uncompressed output changes the OutputDataConfig the SDK sends."""
        name = unique_name("shallow-sft-nocompress")
        trainer = _trainer(
            SFTTrainer,
            sagemaker_session,
            train_data_uri,
            name,
            disable_output_compression=True,
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestRejectedRecipeRequests:
    """Negative cases specific to the recipe path.

    These matter more here than for ``ModelTrainer``: recipe resolution is the
    part of the payload most likely to drift, and an over-permissive change would
    otherwise still yield a green suite.
    """

    def test_nonexistent_training_dataset_is_rejected(
        self, sagemaker_session, nonexistent_data_uri
    ):
        """Dataset existence is checked against S3 before the job is created."""
        trainer = _trainer(
            SFTTrainer,
            sagemaker_session,
            nonexistent_data_uri,
            unique_name("shallow-sft-bad-data"),
        )

        assert_rejected(
            trainer,
            ("does not exist", "ValidationException", "ValidationError", "S3", "not found"),
        )

    def test_nonexistent_validation_dataset_is_rejected(
        self, sagemaker_session, train_data_uri, nonexistent_data_uri
    ):
        """A valid training set must not mask an invalid validation set."""
        trainer = _trainer(
            SFTTrainer,
            sagemaker_session,
            train_data_uri,
            unique_name("shallow-sft-bad-val"),
            validation_dataset=nonexistent_data_uri,
        )

        assert_rejected(
            trainer,
            ("does not exist", "ValidationException", "ValidationError", "S3", "not found"),
        )

    def test_unknown_model_is_rejected(self, sagemaker_session, train_data_uri):
        """Model resolution must fail for a model that does not exist.

        Guards the JumpStart/hub lookup that turns ``model`` into a concrete
        artifact URI in the payload.
        """
        trainer_kwargs = dict(
            model="definitely-not-a-real-model-id-4b91c7",
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=train_data_uri,
            accept_eula=True,
            sagemaker_session=sagemaker_session,
            base_job_name=unique_name("shallow-sft-bad-model"),
        )

        # Model resolution can fail either while constructing the trainer or at
        # submit time depending on how the id is interpreted, so both are allowed
        # here; what matters is that an unknown model never reaches the service.
        with pytest.raises(Exception) as excinfo:
            trainer = SFTTrainer(**trainer_kwargs)
            trainer.train(wait=False)

        message = str(excinfo.value)
        assert any(
            token in message
            for token in (
                "model",
                "Model",
                "not found",
                "does not exist",
                "ResourceNotFound",
                "ValidationException",
                "ValidationError",
            )
        ), f"unexpected rejection reason: {message}"

    def test_invalid_instance_type_is_rejected(self, sagemaker_session, train_data_uri):
        """A nonexistent instance type must be refused on the serverful path."""
        trainer = _trainer(
            SFTTrainer,
            sagemaker_session,
            train_data_uri,
            unique_name("shallow-sft-bad-instance"),
            compute=TrainingJobCompute(instance_type="ml.nonexistent.24xlarge", instance_count=1),
        )

        assert_rejected(
            trainer,
            (
                "instance",
                "Instance",
                "not supported",
                "ValidationException",
                "ValidationError",
            ),
        )
