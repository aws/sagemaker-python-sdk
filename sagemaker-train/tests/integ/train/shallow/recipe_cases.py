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
"""Shared submission cases for the recipe trainers.

Every recipe trainer (SFT, DPO, RLVR, RLAIF, ...) accepts the same core arguments
and must clear the same server-side gates, so the cases live here once and each
``test_<trainer>_trainer.py`` subclasses them. That keeps one file per trainer --
matching the existing ``test_sft_trainer_integration.py`` /
``test_dpo_trainer_integration.py`` layout, so the shallow counterpart of a given
deep test is obvious -- without four near-identical copies of the same bodies.

To add a trainer: create ``test_<name>_trainer.py`` with

    class TestFooTrainerSubmission(RecipeTrainerCases):
        TRAINER = FooTrainer

and override the class attributes below only where the trainer genuinely differs.

This module is deliberately NOT named ``test_*``: pytest must not collect
``RecipeTrainerCases`` directly, since it has no ``TRAINER``.
"""

from __future__ import absolute_import

import pytest
from sagemaker.core import shapes
from sagemaker.core.training.configs import TrainingJobCompute
from sagemaker.train.common import TrainingType

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

# An accelerator type is required for the serverful recipe path: these recipes do
# not resolve onto a CPU instance, so unlike the ModelTrainer suite we cannot use
# ml.m5.large here. The job is still stopped immediately, so this holds capacity
# only transiently.
SERVERFUL_INSTANCE_TYPE = "ml.g5.12xlarge"

# Rejection messages can legitimately come from three layers with different
# wording -- SDK-side validation, the public API model, or the training backend --
# so negative tests accept any of these tokens. Still specific enough to catch a
# *wrong* rejection (e.g. an unrelated credentials error).
_MISSING_DATA_TOKENS = (
    "does not exist",
    "ValidationException",
    "ValidationError",
    "S3",
    "not found",
)


def stopping_condition():
    """Short advertised runtime. Never reached -- the job is stopped long before --
    but it bounds the damage if a stop were ever lost."""
    return shapes.StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS)


class RecipeTrainerCases:
    """Submission cases shared by every recipe trainer.

    Subclasses set ``TRAINER`` and, where the trainer differs, the other class
    attributes. Each test submits a real ``CreateTrainingJob``, asserts the
    service returned an ARN, then stops the job -- see ``harness`` for why a
    returned ARN is a strong assertion.
    """

    #: The trainer class under test. Subclasses must set this.
    TRAINER = None

    #: Extra constructor arguments this trainer requires (e.g. RLAIF's reward
    #: model). Merged on top of the shared kwargs.
    EXTRA_KWARGS = {}

    #: Whether the trainer accepts an explicit ``TrainingJobCompute``. RLAIF does
    #: not take a ``compute`` argument at all, so it has no serverful path.
    SUPPORTS_SERVERFUL = True

    #: Whether the trainer accepts ``training_type`` (LoRA vs full). CPT has no
    #: such distinction.
    SUPPORTS_TRAINING_TYPE = True

    def build(self, sagemaker_session, dataset, name, **overrides):
        """Construct the trainer in its minimal accepted configuration.

        ``accept_eula=True`` is required for gated foundation models; without it
        the request is refused before reaching the validation this suite targets.
        """
        kwargs = dict(
            model=MODEL_ID,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=dataset,
            accept_eula=True,
            sagemaker_session=sagemaker_session,
            base_job_name=name,
            stopping_condition=stopping_condition(),
        )
        if self.SUPPORTS_TRAINING_TYPE:
            kwargs["training_type"] = TrainingType.LORA
        kwargs.update(self.EXTRA_KWARGS)
        kwargs.update(overrides)
        return self.TRAINER(**kwargs)

    def name(self, suffix=""):
        """Job name prefixed with the trainer, so a job in the console is
        traceable back to the test that made it."""
        stem = self.TRAINER.__name__.replace("Trainer", "").lower()
        return unique_name(f"shallow-{stem}{suffix}")

    # -- serverless (recipe-derived compute), the default path ---------------

    def test_minimal_request_is_accepted(self, sagemaker_session, train_data_uri):
        """Baseline: the simplest well-formed request is accepted.

        Recipe selection and resource-config generation happen server-side after
        the request validators, so acceptance here is the cheap proof that the
        SDK's recipe payload is still valid.
        """
        trainer = self.build(sagemaker_session, train_data_uri, self.name())

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_with_validation_dataset(self, sagemaker_session, train_data_uri, validation_data_uri):
        """A validation dataset adds a second channel, resolved against S3
        independently of the training channel."""
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-val"),
            validation_dataset=validation_data_uri,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_dataset_passed_to_train_overrides_constructor(self, sagemaker_session, train_data_uri):
        """``train(training_dataset=...)`` overrides the constructor value.

        Worth asserting server-side: if the override were dropped the payload
        would silently reference the wrong data, and only a real run would show
        it.
        """
        trainer = self.build(sagemaker_session, None, self.name("-override"))

        with submitted(trainer, training_dataset=train_data_uri) as job:
            assert_submitted(job)

    def test_explicit_s3_output_path(self, sagemaker_session, train_data_uri, output_path):
        """A caller-specified output location must validate server-side."""
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-output"),
            s3_output_path=output_path,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    # -- serverful (explicit TrainingJobCompute) -----------------------------

    def test_explicit_compute_is_accepted(self, sagemaker_session, train_data_uri):
        """Explicit compute produces a materially different payload from the
        recipe-derived serverless path, including a resource config the backend
        validates against the recipe."""
        if not self.SUPPORTS_SERVERFUL:
            pytest.skip(f"{self.TRAINER.__name__} takes no compute argument")

        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-serverful"),
            compute=TrainingJobCompute(instance_type=SERVERFUL_INSTANCE_TYPE, instance_count=1),
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    # -- negative cases ------------------------------------------------------

    def test_nonexistent_training_dataset_is_rejected(
        self, sagemaker_session, nonexistent_data_uri
    ):
        """Dataset existence is checked against S3 before the job is created.

        The most valuable negative case here: it proves the backend's
        role-assuming validators actually ran rather than being skipped.
        """
        trainer = self.build(sagemaker_session, nonexistent_data_uri, self.name("-bad-data"))

        assert_rejected(trainer, _MISSING_DATA_TOKENS)

    def test_nonexistent_validation_dataset_is_rejected(
        self, sagemaker_session, train_data_uri, nonexistent_data_uri
    ):
        """A valid training set must not mask an invalid validation set."""
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-bad-val"),
            validation_dataset=nonexistent_data_uri,
        )

        assert_rejected(trainer, _MISSING_DATA_TOKENS)
