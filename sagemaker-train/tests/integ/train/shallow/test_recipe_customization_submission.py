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
"""Shallow submission tests for recipe customization.

Covers the knobs that change the *rendered recipe* rather than the plain request
envelope: ``overrides``, an explicit ``recipe`` file, ``sequence_length``,
``DataMixingConfig``, and direct ``trainer.hyperparameters`` mutation.

Why these matter here specifically
----------------------------------
The training backend does not merely shape-check a recipe request -- it filters
candidate recipes after the request validators have already passed and refuses
the job outright with ``"No valid recipes found for the given request"`` when
nothing matches. A customization that renders into an unsatisfiable recipe is
therefore *only* detectable by actually submitting.

Existing coverage of this area is either client-side or expensive:

* ``test_recipe_override_integration.py`` (35 tests) exercises
  ``get_resolved_recipe`` / ``flatten_resolved_recipe`` and never submits, so it
  cannot catch a recipe that resolves locally but the service rejects.
* ``test_rlvr_trainer_integration.py::test_rlvr_trainer_nemotron_with_kl_and_recipe``
  does submit a recipe+overrides combination, but on a 30B model with a
  two-hour poll loop.

These tests close that gap at submission cost: they prove the customized payload
is *accepted*, without asserting anything about the resulting training run.
"""

from __future__ import absolute_import

import tempfile

import pytest
import yaml
from sagemaker.core import shapes
from sagemaker.train.common import TrainingType
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.sft_trainer import SFTTrainer

from .harness import MAX_RUNTIME_IN_SECONDS, assert_submitted, submitted, unique_name

MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"
MODEL_PACKAGE_GROUP = (
    "arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models"
)


def _stopping_condition():
    return shapes.StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS)


def _sft(sagemaker_session, dataset, name, **overrides):
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
    kwargs.update(overrides)
    return SFTTrainer(**kwargs)


class TestRecipeOverrides:
    """``overrides`` is merged into the rendered recipe before submission."""

    def test_training_config_overrides(self, sagemaker_session, train_data_uri):
        """Override common training_config values.

        Values are chosen to stay inside the recipe's accepted ranges: the point
        is to prove overrides survive rendering into an accepted payload, not to
        probe validation bounds (which the negative tests cover).
        """
        trainer = _sft(
            sagemaker_session,
            train_data_uri,
            unique_name("shallow-sft-overrides"),
            overrides={
                "training_config": {
                    "learning_rate": 2e-5,
                    "max_epochs": 1,
                }
            },
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_explicit_recipe_file(self, sagemaker_session, train_data_uri):
        """A caller-supplied recipe YAML must render into an accepted request.

        Mirrors the shape used by the existing Nemotron test, but on a small
        model and without the two-hour poll loop.
        """
        recipe = {"training_config": {"data": {"max_prompt_length": 1024}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            yaml.dump(recipe, handle)
            recipe_path = handle.name

        trainer = _sft(
            sagemaker_session,
            train_data_uri,
            unique_name("shallow-sft-recipe"),
            recipe=recipe_path,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_recipe_and_overrides_together(self, sagemaker_session, train_data_uri):
        """Recipe file plus overrides: the merge order must still yield an
        accepted payload. This is the combination most likely to break, since
        both paths mutate the same rendered document."""
        recipe = {"training_config": {"data": {"max_prompt_length": 1024}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            yaml.dump(recipe, handle)
            recipe_path = handle.name

        trainer = _sft(
            sagemaker_session,
            train_data_uri,
            unique_name("shallow-sft-recipe-ovr"),
            recipe=recipe_path,
            overrides={"training_config": {"max_epochs": 1}},
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_direct_hyperparameter_mutation(self, sagemaker_session, train_data_uri):
        """``trainer.hyperparameters.<field> = ...`` is a documented pattern
        (used by the existing RLVR tests) and must reach the payload intact."""
        trainer = RLVRTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=train_data_uri,
            accept_eula=True,
            sagemaker_session=sagemaker_session,
            base_job_name=unique_name("shallow-rlvr-hpmutate"),
            stopping_condition=_stopping_condition(),
        )
        trainer.hyperparameters.max_epochs = 1

        with submitted(trainer) as job:
            assert_submitted(job)


class TestSequenceLength:
    """``sequence_length`` selects a different recipe variant.

    Only ``4K`` is parametrized. Verified against AWS: for ``MODEL_ID`` the recipe
    catalogue offers exactly one sequence length --

        ValueError: No recipes found with SequenceLength == 16K.
        Available sequence lengths: ['4K']

    so a ``16K`` case would assert a service-side limitation rather than SDK
    behaviour. Left parametrized (rather than inlined) so another value can be
    added when a model in this account supports one.

    Note this field also requires the bundled service model -- see the
    ``bundled_service_model`` fixture in conftest; the public botocore model has
    no ``ServerlessJobConfig.SequenceLength`` yet.
    """

    @pytest.mark.parametrize("sequence_length", ["4K"])
    def test_sequence_length_variants(self, sagemaker_session, train_data_uri, sequence_length):
        trainer = _sft(
            sagemaker_session,
            train_data_uri,
            unique_name(f"shallow-sft-seq{sequence_length}"),
            sequence_length=sequence_length,
        )

        with submitted(trainer) as job:
            assert_submitted(job)


class TestDataMixing:
    """``DataMixingConfig`` is serialized into flat per-category hyperparameters.

    Nova-only, and Nova is us-east-1 in this repo's fixtures, so these use
    ``sagemaker_session_us_east_1`` (inherited from the parent train conftest)
    rather than the default-region session.

    Marked ``us_east_1`` to match the existing marker convention in
    ``sagemaker-train/tox.ini``; the PR-gate job runs us-west-2 only, so these are
    deselected there and run in the us-east-1 job.
    """

    NOVA_MODEL = "nova-textgeneration-lite-v2"

    @pytest.mark.us_east_1
    def test_data_mixing_with_explicit_percentages(
        self, sagemaker_session_us_east_1, nova_train_data_uri
    ):
        """Per-category percentages must sum to 100 client-side and serialize
        into hyperparameters the service accepts."""
        config = DataMixingConfig(
            customer_data_percent=70.0,
            nova_data_percentages={
                "code": 30.0,
                "math": 20.0,
                "planning": 10.0,
                "instruction-following": 10.0,
                "reasoning-instruction-following": 20.0,
                "reasoning-math": 10.0,
            },
        )
        name = unique_name("shallow-sft-datamix")
        trainer = SFTTrainer(
            model=self.NOVA_MODEL,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=nova_train_data_uri,
            accept_eula=True,
            sagemaker_session=sagemaker_session_us_east_1,
            data_mixing_config=config,
            base_job_name=name,
            # The existing data-mixing test sets the recipe name explicitly;
            # keep that so the rendered recipe matches what the service expects.
            overrides={"name": name},
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    @pytest.mark.us_east_1
    def test_data_mixing_recipe_defaults(self, sagemaker_session_us_east_1, nova_train_data_uri):
        """With ``nova_data_percentages=None`` the recipe template's defaults are
        used at submission time -- a different serialization path from the
        explicit case above."""
        config = DataMixingConfig(customer_data_percent=80.0)
        name = unique_name("shallow-sft-datamix-default")
        trainer = SFTTrainer(
            model=self.NOVA_MODEL,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=nova_train_data_uri,
            accept_eula=True,
            sagemaker_session=sagemaker_session_us_east_1,
            data_mixing_config=config,
            base_job_name=name,
            overrides={"name": name},
        )

        with submitted(trainer) as job:
            assert_submitted(job)
