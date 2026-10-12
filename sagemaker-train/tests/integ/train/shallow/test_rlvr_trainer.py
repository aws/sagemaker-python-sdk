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
"""Shallow submission tests for RLVRTrainer.

Shallow counterpart of test_rlvr_trainer_integration.py. Adds the
recipe-customization cases, since RLVR is where the existing deep suite exercises
recipe files and overrides (on a 30B model with a two-hour poll loop).
"""

from __future__ import absolute_import

import tempfile

import yaml
from sagemaker.train.rlvr_trainer import RLVRTrainer

from .harness import assert_submitted, submitted
from .recipe_cases import RecipeTrainerCases

# Pre-provisioned reward function in the test account, same one the deep suite
# uses (test_rlvr_trainer_integration.py).
REWARD_FUNCTION_ARN = (
    "arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/JsonDoc/rlvr-test-rf/0.0.1"
)

# The preset the deep suite pairs with an ordinary training dataset on this same
# model (test_rlvr_trainer_lora_complete_workflow).
PRESET_REWARD_FUNCTION = "prime_code"


class TestRLVRTrainerSubmission(RecipeTrainerCases):
    """RLVR accepts every shared case, plus recipe customization."""

    TRAINER = RLVRTrainer

    def build(self, sagemaker_session, dataset, name, **overrides):
        """Add a reward signal, which RLVR requires before it will submit.

        ``RLVRTrainer.train()`` raises ``ValueError`` unless
        ``custom_reward_function`` was passed or
        ``hyperparameters.preset_reward_function`` is set. The cases inherited from
        ``RecipeTrainerCases`` pass neither -- they are about recipe rendering and
        dataset handling, not reward configuration -- so the preset is applied once
        here rather than repeated in each test.

        Skipped when the test supplies its own ``custom_reward_function``, so the
        reward-function variants below still exercise exactly what they name.
        """
        trainer = super().build(sagemaker_session, dataset, name, **overrides)
        if not overrides.get("custom_reward_function"):
            trainer.hyperparameters.preset_reward_function = PRESET_REWARD_FUNCTION
        return trainer

    def test_direct_hyperparameter_mutation(self, sagemaker_session, train_data_uri):
        """trainer.hyperparameters.<field> = ... is a documented pattern (used
        by the existing RLVR tests) and must reach the payload intact."""
        trainer = self.build(sagemaker_session, train_data_uri, self.name("-hpmutate"))
        trainer.hyperparameters.max_epochs = 1

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_kl_and_clipping_hyperparameters(self, sagemaker_session, train_data_uri):
        """RLVR-specific GRPO hyperparameters must reach the payload.

        The deep test (test_rlvr_trainer_nemotron_with_kl_and_recipe) sets these
        five fields on a 30B model behind a two-hour poll loop. They are separate
        recipe fields, not one flag, so setting only max_epochs -- as
        test_direct_hyperparameter_mutation does -- would not prove they serialize.
        """
        trainer = self.build(sagemaker_session, train_data_uri, self.name("-kl"))
        trainer.hyperparameters.use_kl_loss = True
        trainer.hyperparameters.kl_loss_coef = 0.05
        trainer.hyperparameters.clip_ratio = 0.2
        trainer.hyperparameters.max_epochs = 1

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_explicit_recipe_file(self, sagemaker_session, train_data_uri):
        """A caller-supplied recipe YAML must render into an accepted request.

        Mirrors the shape used by the existing Nemotron test, but on a small model
        and without the poll loop.
        """
        recipe = {"training_config": {"data": {"max_prompt_length": 1024}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            yaml.dump(recipe, handle)
            recipe_path = handle.name

        trainer = self.build(
            sagemaker_session, train_data_uri, self.name("-recipe"), recipe=recipe_path
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_recipe_and_overrides_together(self, sagemaker_session, train_data_uri):
        """Recipe file plus overrides: the merge order must still yield an accepted
        payload. The combination most likely to break, since both paths mutate the
        same rendered document."""
        recipe = {"training_config": {"data": {"max_prompt_length": 1024}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            yaml.dump(recipe, handle)
            recipe_path = handle.name

        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-recipe-ovr"),
            recipe=recipe_path,
            overrides={"training_config": {"max_epochs": 1}},
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_training_config_overrides(self, sagemaker_session, train_data_uri):
        """Override common training_config values.

        Values stay inside the recipe's accepted ranges: the point is that
        overrides survive rendering into an accepted payload, not to probe
        validation bounds (the negative cases cover that).
        """
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-overrides"),
            overrides={"training_config": {"learning_rate": 2e-5, "max_epochs": 1}},
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    # -- reward-function variants -------------------------------------------
    #
    # RLVR is the only trainer with a pluggable reward function, and the deep
    # suite covers three distinct forms. Each changes what the SDK puts in the
    # payload, so each needs its own acceptance case.

    def test_custom_reward_function_arn(self, sagemaker_session, reward_scored_data_uri):
        """A hub-content reward-function ARN must be accepted.

        Shallow counterpart of test_rlvr_trainer_with_custom_reward_function.
        """
        trainer = self.build(
            sagemaker_session,
            reward_scored_data_uri,
            self.name("-rf-arn"),
            custom_reward_function=REWARD_FUNCTION_ARN,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_custom_reward_function_lambda_arn(
        self, sagemaker_session, reward_scored_data_uri, reward_lambda_arn
    ):
        """A Lambda ARN as the reward function auto-creates an AI Registry
        Evaluator, then submits.

        Shallow counterpart of
        test_rlvr_trainer_with_lambda_arn_auto_creates_evaluator. The Lambda is
        reused from the parent train conftest rather than created here, and the
        test skips if it is unavailable.
        """
        trainer = self.build(
            sagemaker_session,
            reward_scored_data_uri,
            self.name("-rf-lambda"),
            custom_reward_function=reward_lambda_arn,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_custom_reward_function_evaluator_object(
        self, sagemaker_session, reward_scored_data_uri, reward_evaluator
    ):
        """A pre-created ``Evaluator`` object as the reward function must
        serialize to the same accepted payload as an ARN.

        Shallow counterpart of test_rlvr_trainer_with_evaluator_object. Skips when
        the evaluator is absent rather than creating one.
        """
        trainer = self.build(
            sagemaker_session,
            reward_scored_data_uri,
            self.name("-rf-obj"),
            custom_reward_function=reward_evaluator,
        )

        with submitted(trainer) as job:
            assert_submitted(job)
