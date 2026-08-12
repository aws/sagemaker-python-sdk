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


class TestRLVRTrainerSubmission(RecipeTrainerCases):
    """RLVR accepts every shared case, plus recipe customization."""

    TRAINER = RLVRTrainer

    def test_direct_hyperparameter_mutation(self, sagemaker_session, train_data_uri):
        """trainer.hyperparameters.<field> = ... is a documented pattern (used
        by the existing RLVR tests) and must reach the payload intact."""
        trainer = self.build(sagemaker_session, train_data_uri, self.name("-hpmutate"))
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
