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
"""Shallow submission tests for SFTTrainer.

Shallow counterpart of test_sft_trainer_integration.py: submits a real
CreateTrainingJob, asserts the returned ARN, then stops the job. Asserts
acceptance only, never training behaviour.

The shared cases come from RecipeTrainerCases; SFT-specific ones are added
below.
"""

from __future__ import absolute_import

import pytest
from sagemaker.train.sft_trainer import SFTTrainer

from .harness import assert_submitted, submitted
from .recipe_cases import RecipeTrainerCases

_MISSING = object()


def _find_in_nested(obj, key):
    """Return the first value for ``key`` found anywhere in a nested dict/list.

    Returns the ``_MISSING`` sentinel if the key is absent. Used so override
    assertions do not pin a service-owned recipe layout that changes over time.
    """
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_in_nested(value, key)
            if found is not _MISSING:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_in_nested(item, key)
            if found is not _MISSING:
                return found
    return _MISSING


class TestSFTTrainerSubmission(RecipeTrainerCases):
    """SFT accepts every shared case with no deviations."""

    TRAINER = SFTTrainer

    @pytest.mark.parametrize("sequence_length", ["16K"])
    def test_sequence_length_is_accepted(self, sagemaker_session, train_data_uri, sequence_length):
        """sequence_length selects a different recipe variant.

        The value must be a sequence length the MODEL_ID catalogue offers *for
        this trainer's training_type*. The shared cases build with
        TrainingType.LORA, and for MODEL_ID only the 16K SFT recipe is LoRA (the
        8K SFT recipe is full fine-tuning), so 16K is the LoRA-compatible choice.
        Requesting an absent/incompatible length makes the SDK raise before
        submission (e.g. ``No recipes found with SequenceLength == 4K``).

        The catalogue is service-owned and has changed over time (it previously
        offered only '4K'), so this value tracks what the service advertises and
        may need updating again.

        Also requires the bundled service model: the public botocore model has no
        ServerlessJobConfig.SequenceLength (see the bundled_service_model
        fixture in conftest).
        """
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name(f"-seq{sequence_length}"),
            sequence_length=sequence_length,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_disable_output_compression(self, sagemaker_session, train_data_uri):
        """Uncompressed output changes the OutputDataConfig the SDK sends."""
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-nocompress"),
            disable_output_compression=True,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_recipe_overrides_are_accepted(self, sagemaker_session, train_data_uri):
        """``overrides`` is merged into the rendered recipe before submission.

        Shallow counterpart of the override half of
        ``test_sft_trainer_serverful_smtj.py``, which applies
        ``overrides={"training_config": {"max_epochs": 1}}`` and then asserts the
        merge via ``get_resolved_recipe()``.

        Two distinct things are checked, and both matter:

        * ``get_resolved_recipe()`` -- the override reached the *rendered recipe*.
          This is client-side, so it is cheap and exact, and it is what the deep
          test asserts.
        * submission -- the resulting payload is still *accepted* by the service.
          Recipe filtering runs after the request validators and rejects with
          "No valid recipes found for the given request", so a bad merge is only
          caught here.
        """
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-overrides"),
            overrides={"training_config": {"max_epochs": 1, "learning_rate": 2e-5}},
        )

        resolved = trainer.get_resolved_recipe()

        # The ``overrides`` keys (max_epochs, learning_rate) are mapped into the
        # rendered recipe's own fields rather than echoed verbatim: verified
        # against AWS, override {"training_config": {"max_epochs": 1}} lands at
        # training_config.trainer.total_epochs and learning_rate at
        # training_config.optim.lr. (Overrides previously surfaced under
        # training_config.training_args, a key that no longer exists.) Search by
        # the rendered key name anywhere in the resolved recipe so the assertion
        # does not pin the exact nesting. The recipe defaults for this model
        # differ from the overridden values, so finding 1 / 2e-5 proves the
        # override was applied rather than coinciding with a default.
        assert (
            _find_in_nested(resolved, "total_epochs") == 1
        ), f"max_epochs override did not reach the resolved recipe: {resolved}"
        assert (
            _find_in_nested(resolved, "lr") == 2e-5
        ), f"learning_rate override did not reach the resolved recipe: {resolved}"

        with submitted(trainer) as job:
            assert_submitted(job)
