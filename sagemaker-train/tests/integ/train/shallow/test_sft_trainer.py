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


class TestSFTTrainerSubmission(RecipeTrainerCases):
    """SFT accepts every shared case with no deviations."""

    TRAINER = SFTTrainer

    @pytest.mark.parametrize("sequence_length", ["4K"])
    def test_sequence_length_is_accepted(self, sagemaker_session, train_data_uri, sequence_length):
        """sequence_length selects a different recipe variant.

        Only 4K is parametrized. Verified against AWS: for MODEL_ID the recipe
        catalogue offers exactly one sequence length --

            ValueError: No recipes found with SequenceLength == 16K.
            Available sequence lengths: ['4K']

        -- so a 16K case would assert a service-side limitation rather than SDK
        behaviour. Left parametrized so another value can be added against a model
        that supports one.

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
