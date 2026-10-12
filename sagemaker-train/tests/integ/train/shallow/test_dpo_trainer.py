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
"""Shallow submission tests for DPOTrainer.

Shallow counterpart of test_dpo_trainer_integration.py. All cases come from
RecipeTrainerCases; DPO takes the same core arguments as SFT and needs no
overrides.

Note DPOTrainer exposes its submitted job as the *public* latest_training_job
where the others use _latest_training_job; the harness resolves both.
"""

from __future__ import absolute_import

from sagemaker.train.dpo_trainer import DPOTrainer

from .recipe_cases import RecipeTrainerCases


class TestDPOTrainerSubmission(RecipeTrainerCases):
    """DPO accepts every shared case with no deviations."""

    TRAINER = DPOTrainer
