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
"""Shallow submission tests for RLAIFTrainer.

Shallow counterpart of test_rlaif_trainer_integration.py.
"""

from __future__ import absolute_import

from sagemaker.train.rlaif_trainer import RLAIFTrainer

from .recipe_cases import RecipeTrainerCases

# Values match the existing test_rlaif_trainer_integration.py so both suites
# exercise the same already-entitled reward model.
REWARD_MODEL_ID = "openai.gpt-oss-120b-1:0"
REWARD_PROMPT = "Builtin.Summarize"


class TestRLAIFTrainerSubmission(RecipeTrainerCases):
    """RLAIF needs a reward model and prompt, and has no serverful path.

    Verified against the SDK: RLAIFTrainer.__init__ takes no compute
    argument at all, so the shared serverful case is skipped rather than expected
    to fail.
    """

    TRAINER = RLAIFTrainer
    EXTRA_KWARGS = {"reward_model_id": REWARD_MODEL_ID, "reward_prompt": REWARD_PROMPT}
    SUPPORTS_SERVERFUL = False
