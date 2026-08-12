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

from .harness import assert_submitted, submitted
from .recipe_cases import RecipeTrainerCases

# Values match the existing test_rlaif_trainer_integration.py so both suites
# exercise the same already-entitled reward model.
REWARD_MODEL_ID = "openai.gpt-oss-120b-1:0"
REWARD_PROMPT = "Builtin.Summarize"

# Hub-content prompt ARN, the alternative to a Builtin.* prompt name. Same one
# the deep suite uses.
REWARD_PROMPT_ARN = (
    "arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/JsonDoc/rlaif-test-prompt/0.0.1"
)

# An existing fine-tuned model package, used to prove continued fine-tuning
# (model= a model-package ARN rather than a hub model id) still submits.
FINETUNED_MODEL_PACKAGE = (
    "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1"
)


class TestRLAIFTrainerSubmission(RecipeTrainerCases):
    """RLAIF needs a reward model and prompt, and has no serverful path.

    Verified against the SDK: RLAIFTrainer.__init__ takes no compute
    argument at all, so the shared serverful case is skipped rather than expected
    to fail.
    """

    TRAINER = RLAIFTrainer
    EXTRA_KWARGS = {"reward_model_id": REWARD_MODEL_ID, "reward_prompt": REWARD_PROMPT}
    SUPPORTS_SERVERFUL = False

    def test_reward_prompt_as_arn(self, sagemaker_session, train_data_uri):
        """``reward_prompt`` accepts a hub-content ARN as well as a ``Builtin.*``
        name, and the two serialize differently.

        Shallow counterpart of test_rlaif_trainer_with_custom_reward_settings.
        """
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-prompt-arn"),
            reward_prompt=REWARD_PROMPT_ARN,
        )

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_continued_finetuning_from_model_package(self, sagemaker_session, train_data_uri):
        """``model`` as a model-package ARN (continued fine-tuning) must resolve
        and submit, not just a hub model id.

        Shallow counterpart of test_rlaif_trainer_continued_finetuning. Worth
        covering because model resolution takes a different path for an ARN.
        """
        trainer = self.build(
            sagemaker_session,
            train_data_uri,
            self.name("-continued"),
            model=FINETUNED_MODEL_PACKAGE,
        )

        with submitted(trainer) as job:
            assert_submitted(job)
