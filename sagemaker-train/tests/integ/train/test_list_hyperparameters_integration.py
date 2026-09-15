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
"""Integration tests for list_hyperparameters utility."""
from __future__ import absolute_import

import pytest
from sagemaker.train.common_utils.finetune_utils import list_hyperparameters
from sagemaker.train.common import FineTuningOptions, TrainingType, CustomizationTechnique


class TestListHyperparametersInteg:
    """Integration tests for list_hyperparameters against live SageMakerPublicHub."""

    def test_sft_lora_returns_expected_params(self):
        """SFT LORA returns a FineTuningOptions with known hyperparameters."""
        hp = list_hyperparameters("meta-textgeneration-llama-3-2-1b-instruct", "SFT", "LORA")

        assert isinstance(hp, FineTuningOptions)
        assert "learning_rate" in hp._specs
        assert "global_batch_size" in hp._specs
        assert "lora_rank" in hp._specs
        assert hp._specs["learning_rate"]["type"] == "float"

    def test_dpo_lora_has_additional_params(self):
        """DPO LORA returns params including adam_beta (not present in SFT)."""
        hp = list_hyperparameters("meta-textgeneration-llama-3-2-1b-instruct", "DPO", "LORA")

        assert isinstance(hp, FineTuningOptions)
        assert "adam_beta" in hp._specs
        assert "learning_rate" in hp._specs

    def test_rlvr_lora_returns_params(self):
        """RLVR LORA returns FineTuningOptions with RL-specific params."""
        hp = list_hyperparameters("meta-textgeneration-llama-3-2-1b-instruct", "RLVR", "LORA")

        assert isinstance(hp, FineTuningOptions)
        assert "learning_rate" in hp._specs
        assert len(hp._specs) > 10

    def test_accepts_enum_values(self):
        """Accepts CustomizationTechnique and TrainingType enums."""
        hp = list_hyperparameters(
            "meta-textgeneration-llama-3-2-1b-instruct",
            CustomizationTechnique.SFT,
            TrainingType.LORA,
        )

        assert isinstance(hp, FineTuningOptions)
        assert "learning_rate" in hp._specs

    def test_get_info_does_not_raise(self):
        """get_info() runs without error on returned object."""
        hp = list_hyperparameters("meta-textgeneration-llama-3-2-1b-instruct", "SFT", "LORA")
        # Should print without raising
        hp.get_info("learning_rate")

    def test_invalid_model_raises(self):
        """Non-existent model raises an error."""
        with pytest.raises(Exception):
            list_hyperparameters("nonexistent-model-xyz-123", "SFT", "LORA")

    def test_invalid_technique_raises(self):
        """Technique not available for model raises an error."""
        # PPO is not available on Llama 3.2
        with pytest.raises(Exception):
            list_hyperparameters("meta-textgeneration-llama-3-2-1b-instruct", "PPO", "LORA")
