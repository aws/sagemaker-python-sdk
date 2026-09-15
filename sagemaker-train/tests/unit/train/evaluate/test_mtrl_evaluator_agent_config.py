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
"""Unit tests for _resolve_trainer_defaults agent_config dict parsing."""
from __future__ import absolute_import

import pytest
from unittest.mock import MagicMock

from sagemaker.train.evaluate.multi_turn_rl_evaluator import MultiTurnRLEvaluator


AGENT_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-agent-aBcDeFgHiJ"
LAMBDA_ARN = "arn:aws:lambda:us-west-2:123456789012:function:my-agent"
SOURCE_MP_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-mpg/1"
BASE_MODEL = "huggingface-reasoning-qwen3-32b"


def _make_evaluator_with_trainer(agent_config_dict):
    """Create a mock evaluator + trainer to test _resolve_trainer_defaults directly."""
    trainer = MagicMock()
    type(trainer).__name__ = "MultiTurnRLTrainer"
    trainer.output_model_package_arn = SOURCE_MP_ARN
    trainer.model_package_arn = SOURCE_MP_ARN
    trainer.base_model_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/test"
    trainer.base_model_name = BASE_MODEL
    trainer.agent_config = agent_config_dict
    trainer._agent_config = None
    trainer.agent_env = None
    trainer.agent_qualifier = None
    trainer._agent_qualifier = None

    evaluator = MagicMock()
    evaluator.model = trainer
    evaluator.agent_config = None
    evaluator.agent_qualifier = None
    evaluator._source_model_package_arn_cache = None
    evaluator._base_model_arn_cache = None
    evaluator._base_model_name_cache = None
    return evaluator


class TestResolveTrainerAgentConfig:
    """Tests for _resolve_trainer_defaults agent_config dict parsing."""

    def test_nested_bedrock_agent_core_config(self):
        """Test that nested BedrockAgentCoreConfig dict is correctly parsed."""
        evaluator = _make_evaluator_with_trainer(
            {"BedrockAgentCoreConfig": {"AgentRuntimeArn": AGENT_ARN}}
        )
        MultiTurnRLEvaluator._resolve_trainer_defaults(evaluator)
        assert evaluator.agent_config == AGENT_ARN

    def test_nested_custom_agent_lambda_config(self):
        """Test that nested CustomAgentLambdaConfig dict is correctly parsed."""
        evaluator = _make_evaluator_with_trainer(
            {"CustomAgentLambdaConfig": {"LambdaArn": LAMBDA_ARN}}
        )
        MultiTurnRLEvaluator._resolve_trainer_defaults(evaluator)
        assert evaluator.agent_config == LAMBDA_ARN

    def test_flat_dict_fallback_agent_runtime_arn(self):
        """Test that flat AgentRuntimeArn key still works as a fallback."""
        evaluator = _make_evaluator_with_trainer(
            {"AgentRuntimeArn": AGENT_ARN}
        )
        MultiTurnRLEvaluator._resolve_trainer_defaults(evaluator)
        assert evaluator.agent_config == AGENT_ARN

    def test_flat_dict_fallback_lambda_arn(self):
        """Test that flat LambdaArn key still works as a fallback."""
        evaluator = _make_evaluator_with_trainer(
            {"LambdaArn": LAMBDA_ARN}
        )
        MultiTurnRLEvaluator._resolve_trainer_defaults(evaluator)
        assert evaluator.agent_config == LAMBDA_ARN

    def test_customer_provided_agent_config_not_overwritten(self):
        """Test that customer-provided agent_config is not overwritten by trainer."""
        customer_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/customer-agent"
        evaluator = _make_evaluator_with_trainer(
            {"BedrockAgentCoreConfig": {"AgentRuntimeArn": AGENT_ARN}}
        )
        evaluator.agent_config = customer_arn
        MultiTurnRLEvaluator._resolve_trainer_defaults(evaluator)
        assert evaluator.agent_config == customer_arn
