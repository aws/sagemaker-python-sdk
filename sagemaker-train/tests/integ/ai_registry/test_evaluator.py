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

"""Integration tests for Evaluator."""
import time

import pytest
from sagemaker.ai_registry.evaluator import Evaluator, EvaluatorMethod
from sagemaker.ai_registry.air_constants import HubContentStatus, REWARD_FUNCTION, REWARD_PROMPT


class TestEvaluatorIntegration:
    """Integration tests for Evaluator operations."""

    def test_create_reward_prompt_from_local_file(self, unique_name, sample_prompt_file, cleanup_list):
        """Test creating reward prompt evaluator from local file."""
        evaluator = Evaluator.create(
            name=unique_name,
            type=REWARD_PROMPT,
            source=sample_prompt_file,
            wait=False
        )
        cleanup_list.append(evaluator)
        assert evaluator.name == unique_name
        assert evaluator.type == REWARD_PROMPT
        assert evaluator.arn is not None
        assert evaluator.version is not None

    def test_create_reward_prompt_from_s3_uri(self, unique_name, test_bucket, cleanup_list):
        """Test creating reward prompt evaluator from S3 URI."""
        s3_uri = f"s3://{test_bucket}/prompts/{unique_name}.txt"
        evaluator = Evaluator.create(
            name=unique_name,
            type=REWARD_PROMPT,
            source=s3_uri,
            wait=False
        )
        cleanup_list.append(evaluator)
        assert evaluator.name == unique_name
        assert evaluator.type == REWARD_PROMPT

    def test_create_reward_function_from_lambda_arn(self, unique_name, cleanup_list):
        """Test creating reward function evaluator from existing Lambda ARN."""
        lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
        evaluator = Evaluator.create(
            name=unique_name,
            type=REWARD_FUNCTION,
            source=lambda_arn,
            wait=False
        )
        cleanup_list.append(evaluator)
        assert evaluator.name == unique_name
        assert evaluator.type == REWARD_FUNCTION
        assert evaluator.method == EvaluatorMethod.LAMBDA
        assert evaluator.reference == lambda_arn

    def test_create_reward_function_from_local_code(self, unique_name, sample_lambda_code, test_role, cleanup_list):
        """Test creating reward function evaluator from local code (BYOC)."""
        evaluator = Evaluator.create(
            name=unique_name,
            type=REWARD_FUNCTION,
            source=sample_lambda_code,
            role=test_role,
            wait=False
        )
        cleanup_list.append(evaluator)
        assert evaluator.name == unique_name
        assert evaluator.type == REWARD_FUNCTION
        assert evaluator.method == EvaluatorMethod.BYOC
        assert evaluator.reference is not None

    def test_get_evaluator(self, unique_name, sample_prompt_file, cleanup_list):
        """Test retrieving evaluator by name."""
        created = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=False)
        cleanup_list.append(created)
        retrieved = Evaluator.get(unique_name)
        assert retrieved.name == created.name
        assert retrieved.arn == created.arn
        assert retrieved.type == created.type

    def test_get_all_evaluators(self):
        """Test listing all evaluators."""
        evaluators = list(Evaluator.get_all(max_results=5))
        assert isinstance(evaluators, list)

    def test_get_all_evaluators_filtered_by_type(self):
        """Test listing evaluators filtered by type."""
        evaluators = list(Evaluator.get_all(type=REWARD_PROMPT, max_results=3))
        assert isinstance(evaluators, list)
        for evaluator in evaluators:
            assert evaluator.type == REWARD_PROMPT

    def test_evaluator_refresh(self, unique_name, sample_prompt_file, cleanup_list):
        """Test refreshing evaluator status."""
        evaluator = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=False)
        cleanup_list.append(evaluator)
        time.sleep(3)
        evaluator.refresh()
        assert evaluator.status in [HubContentStatus.IMPORTING.value, HubContentStatus.AVAILABLE.value]

    def test_evaluator_get_versions(self, unique_name, sample_prompt_file, cleanup_list):
        """Test getting evaluator versions."""
        evaluator = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=False)
        cleanup_list.append(evaluator)
        versions = evaluator.get_versions()
        assert len(versions) >= 1
        assert all(isinstance(v, Evaluator) for v in versions)

    def test_evaluator_wait(self, unique_name, sample_prompt_file, cleanup_list):
        """Test waiting for evaluator to be available."""
        evaluator = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=True)
        cleanup_list.append(evaluator)
        time.sleep(3)
        assert evaluator.status == HubContentStatus.AVAILABLE.value

    def test_create_evaluator_version(self, unique_name, sample_prompt_file, cleanup_list):
        """Test creating new evaluator version."""
        Evaluator.delete_by_name(name=unique_name)
        evaluator = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=False)
        # cleanup_list.append(evaluator)
        result = evaluator.create_version(source=sample_prompt_file)
        assert result is True
        Evaluator.delete_by_name(name=unique_name)

    def test_create_reward_prompt_without_source_fails(self, unique_name):
        """Test that creating reward prompt without source fails."""
        with pytest.raises(ValueError, match="source must be provided for RewardPrompt"):
            Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=None)

    def test_create_reward_function_without_source_fails(self, unique_name):
        """Test that creating reward function without source fails."""
        with pytest.raises(ValueError, match="source must be provided for RewardFunction"):
            Evaluator.create(name=unique_name, type=REWARD_FUNCTION, source=None)

    def test_create_unsupported_evaluator_type_fails(self, unique_name, sample_prompt_file):
        """Test that creating unsupported evaluator type fails."""
        with pytest.raises(ValueError, match="Unsupported evaluator type"):
            Evaluator.create(name=unique_name, type="UnsupportedType", source=sample_prompt_file)

    def test_evaluator_repr(self, unique_name, sample_prompt_file, cleanup_list):
        """Test evaluator string representation."""
        evaluator = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=False)
        cleanup_list.append(evaluator)
        repr_str = repr(evaluator)
        assert "Evaluator(" in repr_str
        assert f"name='{unique_name}'" in repr_str
        assert f"type='{REWARD_PROMPT}'" in repr_str

    def test_evaluator_str(self, unique_name, sample_prompt_file, cleanup_list):
        """Test evaluator string conversion."""
        evaluator = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=False)
        cleanup_list.append(evaluator)
        str_repr = str(evaluator)
        assert "Evaluator(" in str_repr

    def test_evaluator_method_enum(self):
        """Test EvaluatorMethod enum values."""
        assert EvaluatorMethod.BYOC.value == "byoc"
        assert EvaluatorMethod.LAMBDA.value == "lambda"

    def test_create_multiple_evaluators_same_session(self, unique_name, sample_prompt_file, sample_lambda_code, cleanup_list):
        """Test creating multiple evaluators in same session."""
        prompt_name = f"{unique_name}-prompt"
        function_name = f"{unique_name}-function"
        
        prompt_evaluator = Evaluator.create(
            name=prompt_name,
            type=REWARD_PROMPT,
            source=sample_prompt_file,
            wait=False
        )
        cleanup_list.append(prompt_evaluator)
        
        function_evaluator = Evaluator.create(
            name=function_name,
            type=REWARD_FUNCTION,
            source=sample_lambda_code,
            wait=False
        )
        cleanup_list.append(function_evaluator)
        
        assert prompt_evaluator.name == prompt_name
        assert function_evaluator.name == function_name
        assert prompt_evaluator.type == REWARD_PROMPT
        assert function_evaluator.type == REWARD_FUNCTION

    def test_evaluator_with_custom_role(self, unique_name, sample_lambda_code, test_role, cleanup_list):
        """Test creating evaluator with custom IAM role."""
        evaluator = Evaluator.create(
            name=unique_name,
            type=REWARD_FUNCTION,
            source=sample_lambda_code,
            role=test_role,
            wait=False
        )
        cleanup_list.append(evaluator)
        assert evaluator.name == unique_name
        assert evaluator.type == REWARD_FUNCTION

    def test_evaluator_lambda_function_creation_idempotent(self, unique_name, sample_lambda_code, test_role, cleanup_list):
        """Test that Lambda function creation is idempotent."""
        # Create first evaluator
        evaluator1 = Evaluator.create(
            name=unique_name,
            type=REWARD_FUNCTION,
            source=sample_lambda_code,
            role=test_role,
            wait=False
        )
        cleanup_list.append(evaluator1)
        
        # Create second evaluator with same name (should update existing Lambda)
        evaluator2 = Evaluator.create(
            name=unique_name,
            type=REWARD_FUNCTION,
            source=sample_lambda_code,
            role=test_role,
            wait=False
        )
        
        assert evaluator1.name == evaluator2.name
        assert evaluator1.type == evaluator2.type

    def test_evaluator_list_operations(self):
        """Test EvaluatorList wrapper functionality."""
        from sagemaker.ai_registry.evaluator import EvaluatorList
        
        # Create mock evaluators
        evaluators = [
            Evaluator(name="test1", type=REWARD_PROMPT),
            Evaluator(name="test2", type=REWARD_FUNCTION)
        ]
        
        evaluator_list = EvaluatorList(evaluators, next_token="token123")
        
        assert len(evaluator_list) == 2
        assert evaluator_list[0].name == "test1"
        assert evaluator_list[1].name == "test2"
        assert evaluator_list.next_token == "token123"
        assert "test1" in str(evaluator_list)
        assert "test2" in repr(evaluator_list)

    def test_evaluator_hub_content_type_property(self, unique_name, sample_prompt_file, cleanup_list):
        """Test hub_content_type property."""
        evaluator = Evaluator.create(name=unique_name, type=REWARD_PROMPT, source=sample_prompt_file, wait=False)
        cleanup_list.append(evaluator)
        assert evaluator.hub_content_type == "JsonDoc"

    def test_evaluator_get_hub_content_type_for_list(self):
        """Test class method for getting hub content type."""
        assert Evaluator._get_hub_content_type_for_list() == "JsonDoc"