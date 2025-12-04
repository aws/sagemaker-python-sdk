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
"""Unit tests for domain-id tagging in Evaluator."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.ai_registry.evaluator import Evaluator, EvaluatorMethod


class TestEvaluatorDomainId:
    """Test domain-id is added to SearchKeywords when available."""
    
    @patch('sagemaker.core.helper.session_helper.Session')
    @patch('sagemaker.ai_registry.evaluator._get_current_domain_id')
    @patch('sagemaker.ai_registry.evaluator.AIRHub')
    def test_domain_id_added_when_available(
        self, mock_air_hub, mock_get_domain_id, mock_session
    ):
        """Test that domain-id is added to tags when available."""
        # Setup mocks
        mock_domain_id = "d-test123456"
        mock_get_domain_id.return_value = mock_domain_id
        mock_session.return_value = Mock()
        
        # Mock AIRHub methods
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value={
            'HubContentName': 'test-evaluator',
            'HubContentArn': 'arn:aws:sagemaker:us-west-2:123:hub-content/test',
            'HubContentVersion': '1.0.0',
            'HubContentStatus': 'Available',
            'CreationTime': '2024-01-01',
            'LastModifiedTime': '2024-01-01',
            'HubContentDocument': '{"SubType": "AWS/Evaluator", "JsonContent": "{}"}'
        })
        
        # Create evaluator
        with patch('sagemaker.ai_registry.evaluator.Evaluator.wait'):
            with patch('sagemaker.ai_registry.evaluator.Evaluator._handle_reward_function', return_value=(EvaluatorMethod.LAMBDA, 'arn:aws:lambda:...')):
                evaluator = Evaluator.create(
                    name="test-evaluator",
                    type="RewardFunction",
                    source="arn:aws:lambda:us-west-2:123:function:test"
                )
        
        # Verify import_hub_content was called
        assert mock_air_hub.import_hub_content.called
        
        # Get the tags argument
        call_args = mock_air_hub.import_hub_content.call_args
        tags = call_args[1]['tags']
        
        # Verify domain-id is in tags
        assert any(tag[0] == '@domain' and tag[1] == mock_domain_id for tag in tags)
        
    @patch('sagemaker.core.helper.session_helper.Session')
    @patch('sagemaker.ai_registry.evaluator._get_current_domain_id')
    @patch('sagemaker.ai_registry.evaluator.AIRHub')
    def test_domain_id_not_added_when_unavailable(
        self, mock_air_hub, mock_get_domain_id, mock_session
    ):
        """Test that domain-id is not added when unavailable (non-Studio)."""
        # Setup mocks - domain_id returns None
        mock_get_domain_id.return_value = None
        mock_session.return_value = Mock()
        
        # Mock AIRHub methods
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value={
            'HubContentName': 'test-evaluator',
            'HubContentArn': 'arn:aws:sagemaker:us-west-2:123:hub-content/test',
            'HubContentVersion': '1.0.0',
            'HubContentStatus': 'Available',
            'CreationTime': '2024-01-01',
            'LastModifiedTime': '2024-01-01',
            'HubContentDocument': '{"SubType": "AWS/Evaluator", "JsonContent": "{}"}'
        })
        
        # Create evaluator
        with patch('sagemaker.ai_registry.evaluator.Evaluator.wait'):
            with patch('sagemaker.ai_registry.evaluator.Evaluator._handle_reward_function', return_value=(EvaluatorMethod.LAMBDA, 'arn:aws:lambda:...')):
                evaluator = Evaluator.create(
                    name="test-evaluator",
                    type="RewardFunction",
                    source="arn:aws:lambda:us-west-2:123:function:test"
                )
        
        # Verify import_hub_content was called
        assert mock_air_hub.import_hub_content.called
        
        # Get the tags argument
        call_args = mock_air_hub.import_hub_content.call_args
        tags = call_args[1]['tags']
        
        # Verify domain-id is NOT in tags
        assert not any(tag[0] == '@domain' for tag in tags)
