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
"""Unit tests for domain-id tagging in DataSet."""
import json
import tempfile
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique


# Sample RLVR format dataset (GSM8K style)
SAMPLE_DATASET = {
    "data_source": "openai/gsm8k",
    "prompt": [{"content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\".", "role": "user"}],
    "ability": "math",
    "reward_model": {"ground_truth": "72", "style": "rule"},
    "extra_info": {"answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72", "index": 0, "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "split": "train"}
}


@pytest.fixture
def sample_dataset_file():
    """Create a temporary JSONL file with sample dataset."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        json.dump(SAMPLE_DATASET, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestDataSetDomainId:
    """Test domain-id is added to SearchKeywords when available."""
    
    @patch('sagemaker.core.helper.session_helper.Session')
    @patch('sagemaker.ai_registry.dataset._get_current_domain_id')
    @patch('sagemaker.ai_registry.dataset.AIRHub')
    @patch('sagemaker.train.defaults.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.defaults.TrainDefaults.get_role')
    def test_domain_id_added_when_available(
        self, mock_get_role, mock_get_session, mock_air_hub, mock_get_domain_id, mock_session, sample_dataset_file
    ):
        """Test that domain-id is added to tags when available."""
        # Setup mocks
        mock_domain_id = "d-test123456"
        mock_get_domain_id.return_value = mock_domain_id
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"
        
        # Mock AIRHub methods
        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value={
            'HubContentName': 'test-dataset',
            'HubContentArn': 'arn:aws:sagemaker:us-west-2:123:hub-content/test',
            'HubContentVersion': '1.0.0',
            'HubContentStatus': 'Available',
            'CreationTime': '2024-01-01',
            'LastModifiedTime': '2024-01-01',
            'HubContentDocument': '{"DatasetS3Bucket": "bucket", "DatasetS3Prefix": "prefix"}'
        })
        
        # Create dataset with real file
        with patch('sagemaker.ai_registry.dataset.DataSet.wait'):
            dataset = DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                customization_technique=CustomizationTechnique.SFT
            )
        
        # Verify import_hub_content was called
        assert mock_air_hub.import_hub_content.called
        
        # Get the tags argument
        call_args = mock_air_hub.import_hub_content.call_args
        tags = call_args[1]['tags']
        
        # Verify domain-id is in tags
        assert any(tag[0] == '@domain' and tag[1] == mock_domain_id for tag in tags)
        
    @patch('sagemaker.core.helper.session_helper.Session')
    @patch('sagemaker.ai_registry.dataset._get_current_domain_id')
    @patch('sagemaker.ai_registry.dataset.AIRHub')
    @patch('sagemaker.train.defaults.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.defaults.TrainDefaults.get_role')
    def test_domain_id_not_added_when_unavailable(
        self, mock_get_role, mock_get_session, mock_air_hub, mock_get_domain_id, mock_session, sample_dataset_file
    ):
        """Test that domain-id is not added when unavailable (non-Studio)."""
        # Setup mocks - domain_id returns None
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"
        
        # Mock AIRHub methods
        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value={
            'HubContentName': 'test-dataset',
            'HubContentArn': 'arn:aws:sagemaker:us-west-2:123:hub-content/test',
            'HubContentVersion': '1.0.0',
            'HubContentStatus': 'Available',
            'CreationTime': '2024-01-01',
            'LastModifiedTime': '2024-01-01',
            'HubContentDocument': '{"DatasetS3Bucket": "bucket", "DatasetS3Prefix": "prefix"}'
        })
        
        # Create dataset with real file
        with patch('sagemaker.ai_registry.dataset.DataSet.wait'):
            dataset = DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                customization_technique=CustomizationTechnique.SFT
            )
        
        # Verify import_hub_content was called
        assert mock_air_hub.import_hub_content.called
        
        # Get the tags argument
        call_args = mock_air_hub.import_hub_content.call_args
        tags = call_args[1]['tags']
        
        # Verify domain-id is NOT in tags
        assert not any(tag[0] == '@domain' for tag in tags)
        
    @patch('sagemaker.core.helper.session_helper.Session')
    @patch('sagemaker.ai_registry.dataset._get_current_domain_id')
    @patch('sagemaker.ai_registry.dataset.AIRHub')
    @patch('sagemaker.train.defaults.TrainDefaults.get_sagemaker_session')
    @patch('sagemaker.train.defaults.TrainDefaults.get_role')
    def test_domain_id_added_without_customization_technique(
        self, mock_get_role, mock_get_session, mock_air_hub, mock_get_domain_id, mock_session, sample_dataset_file
    ):
        """Test that domain-id is added even without customization_technique."""
        # Setup mocks
        mock_domain_id = "d-test789"
        mock_get_domain_id.return_value = mock_domain_id
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"
        
        # Mock AIRHub methods
        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value={
            'HubContentName': 'test-dataset',
            'HubContentArn': 'arn:aws:sagemaker:us-west-2:123:hub-content/test',
            'HubContentVersion': '1.0.0',
            'HubContentStatus': 'Available',
            'CreationTime': '2024-01-01',
            'LastModifiedTime': '2024-01-01',
            'HubContentDocument': '{"DatasetS3Bucket": "bucket", "DatasetS3Prefix": "prefix"}'
        })
        
        # Create dataset WITHOUT customization_technique using real file
        with patch('sagemaker.ai_registry.dataset.DataSet.wait'):
            dataset = DataSet.create(
                name="test-dataset",
                source=sample_dataset_file
                # No customization_technique
            )
        
        # Verify import_hub_content was called
        assert mock_air_hub.import_hub_content.called
        
        # Get the tags argument
        call_args = mock_air_hub.import_hub_content.call_args
        tags = call_args[1]['tags']
        
        # Verify domain-id is still in tags
        assert any(tag[0] == '@domain' and tag[1] == mock_domain_id for tag in tags)
