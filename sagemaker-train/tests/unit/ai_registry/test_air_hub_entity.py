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

"""Tests for AIRHubEntity base class."""

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.core.utils.exceptions import FailedStatusError, TimeoutExceededError

from sagemaker.ai_registry.air_hub_entity import AIRHubEntity
from sagemaker.ai_registry.air_constants import (
    RESPONSE_KEY_HUB_CONTENT_VERSION,
    RESPONSE_KEY_HUB_CONTENT_NAME,
    RESPONSE_KEY_HUB_CONTENT_ARN,
    RESPONSE_KEY_HUB_CONTENT_STATUS,
    RESPONSE_KEY_CREATION_TIME
)


class ConcreteEntity(AIRHubEntity):
    """Concrete implementation for testing."""
    
    @property
    def hub_content_type(self) -> str:
        return "TestType"
    
    @classmethod
    def _get_hub_content_type_for_list(cls) -> str:
        return "TestType"


class TestAIRHubEntity:
    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_initialization(self, mock_air_hub):
        """Test entity initialization."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        
        entity = ConcreteEntity(
            name="test-entity",
            version="1.0.0",
            arn="test-arn",
            status="Available",
            created_time="2024-01-01",
            updated_time="2024-01-02",
            description="Test description"
        )
        
        assert entity.name == "test-entity"
        assert entity.version == "1.0.0"
        assert entity.arn == "test-arn"
        assert entity.status == "Available"
        assert entity.created == "2024-01-01"
        assert entity.updated == "2024-01-02"
        assert entity.description == "Test description"
        assert entity.hub_name == "test-hub"

    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_list(self, mock_air_hub):
        """Test listing entities."""
        mock_air_hub.list_hub_content.return_value = {
            "items": [{"name": "entity1"}, {"name": "entity2"}],
            "next_token": None
        }
        
        result = ConcreteEntity.list(max_results=10)
        
        mock_air_hub.list_hub_content.assert_called_once_with("TestType", 10, None)
        assert result["items"][0]["name"] == "entity1"

    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_get_versions(self, mock_air_hub):
        """Test getting entity versions."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.list_hub_content_versions.return_value = [
            {
                RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
                RESPONSE_KEY_HUB_CONTENT_NAME: "test",
                RESPONSE_KEY_HUB_CONTENT_ARN: "arn1",
                RESPONSE_KEY_HUB_CONTENT_STATUS: "Available",
                RESPONSE_KEY_CREATION_TIME: "2024-01-01"
            },
            {
                RESPONSE_KEY_HUB_CONTENT_VERSION: "2.0.0",
                RESPONSE_KEY_HUB_CONTENT_NAME: "test",
                RESPONSE_KEY_HUB_CONTENT_ARN: "arn2",
                RESPONSE_KEY_HUB_CONTENT_STATUS: "Available",
                RESPONSE_KEY_CREATION_TIME: "2024-01-02"
            }
        ]
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Available")
        versions = entity.get_versions()
        
        assert len(versions) == 2
        assert versions[0]["version"] == "1.0.0"
        assert versions[1]["version"] == "2.0.0"

    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_delete_single_version(self, mock_air_hub):
        """Test deleting single version."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.delete_hub_content.return_value = {}
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Available")
        result = entity.delete(version="1.0.0")
        
        assert result is True
        mock_air_hub.delete_hub_content.assert_called_once_with("TestType", "test", "1.0.0")

    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_delete_all_versions(self, mock_air_hub):
        """Test deleting all versions."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.list_hub_content_versions.return_value = [
            {RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0"},
            {RESPONSE_KEY_HUB_CONTENT_VERSION: "2.0.0"}
        ]
        mock_air_hub.delete_hub_content.return_value = {}
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Available")
        result = entity.delete()
        
        assert result is True
        assert mock_air_hub.delete_hub_content.call_count == 2

    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_delete_failure(self, mock_air_hub):
        """Test delete failure handling."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.delete_hub_content.side_effect = Exception("Delete failed")
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Available")
        result = entity.delete(version="1.0.0")
        
        assert result is False

    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_delete_by_name(self, mock_air_hub):
        """Test deleting by name."""
        mock_air_hub.delete_hub_content.return_value = {}
        
        result = ConcreteEntity.delete_by_name("test", version="1.0.0")
        
        assert result is True
        mock_air_hub.delete_hub_content.assert_called_once_with("TestType", "test", "1.0.0")

    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_refresh(self, mock_air_hub):
        """Test refreshing entity status."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_STATUS: "Available"
        }
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Importing")
        entity.refresh()
        
        assert entity.status == "Available"
        mock_air_hub.describe_hub_content.assert_called_once_with("TestType", "test", "1.0.0")

    @patch('sagemaker.ai_registry.air_hub_entity.time')
    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_wait_success(self, mock_air_hub, mock_time):
        """Test waiting for entity to become available."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_STATUS: "Available"
        }
        mock_time.time.return_value = 0
        mock_time.sleep.return_value = None
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Importing")
        
        with patch('builtins.print'):
            entity.wait(poll=1, timeout=10)
        
        assert entity.status == "Available"

    @patch('sagemaker.ai_registry.air_hub_entity.time')
    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_wait_failed_status(self, mock_air_hub, mock_time):
        """Test waiting fails when entity reaches failed state."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_STATUS: "ImportFailed"
        }
        mock_time.time.return_value = 0
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Importing")
        
        with pytest.raises(FailedStatusError):
            with patch('builtins.print'):
                entity.wait(poll=1, timeout=10)

    @patch('sagemaker.ai_registry.air_hub_entity.time')
    @patch('sagemaker.ai_registry.air_hub_entity.AIRHub')
    def test_wait_timeout(self, mock_air_hub, mock_time):
        """Test waiting times out."""
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_STATUS: "Importing"
        }
        
        call_count = [0]
        def mock_time_func():
            call_count[0] += 1
            return call_count[0] * 10
        
        mock_time.time.side_effect = mock_time_func
        mock_time.sleep.return_value = None
        
        entity = ConcreteEntity("test", "1.0.0", "arn", "Importing")
        
        with pytest.raises(TimeoutExceededError):
            with patch('builtins.print'):
                entity.wait(poll=1, timeout=5)
