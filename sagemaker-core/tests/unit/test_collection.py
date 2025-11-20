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
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, MagicMock, patch
from botocore.exceptions import ClientError

from sagemaker.core.collection import Collection


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session."""
    session = Mock()
    session.sagemaker_client = Mock()
    return session


@pytest.fixture
def collection(mock_session):
    """Create a Collection instance with mock session."""
    return Collection(mock_session)


def test_collection_initialization_with_session(mock_session):
    """Test Collection initialization with provided session."""
    coll = Collection(mock_session)
    assert coll.sagemaker_session == mock_session


def test_collection_initialization_without_session():
    """Test Collection initialization without session creates default."""
    with patch('sagemaker.core.collection.Session') as mock_session_class:
        mock_session_instance = Mock()
        mock_session_class.return_value = mock_session_instance
        
        coll = Collection(None)
        assert coll.sagemaker_session == mock_session_instance


def test_check_access_error_with_access_denied(collection):
    """Test _check_access_error raises exception for AccessDeniedException."""
    error = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
        "operation"
    )
    
    with pytest.raises(Exception, match="AccessDeniedException"):
        collection._check_access_error(error)


def test_check_access_error_with_other_error(collection):
    """Test _check_access_error does nothing for other errors."""
    error = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Validation error"}},
        "operation"
    )
    
    # Should not raise
    collection._check_access_error(error)


def test_add_model_group_success(collection, mock_session):
    """Test _add_model_group successfully adds a model group."""
    mock_session.sagemaker_client.describe_model_package_group.return_value = {
        "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group"
    }
    
    collection._add_model_group("test-group", "tag-key", "tag-value")
    
    mock_session.sagemaker_client.describe_model_package_group.assert_called_once_with(
        ModelPackageGroupName="test-group"
    )
    mock_session.sagemaker_client.add_tags.assert_called_once()


def test_remove_model_group_success(collection, mock_session):
    """Test _remove_model_group successfully removes a model group."""
    mock_session.sagemaker_client.describe_model_package_group.return_value = {
        "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group"
    }
    
    collection._remove_model_group("test-group", "tag-key")
    
    mock_session.sagemaker_client.describe_model_package_group.assert_called_once_with(
        ModelPackageGroupName="test-group"
    )
    mock_session.sagemaker_client.delete_tags.assert_called_once()


def test_create_collection_at_root(collection, mock_session):
    """Test create collection at root level."""
    mock_session.create_group.return_value = {
        "Group": {
            "Name": "test-collection",
            "GroupArn": "arn:aws:resource-groups:us-west-2:123456789012:group/test-collection"
        }
    }
    
    result = collection.create("test-collection")
    
    assert result["Name"] == "test-collection"
    assert "Arn" in result
    mock_session.create_group.assert_called_once()


def test_create_collection_with_parent(collection, mock_session):
    """Test create collection with parent collection."""
    mock_session.get_resource_group_query.return_value = {
        "GroupQuery": {
            "ResourceQuery": {
                "Query": '{"TagFilters": [{"Key": "parent-key", "Values": ["parent-value"]}]}'
            }
        }
    }
    mock_session.create_group.return_value = {
        "Group": {
            "Name": "child-collection",
            "GroupArn": "arn:aws:resource-groups:us-west-2:123456789012:group/child-collection"
        }
    }
    
    result = collection.create("child-collection", "parent-collection")
    
    assert result["Name"] == "child-collection"
    mock_session.get_resource_group_query.assert_called_once()


def test_create_collection_already_exists(collection, mock_session):
    """Test create collection raises ValueError when collection already exists."""
    mock_session.create_group.side_effect = ClientError(
        {"Error": {"Code": "BadRequestException", "Message": "group already exists"}},
        "CreateGroup"
    )
    
    with pytest.raises(ValueError, match="Collection with the given name already exists"):
        collection.create("existing-collection")


def test_delete_collections_success(collection, mock_session):
    """Test delete collections successfully."""
    mock_session.list_group_resources.return_value = {"Resources": []}
    mock_session.delete_resource_group.return_value = {}
    
    result = collection.delete(["collection1", "collection2"])
    
    assert len(result["deleted_collections"]) == 2
    assert len(result["delete_collection_failures"]) == 0


def test_delete_collections_too_many(collection):
    """Test delete collections raises ValueError for more than 10 collections."""
    collections = [f"collection{i}" for i in range(11)]
    
    with pytest.raises(ValueError, match="Can delete upto 10 collections at a time"):
        collection.delete(collections)


def test_delete_collections_not_empty(collection, mock_session):
    """Test delete collections fails when collection is not empty."""
    mock_session.list_group_resources.return_value = {
        "Resources": [{"ResourceArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test"}]
    }
    
    result = collection.delete(["non-empty-collection"])
    
    assert len(result["deleted_collections"]) == 0
    assert len(result["delete_collection_failures"]) == 1
    assert "Collection not empty" in result["delete_collection_failures"][0]["message"]


def test_get_collection_tag_rule_success(collection, mock_session):
    """Test _get_collection_tag_rule returns tag rules."""
    mock_session.get_resource_group_query.return_value = {
        "GroupQuery": {
            "ResourceQuery": {
                "Query": '{"TagFilters": [{"Key": "test-key", "Values": ["test-value"]}]}'
            }
        }
    }
    
    result = collection._get_collection_tag_rule("test-collection")
    
    assert result["tag_rule_key"] == "test-key"
    assert result["tag_rule_value"] == "test-value"


def test_get_collection_tag_rule_not_found(collection, mock_session):
    """Test _get_collection_tag_rule raises ValueError when collection not found."""
    mock_session.get_resource_group_query.side_effect = ClientError(
        {"Error": {"Code": "NotFoundException", "Message": "Not found"}},
        "GetGroupQuery"
    )
    
    with pytest.raises(ValueError, match="Cannot find collection"):
        collection._get_collection_tag_rule("non-existent")


def test_get_collection_tag_rule_none_name(collection):
    """Test _get_collection_tag_rule raises ValueError for None name."""
    with pytest.raises(ValueError, match="Collection name is required"):
        collection._get_collection_tag_rule(None)


def test_add_model_groups_success(collection, mock_session):
    """Test add_model_groups successfully adds model groups."""
    mock_session.get_resource_group_query.return_value = {
        "GroupQuery": {
            "ResourceQuery": {
                "Query": '{"TagFilters": [{"Key": "test-key", "Values": ["test-value"]}]}'
            }
        }
    }
    mock_session.sagemaker_client.describe_model_package_group.return_value = {
        "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test"
    }
    
    result = collection.add_model_groups("test-collection", ["model-group-1", "model-group-2"])
    
    assert len(result["added_groups"]) == 2
    assert len(result["failure"]) == 0


def test_add_model_groups_too_many(collection):
    """Test add_model_groups raises exception for more than 10 groups."""
    model_groups = [f"group{i}" for i in range(11)]
    
    with pytest.raises(Exception, match="Model groups can have a maximum length of 10"):
        collection.add_model_groups("test-collection", model_groups)


def test_remove_model_groups_success(collection, mock_session):
    """Test remove_model_groups successfully removes model groups."""
    mock_session.get_resource_group_query.return_value = {
        "GroupQuery": {
            "ResourceQuery": {
                "Query": '{"TagFilters": [{"Key": "test-key", "Values": ["test-value"]}]}'
            }
        }
    }
    mock_session.sagemaker_client.describe_model_package_group.return_value = {
        "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test"
    }
    
    result = collection.remove_model_groups("test-collection", ["model-group-1"])
    
    assert len(result["removed_groups"]) == 1
    assert len(result["failure"]) == 0


def test_remove_model_groups_too_many(collection):
    """Test remove_model_groups raises exception for more than 10 groups."""
    model_groups = [f"group{i}" for i in range(11)]
    
    with pytest.raises(Exception, match="Model groups can have a maximum length of 10"):
        collection.remove_model_groups("test-collection", model_groups)


def test_move_model_group_success(collection, mock_session):
    """Test move_model_group successfully moves a model group."""
    mock_session.get_resource_group_query.return_value = {
        "GroupQuery": {
            "ResourceQuery": {
                "Query": '{"TagFilters": [{"Key": "test-key", "Values": ["test-value"]}]}'
            }
        }
    }
    mock_session.sagemaker_client.describe_model_package_group.return_value = {
        "ModelPackageGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test"
    }
    
    result = collection.move_model_group("source-collection", "model-group", "dest-collection")
    
    assert result["moved_success"] == "model-group"


def test_convert_tag_collection_response(collection):
    """Test _convert_tag_collection_response converts response correctly."""
    tag_collections = [
        {"ResourceARN": "arn:aws:resource-groups:us-west-2:123456789012:group/collection1"},
        {"ResourceARN": "arn:aws:resource-groups:us-west-2:123456789012:group/collection2"}
    ]
    
    result = collection._convert_tag_collection_response(tag_collections)
    
    assert len(result) == 2
    assert result[0]["Name"] == "collection1"
    assert result[0]["Type"] == "Collection"
    assert result[1]["Name"] == "collection2"


def test_convert_group_resource_response(collection):
    """Test _convert_group_resource_response converts response correctly."""
    group_resource_details = {
        "Resources": [
            {
                "Identifier": {
                    "ResourceArn": "arn:aws:resource-groups:us-west-2:123456789012:group/collection1",
                    "ResourceType": "AWS::ResourceGroups::Group"
                }
            }
        ]
    }
    
    result = collection._convert_group_resource_response(group_resource_details)
    
    assert len(result) == 1
    assert result[0]["Name"] == "collection1"
    assert result[0]["Type"] == "Collection"


def test_convert_group_resource_response_model_group(collection):
    """Test _convert_group_resource_response with model group type."""
    group_resource_details = {
        "Resources": [
            {
                "Identifier": {
                    "ResourceArn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/model1",
                    "ResourceType": "AWS::SageMaker::ModelPackageGroup"
                }
            }
        ]
    }
    
    result = collection._convert_group_resource_response(group_resource_details, is_model_group=True)
    
    assert len(result) == 1
    assert result[0]["Type"] == "AWS::SageMaker::ModelPackageGroup"


def test_get_full_list_resource_no_pagination(collection, mock_session):
    """Test _get_full_list_resource without pagination."""
    mock_session.list_group_resources.return_value = {
        "Resources": [{"test": "resource"}],
        "ResourceIdentifiers": [{"id": "1"}]
    }
    
    result = collection._get_full_list_resource("test-collection", [])
    
    assert len(result["Resources"]) == 1
    mock_session.list_group_resources.assert_called_once()


def test_get_full_list_resource_with_pagination(collection, mock_session):
    """Test _get_full_list_resource with pagination."""
    mock_session.list_group_resources.side_effect = [
        {
            "Resources": [{"test": "resource1"}],
            "ResourceIdentifiers": [{"id": "1"}],
            "NextToken": "token1"
        },
        {
            "Resources": [{"test": "resource2"}],
            "ResourceIdentifiers": [{"id": "2"}],
            "NextToken": None
        }
    ]
    
    result = collection._get_full_list_resource("test-collection", [])
    
    assert len(result["Resources"]) == 2
    assert mock_session.list_group_resources.call_count == 2


def test_list_collection_at_root(collection, mock_session):
    """Test list_collection at root level."""
    mock_session.get_tagging_resources.return_value = [
        {"ResourceARN": "arn:aws:resource-groups:us-west-2:123456789012:group/collection1"}
    ]
    
    result = collection.list_collection()
    
    assert len(result) == 1
    assert result[0]["Name"] == "collection1"


def test_list_collection_with_name(collection, mock_session):
    """Test list_collection with collection name."""
    mock_session.list_group_resources.side_effect = [
        {"Resources": [], "ResourceIdentifiers": []},
        {"Resources": [], "ResourceIdentifiers": []}
    ]
    
    result = collection.list_collection("test-collection")
    
    assert isinstance(result, list)
    assert mock_session.list_group_resources.call_count == 2
