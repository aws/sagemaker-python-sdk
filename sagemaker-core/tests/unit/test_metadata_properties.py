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

from unittest.mock import Mock

from sagemaker.core.metadata_properties import MetadataProperties


def test_metadata_properties_initialization_empty():
    """Test MetadataProperties initialization with no parameters."""
    metadata = MetadataProperties()

    assert metadata.commit_id is None
    assert metadata.repository is None
    assert metadata.generated_by is None
    assert metadata.project_id is None


def test_metadata_properties_to_request_dict_empty():
    """Test _to_request_dict with no parameters returns empty dict."""
    metadata = MetadataProperties()

    request_dict = metadata._to_request_dict()

    assert request_dict == {}


def test_metadata_properties_with_commit_id():
    """Test MetadataProperties with commit_id."""
    metadata = MetadataProperties(commit_id="abc123def456")

    request_dict = metadata._to_request_dict()

    assert request_dict == {"CommitId": "abc123def456"}


def test_metadata_properties_with_repository():
    """Test MetadataProperties with repository."""
    metadata = MetadataProperties(repository="https://github.com/test/repo.git")

    request_dict = metadata._to_request_dict()

    assert request_dict == {"Repository": "https://github.com/test/repo.git"}


def test_metadata_properties_with_generated_by():
    """Test MetadataProperties with generated_by."""
    metadata = MetadataProperties(generated_by="SageMaker Training Job")

    request_dict = metadata._to_request_dict()

    assert request_dict == {"GeneratedBy": "SageMaker Training Job"}


def test_metadata_properties_with_project_id():
    """Test MetadataProperties with project_id."""
    metadata = MetadataProperties(project_id="project-12345")

    request_dict = metadata._to_request_dict()

    assert request_dict == {"ProjectId": "project-12345"}


def test_metadata_properties_all_parameters():
    """Test MetadataProperties with all parameters."""
    metadata = MetadataProperties(
        commit_id="abc123",
        repository="https://github.com/test/repo.git",
        generated_by="Training Job",
        project_id="proj-123",
    )

    request_dict = metadata._to_request_dict()

    assert request_dict == {
        "CommitId": "abc123",
        "Repository": "https://github.com/test/repo.git",
        "GeneratedBy": "Training Job",
        "ProjectId": "proj-123",
    }


def test_metadata_properties_with_pipeline_variable():
    """Test MetadataProperties with PipelineVariable."""
    mock_pipeline_var = Mock()
    mock_pipeline_var.__str__ = Mock(return_value="pipeline_var")

    metadata = MetadataProperties(commit_id=mock_pipeline_var)

    assert metadata.commit_id == mock_pipeline_var
    request_dict = metadata._to_request_dict()
    assert "CommitId" in request_dict


def test_metadata_properties_partial_parameters():
    """Test MetadataProperties with partial parameters."""
    metadata = MetadataProperties(commit_id="abc123", project_id="proj-456")

    request_dict = metadata._to_request_dict()

    assert request_dict == {"CommitId": "abc123", "ProjectId": "proj-456"}
    assert "Repository" not in request_dict
    assert "GeneratedBy" not in request_dict


def test_metadata_properties_empty_string_values():
    """Test MetadataProperties with empty string values are excluded."""
    metadata = MetadataProperties(
        commit_id="",
        repository="https://github.com/test/repo.git",
        generated_by="",
        project_id="proj-123",
    )

    request_dict = metadata._to_request_dict()

    # Empty strings are falsy, so they should not be included
    assert "CommitId" not in request_dict
    assert "GeneratedBy" not in request_dict
    assert request_dict == {
        "Repository": "https://github.com/test/repo.git",
        "ProjectId": "proj-123",
    }


def test_metadata_properties_modification():
    """Test modifying MetadataProperties attributes after initialization."""
    metadata = MetadataProperties(commit_id="initial")

    metadata.commit_id = "modified"
    metadata.repository = "https://github.com/new/repo.git"

    request_dict = metadata._to_request_dict()

    assert request_dict["CommitId"] == "modified"
    assert request_dict["Repository"] == "https://github.com/new/repo.git"


def test_metadata_properties_long_commit_id():
    """Test MetadataProperties with long commit ID."""
    long_commit = "a" * 40  # SHA-1 hash length
    metadata = MetadataProperties(commit_id=long_commit)

    request_dict = metadata._to_request_dict()

    assert request_dict["CommitId"] == long_commit


def test_metadata_properties_special_characters():
    """Test MetadataProperties with special characters in values."""
    metadata = MetadataProperties(
        repository="git@github.com:user/repo.git",
        generated_by="SageMaker Training Job (v2.0)",
        project_id="proj-test-123_456",
    )

    request_dict = metadata._to_request_dict()

    assert request_dict["Repository"] == "git@github.com:user/repo.git"
    assert request_dict["GeneratedBy"] == "SageMaker Training Job (v2.0)"
    assert request_dict["ProjectId"] == "proj-test-123_456"
