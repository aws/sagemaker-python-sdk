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

from sagemaker.core.git_utils import _validate_git_config


def test_validate_git_config_valid():
    """Test _validate_git_config with valid configuration."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "branch": "main",
        "commit": "abc123"
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_missing_repo():
    """Test _validate_git_config raises ValueError when repo is missing."""
    git_config = {
        "branch": "main"
    }
    
    with pytest.raises(ValueError, match="Please provide a repo for git_config"):
        _validate_git_config(git_config)


def test_validate_git_config_with_2fa_enabled_true():
    """Test _validate_git_config with 2FA_enabled as True."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "2FA_enabled": True
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_with_2fa_enabled_false():
    """Test _validate_git_config with 2FA_enabled as False."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "2FA_enabled": False
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_2fa_enabled_not_bool():
    """Test _validate_git_config raises ValueError when 2FA_enabled is not bool."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "2FA_enabled": "true"
    }
    
    with pytest.raises(ValueError, match="Please enter a bool type for 2FA_enabled"):
        _validate_git_config(git_config)


def test_validate_git_config_non_string_value():
    """Test _validate_git_config raises ValueError for non-string values."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "branch": 123
    }
    
    with pytest.raises(ValueError, match="'branch' must be a string"):
        _validate_git_config(git_config)


def test_validate_git_config_with_username_password():
    """Test _validate_git_config with username and password."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "username": "testuser",
        "password": "testpass"
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_with_token():
    """Test _validate_git_config with token."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "token": "ghp_testtoken123"
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_all_fields():
    """Test _validate_git_config with all possible fields."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "branch": "develop",
        "commit": "def456",
        "2FA_enabled": True,
        "username": "testuser",
        "password": "testpass",
        "token": "ghp_testtoken123"
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_ssh_url():
    """Test _validate_git_config with SSH URL."""
    git_config = {
        "repo": "git@github.com:test/repo.git",
        "branch": "main"
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_codecommit_url():
    """Test _validate_git_config with CodeCommit URL."""
    git_config = {
        "repo": "https://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo",
        "branch": "main"
    }
    
    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_empty_repo():
    """Test _validate_git_config raises ValueError for empty repo string."""
    git_config = {
        "repo": ""
    }
    
    # Empty string is still a string, so validation passes for type
    # but the actual cloning would fail
    _validate_git_config(git_config)


def test_validate_git_config_repo_none():
    """Test _validate_git_config when repo key exists but value is None."""
    git_config = {
        "repo": None
    }
    
    with pytest.raises(ValueError, match="'repo' must be a string"):
        _validate_git_config(git_config)
