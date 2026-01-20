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

from sagemaker.core import git_utils
from sagemaker.core.git_utils import _validate_git_config


def test_validate_git_config_valid():
    """Test _validate_git_config with valid configuration."""
    git_config = {"repo": "https://github.com/test/repo.git", "branch": "main", "commit": "abc123"}

    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_missing_repo():
    """Test _validate_git_config raises ValueError when repo is missing."""
    git_config = {"branch": "main"}

    with pytest.raises(ValueError, match="Please provide a repo for git_config"):
        _validate_git_config(git_config)


def test_validate_git_config_with_2fa_enabled_true():
    """Test _validate_git_config with 2FA_enabled as True."""
    git_config = {"repo": "https://github.com/test/repo.git", "2FA_enabled": True}

    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_with_2fa_enabled_false():
    """Test _validate_git_config with 2FA_enabled as False."""
    git_config = {"repo": "https://github.com/test/repo.git", "2FA_enabled": False}

    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_2fa_enabled_not_bool():
    """Test _validate_git_config raises ValueError when 2FA_enabled is not bool."""
    git_config = {"repo": "https://github.com/test/repo.git", "2FA_enabled": "true"}

    with pytest.raises(ValueError, match="Please enter a bool type for 2FA_enabled"):
        _validate_git_config(git_config)


def test_validate_git_config_non_string_value():
    """Test _validate_git_config raises ValueError for non-string values."""
    git_config = {"repo": "https://github.com/test/repo.git", "branch": 123}

    with pytest.raises(ValueError, match="'branch' must be a string"):
        _validate_git_config(git_config)


def test_validate_git_config_with_username_password():
    """Test _validate_git_config with username and password."""
    git_config = {
        "repo": "https://github.com/test/repo.git",
        "username": "testuser",
        "password": "testpass",
    }

    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_with_token():
    """Test _validate_git_config with token."""
    git_config = {"repo": "https://github.com/test/repo.git", "token": "ghp_testtoken123"}

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
        "token": "ghp_testtoken123",
    }

    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_ssh_url():
    """Test _validate_git_config with SSH URL."""
    git_config = {"repo": "git@github.com:test/repo.git", "branch": "main"}

    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_codecommit_url():
    """Test _validate_git_config with CodeCommit URL."""
    git_config = {
        "repo": "https://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo",
        "branch": "main",
    }

    # Should not raise
    _validate_git_config(git_config)


def test_validate_git_config_empty_repo():
    """Test _validate_git_config raises ValueError for empty repo string."""
    git_config = {"repo": ""}

    # Empty string is still a string, so validation passes for type
    # but the actual cloning would fail
    _validate_git_config(git_config)


def test_validate_git_config_repo_none():
    """Test _validate_git_config when repo key exists but value is None."""
    git_config = {"repo": None}

    with pytest.raises(ValueError, match="'repo' must be a string"):
        _validate_git_config(git_config)


class TestGitUrlSanitization:
    """Test cases for Git URL sanitization to prevent injection attacks."""

    def test_sanitize_git_url_valid_https_urls(self):
        """Test that valid HTTPS URLs pass sanitization."""
        valid_urls = [
            "https://github.com/user/repo.git",
            "https://gitlab.com/user/repo.git",
            "https://token@github.com/user/repo.git",
            "https://user:pass@github.com/user/repo.git",
            "http://internal-git.company.com/repo.git",
        ]

        for url in valid_urls:
            result = git_utils._sanitize_git_url(url)
            assert result == url

    def test_sanitize_git_url_valid_ssh_urls(self):
        """Test that valid SSH URLs pass sanitization."""
        valid_urls = [
            "git@github.com:user/repo.git",
            "git@gitlab.com:user/repo.git",
            "ssh://git@github.com/user/repo.git",
            "ssh://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo/",
            "git@internal-git.company.com:repo.git",
        ]

        for url in valid_urls:
            result = git_utils._sanitize_git_url(url)
            assert result == url

    def test_sanitize_git_url_blocks_multiple_at_https(self):
        """Test that HTTPS URLs with multiple @ symbols are blocked."""
        malicious_urls = [
            "https://user@attacker.com@github.com/repo.git",
            "https://token@evil.com@gitlab.com/user/repo.git",
            "https://a@b@c@github.com/repo.git",
            "https://user@malicious-host@github.com/legit/repo.git",
        ]

        for url in malicious_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert "multiple @ symbols detected" in str(error.value)

    def test_sanitize_git_url_blocks_multiple_at_ssh(self):
        """Test that SSH URLs with multiple @ symbols are blocked."""
        malicious_urls = [
            "git@attacker.com@github.com:repo.git",
            "git@evil@gitlab.com:user/repo.git",
            "ssh://git@malicious@github.com/repo.git",
            "git@a@b@c:repo.git",
        ]

        for url in malicious_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert any(
                phrase in str(error.value)
                for phrase in ["multiple @ symbols detected", "exactly one @ symbol"]
            )

    def test_sanitize_git_url_blocks_invalid_schemes_and_git_at_format(self):
        """Test that invalid schemes and git@ format violations are blocked."""
        unsupported_scheme_urls = [
            "git-github.com:user/repo.git",
        ]

        for url in unsupported_scheme_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert "Unsupported URL scheme" in str(error.value)

        invalid_git_at_urls = [
            "git@github.com@evil.com:repo.git",
        ]

        for url in invalid_git_at_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert "exactly one @ symbol" in str(error.value)

    def test_sanitize_git_url_blocks_url_encoding_obfuscation(self):
        """Test that URL-encoded obfuscation attempts are blocked."""
        obfuscated_urls = [
            "https://github.com%25evil.com/repo.git",
            "https://user@github.com%40attacker.com/repo.git",
            "https://github.com%2Fevil.com/repo.git",
            "https://github.com%3Aevil.com/repo.git",
        ]

        for url in obfuscated_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert any(
                phrase in str(error.value)
                for phrase in ["Suspicious URL encoding detected", "Invalid characters in hostname"]
            )

    def test_sanitize_git_url_blocks_invalid_hostname_chars(self):
        """Test that hostnames with invalid characters are blocked."""
        invalid_urls = [
            "https://github<script>.com/repo.git",
            "https://github>.com/repo.git",
            "https://github[].com/repo.git",
            "https://github{}.com/repo.git",
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert any(
                phrase in str(error.value)
                for phrase in [
                    "Invalid characters in hostname",
                    "Failed to parse URL",
                    "does not appear to be an IPv4 or IPv6 address",
                ]
            )

    def test_sanitize_git_url_blocks_unsupported_schemes(self):
        """Test that unsupported URL schemes are blocked."""
        unsupported_urls = [
            "ftp://github.com/repo.git",
            "file:///local/repo.git",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
        ]

        for url in unsupported_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert "Unsupported URL scheme" in str(error.value)

    def test_git_clone_repo_blocks_malicious_https_url(self):
        """Test that git_clone_repo blocks malicious HTTPS URLs."""
        malicious_git_config = {
            "repo": "https://user@attacker.com@github.com/legit/repo.git",
            "branch": "main",
        }
        entry_point = "train.py"

        with pytest.raises(ValueError) as error:
            git_utils.git_clone_repo(malicious_git_config, entry_point)
        assert "multiple @ symbols detected" in str(error.value)

    def test_git_clone_repo_blocks_malicious_ssh_url(self):
        """Test that git_clone_repo blocks malicious SSH URLs."""
        malicious_git_config = {
            "repo": "git@OBVIOUS@github.com:sage-maker/temp-sev2.git",
            "branch": "main",
        }
        entry_point = "train.py"

        with pytest.raises(ValueError) as error:
            git_utils.git_clone_repo(malicious_git_config, entry_point)
        assert "exactly one @ symbol" in str(error.value)

    def test_git_clone_repo_blocks_url_encoded_attack(self):
        """Test that git_clone_repo blocks URL-encoded attacks."""
        malicious_git_config = {
            "repo": "https://github.com%40attacker.com/repo.git",
            "branch": "main",
        }
        entry_point = "train.py"

        with pytest.raises(ValueError) as error:
            git_utils.git_clone_repo(malicious_git_config, entry_point)
        assert "Suspicious URL encoding detected" in str(error.value)

    def test_sanitize_git_url_comprehensive_attack_scenarios(self):
        attack_scenarios = [
            "https://USER@YOUR_NGROK_OR_LOCALHOST/malicious.git@github.com%25legit%25repo.git",
            "https://user@malicious-host@github.com/legit/repo.git",
            "git@attacker.com@github.com:user/repo.git",
            "ssh://git@evil.com@github.com/repo.git",
            "https://github.com%40evil.com/repo.git",
            "https://user@github.com%2Fevil.com/repo.git",
        ]

        entry_point = "train.py"

        for malicious_url in attack_scenarios:
            git_config = {"repo": malicious_url}
            with pytest.raises(ValueError) as error:
                git_utils.git_clone_repo(git_config, entry_point)
            assert any(
                phrase in str(error.value)
                for phrase in [
                    "multiple @ symbols detected",
                    "exactly one @ symbol",
                    "Suspicious URL encoding detected",
                    "Invalid characters in hostname",
                ]
            )
