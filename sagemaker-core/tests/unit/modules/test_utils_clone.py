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
"""Unit tests for _run_clone_command_silent in sagemaker.core.modules.utils."""

import logging
import subprocess
from unittest.mock import patch

import pytest

from sagemaker.core.modules.utils import _run_clone_command_silent

MODULE = "sagemaker.core.modules.utils"
DEST = "/tmp/dest"


@pytest.mark.parametrize(
    "repo_url,userinfo",
    [
        ("https://oauth2:abc123@example.com/org/repo.git", "oauth2:abc123"),
        ("https://user:p%40ss@example.com/org/repo.git", "user:p%40ss"),
    ],
)
def test_clone_failure_uses_sanitized_url(repo_url, userinfo, caplog):
    with patch(
        f"{MODULE}.subprocess.check_call",
        side_effect=subprocess.CalledProcessError(128, ["git", "clone", repo_url, DEST]),
    ):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                _run_clone_command_silent(repo_url, DEST)

    error = exc_info.value
    expected_url = "https://<credentials-redacted>@example.com/org/repo.git"
    assert error.returncode == 128
    assert error.cmd == ["git", "clone", expected_url, DEST]
    assert error.__suppress_context__
    assert userinfo not in str(error)
    assert userinfo not in caplog.text
    assert expected_url in caplog.text


def test_clone_failure_without_userinfo_keeps_url(caplog):
    repo_url = "https://example.com/org/repo.git"
    with patch(
        f"{MODULE}.subprocess.check_call",
        side_effect=subprocess.CalledProcessError(128, ["git", "clone", repo_url, DEST]),
    ):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                _run_clone_command_silent(repo_url, DEST)

    assert exc_info.value.cmd == ["git", "clone", repo_url, DEST]
    assert repo_url in caplog.text


def test_clone_success_uses_original_url():
    repo_url = "https://oauth2:abc123@example.com/org/repo.git"
    with patch(f"{MODULE}.subprocess.check_call") as mock_check_call:
        _run_clone_command_silent(repo_url, DEST)

    args, kwargs = mock_check_call.call_args
    assert args[0] == ["git", "clone", repo_url, DEST]
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
