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

import os
import subprocess
from pathlib import Path

import pytest
from mock import ANY, patch

from sagemaker import git_utils

REPO_DIR = "/tmp/repo_dir"
PUBLIC_GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
PUBLIC_BRANCH = "test-branch-git-config"
PUBLIC_COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"
PRIVATE_GIT_REPO_SSH = "git@github.com:testAccount/private-repo.git"
PRIVATE_GIT_REPO = "https://github.com/testAccount/private-repo.git"
PRIVATE_BRANCH = "test-branch"
PRIVATE_COMMIT = "329bfcf884482002c05ff7f44f62599ebc9f445a"
CODECOMMIT_REPO = "https://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo/"
CODECOMMIT_REPO_SSH = "ssh://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo/"
CODECOMMIT_BRANCH = "master"


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_succeed(exists, isdir, isfile, tempdir, mkdtemp, check_call):
    git_config = {"repo": PUBLIC_GIT_REPO, "branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    ret = git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    check_call.assert_any_call(["git", "clone", git_config["repo"], REPO_DIR], env=env)
    check_call.assert_any_call(args=["git", "checkout", PUBLIC_BRANCH], cwd=REPO_DIR)
    check_call.assert_any_call(args=["git", "checkout", PUBLIC_COMMIT], cwd=REPO_DIR)
    mkdtemp.assert_called_once()
    assert ret["entry_point"] == "entry_point"
    assert ret["source_dir"] == "/tmp/repo_dir/source_dir"
    assert ret["dependencies"] == ["/tmp/repo_dir/foo", "/tmp/repo_dir/bar"]


def test_git_clone_repo_repo_not_provided():
    git_config = {"branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point_that_does_not_exist"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "Please provide a repo for git_config." in str(error)


def test_git_clone_repo_git_argument_wrong_format():
    git_config = {
        "repo": PUBLIC_GIT_REPO,
        "branch": PUBLIC_BRANCH,
        "commit": PUBLIC_COMMIT,
        "token": 42,
    }
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "'token' must be a string." in str(error)


@patch(
    "subprocess.check_call",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(PUBLIC_GIT_REPO, REPO_DIR)
    ),
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_clone_fail(tempdir, mkdtemp, check_call):
    git_config = {"repo": PUBLIC_GIT_REPO, "branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "subprocess.check_call",
    side_effect=[True, subprocess.CalledProcessError(returncode=1, cmd="git checkout banana")],
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_branch_not_exist(tempdir, mkdtemp, check_call):
    git_config = {"repo": PUBLIC_GIT_REPO, "branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "subprocess.check_call",
    side_effect=[
        True,
        True,
        subprocess.CalledProcessError(returncode=1, cmd="git checkout {}".format(PUBLIC_COMMIT)),
    ],
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_commit_not_exist(tempdir, mkdtemp, check_call):
    git_config = {"repo": PUBLIC_GIT_REPO, "branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "returned non-zero exit status" in str(error.value)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=False)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_entry_point_not_exist(exists, isdir, isfile, tempdir, mkdtemp, heck_call):
    git_config = {"repo": PUBLIC_GIT_REPO, "branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point_that_does_not_exist"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "Entry point does not exist in the repo." in str(error)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=False)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_source_dir_not_exist(exists, isdir, isfile, tempdir, mkdtemp, check_call):
    git_config = {"repo": PUBLIC_GIT_REPO, "branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir_that_does_not_exist"
    dependencies = ["foo", "bar"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "Source directory does not exist in the repo." in str(error)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", side_effect=[True, False])
def test_git_clone_repo_dependencies_not_exist(exists, isdir, isfile, tempdir, mkdtemp, check_call):
    git_config = {"repo": PUBLIC_GIT_REPO, "branch": PUBLIC_BRANCH, "commit": PUBLIC_COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "dep_that_does_not_exist"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "does not exist in the repo." in str(error)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
def test_git_clone_repo_with_username_password_no_2fa(isfile, tempdir, mkdtemp, check_call):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "username": "username",
        "password": "passw0rd!",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    ret = git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    check_call.assert_any_call(
        [
            "git",
            "clone",
            "https://username:passw0rd%21@github.com/testAccount/private-repo.git",
            REPO_DIR,
        ],
        env=env,
    )
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_BRANCH], cwd=REPO_DIR)
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_COMMIT], cwd=REPO_DIR)
    assert ret["entry_point"] == "/tmp/repo_dir/entry_point"
    assert ret["source_dir"] is None
    assert ret["dependencies"] is None


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
def test_git_clone_repo_with_token_no_2fa(isfile, tempdir, mkdtemp, check_call):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "token": "my-token",
        "2FA_enabled": False,
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    ret = git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    check_call.assert_any_call(
        ["git", "clone", "https://my-token@github.com/testAccount/private-repo.git", REPO_DIR],
        env=env,
    )
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_BRANCH], cwd=REPO_DIR)
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_COMMIT], cwd=REPO_DIR)
    assert ret["entry_point"] == "/tmp/repo_dir/entry_point"
    assert ret["source_dir"] is None
    assert ret["dependencies"] is None


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
def test_git_clone_repo_with_token_2fa(isfile, tempdirm, mkdtemp, check_call):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "2FA_enabled": True,
        "username": "username",
        "token": "my-token",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    ret = git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    check_call.assert_any_call(
        ["git", "clone", "https://my-token@github.com/testAccount/private-repo.git", REPO_DIR],
        env=env,
    )
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_BRANCH], cwd=REPO_DIR)
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_COMMIT], cwd=REPO_DIR)
    assert ret["entry_point"] == "/tmp/repo_dir/entry_point"
    assert ret["source_dir"] is None
    assert ret["dependencies"] is None


@patch("subprocess.check_call")
@patch("os.chmod")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
def test_git_clone_repo_ssh(isfile, tempdir, mkdtemp, chmod, check_call):
    Path(REPO_DIR).mkdir(parents=True, exist_ok=True)
    git_config = {"repo": PRIVATE_GIT_REPO_SSH, "branch": PRIVATE_BRANCH, "commit": PRIVATE_COMMIT}
    entry_point = "entry_point"
    ret = git_utils.git_clone_repo(git_config, entry_point)
    chmod.assert_any_call(ANY, 0o511)
    assert ret["entry_point"] == "/tmp/repo_dir/entry_point"
    assert ret["source_dir"] is None
    assert ret["dependencies"] is None


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
def test_git_clone_repo_with_token_no_2fa_unnecessary_creds_provided(
    isfile, tempdir, mkdtemp, check_call
):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "username": "username",
        "password": "passw0rd!",
        "token": "my-token",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    with pytest.warns(UserWarning) as warn:
        ret = git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    assert (
        "Using token for authentication, other credentials will be ignored."
        in warn[0].message.args[0]
    )
    check_call.assert_any_call(
        ["git", "clone", "https://my-token@github.com/testAccount/private-repo.git", REPO_DIR],
        env=env,
    )
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_BRANCH], cwd=REPO_DIR)
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_COMMIT], cwd=REPO_DIR)
    assert ret["entry_point"] == "/tmp/repo_dir/entry_point"
    assert ret["source_dir"] is None
    assert ret["dependencies"] is None


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
def test_git_clone_repo_with_token_2fa_unnecessary_creds_provided(
    isfile, tempdir, mkdtemp, check_call
):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "2FA_enabled": True,
        "username": "username",
        "token": "my-token",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    with pytest.warns(UserWarning) as warn:
        ret = git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    assert (
        "Using token for authentication, other credentials will be ignored."
        in warn[0].message.args[0]
    )
    check_call.assert_any_call(
        ["git", "clone", "https://my-token@github.com/testAccount/private-repo.git", REPO_DIR],
        env=env,
    )
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_BRANCH], cwd=REPO_DIR)
    check_call.assert_any_call(args=["git", "checkout", PRIVATE_COMMIT], cwd=REPO_DIR)
    assert ret["entry_point"] == "/tmp/repo_dir/entry_point"
    assert ret["source_dir"] is None
    assert ret["dependencies"] is None


@patch(
    "subprocess.check_call",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(PRIVATE_GIT_REPO, REPO_DIR)
    ),
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_with_username_and_password_wrong_creds(tempdir, mkdtemp, check_call):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "2FA_enabled": False,
        "username": "username",
        "password": "wrong-password",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "subprocess.check_call",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(PRIVATE_GIT_REPO, REPO_DIR)
    ),
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_with_token_wrong_creds(tempdir, mkdtemp, check_call):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "2FA_enabled": False,
        "token": "wrong-token",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "subprocess.check_call",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(PRIVATE_GIT_REPO, REPO_DIR)
    ),
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_with_and_token_2fa_wrong_creds(tempdir, mkdtemp, check_call):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "2FA_enabled": False,
        "token": "wrong-token",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    assert "returned non-zero exit status" in str(error.value)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
def test_git_clone_repo_codecommit_https_with_username_and_password(
    isfile, tempdir, mkdtemp, check_call
):
    git_config = {
        "repo": CODECOMMIT_REPO,
        "branch": CODECOMMIT_BRANCH,
        "username": "username",
        "password": "my-codecommit-password",
    }
    entry_point = "entry_point"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    ret = git_utils.git_clone_repo(git_config=git_config, entry_point=entry_point)
    check_call.assert_any_call(
        [
            "git",
            "clone",
            "https://username:my-codecommit-password@git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo/",
            REPO_DIR,
        ],
        env=env,
    )
    check_call.assert_any_call(args=["git", "checkout", CODECOMMIT_BRANCH], cwd=REPO_DIR)
    assert ret["entry_point"] == "/tmp/repo_dir/entry_point"
    assert ret["source_dir"] is None
    assert ret["dependencies"] is None


@patch(
    "subprocess.check_call",
    side_effect=subprocess.CalledProcessError(
        returncode=128, cmd="git clone {} {}".format(CODECOMMIT_REPO_SSH, REPO_DIR)
    ),
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_codecommit_ssh_passphrase_required(tempdir, mkdtemp, check_call):
    Path(REPO_DIR).mkdir(parents=True, exist_ok=True)
    git_config = {"repo": CODECOMMIT_REPO_SSH, "branch": CODECOMMIT_BRANCH}
    entry_point = "entry_point"
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point)
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "subprocess.check_call",
    side_effect=subprocess.CalledProcessError(
        returncode=128, cmd="git clone {} {}".format(CODECOMMIT_REPO, REPO_DIR)
    ),
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("tempfile.TemporaryDirectory.__enter__", return_value=REPO_DIR)
def test_git_clone_repo_codecommit_https_creds_not_stored_locally(tempdir, mkdtemp, check_call):
    git_config = {"repo": CODECOMMIT_REPO, "branch": CODECOMMIT_BRANCH}
    entry_point = "entry_point"
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point)
    assert "returned non-zero exit status" in str(error.value)


# ============================================================================
# URL Sanitization Tests - Security vulnerability prevention
# ============================================================================

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
            # Should not raise any exception
            result = git_utils._sanitize_git_url(url)
            assert result == url

    def test_sanitize_git_url_valid_ssh_urls(self):
        """Test that valid SSH URLs pass sanitization."""
        valid_urls = [
            "git@github.com:user/repo.git",
            "git@gitlab.com:user/repo.git",
            "ssh://git@github.com/user/repo.git",
            "ssh://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo/",  # 0 @ symbols - valid for ssh://
            "git@internal-git.company.com:repo.git",
        ]
        
        for url in valid_urls:
            # Should not raise any exception
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
            # git@ URLs should give "exactly one @ symbol" error
            # ssh:// URLs should give "multiple @ symbols detected" error
            assert any(phrase in str(error.value) for phrase in [
                "multiple @ symbols detected",
                "exactly one @ symbol"
            ])

    def test_sanitize_git_url_blocks_invalid_schemes_and_git_at_format(self):
        """Test that invalid schemes and git@ format violations are blocked."""
        # Test unsupported schemes
        unsupported_scheme_urls = [
            "git-github.com:user/repo.git",  # Doesn't start with git@, ssh://, http://, https://
        ]
        
        for url in unsupported_scheme_urls:
            with pytest.raises(ValueError) as error:
                git_utils._sanitize_git_url(url)
            assert "Unsupported URL scheme" in str(error.value)
        
        # Test git@ URLs with wrong @ count
        invalid_git_at_urls = [
            "git@github.com@evil.com:repo.git",  # 2 @ symbols
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
            # The error could be either suspicious encoding or invalid characters
            assert any(phrase in str(error.value) for phrase in [
                "Suspicious URL encoding detected",
                "Invalid characters in hostname"
            ])

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
            # The error could be various types due to URL parsing edge cases
            assert any(phrase in str(error.value) for phrase in [
                "Invalid characters in hostname",
                "Failed to parse URL",
                "does not appear to be an IPv4 or IPv6 address"
            ])

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
            "branch": "main"
        }
        entry_point = "train.py"
        
        with pytest.raises(ValueError) as error:
            git_utils.git_clone_repo(malicious_git_config, entry_point)
        assert "multiple @ symbols detected" in str(error.value)

    def test_git_clone_repo_blocks_malicious_ssh_url(self):
        """Test that git_clone_repo blocks malicious SSH URLs."""
        malicious_git_config = {
            "repo": "git@OBVIOUS@github.com:sage-maker/temp-sev2.git",
            "branch": "main"
        }
        entry_point = "train.py"
        
        with pytest.raises(ValueError) as error:
            git_utils.git_clone_repo(malicious_git_config, entry_point)
        assert "exactly one @ symbol" in str(error.value)

    def test_git_clone_repo_blocks_url_encoded_attack(self):
        """Test that git_clone_repo blocks URL-encoded attacks."""
        malicious_git_config = {
            "repo": "https://github.com%40attacker.com/repo.git",
            "branch": "main"
        }
        entry_point = "train.py"
        
        with pytest.raises(ValueError) as error:
            git_utils.git_clone_repo(malicious_git_config, entry_point)
        assert "Suspicious URL encoding detected" in str(error.value)

    def test_sanitize_git_url_comprehensive_attack_scenarios(self):
        """Test comprehensive attack scenarios from the vulnerability report."""
        # These are the exact attack patterns from the security report
        attack_scenarios = [
            # Original PoC attack
            "https://USER@YOUR_NGROK_OR_LOCALHOST/malicious.git@github.com%25legit%25repo.git",
            # Variations of the attack
            "https://user@malicious-host@github.com/legit/repo.git",
            "git@attacker.com@github.com:user/repo.git",
            "ssh://git@evil.com@github.com/repo.git",
            # URL encoding variations
            "https://github.com%40evil.com/repo.git",
            "https://user@github.com%2Fevil.com/repo.git",
        ]
        
        entry_point = "train.py"
        
        for malicious_url in attack_scenarios:
            git_config = {"repo": malicious_url}
            with pytest.raises(ValueError) as error:
                git_utils.git_clone_repo(git_config, entry_point)
            # Should be blocked by sanitization
            assert any(phrase in str(error.value) for phrase in [
                "multiple @ symbols detected",
                "exactly one @ symbol",
                "Suspicious URL encoding detected",
                "Invalid characters in hostname"
            ])
