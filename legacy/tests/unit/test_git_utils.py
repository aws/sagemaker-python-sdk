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
import os
from pathlib import Path
import subprocess
from mock import patch, ANY

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
