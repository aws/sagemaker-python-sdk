# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import subprocess
from mock import patch

from sagemaker import git_utils

REPO_DIR = "/tmp/repo_dir"
GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "329bfcf884482002c05ff7f44f62599ebc9f445a"


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_succeed(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    ret = git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    check_call.assert_any_call(["git", "clone", git_config["repo"], REPO_DIR])
    check_call.assert_any_call(args=["git", "checkout", BRANCH], cwd=REPO_DIR)
    check_call.assert_any_call(args=["git", "checkout", COMMIT], cwd=REPO_DIR)
    mkdtemp.assert_called_once()
    assert ret["entry_point"] == "entry_point"
    assert ret["source_dir"] == "/tmp/repo_dir/source_dir"
    assert ret["dependencies"] == ["/tmp/repo_dir/foo", "/tmp/repo_dir/bar"]


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_repo_not_provided(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point_that_does_not_exist"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "Please provide a repo for git_config." in str(error)


@patch(
    "subprocess.check_call",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(GIT_REPO, REPO_DIR)
    ),
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_clone_fail(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "returned non-zero exit status" in str(error)


@patch(
    "subprocess.check_call",
    side_effect=[True, subprocess.CalledProcessError(returncode=1, cmd="git checkout banana")],
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_branch_not_exist(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "returned non-zero exit status" in str(error)


@patch(
    "subprocess.check_call",
    side_effect=[
        True,
        True,
        subprocess.CalledProcessError(returncode=1, cmd="git checkout {}".format(COMMIT)),
    ],
)
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_commit_not_exist(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "returned non-zero exit status" in str(error)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=False)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_entry_point_not_exist(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point_that_does_not_exist"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "Entry point does not exist in the repo." in str(error)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=False)
@patch("os.path.exists", return_value=True)
def test_git_clone_repo_source_dir_not_exist(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir_that_does_not_exist"
    dependencies = ["foo", "bar"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "Source directory does not exist in the repo." in str(error)


@patch("subprocess.check_call")
@patch("tempfile.mkdtemp", return_value=REPO_DIR)
@patch("os.path.isfile", return_value=True)
@patch("os.path.isdir", return_value=True)
@patch("os.path.exists", side_effect=[True, False])
def test_git_clone_repo_dependencies_not_exist(exists, isdir, isfile, mkdtemp, check_call):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "dep_that_does_not_exist"]
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert "does not exist in the repo." in str(error)
