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

import os
import subprocess
import tempfile


def git_clone_repo(git_config, entry_point, source_dir=None, dependencies=None):
    """Git clone repo containing the training code and serving code. This method also validate ``git_config``,
    and set ``entry_point``, ``source_dir`` and ``dependencies`` to the right file or directory in the repo cloned.

    Args:
        git_config (dict[str, str]): Git configurations used for cloning files, including ``repo``, ``branch``
            and ``commit``. ``branch`` and ``commit`` are optional. If ``branch`` is not specified, master branch
            will be used. If ``commit`` is not specified, the latest commit in the required branch will be used.
        entry_point (str): A relative location to the Python source file which should be executed as the entry point
            to training or model hosting in the Git repo.
        source_dir (str): A relative location to a directory with other training or model hosting source code
            dependencies aside from the entry point file in the Git repo (default: None). Structure within this
            directory are preserved when training on Amazon SageMaker.
        dependencies (list[str]): A list of relative locations to directories with any additional libraries that will
            be exported to the container in the Git repo (default: []).

    Raises:
        CalledProcessError: If 1. failed to clone git repo
                               2. failed to checkout the required branch
                               3. failed to checkout the required commit
        ValueError: If 1. entry point specified does not exist in the repo
                       2. source dir specified does not exist in the repo

    Returns:
        dict: A dict that contains the updated values of entry_point, source_dir and dependencies
    """
    _validate_git_config(git_config)
    repo_dir = tempfile.mkdtemp()
    subprocess.check_call(["git", "clone", git_config["repo"], repo_dir])

    _checkout_branch_and_commit(git_config, repo_dir)

    ret = {"entry_point": entry_point, "source_dir": source_dir, "dependencies": dependencies}
    # check if the cloned repo contains entry point, source directory and dependencies
    if source_dir:
        if not os.path.isdir(os.path.join(repo_dir, source_dir)):
            raise ValueError("Source directory does not exist in the repo.")
        if not os.path.isfile(os.path.join(repo_dir, source_dir, entry_point)):
            raise ValueError("Entry point does not exist in the repo.")
        ret["source_dir"] = os.path.join(repo_dir, source_dir)
    else:
        if not os.path.isfile(os.path.join(repo_dir, entry_point)):
            raise ValueError("Entry point does not exist in the repo.")
        ret["entry_point"] = os.path.join(repo_dir, entry_point)

    ret["dependencies"] = []
    for path in dependencies:
        if not os.path.exists(os.path.join(repo_dir, path)):
            raise ValueError("Dependency {} does not exist in the repo.".format(path))
        ret["dependencies"].append(os.path.join(repo_dir, path))
    return ret


def _validate_git_config(git_config):
    """check if a git_config param is valid

    Args:
        git_config ((dict[str, str]): Git configurations used for cloning files, including ``repo``, ``branch``
            and ``commit``.

    Raises:
        ValueError: If:
            1. git_config has no key 'repo'
            2. git_config['repo'] is in the wrong format.
    """
    if "repo" not in git_config:
        raise ValueError("Please provide a repo for git_config.")


def _checkout_branch_and_commit(git_config, repo_dir):
    """Checkout the required branch and commit.

    Args:
        git_config: (dict[str, str]): Git configurations used for cloning files, including ``repo``, ``branch``
            and ``commit``.
        repo_dir (str): the directory where the repo is cloned

    Raises:
        ValueError: If 1. entry point specified does not exist in the repo
                       2. source dir specified does not exist in the repo
    """
    if "branch" in git_config:
        subprocess.check_call(args=["git", "checkout", git_config["branch"]], cwd=str(repo_dir))
    if "commit" in git_config:
        subprocess.check_call(args=["git", "checkout", git_config["commit"]], cwd=str(repo_dir))
