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
import six
import subprocess
import tempfile
import warnings
from six.moves import urllib


def git_clone_repo(git_config, entry_point, source_dir=None, dependencies=None):
    """Git clone repo containing the training code and serving code. This method also validate ``git_config``,
    and set ``entry_point``, ``source_dir`` and ``dependencies`` to the right file or directory in the repo cloned.

    Args:
        git_config (dict[str, str]): Git configurations used for cloning files, including ``repo``, ``branch``,
            ``commit``, ``2FA_enabled``, ``username``, ``password`` and ``token``. The fields are optional except
            ``repo``. If ``branch`` is not specified, master branch will be used. If ``commit`` is not specified,
            the latest commit in the required branch will be used. ``2FA_enabled``, ``username``, ``password`` and
            ``token`` are for authentication purpose.
            If ``2FA_enabled`` is not provided, we consider 2FA as disabled. For GitHub and GitHub-like repos, when
            ssh urls are provided, it does not make a difference whether 2FA is enabled or disabled; an ssh passphrase
            should be in local storage. When https urls are provided: if 2FA is disabled, then either token or
            username+password will be used for authentication if provided (token prioritized); if 2FA is enabled,
            only token will be used for authentication if provided. If required authentication info is not provided,
            python SDK will try to use local credentials storage to authenticate. If that fails either, an error message
            will be thrown.
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
                       3. dependencies specified do not exist in the repo
                       4. wrong format is provided for git_config

    Returns:
        dict: A dict that contains the updated values of entry_point, source_dir and dependencies.
    """
    _validate_git_config(git_config)
    repo_dir = tempfile.mkdtemp()
    _generate_and_run_clone_command(git_config, repo_dir)

    _checkout_branch_and_commit(git_config, repo_dir)

    updated_paths = {
        "entry_point": entry_point,
        "source_dir": source_dir,
        "dependencies": dependencies,
    }

    # check if the cloned repo contains entry point, source directory and dependencies
    if source_dir:
        if not os.path.isdir(os.path.join(repo_dir, source_dir)):
            raise ValueError("Source directory does not exist in the repo.")
        if not os.path.isfile(os.path.join(repo_dir, source_dir, entry_point)):
            raise ValueError("Entry point does not exist in the repo.")
        updated_paths["source_dir"] = os.path.join(repo_dir, source_dir)
    else:
        if os.path.isfile(os.path.join(repo_dir, entry_point)):
            updated_paths["entry_point"] = os.path.join(repo_dir, entry_point)
        else:
            raise ValueError("Entry point does not exist in the repo.")
    if dependencies is None:
        updated_paths["dependencies"] = None
    else:
        updated_paths["dependencies"] = []
        for path in dependencies:
            if os.path.exists(os.path.join(repo_dir, path)):
                updated_paths["dependencies"].append(os.path.join(repo_dir, path))
            else:
                raise ValueError("Dependency {} does not exist in the repo.".format(path))
    return updated_paths


def _validate_git_config(git_config):
    if "repo" not in git_config:
        raise ValueError("Please provide a repo for git_config.")
    string_args = ["repo", "branch", "commit", "username", "password", "token"]
    for key in string_args:
        if key in git_config and not isinstance(git_config[key], six.string_types):
            raise ValueError("'{}' must be a string.".format(key))
    if "2FA_enabled" in git_config and not isinstance(git_config["2FA_enabled"], bool):
        raise ValueError("'2FA_enabled' must be a bool value.")
    allowed_keys = ["repo", "branch", "commit", "2FA_enabled", "username", "password", "token"]
    for k in list(git_config):
        if k not in allowed_keys:
            raise ValueError("Unexpected git_config argument(s) provided!")


def _generate_and_run_clone_command(git_config, repo_dir):
    """check if a git_config param is valid, if it is, create the command to git clone the repo, and run it.

    Args:
        git_config ((dict[str, str]): Git configurations used for cloning files, including ``repo``, ``branch``
            and ``commit``.
        repo_dir (str): The local directory to clone the Git repo into.

    Raises:
        CalledProcessError: If failed to clone git repo.
    """
    exists = {
        "2FA_enabled": "2FA_enabled" in git_config and git_config["2FA_enabled"] is True,
        "username": "username" in git_config,
        "password": "password" in git_config,
        "token": "token" in git_config,
    }
    _clone_command_for_github_like(git_config, repo_dir, exists)


def _clone_command_for_github_like(git_config, repo_dir, exists):
    """check if a git_config param representing a GitHub (or like) repo is valid, if it is, create the command to
    git clone the repo, and run it.

    Args:
        git_config ((dict[str, str]): Git configurations used for cloning files, including ``repo``, ``branch``
            and ``commit``.
        repo_dir (str): The local directory to clone the Git repo into.

    Raises:
        ValueError: If git_config['repo'] is in the wrong format.
        CalledProcessError: If failed to clone git repo.
    """
    is_https = git_config["repo"].startswith("https://")
    is_ssh = git_config["repo"].startswith("git@")
    if not is_https and not is_ssh:
        raise ValueError("Invalid Git url provided.")
    if is_ssh:
        _clone_command_for_github_like_ssh(git_config, repo_dir, exists)
    elif exists["2FA_enabled"]:
        _clone_command_for_github_like_https_2fa_enabled(git_config, repo_dir, exists)
    else:
        _clone_command_for_github_like_https_2fa_disabled(git_config, repo_dir, exists)


def _clone_command_for_github_like_ssh(git_config, repo_dir, exists):
    if exists["username"] or exists["password"] or exists["token"]:
        warnings.warn("Unnecessary credential argument(s) provided.")
    _run_clone_command(git_config["repo"], repo_dir)


def _clone_command_for_github_like_https_2fa_disabled(git_config, repo_dir, exists):
    updated_url = git_config["repo"]
    if exists["token"]:
        if exists["username"] or exists["password"]:
            warnings.warn(
                "Using token for authentication, "
                "but unnecessary credential argument(s) provided."
            )
        updated_url = _insert_token_to_repo_url(url=git_config["repo"], token=git_config["token"])
    elif exists["username"] and exists["password"]:
        updated_url = _insert_username_and_password_to_repo_url(
            url=git_config["repo"], username=git_config["username"], password=git_config["password"]
        )
    elif exists["username"] or exists["password"]:
        warnings.warn("Unnecessary credential argument(s) provided.")
    _run_clone_command(updated_url, repo_dir)


def _clone_command_for_github_like_https_2fa_enabled(git_config, repo_dir, exists):
    updated_url = git_config["repo"]
    if exists["token"]:
        if exists["username"] or exists["password"]:
            warnings.warn(
                "Using token for authentication, "
                "but unnecessary credential argument(s) provided."
            )
        updated_url = _insert_token_to_repo_url(url=git_config["repo"], token=git_config["token"])
    elif exists["username"] or exists["password"] or exists["token"]:
        warnings.warn(
            "Unnecessary credential argument(s) provided."
            "Hint: since two factor authentication is enabled, you have to provide token."
        )
    _run_clone_command(updated_url, repo_dir)


def _run_clone_command(repo_url, repo_dir):
    """Run the 'git clone' command with the repo url and the directory to clone the repo into.

    Args:
        repo_url (str): Git repo url to be cloned.
        repo_dir: (str): Local path where the repo should be cloned into.

    Raises:
        CalledProcessError: If failed to clone git repo.
    """
    my_env = os.environ.copy()
    if repo_url.startswith("https://"):
        my_env["GIT_TERMINAL_PROMPT"] = "0"
    elif repo_url.startswith("git@"):
        f = tempfile.NamedTemporaryFile()
        w = open(f.name, "w")
        w.write("ssh -oBatchMode=yes $@")
        w.close()
        # 511 in decimal is same as 777 in octal
        os.chmod(f.name, 511)
        my_env["GIT_SSH"] = f.name
    subprocess.check_call(["git", "clone", repo_url, repo_dir], env=my_env)


def _insert_token_to_repo_url(url, token):
    """Insert the token to the Git repo url, to make a component of the git clone command. This method can
    only be called when repo_url is an https url.

    Args:
        url (str): Git repo url where the token should be inserted into.
        token (str): Token to be inserted.

    Returns:
        str: the component needed fot the git clone command.
    """
    index = len("https://")
    return url[:index] + token + "@" + url[index:]


def _insert_username_and_password_to_repo_url(url, username, password):
    """Insert the username and the password to the Git repo url, to make a component of the git clone command.
    This method can only be called when repo_url is an https url.

    Args:
        url (str): Git repo url where the token should be inserted into.
        username (str): Username to be inserted.
        password (str): Password to be inserted.

    Returns:
        str: the component needed fot the git clone command.
    """
    password = urllib.parse.quote_plus(password)
    # urllib parses ' ' as '+', but what we need is '%20' here
    password = password.replace("+", "%20")
    index = len("https://")
    return url[:index] + username + ":" + password + "@" + url[index:]


def _checkout_branch_and_commit(git_config, repo_dir):
    """Checkout the required branch and commit.

    Args:
        git_config (dict[str, str]): Git configurations used for cloning files, including ``repo``, ``branch``
            and ``commit``.
        repo_dir (str): the directory where the repo is cloned

    Raises:
        CalledProcessError: If 1. failed to checkout the required branch
                               2. failed to checkout the required commit
    """
    if "branch" in git_config:
        subprocess.check_call(args=["git", "checkout", git_config["branch"]], cwd=str(repo_dir))
    if "commit" in git_config:
        subprocess.check_call(args=["git", "checkout", git_config["commit"]], cwd=str(repo_dir))
