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
"""CodeArtifact-aware pip requirements installer.

Reads ``CA_REPOSITORY_ARN`` from the environment and authenticates with
CodeArtifact before installing packages.  Tries boto3 first (matching
``sagemaker-training-toolkit``), falls back to AWS CLI, and hard-fails
when the env var is set but neither mechanism is available.

Can be used as:
  - An importable module:

    - ``configure_pip()`` — returns an authenticated pip index URL (or ``None``).
      Use when you need to build your own pip command with custom flags.
    - ``install_requirements(path)`` — configures pip and installs with ``uv``.
      Use when you just want requirements installed.

    ::

        from sagemaker.core.utils.install_requirements import configure_pip, install_requirements

  - A standalone script:   ``python install_requirements.py requirements.txt``
"""

from __future__ import absolute_import

import enum
import logging
import os
import re
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

CA_REPOSITORY_ARN_ENV = "CA_REPOSITORY_ARN"

_ARN_RE = re.compile(r"arn:([^:]+):codeartifact:([^:]+):([^:]+):repository/([^/]+)/(.+)")


class CodeArtifactAuthMethod(enum.Enum):
    """Authentication method for CodeArtifact pip configuration."""

    BOTO3 = "boto3"
    """Use boto3 only. Fails if boto3 is not available."""

    AWS_CLI = "aws_cli"
    """Use AWS CLI only. Fails if AWS CLI is not available."""

    AUTO = "auto"
    """Try boto3 first, fall back to AWS CLI, hard-fail if neither is available."""


def _parse_arn(arn):
    """Parse a CodeArtifact repository ARN into its components.

    Returns:
        Tuple of (region, account, domain, repository) or raises ValueError.
    """
    m = _ARN_RE.match(arn)
    if not m:
        raise ValueError(f"Invalid {CA_REPOSITORY_ARN_ENV}: {arn}")
    _, region, account, domain, repo = m.groups()
    return region, account, domain, repo


def _get_index_boto3(region, account, domain, repo):
    """Build an authenticated pip index URL using boto3."""
    import boto3  # noqa: delay import — may not be installed

    ca = boto3.client("codeartifact", region_name=region)
    token = ca.get_authorization_token(domain=domain, domainOwner=account)["authorizationToken"]
    endpoint = ca.get_repository_endpoint(
        domain=domain, domainOwner=account, repository=repo, format="pypi"
    )["repositoryEndpoint"]
    return re.sub(
        "https://",
        f"https://aws:{token}@",
        re.sub(f"{repo}/?$", f"{repo}/simple/", endpoint),
    )


def _login_awscli(region, account, domain, repo):
    """Configure pip globally via ``aws codeartifact login``."""
    subprocess.check_call(
        [
            "aws",
            "codeartifact",
            "login",
            "--tool",
            "pip",
            "--domain",
            domain,
            "--domain-owner",
            account,
            "--repository",
            repo,
            "--region",
            region,
        ]
    )


def _set_pip_index(python_executable, index_url):
    """Persist an authenticated index URL into pip config.

    Mirrors what ``aws codeartifact login`` does for the CLI path: writes
    ``global.index-url`` so the container's pip — including the ``uv`` bootstrap in
    :func:`_ensure_uv` — pulls from CodeArtifact. This matters in isolated
    environments with no public PyPI access. (``uv`` ignores ``pip.conf``, so the
    index is still surfaced to uv separately via ``UV_INDEX_URL``.)
    """
    subprocess.check_call(
        [python_executable, "-m", "pip", "config", "set", "global.index-url", index_url]
    )


def configure_pip(auth_method=CodeArtifactAuthMethod.AUTO, python_executable=None):
    """Configure pip for CodeArtifact if ``CA_REPOSITORY_ARN`` is set.

    Both auth paths persist the authenticated index into pip config, so any later
    pip invocation — including the ``uv`` bootstrap in :func:`_ensure_uv` — uses
    CodeArtifact even in isolated environments without public PyPI access.

    Args:
        auth_method: Authentication mechanism to use. Defaults to ``CodeArtifactAuthMethod.AUTO``
            (try boto3 first, fall back to AWS CLI).
        python_executable: Python executable whose pip config is written on the
            boto3 path. Defaults to ``sys.executable``.

    Returns:
        An authenticated pip index URL (str) when boto3 succeeds (also written to
        pip config), ``None`` when AWS CLI was used (pip config modified globally),
        or ``None`` when ``CA_REPOSITORY_ARN`` is not set.

    Raises:
        SystemExit: When ``CA_REPOSITORY_ARN`` is set but the requested
            auth method is not available.
        ValueError: When the ARN format is invalid.
    """
    python_executable = python_executable or sys.executable
    arn = os.environ.get(CA_REPOSITORY_ARN_ENV)
    if not arn:
        return None

    region, account, domain, repo = _parse_arn(arn)
    logger.info(
        "Configuring pip for CodeArtifact "
        "(domain=%s, domain_owner=%s, repository=%s, region=%s)",
        domain,
        account,
        repo,
        region,
    )

    if auth_method in (CodeArtifactAuthMethod.BOTO3, CodeArtifactAuthMethod.AUTO):
        try:
            index = _get_index_boto3(region, account, domain, repo)
        except ImportError:
            if auth_method == CodeArtifactAuthMethod.BOTO3:
                logger.error("boto3 is not available")
                sys.exit(1)
            logger.info("boto3 not available, trying AWS CLI fallback")
        else:
            _set_pip_index(python_executable, index)
            return index

    if auth_method in (CodeArtifactAuthMethod.AWS_CLI, CodeArtifactAuthMethod.AUTO):
        try:
            _login_awscli(region, account, domain, repo)
            return None
        except FileNotFoundError:
            if auth_method == CodeArtifactAuthMethod.AWS_CLI:
                logger.error("AWS CLI is not available")
                sys.exit(1)
            logger.info("AWS CLI not available")

    # Hard fail — CA is configured but we can't authenticate
    logger.error(
        "%s is set but neither boto3 nor AWS CLI is available "
        "to authenticate with CodeArtifact.",
        CA_REPOSITORY_ARN_ENV,
    )
    sys.exit(1)


def _ensure_uv(python_executable):
    """Return a path to a ``uv`` executable, bootstrapping it with pip if absent.

    Some containers don't ship ``uv``. When it's missing we install it with the
    container's pip (which has already been pointed at CodeArtifact, if configured).

    Args:
        python_executable: Python executable whose pip is used to bootstrap ``uv``.

    Returns:
        Path to the ``uv`` executable (str).
    """
    uv = shutil.which("uv")
    if uv:
        return uv
    logger.info("uv not found; bootstrapping it with pip")
    subprocess.check_call([python_executable, "-m", "pip", "install", "uv"])
    return shutil.which("uv") or "uv"


def _pip_config_get(python_executable, key):
    """Read a pip config value (e.g. set by ``aws codeartifact login``).

    ``uv`` ignores ``pip.conf``, so any index configured globally on pip must be
    read back and propagated to ``uv`` explicitly. Returns ``None`` when the key
    is unset or pip can't report it.

    Args:
        python_executable: Python executable whose pip config is queried.
        key: pip config key, e.g. ``global.index-url``.
    """
    try:
        out = subprocess.check_output(
            [python_executable, "-m", "pip", "config", "get", key],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = out.decode().strip()
    return value or None


def _build_uv_index_env(python_executable, index):
    """Build the ``UV_*`` environment for index configuration.

    Bridges both CodeArtifact auth paths and any pre-existing pip config into the
    environment variables ``uv`` understands:

    - ``index`` (from boto3 auth) becomes ``UV_INDEX_URL``.
    - When boto3 didn't yield an index (CLI login wrote ``pip.conf``, or the user
      pre-configured pip), read ``global.index-url`` back from pip config.
    - ``global.extra-index-url`` / ``global.trusted-host`` are always propagated
      when present, to stay general across private-index setups.

    Args:
        python_executable: Python executable whose pip config is consulted.
        index: Authenticated index URL from :func:`configure_pip`, or ``None``.

    Returns:
        A dict of ``UV_*`` overrides to merge into the subprocess environment.
    """
    env = {}

    index_url = index or _pip_config_get(python_executable, "global.index-url")
    if index_url:
        env["UV_INDEX_URL"] = index_url

    extra_index_url = _pip_config_get(python_executable, "global.extra-index-url")
    if extra_index_url:
        env["UV_EXTRA_INDEX_URL"] = extra_index_url

    trusted_host = _pip_config_get(python_executable, "global.trusted-host")
    if trusted_host:
        env["UV_INSECURE_HOST"] = trusted_host

    return env


def install_requirements(
    requirements_file="requirements.txt", python_executable=None, auth_method=CodeArtifactAuthMethod.AUTO
):
    """Install requirements with ``uv`` and optional CodeArtifact authentication.

    Configures CodeArtifact (if ``CA_REPOSITORY_ARN`` is set), bootstraps ``uv``
    when the container doesn't ship it, propagates any private-index configuration
    into ``uv``'s environment, and installs the requirements into the system
    interpreter.

    Args:
        requirements_file: Path to the requirements file.
        python_executable: Python executable used to bootstrap ``uv`` and read pip
            config. Defaults to ``sys.executable``.
        auth_method: Authentication mechanism for CodeArtifact. Defaults to ``CodeArtifactAuthMethod.AUTO``.
    """
    python_executable = python_executable or sys.executable

    index = configure_pip(auth_method=auth_method, python_executable=python_executable)

    uv = _ensure_uv(python_executable)
    env = os.environ.copy()
    env.update(_build_uv_index_env(python_executable, index))

    install_cmd = [uv, "pip", "install", "--system", "-r", requirements_file]
    logger.info("Running: %s", " ".join(install_cmd))
    subprocess.check_call(install_cmd, env=env)


def main():
    """CLI entry point."""
    req_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    install_requirements(req_file)


if __name__ == "__main__":
    main()
