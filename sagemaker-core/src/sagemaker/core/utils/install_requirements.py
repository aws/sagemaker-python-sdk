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
    - ``install_requirements(path)`` — configures pip and runs ``pip install -r``.
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


def configure_pip(auth_method=CodeArtifactAuthMethod.AUTO):
    """Configure pip for CodeArtifact if ``CA_REPOSITORY_ARN`` is set.

    Args:
        auth_method: Authentication mechanism to use. Defaults to ``CodeArtifactAuthMethod.AUTO``
            (try boto3 first, fall back to AWS CLI).

    Returns:
        An authenticated pip index URL (str) when boto3 succeeds,
        ``None`` when AWS CLI was used (pip config modified globally),
        or ``None`` when ``CA_REPOSITORY_ARN`` is not set.

    Raises:
        SystemExit: When ``CA_REPOSITORY_ARN`` is set but the requested
            auth method is not available.
        ValueError: When the ARN format is invalid.
    """
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
            return _get_index_boto3(region, account, domain, repo)
        except ImportError:
            if auth_method == CodeArtifactAuthMethod.BOTO3:
                logger.error("boto3 is not available")
                sys.exit(1)
            logger.info("boto3 not available, trying AWS CLI fallback")

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


def install_requirements(
    requirements_file="requirements.txt", python_executable=None, auth_method=CodeArtifactAuthMethod.AUTO
):
    """Install pip requirements with optional CodeArtifact authentication.

    Args:
        requirements_file: Path to the requirements file.
        python_executable: Python executable to use for pip. Defaults to ``sys.executable``.
        auth_method: Authentication mechanism for CodeArtifact. Defaults to ``CodeArtifactAuthMethod.AUTO``.
    """
    python_executable = python_executable or sys.executable
    pip_cmd = [python_executable, "-m", "pip", "install", "-r", requirements_file]
    index = configure_pip(auth_method=auth_method)
    if index:
        pip_cmd.extend(["-i", index])
    logger.info("Running: %s", " ".join(pip_cmd))
    subprocess.check_call(pip_cmd)


def main():
    """CLI entry point."""
    req_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    install_requirements(req_file)


if __name__ == "__main__":
    main()
