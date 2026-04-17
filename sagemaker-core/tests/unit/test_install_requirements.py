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

import subprocess
import sys
from unittest import mock

import pytest

from sagemaker.core.utils.install_requirements import (
    CA_REPOSITORY_ARN_ENV,
    CodeArtifactAuthMethod,
    _parse_arn,
    configure_pip,
    install_requirements,
    main,
)

_MODULE = "sagemaker.core.utils.install_requirements"

VALID_ARN = "arn:aws:codeartifact:us-west-2:123456789012:repository/my-domain/my-repo"
PARSED = ("us-west-2", "123456789012", "my-domain", "my-repo")
FAKE_TOKEN = "fake-auth-token"
FAKE_ENDPOINT = (
    "https://my-domain-123456789012.d.codeartifact.us-west-2.amazonaws.com/pypi/my-repo/"
)
EXPECTED_INDEX = (
    f"https://aws:{FAKE_TOKEN}"
    "@my-domain-123456789012.d.codeartifact.us-west-2.amazonaws.com/pypi/my-repo/simple/"
)
EXPECTED_CLI_CMD = [
    "aws", "codeartifact", "login", "--tool", "pip",
    "--domain", "my-domain", "--domain-owner", "123456789012",
    "--repository", "my-repo", "--region", "us-west-2",
]  # fmt: skip


@pytest.fixture()
def ca_env():
    """Set CA_REPOSITORY_ARN in the environment for the duration of a test."""
    with mock.patch.dict("os.environ", {CA_REPOSITORY_ARN_ENV: VALID_ARN}):
        yield


@pytest.fixture()
def mock_boto3_ca():
    """Mock boto3 CodeArtifact client with valid responses."""
    client = mock.MagicMock()
    client.get_authorization_token.return_value = {"authorizationToken": FAKE_TOKEN}
    client.get_repository_endpoint.return_value = {"repositoryEndpoint": FAKE_ENDPOINT}
    with mock.patch("boto3.client", return_value=client) as factory:
        yield factory, client


def _pip_cmd(*extra):
    return [sys.executable, "-m", "pip", "install", "-r", "reqs.txt", *extra]


# ---------------------------------------------------------------------------
# _parse_arn
# ---------------------------------------------------------------------------
class TestParseArn:
    def test_valid_arn(self):
        assert _parse_arn(VALID_ARN) == PARSED

    def test_invalid_arn(self):
        with pytest.raises(ValueError, match="Invalid CA_REPOSITORY_ARN"):
            _parse_arn("not-an-arn")

    def test_arn_with_nested_repo(self):
        region, account, domain, repo = _parse_arn(
            "arn:aws:codeartifact:eu-west-1:111111111111:repository/dom/nested/repo"
        )
        assert (region, account, domain, repo) == (
            "eu-west-1",
            "111111111111",
            "dom",
            "nested/repo",
        )


# ---------------------------------------------------------------------------
# configure_pip — AUTO (default)
# ---------------------------------------------------------------------------
class TestConfigurePipAuto:
    def test_no_env_var_returns_none(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            assert configure_pip() is None

    def test_invalid_arn_raises(self, ca_env):
        with mock.patch.dict("os.environ", {CA_REPOSITORY_ARN_ENV: "garbage"}):
            with pytest.raises(ValueError, match="Invalid"):
                configure_pip()

    def test_boto3_success(self, ca_env, mock_boto3_ca):
        factory, client = mock_boto3_ca
        result = configure_pip()

        factory.assert_called_once_with("codeartifact", region_name="us-west-2")
        client.get_authorization_token.assert_called_once_with(
            domain="my-domain", domainOwner="123456789012"
        )
        client.get_repository_endpoint.assert_called_once_with(
            domain="my-domain", domainOwner="123456789012", repository="my-repo", format="pypi"
        )
        assert result == EXPECTED_INDEX

    def test_falls_back_to_cli_when_no_boto3(self, ca_env):
        with mock.patch(f"{_MODULE}._get_index_boto3", side_effect=ImportError):
            with mock.patch("subprocess.check_call") as mock_call:
                result = configure_pip()

        mock_call.assert_called_once_with(EXPECTED_CLI_CMD)
        assert result is None

    def test_hard_fails_when_nothing_available(self, ca_env):
        with mock.patch(f"{_MODULE}._get_index_boto3", side_effect=ImportError):
            with mock.patch(f"{_MODULE}._login_awscli", side_effect=FileNotFoundError):
                with pytest.raises(SystemExit):
                    configure_pip()

    def test_boto3_api_error_propagates_not_fallback(self, ca_env):
        """boto3 available but API call fails → raise, don't fall back to CLI."""
        with mock.patch(f"{_MODULE}._get_index_boto3", side_effect=Exception("AccessDenied")):
            with mock.patch(f"{_MODULE}._login_awscli") as mock_cli:
                with pytest.raises(Exception, match="AccessDenied"):
                    configure_pip()
            mock_cli.assert_not_called()


# ---------------------------------------------------------------------------
# configure_pip — BOTO3 only
# ---------------------------------------------------------------------------
class TestConfigurePipBoto3Only:
    def test_fails_when_unavailable(self, ca_env):
        with mock.patch(f"{_MODULE}._get_index_boto3", side_effect=ImportError):
            with pytest.raises(SystemExit):
                configure_pip(auth_method=CodeArtifactAuthMethod.BOTO3)

    def test_does_not_try_cli(self, ca_env, mock_boto3_ca):
        with mock.patch(f"{_MODULE}._login_awscli") as mock_cli:
            configure_pip(auth_method=CodeArtifactAuthMethod.BOTO3)
        mock_cli.assert_not_called()


# ---------------------------------------------------------------------------
# configure_pip — AWS_CLI only
# ---------------------------------------------------------------------------
class TestConfigurePipCliOnly:
    def test_succeeds(self, ca_env):
        with mock.patch("subprocess.check_call") as mock_call:
            result = configure_pip(auth_method=CodeArtifactAuthMethod.AWS_CLI)
        assert result is None
        mock_call.assert_called_once()

    def test_fails_when_unavailable(self, ca_env):
        with mock.patch(f"{_MODULE}._login_awscli", side_effect=FileNotFoundError):
            with pytest.raises(SystemExit):
                configure_pip(auth_method=CodeArtifactAuthMethod.AWS_CLI)

    def test_does_not_try_boto3(self, ca_env):
        with mock.patch(f"{_MODULE}._get_index_boto3") as mock_boto3:
            with mock.patch("subprocess.check_call"):
                configure_pip(auth_method=CodeArtifactAuthMethod.AWS_CLI)
        mock_boto3.assert_not_called()


# ---------------------------------------------------------------------------
# install_requirements
# ---------------------------------------------------------------------------
class TestInstallRequirements:
    def test_without_codeartifact(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("subprocess.check_call") as mock_call:
                install_requirements("reqs.txt")
        mock_call.assert_called_once_with(_pip_cmd())

    def test_with_codeartifact_index(self):
        with mock.patch(f"{_MODULE}.configure_pip", return_value=EXPECTED_INDEX):
            with mock.patch("subprocess.check_call") as mock_call:
                install_requirements("reqs.txt")
        mock_call.assert_called_once_with(_pip_cmd("-i", EXPECTED_INDEX))

    def test_with_cli_fallback_no_index_flag(self):
        with mock.patch(f"{_MODULE}.configure_pip", return_value=None):
            with mock.patch("subprocess.check_call") as mock_call:
                install_requirements("reqs.txt")
        mock_call.assert_called_once_with(_pip_cmd())

    def test_custom_python_executable(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("subprocess.check_call") as mock_call:
                install_requirements("reqs.txt", python_executable="/usr/bin/python3")
        mock_call.assert_called_once_with(
            ["/usr/bin/python3", "-m", "pip", "install", "-r", "reqs.txt"]
        )

    def test_pip_failure_propagates(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(
                "subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip")
            ):
                with pytest.raises(subprocess.CalledProcessError):
                    install_requirements("reqs.txt")

    def test_auth_method_passed_through(self):
        with mock.patch(f"{_MODULE}.configure_pip", return_value=None) as mock_configure:
            with mock.patch("subprocess.check_call"):
                install_requirements("reqs.txt", auth_method=CodeArtifactAuthMethod.BOTO3)
        mock_configure.assert_called_once_with(auth_method=CodeArtifactAuthMethod.BOTO3)


# ---------------------------------------------------------------------------
# main (CLI entry point)
# ---------------------------------------------------------------------------
class TestMain:
    def test_default_requirements_file(self):
        with mock.patch(f"{_MODULE}.install_requirements") as mock_install:
            with mock.patch("sys.argv", ["install_requirements.py"]):
                main()
        mock_install.assert_called_once_with("requirements.txt")

    def test_custom_requirements_file(self):
        with mock.patch(f"{_MODULE}.install_requirements") as mock_install:
            with mock.patch("sys.argv", ["install_requirements.py", "custom.txt"]):
                main()
        mock_install.assert_called_once_with("custom.txt")
