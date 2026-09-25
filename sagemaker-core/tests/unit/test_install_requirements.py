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

UV_PATH = "/usr/bin/uv"

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


def _uv_cmd(requirements="reqs.txt"):
    return [UV_PATH, "pip", "install", "--system", "-r", requirements]


def _pip_config_set_cmd(index=EXPECTED_INDEX):
    return [sys.executable, "-m", "pip", "config", "set", "global.index-url", index]


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
        with mock.patch("subprocess.check_call") as mock_call:
            result = configure_pip()

        factory.assert_called_once_with("codeartifact", region_name="us-west-2")
        client.get_authorization_token.assert_called_once_with(
            domain="my-domain", domainOwner="123456789012"
        )
        client.get_repository_endpoint.assert_called_once_with(
            domain="my-domain", domainOwner="123456789012", repository="my-repo", format="pypi"
        )
        assert result == EXPECTED_INDEX
        # boto3 path also persists the index to pip config so the uv bootstrap
        # (and any other pip call) inherits CodeArtifact in isolated environments.
        mock_call.assert_called_once_with(_pip_config_set_cmd())

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
            with mock.patch("subprocess.check_call"):
                configure_pip(auth_method=CodeArtifactAuthMethod.BOTO3)
        mock_cli.assert_not_called()

    def test_writes_pip_config(self, ca_env, mock_boto3_ca):
        """boto3 path persists the index to pip config for the uv bootstrap."""
        with mock.patch("subprocess.check_call") as mock_call:
            configure_pip(auth_method=CodeArtifactAuthMethod.BOTO3)
        mock_call.assert_called_once_with(_pip_config_set_cmd())


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
        """No CA, uv present, no pip index config → bare uv install, clean env."""
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(f"{_MODULE}.shutil.which", return_value=UV_PATH):
                with mock.patch(f"{_MODULE}._pip_config_get", return_value=None):
                    with mock.patch("subprocess.check_call") as mock_call:
                        install_requirements("reqs.txt")
        mock_call.assert_called_once_with(_uv_cmd(), env=mock.ANY)
        assert "UV_INDEX_URL" not in mock_call.call_args.kwargs["env"]

    def test_with_codeartifact_index_sets_uv_index_url(self):
        """boto3 index URL is propagated to uv via UV_INDEX_URL."""
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(f"{_MODULE}.configure_pip", return_value=EXPECTED_INDEX):
                with mock.patch(f"{_MODULE}.shutil.which", return_value=UV_PATH):
                    with mock.patch(f"{_MODULE}._pip_config_get", return_value=None):
                        with mock.patch("subprocess.check_call") as mock_call:
                            install_requirements("reqs.txt")
        mock_call.assert_called_once_with(_uv_cmd(), env=mock.ANY)
        assert mock_call.call_args.kwargs["env"]["UV_INDEX_URL"] == EXPECTED_INDEX

    def test_cli_fallback_index_recovered_from_pip_config(self):
        """CLI login returns no index but writes pip.conf → recovered into UV_INDEX_URL."""

        def fake_pip_config(_exe, key):
            return EXPECTED_INDEX if key == "global.index-url" else None

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(f"{_MODULE}.configure_pip", return_value=None):
                with mock.patch(f"{_MODULE}.shutil.which", return_value=UV_PATH):
                    with mock.patch(f"{_MODULE}._pip_config_get", side_effect=fake_pip_config):
                        with mock.patch("subprocess.check_call") as mock_call:
                            install_requirements("reqs.txt")
        assert mock_call.call_args.kwargs["env"]["UV_INDEX_URL"] == EXPECTED_INDEX

    def test_extra_index_and_trusted_host_propagated(self):
        """extra-index-url and trusted-host from pip config flow to uv env vars."""

        def fake_pip_config(_exe, key):
            return {
                "global.index-url": "https://primary/simple/",
                "global.extra-index-url": "https://extra/simple/",
                "global.trusted-host": "extra",
            }.get(key)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(f"{_MODULE}.configure_pip", return_value=None):
                with mock.patch(f"{_MODULE}.shutil.which", return_value=UV_PATH):
                    with mock.patch(f"{_MODULE}._pip_config_get", side_effect=fake_pip_config):
                        with mock.patch("subprocess.check_call") as mock_call:
                            install_requirements("reqs.txt")
        env = mock_call.call_args.kwargs["env"]
        assert env["UV_INDEX_URL"] == "https://primary/simple/"
        assert env["UV_EXTRA_INDEX_URL"] == "https://extra/simple/"
        assert env["UV_INSECURE_HOST"] == "extra"

    def test_bootstraps_uv_when_missing(self):
        """uv absent → pip install uv, then uv install."""
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(
                f"{_MODULE}.shutil.which", side_effect=[None, UV_PATH]
            ):
                with mock.patch(f"{_MODULE}._pip_config_get", return_value=None):
                    with mock.patch("subprocess.check_call") as mock_call:
                        install_requirements("reqs.txt", python_executable="/usr/bin/python3")
        bootstrap_call = mock.call(["/usr/bin/python3", "-m", "pip", "install", "uv"])
        install_call = mock.call(_uv_cmd(), env=mock.ANY)
        mock_call.assert_has_calls([bootstrap_call, install_call])

    def test_custom_python_executable(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(f"{_MODULE}.shutil.which", return_value=UV_PATH):
                with mock.patch(f"{_MODULE}._pip_config_get", return_value=None):
                    with mock.patch("subprocess.check_call") as mock_call:
                        install_requirements("reqs.txt", python_executable="/usr/bin/python3")
        mock_call.assert_called_once_with(_uv_cmd(), env=mock.ANY)

    def test_install_failure_propagates(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(f"{_MODULE}.shutil.which", return_value=UV_PATH):
                with mock.patch(f"{_MODULE}._pip_config_get", return_value=None):
                    with mock.patch(
                        "subprocess.check_call",
                        side_effect=subprocess.CalledProcessError(1, "uv"),
                    ):
                        with pytest.raises(subprocess.CalledProcessError):
                            install_requirements("reqs.txt")

    def test_auth_method_passed_through(self):
        with mock.patch(f"{_MODULE}.configure_pip", return_value=None) as mock_configure:
            with mock.patch(f"{_MODULE}.shutil.which", return_value=UV_PATH):
                with mock.patch(f"{_MODULE}._pip_config_get", return_value=None):
                    with mock.patch("subprocess.check_call"):
                        install_requirements("reqs.txt", auth_method=CodeArtifactAuthMethod.BOTO3)
        mock_configure.assert_called_once_with(
            auth_method=CodeArtifactAuthMethod.BOTO3, python_executable=sys.executable
        )


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
