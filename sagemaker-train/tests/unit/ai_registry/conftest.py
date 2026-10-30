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
"""Shared fixtures for the ai_registry unit test suite.

The ``_block_aws_access`` fixture applies to every test in this subtree
(autouse=True).  It strips real credential env vars and redirects boto3's
config/credentials file lookups to ``/dev/null`` so that:

* Tests always fail loudly (NoCredentialsError) when they accidentally reach
  a real AWS endpoint, regardless of whether the developer's shell happens
  to have credentials configured.
* The credential-free gate command and a plain ``pytest`` run produce
  identical results — passing credentials in the environment never silently
  hides a missing mock.

Any test that genuinely needs to interact with AWS (integration tests) lives
under ``tests/integ/``, not here, and carries its own credential fixtures.
"""

from __future__ import absolute_import

import pytest


@pytest.fixture(autouse=True)
def _block_aws_access(monkeypatch):
    """Prevent real AWS credential access in every ai_registry unit test.

    Mirrors the env-stripping done by the official gate command::

        env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \\
            -u AWS_SESSION_TOKEN -u AWS_PROFILE \\
            AWS_SHARED_CREDENTIALS_FILE=/dev/null \\
            AWS_CONFIG_FILE=/dev/null \\
            AWS_EC2_METADATA_DISABLED=true \\
            AWS_DEFAULT_REGION=us-west-2 \\
            python3 -m pytest tests/unit/ai_registry ...

    When a test needs to mock an AWS call, it must do so explicitly via
    ``@patch`` or ``monkeypatch``; it cannot rely on ambient credentials.
    If an existing test depends on something this guard removes, fix that
    test with a mock rather than exempting it from this fixture.
    """
    # Strip any real credential env vars that may be present in the shell.
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)

    # Redirect boto3's file-based credential/config lookups to /dev/null so
    # no credentials can be loaded from ~/.aws/credentials or ~/.aws/config.
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")

    # Disable the EC2 instance metadata service so boto3 cannot fall back to
    # IMDS when no static credentials are present.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    # Provide a synthetic default region so tests that inspect the region
    # (without making real calls) see a predictable value.
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
