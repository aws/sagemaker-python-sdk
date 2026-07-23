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
"""Shared pytest configuration for sagemaker-serve integration tests.

These tests run under ``pytest -n auto`` (dozens of xdist workers). Several of
them resolve/validate an IAM execution role during ``ModelBuilder.build()`` /
``.deploy()``, which internally calls the low-TPS ``iam:SimulatePrincipalPolicy``
API. With many workers hitting it at once IAM throttles the request, surfacing as
``ClientError: (Throttling) ... Rate exceeded`` and failing the build.

This is purely a test-harness concurrency problem, so the mitigation lives here in
the test layer rather than in SDK source:

* ``_configure_default_boto_retries`` (autouse) — set an adaptive retry policy on
  boto3's *default* session. The role resolver, when no explicit session is
  passed, falls back to ``sagemaker.core...Session()`` whose boto session is
  ``boto3.DEFAULT_SESSION`` (see session_helper._initialize). Clients created from
  it therefore inherit these retries, letting the internal IAM calls ride out
  transient throttling without any source change.

Throttling that still exhausts the adaptive retry budget is deliberately left to
fail the test loudly (rather than being converted to a skip), so a persistent
rate-limit regression stays visible instead of silently disappearing from the
results.
"""
from __future__ import absolute_import

import boto3
import pytest
from botocore.config import Config

# Adaptive retry budget for throttling-prone IAM validation calls.
_ADAPTIVE_RETRY_CONFIG = Config(retries={"max_attempts": 10, "mode": "adaptive"})


@pytest.fixture(autouse=True, scope="session")
def _configure_default_boto_retries():
    """Give boto3's default session adaptive retries so IAM clients built by the
    role resolver absorb transient SimulatePrincipalPolicy throttling."""
    boto3.setup_default_session()
    boto3.DEFAULT_SESSION._session.set_default_client_config(_ADAPTIVE_RETRY_CONFIG)
    yield
